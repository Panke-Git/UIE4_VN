#!/usr/bin/env python3
"""Screen a complete v16 LSUI test split for paper-worthy UICF responses."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.uicf_inr import UICFINROutput
from src.shared.uicf_models import UICFPreBackbone
from src.v16.dataset import LSUIDataset, ManifestEntry, validate_split_protocol
from src.v16.metrics import batch_metrics
from src.v16.test import _checkpoint_path, _torch_load
from src.v16.utils import (
    atomic_json,
    load_yaml,
    project_path,
    seed_everything,
    select_device,
    sha256_file,
    tensor_to_image,
)
from tools.visualize_v16_uicf import (
    CapturedSample,
    _diverging_rgb,
    _safe_sample_id,
    assert_uicf_consistency,
    build_and_load_v16_model,
    capture_sample,
    save_visualization_sample,
    split_correction_channels,
    symmetric_heatmap_range,
)


SCRIPT_VERSION = "1.0"
EXPECTED_VERSION = "v16"
EPSILON = 1e-12
DEFAULT_VIZ_WEIGHTS = {
    "spatial": 0.45,
    "channel": 0.35,
    "coherence": 0.15,
    "delta": 0.05,
}
CHANNEL_KEYS = ("R_r", "R_g", "R_b")
CHANNEL_STATISTICS = ("min", "max", "mean", "std", "mean_abs", "p05", "p50", "p95", "robust_range")
PERCENTILE_METRICS = {
    "P_spatial": "spatial_nonuniformity",
    "P_channel": "channel_specificity",
    "P_coherence": "spatial_coherence",
    "P_delta": "image_correction_strength",
}
BASE_FIELDS = ["sample_index", "sample_id", "filename"]
CHANNEL_FIELDS = [
    f"{channel}_{statistic}"
    for channel in CHANNEL_KEYS
    for statistic in CHANNEL_STATISTICS
]
METRIC_FIELDS = [
    *BASE_FIELDS,
    *CHANNEL_FIELDS,
    "field_mean_abs",
    "field_std",
    "spatial_nonuniformity",
    "spatial_std",
    "spatial_robust_range",
    "channel_specificity",
    "channel_difference",
    "D_rg",
    "D_rb",
    "D_gb",
    "channel_mean_spread",
    "spatial_coherence",
    "correction_magnitude",
    "image_correction_strength",
    "b_r",
    "b_g",
    "b_b",
    "anchor_mean",
    "anchor_std",
    "anchor_channel_spread",
    "nonfinite_ratio",
    "field_abs_max",
    "psnr",
    "ssim",
    "thumbnail_vmin",
    "thumbnail_vmax",
    "thumbnail_path",
]
RANKING_FIELDS = [
    "rank",
    *BASE_FIELDS,
    "viz_score",
    "P_spatial",
    "P_channel",
    "P_coherence",
    "P_delta",
    *[field for field in METRIC_FIELDS if field not in BASE_FIELDS],
    "top_folder",
]
FAILED_FIELDS = [
    "sample_index",
    "sample_id",
    "filename",
    "error_type",
    "error",
    "nonfinite_ratio",
    "field_abs_max",
]


class SafeTestDataset(Dataset[dict[str, Any]]):
    """Turn individual image-loading failures into serializable result rows."""

    def __init__(self, dataset: LSUIDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.dataset.entries[index]
        try:
            item = self.dataset[index]
        except Exception as error:
            return {
                "ok": False,
                "index": index,
                "sample_id": entry.sample_id,
                "filename": Path(entry.input_relative).name,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        return {"ok": True, "index": index, **item}


def _list_collate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank the complete LSUI test split by UICF visualization value"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config AMP for inference (CUDA only)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--thumbnail-size", type=int, default=180)
    parser.add_argument("--robust-percentile", type=float, default=99.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _channel_statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    percentiles = np.percentile(array, (5.0, 50.0, 95.0))
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "mean_abs": float(np.abs(array).mean()),
        "p05": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "robust_range": float(percentiles[2] - percentiles[0]),
    }


def compute_spatial_coherence(correction_field: np.ndarray, eps: float = EPSILON) -> float:
    field = np.asarray(correction_field, dtype=np.float32)
    split_correction_channels(field)
    tensor = torch.from_numpy(field).unsqueeze(0)
    padded = F.pad(tensor, (2, 2, 2, 2), mode="replicate")
    smooth = F.avg_pool2d(padded, kernel_size=5, stride=1)
    original_variance = tensor.var(dim=(-2, -1), unbiased=False)
    smooth_variance = smooth.var(dim=(-2, -1), unbiased=False)
    coherence = torch.where(
        original_variance > eps,
        smooth_variance / (original_variance + eps),
        torch.zeros_like(original_variance),
    )
    return float(coherence.clamp(0.0, 1.0).mean())


def compute_field_metrics(
    correction_field: np.ndarray,
    input_image: np.ndarray,
    corrected_image: np.ndarray,
    chromatic_anchor: np.ndarray,
    *,
    eps: float = EPSILON,
) -> dict[str, float]:
    """Compute raw, unnormalized UICF diagnostics for one sample."""
    field = np.asarray(correction_field, dtype=np.float64)
    inputs = np.asarray(input_image, dtype=np.float64)
    corrected = np.asarray(corrected_image, dtype=np.float64)
    anchor = np.asarray(chromatic_anchor, dtype=np.float64).reshape(-1)
    if field.ndim != 3 or field.shape[0] != 3:
        raise ValueError(f"Expected correction field [3,H,W], got {field.shape}")
    if inputs.shape != field.shape or corrected.shape != field.shape:
        raise ValueError(
            f"Input/Ic/field shapes differ: input={inputs.shape}, Ic={corrected.shape}, R={field.shape}"
        )
    if anchor.shape != (3,):
        raise ValueError(f"Expected chromatic anchor [3], got {anchor.shape}")
    arrays = (field, inputs, corrected, anchor)
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError("UICF sample contains non-finite values")

    channels = split_correction_channels(field)
    channel_stats = [_channel_statistics(channel) for channel in channels]
    result: dict[str, float] = {}
    for key, statistics in zip(CHANNEL_KEYS, channel_stats, strict=True):
        result.update({f"{key}_{name}": value for name, value in statistics.items()})

    field_mean_abs = float(np.abs(field).mean())
    robust_ranges = [statistics["robust_range"] for statistics in channel_stats]
    channel_means = [statistics["mean"] for statistics in channel_stats]
    d_rg = float(np.abs(channels[0] - channels[1]).mean())
    d_rb = float(np.abs(channels[0] - channels[2]).mean())
    d_gb = float(np.abs(channels[1] - channels[2]).mean())
    channel_difference = (d_rg + d_rb + d_gb) / 3.0
    spatial_robust_range = float(np.mean(robust_ranges))
    result.update(
        {
            "field_mean_abs": field_mean_abs,
            "field_std": float(field.std()),
            "spatial_nonuniformity": spatial_robust_range / (field_mean_abs + eps),
            "spatial_std": float(np.mean([statistics["std"] for statistics in channel_stats])),
            "spatial_robust_range": spatial_robust_range,
            "channel_specificity": channel_difference / (field_mean_abs + eps),
            "channel_difference": channel_difference,
            "D_rg": d_rg,
            "D_rb": d_rb,
            "D_gb": d_gb,
            "channel_mean_spread": float(np.std(channel_means)),
            "spatial_coherence": compute_spatial_coherence(field, eps),
            "correction_magnitude": field_mean_abs,
            "image_correction_strength": float(np.abs(corrected - inputs).mean()),
            "b_r": float(anchor[0]),
            "b_g": float(anchor[1]),
            "b_b": float(anchor[2]),
            "anchor_mean": float(anchor.mean()),
            "anchor_std": float(anchor.std()),
            "anchor_channel_spread": float(anchor.max() - anchor.min()),
            "nonfinite_ratio": 0.0,
            "field_abs_max": float(np.abs(field).max()),
        }
    )
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Computed UICF metrics contain non-finite values")
    return result


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Average-rank percentiles in [0,1], with deterministic tie handling."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Percentile ranks require a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise FloatingPointError("Percentile-rank inputs must be finite")
    if array.size == 1:
        return [1.0]
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        average_position = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_position / (array.size - 1)
        start = end
    return [float(value) for value in ranks]


def rank_samples(
    rows: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float] = DEFAULT_VIZ_WEIGHTS,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot rank an empty sample set")
    required_weights = {"spatial", "channel", "coherence", "delta"}
    if set(weights) != required_weights or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Visualization weights must contain four terms summing to 1")
    ranked = [dict(row) for row in rows]
    for percentile_key, metric_key in PERCENTILE_METRICS.items():
        values = [float(row[metric_key]) for row in ranked]
        for row, percentile in zip(ranked, percentile_ranks(values), strict=True):
            row[percentile_key] = percentile
    for row in ranked:
        row["viz_score"] = float(
            weights["spatial"] * row["P_spatial"]
            + weights["channel"] * row["P_channel"]
            + weights["coherence"] * row["P_coherence"]
            + weights["delta"] * row["P_delta"]
        )
    ranked.sort(key=lambda row: (-row["viz_score"], int(row["sample_index"]), row["sample_id"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked


def _finite_field_diagnostics(field: Tensor) -> tuple[float, float | None]:
    finite = torch.isfinite(field)
    nonfinite_ratio = 1.0 - float(finite.float().mean().detach().cpu())
    finite_values = field[finite]
    field_abs_max = (
        None
        if finite_values.numel() == 0
        else float(finite_values.abs().max().detach().float().cpu())
    )
    return nonfinite_ratio, field_abs_max


def _validate_batch_shapes(inputs: Tensor, prediction: Any, details: Any) -> str | None:
    if not isinstance(prediction, Tensor):
        raise TypeError("v16 diagnostics forward did not return a prediction tensor")
    if not isinstance(details, UICFINROutput):
        raise TypeError(f"Expected UICFINROutput, got {type(details).__name__}")
    expected = tuple(inputs.shape)
    image_shapes = {
        "prediction": tuple(prediction.shape),
        "enhanced": tuple(details.enhanced.shape),
        "correction_field": tuple(details.correction_field.shape),
    }
    for name, shape in image_shapes.items():
        if shape != expected:
            return f"{name} shape is {shape}, expected {expected}"
    if tuple(details.chromatic_anchor.shape) != (inputs.shape[0], 3):
        return (
            f"chromatic_anchor shape is {tuple(details.chromatic_anchor.shape)}, "
            f"expected {(inputs.shape[0], 3)}"
        )
    if details.global_feature.ndim != 2 or details.global_feature.shape[0] != inputs.shape[0]:
        return f"global_feature must be [B,C], got {tuple(details.global_feature.shape)}"
    return None


def _slice_details(details: UICFINROutput, position: int) -> UICFINROutput:
    selection = slice(position, position + 1)
    return UICFINROutput(
        enhanced=details.enhanced[selection],
        correction_field=details.correction_field[selection],
        chromatic_anchor=details.chromatic_anchor[selection],
        global_feature=details.global_feature[selection],
    )


def _center_text(draw: ImageDraw.ImageDraw, center: tuple[int, int], text: str) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text)
    draw.text(
        (center[0] - (right - left) // 2, center[1] - (bottom - top) // 2),
        text,
        fill="black",
    )


def build_thumbnail(
    sample: CapturedSample,
    path: Path,
    *,
    thumbnail_size: int,
    robust_percentile: float,
) -> tuple[float, float]:
    if thumbnail_size < 64:
        raise ValueError("--thumbnail-size must be at least 64")
    field = sample.correction_field.numpy()
    channels = split_correction_channels(field)
    vmin, vmax = symmetric_heatmap_range([field], robust_percentile)
    source_images = [
        tensor_to_image(sample.input_tensor),
        *[Image.fromarray(_diverging_rgb(channel, vmin, vmax), mode="RGB") for channel in channels],
        tensor_to_image(sample.enhanced),
    ]
    width, height = source_images[0].size
    cell_height = thumbnail_size
    cell_width = max(1, int(round(width * cell_height / height)))
    images = [
        image.resize((cell_width, cell_height), Image.Resampling.BILINEAR)
        for image in source_images
    ]
    title_height, gap = 20, 3
    panel = Image.new(
        "RGB", (5 * cell_width + 4 * gap, title_height + cell_height), "white"
    )
    draw = ImageDraw.Draw(panel)
    for column, (image, title) in enumerate(
        zip(images, ("Input", "R_r", "R_g", "R_b", "I^c"), strict=True)
    ):
        x = column * (cell_width + gap)
        panel.paste(image, (x, title_height))
        _center_text(draw, (x + cell_width // 2, title_height // 2), title)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path, dpi=(150, 150))
    return vmin, vmax


def _failure_row(
    entry: ManifestEntry,
    index: int,
    error: Exception | str,
    *,
    error_type: str | None = None,
    nonfinite_ratio: float | None = None,
    field_abs_max: float | None = None,
) -> dict[str, Any]:
    return {
        "sample_index": index,
        "sample_id": entry.sample_id,
        "filename": Path(entry.input_relative).name,
        "error_type": error_type or type(error).__name__,
        "error": str(error),
        "nonfinite_ratio": nonfinite_ratio,
        "field_abs_max": field_abs_max,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    for row in rows:
        for value in row.values():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                raise FloatingPointError(f"Refusing to serialize non-finite CSV value in {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def export_ranking_html(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    cards = []
    for row in rows:
        thumbnail = html.escape(str(row["thumbnail_path"]), quote=True)
        sample_id = html.escape(str(row["sample_id"]))
        cards.append(
            f"""
            <article class="card">
              <h2>Rank #{int(row['rank'])} · {sample_id}</h2>
              <img src="{thumbnail}" alt="UICF thumbnail for {sample_id}">
              <dl>
                <dt>Viz score</dt><dd>{float(row['viz_score']):.6f}</dd>
                <dt>Spatial</dt><dd>{float(row['P_spatial']):.6f}</dd>
                <dt>Channel</dt><dd>{float(row['P_channel']):.6f}</dd>
                <dt>Coherence</dt><dd>{float(row['P_coherence']):.6f}</dd>
                <dt>Image correction</dt><dd>{float(row['P_delta']):.6f}</dd>
              </dl>
            </article>"""
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v16 UICF screening ranking</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #f3f5f7; color: #17202a; font-family: Arial, sans-serif; }}
    h1 {{ margin: 0 0 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(620px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #d8dee4; border-radius: 8px; padding: 14px; box-shadow: 0 2px 7px #00000012; }}
    .card h2 {{ font-size: 17px; margin: 0 0 10px; }}
    .card img {{ display: block; width: 100%; height: auto; background: #eee; }}
    dl {{ display: grid; grid-template-columns: repeat(5, max-content); gap: 4px 14px; margin: 10px 0 0; }}
    dt {{ font-weight: bold; }} dd {{ margin: 0; }}
  </style>
</head>
<body>
  <h1>v16 UICF correction-field screening</h1>
  <main class="grid">{''.join(cards)}</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def build_contact_sheet(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, path: Path, columns: int = 4
) -> None:
    if not rows:
        raise ValueError("Contact sheet requires at least one candidate")
    columns = max(1, min(columns, len(rows)))
    thumbnails: list[Image.Image] = []
    for row in rows:
        with Image.open(output_dir / str(row["thumbnail_path"])) as image:
            thumbnails.append(image.convert("RGB"))
    image_width = max(image.width for image in thumbnails)
    image_height = max(image.height for image in thumbnails)
    label_height, gap = 30, 10
    card_width, card_height = image_width, label_height + image_height
    rows_count = math.ceil(len(thumbnails) / columns)
    sheet = Image.new(
        "RGB",
        (
            columns * card_width + (columns - 1) * gap,
            rows_count * card_height + (rows_count - 1) * gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for position, (row, image) in enumerate(zip(rows, thumbnails, strict=True)):
        grid_row, grid_column = divmod(position, columns)
        x = grid_column * (card_width + gap)
        y = grid_row * (card_height + gap)
        sheet.paste(image, (x, y + label_height))
        label = f"Rank {int(row['rank']):02d}  {row['sample_id']}"
        draw.text((x + 5, y + 8), label, fill="black")
    sheet.save(path, dpi=(200, 200))


def _prepare_output_directory(
    path: Path,
    *,
    overwrite: bool,
    run_dir: Path,
    data_root: Path,
) -> None:
    resolved = path.resolve()
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
        run_dir.resolve(),
        (run_dir / "result").resolve(),
        data_root.resolve(),
    }
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {path}")
        contains_protected_path = any(
            protected_path == resolved or protected_path.is_relative_to(resolved)
            for protected_path in protected
        )
        default_output = (run_dir / "result" / "uicf_screening").resolve()
        custom_output_is_unmarked = (
            resolved != default_output
            and not (resolved / "screening_config.json").is_file()
        )
        if contains_protected_path or len(resolved.parts) < 4 or custom_output_is_unmarked:
            raise ValueError(f"Refusing to overwrite protected directory: {path}")
        shutil.rmtree(path)
    (path / "thumbnails").mkdir(parents=True)


def _group_valid_items(items: Sequence[dict[str, Any]]) -> dict[tuple[int, ...], list[dict[str, Any]]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for item in items:
        input_tensor, target_tensor = item.get("input"), item.get("target")
        if not isinstance(input_tensor, Tensor) or not isinstance(target_tensor, Tensor):
            item["shape_error"] = "Dataset input/target is not a tensor"
            continue
        if input_tensor.ndim != 3 or input_tensor.shape[0] != 3:
            item["shape_error"] = f"Input must be [3,H,W], got {tuple(input_tensor.shape)}"
            continue
        if tuple(target_tensor.shape) != tuple(input_tensor.shape):
            item["shape_error"] = (
                f"GT shape {tuple(target_tensor.shape)} differs from input {tuple(input_tensor.shape)}"
            )
            continue
        groups.setdefault(tuple(input_tensor.shape), []).append(item)
    return groups


def screen_test_set(
    model: UICFPreBackbone,
    dataset: LSUIDataset,
    entries: Sequence[ManifestEntry],
    device: torch.device,
    output_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    amp_enabled: bool,
    thumbnail_size: int,
    robust_percentile: float,
    metric_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    safe_dataset = SafeTestDataset(dataset)
    loader = DataLoader(
        safe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_list_collate,
    )
    metrics_rows: list[dict[str, Any]] = []
    failures_by_index: dict[int, dict[str, Any]] = {}
    normal_forward_checked = False
    index_width = max(4, len(str(len(entries))))
    progress = tqdm(
        total=len(entries),
        desc="[Phase 1/2] Screening LSUI test set",
        unit="sample",
    )
    forward = getattr(model, "forward_with_uicf_details", None)
    if not callable(forward):
        raise TypeError("v16 model lacks forward_with_uicf_details")
    for loaded_items in loader:
        progress.update(len(loaded_items))
        valid_items: list[dict[str, Any]] = []
        for item in loaded_items:
            index = int(item["index"])
            if not item["ok"]:
                failures_by_index[index] = _failure_row(
                    entries[index],
                    index,
                    item["error"],
                    error_type=str(item["error_type"]),
                )
            else:
                valid_items.append(item)
        groups = _group_valid_items(valid_items)
        for item in valid_items:
            if "shape_error" in item:
                index = int(item["index"])
                failures_by_index[index] = _failure_row(
                    entries[index], index, str(item["shape_error"]), error_type="ShapeError"
                )
        for group in groups.values():
            inputs = torch.stack([item["input"] for item in group]).to(
                device, non_blocking=True
            )
            targets = torch.stack([item["target"] for item in group]).to(
                device, non_blocking=True
            )
            try:
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, enabled=amp_enabled
                ):
                    prediction, details = forward(inputs)
            except Exception as error:
                if isinstance(error, (TypeError, AttributeError)):
                    progress.close()
                    raise
                for item in group:
                    index = int(item["index"])
                    failures_by_index[index] = _failure_row(entries[index], index, error)
                continue
            shape_error = _validate_batch_shapes(inputs, prediction, details)
            if shape_error is not None:
                for item in group:
                    index = int(item["index"])
                    failures_by_index[index] = _failure_row(
                        entries[index], index, shape_error, error_type="ShapeError"
                    )
                continue

            for position, item in enumerate(group):
                index = int(item["index"])
                entry = entries[index]
                sample_details = _slice_details(details, position)
                nonfinite_ratio, field_abs_max = _finite_field_diagnostics(
                    sample_details.correction_field
                )
                sample_tensors = (
                    inputs[position : position + 1],
                    targets[position : position + 1],
                    prediction[position : position + 1],
                    sample_details.enhanced,
                    sample_details.chromatic_anchor,
                    sample_details.global_feature,
                )
                if nonfinite_ratio > 0.0 or not all(
                    torch.isfinite(tensor).all() for tensor in sample_tensors
                ):
                    failures_by_index[index] = _failure_row(
                        entry,
                        index,
                        "Non-finite model diagnostics",
                        error_type="FloatingPointError",
                        nonfinite_ratio=nonfinite_ratio,
                        field_abs_max=field_abs_max,
                    )
                    continue

                # Formula disagreement and normal/diagnostics disagreement invalidate
                # the whole scientific export, so these checks intentionally abort.
                prediction_normal = None
                if not normal_forward_checked:
                    with torch.inference_mode(), torch.autocast(
                        device_type=device.type, enabled=amp_enabled
                    ):
                        prediction_normal = model(inputs[position : position + 1])
                assert_uicf_consistency(
                    inputs[position : position + 1],
                    prediction[position : position + 1],
                    sample_details,
                    prediction_normal,
                )
                normal_forward_checked = True

                sample = CapturedSample(
                    index=index,
                    sample_id=entry.sample_id,
                    filename=Path(entry.input_relative).name,
                    input_tensor=inputs[position].detach().float().cpu(),
                    target_tensor=targets[position].detach().float().cpu(),
                    prediction=prediction[position].detach().float().cpu(),
                    enhanced=sample_details.enhanced[0].detach().float().cpu(),
                    correction_field=sample_details.correction_field[0].detach().float().cpu(),
                    chromatic_anchor=sample_details.chromatic_anchor[0].detach().float().cpu(),
                    global_feature=sample_details.global_feature[0].detach().float().cpu(),
                )
                thumbnail_name = (
                    f"{index + 1:0{index_width}d}_{_safe_sample_id(entry.sample_id)}.png"
                )
                thumbnail_relative = Path("thumbnails") / thumbnail_name
                try:
                    row = {
                        "sample_index": index,
                        "sample_id": entry.sample_id,
                        "filename": sample.filename,
                        **compute_field_metrics(
                            sample.correction_field.numpy(),
                            sample.input_tensor.numpy(),
                            sample.enhanced.numpy(),
                            sample.chromatic_anchor.numpy(),
                        ),
                    }
                    psnr, ssim = batch_metrics(
                        prediction[position : position + 1].float().clamp(0.0, 1.0),
                        targets[position : position + 1].float(),
                        dict(metric_config),
                    )
                    row["psnr"] = float(psnr[0])
                    row["ssim"] = float(ssim[0])
                    thumbnail_vmin, thumbnail_vmax = build_thumbnail(
                        sample,
                        output_dir / thumbnail_relative,
                        thumbnail_size=thumbnail_size,
                        robust_percentile=robust_percentile,
                    )
                    row["thumbnail_vmin"] = thumbnail_vmin
                    row["thumbnail_vmax"] = thumbnail_vmax
                    row["thumbnail_path"] = thumbnail_relative.as_posix()
                    metrics_rows.append(row)
                except Exception as error:
                    (output_dir / thumbnail_relative).unlink(missing_ok=True)
                    failures_by_index[index] = _failure_row(
                        entry,
                        index,
                        error,
                        nonfinite_ratio=nonfinite_ratio,
                        field_abs_max=field_abs_max,
                    )
    progress.close()
    metrics_rows.sort(key=lambda row: int(row["sample_index"]))
    failures = [failures_by_index[index] for index in sorted(failures_by_index)]
    if len(metrics_rows) + len(failures) != len(entries):
        raise RuntimeError(
            "Processed sample count mismatch: "
            f"success={len(metrics_rows)} failed={len(failures)} expected={len(entries)}"
        )
    if metrics_rows and not normal_forward_checked:
        raise RuntimeError("No valid sample was available for normal-forward equivalence check")
    return metrics_rows, failures


def _write_top_summary(
    rows: Sequence[Mapping[str, Any]], csv_path: Path, text_path: Path
) -> None:
    fields = [
        "rank",
        "sample_index",
        "sample_id",
        "filename",
        "viz_score",
        "spatial_nonuniformity",
        "channel_specificity",
        "spatial_coherence",
        "image_correction_strength",
        "b_r",
        "b_g",
        "b_b",
        "top_folder",
    ]
    _write_csv(csv_path, rows, fields)
    blocks = []
    for row in rows:
        blocks.append(
            "\n".join(
                (
                    f"Rank {int(row['rank'])}",
                    f"sample_id: {row['sample_id']}",
                    f"viz_score: {float(row['viz_score']):.9f}",
                    f"spatial_nonuniformity: {float(row['spatial_nonuniformity']):.9f}",
                    f"channel_specificity: {float(row['channel_specificity']):.9f}",
                    f"spatial_coherence: {float(row['spatial_coherence']):.9f}",
                    f"b: [{float(row['b_r']):.9f}, {float(row['b_g']):.9f}, {float(row['b_b']):.9f}]",
                    f"folder: {row['top_folder']}",
                )
            )
        )
    text_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def export_top_k(
    model: UICFPreBackbone,
    dataset: LSUIDataset,
    ranked_rows: Sequence[dict[str, Any]],
    device: torch.device,
    output_dir: Path,
    *,
    checkpoint_path: Path,
    checkpoint_selector: str,
    checkpoint_epoch: int,
    amp_enabled: bool,
    robust_percentile: float,
) -> list[dict[str, Any]]:
    top_rows = [dict(row) for row in ranked_rows]
    top_dir_name = f"top{len(top_rows)}"
    top_dir = output_dir / top_dir_name
    top_dir.mkdir()
    progress = tqdm(top_rows, desc=f"[Phase 2/2] Exporting Top-{len(top_rows)}", unit="sample")
    for row in progress:
        rank = int(row["rank"])
        sample = capture_sample(
            model,
            dataset,
            int(row["sample_index"]),
            device,
            amp_enabled=amp_enabled,
            compare_normal_forward=False,
        )
        vmin, vmax = symmetric_heatmap_range(
            [sample.correction_field.numpy()], robust_percentile
        )
        folder_name = f"rank{rank:02d}_{_safe_sample_id(sample.sample_id)}"
        row["top_folder"] = f"{top_dir_name}/{folder_name}"
        folder = top_dir / folder_name
        metadata = save_visualization_sample(
            sample,
            folder,
            checkpoint_path=checkpoint_path,
            checkpoint_selector=checkpoint_selector,
            checkpoint_epoch=checkpoint_epoch,
            heatmap_vmin=vmin,
            heatmap_vmax=vmax,
            robust_percentile=robust_percentile,
        )
        metadata["screening"] = {
            "rank": rank,
            "viz_score": float(row["viz_score"]),
            "P_spatial": float(row["P_spatial"]),
            "P_channel": float(row["P_channel"]),
            "P_coherence": float(row["P_coherence"]),
            "P_delta": float(row["P_delta"]),
            "spatial_nonuniformity": float(row["spatial_nonuniformity"]),
            "channel_specificity": float(row["channel_specificity"]),
            "spatial_coherence": float(row["spatial_coherence"]),
            "image_correction_strength": float(row["image_correction_strength"]),
        }
        atomic_json(folder / "uicf_metadata.json", metadata)
    return top_rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.thumbnail_size < 64:
        raise ValueError("--thumbnail-size must be at least 64")
    if not 0.0 < args.robust_percentile <= 100.0:
        raise ValueError("--robust-percentile must be in (0,100]")

    run_dir = project_path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run lacks config_resolved.yaml: {run_dir}")
    config = load_yaml(config_path)
    if config["experiment"]["version"] != EXPECTED_VERSION:
        raise ValueError(
            f"{EXPECTED_VERSION} screening refuses run version "
            f"{config['experiment']['version']!r}"
        )
    dataset_name = config["data"].get("dataset")
    if dataset_name not in (None, "LSUI19"):
        raise ValueError(f"This tool screens LSUI runs, got data.dataset={dataset_name!r}")
    if args.data_root is not None:
        config["data"]["root"] = str(Path(args.data_root).expanduser().resolve())
    data_root = Path(config["data"]["root"]).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data.root is unavailable: {data_root}")

    selector = args.checkpoint or config["test"]["checkpoint"]
    checkpoint_path = _checkpoint_path(run_dir, selector)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    snapshot = run_dir / "split_snapshot"
    manifests = {
        name: snapshot / f"{name}.tsv" for name in ("train", "validation", "test")
    }
    split_entries = validate_split_protocol(
        manifests, config["data"].get("expected_counts")
    )
    test_entries = split_entries["test"]
    data = config["data"]
    test_dataset = LSUIDataset(
        manifests["test"],
        data_root,
        "test",
        int(data["patch_size"]),
        data["augmentation"],
        bool(data["pad_if_smaller"]),
        str(data["pad_mode"]),
        config["evaluation"],
        verify_files=False,
    )
    if test_dataset.entries != test_entries:
        raise RuntimeError("Validated test manifest order differs from Dataset order")

    output_dir = (
        (run_dir / "result" / "uicf_screening")
        if args.output_dir is None
        else project_path(args.output_dir).expanduser().resolve()
    )
    output_dir = output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists; pass --overwrite to replace it: {output_dir}"
        )
    device = select_device(args.gpu)
    seed_everything(int(config["experiment"]["seed"]), deterministic=True)
    checkpoint = _torch_load(checkpoint_path, device)
    model = build_and_load_v16_model(config, checkpoint, device)
    if not isinstance(model, UICFPreBackbone):
        raise TypeError(f"Expected UICFPreBackbone, got {type(model).__name__}")
    amp_requested = bool(config["training"]["amp"]) if args.amp is None else args.amp
    amp_enabled = amp_requested and device.type == "cuda"
    num_workers = int(data["num_workers"]) if args.num_workers is None else args.num_workers
    if num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    _prepare_output_directory(
        output_dir,
        overwrite=args.overwrite,
        run_dir=run_dir,
        data_root=data_root,
    )

    metrics_rows, failures = screen_test_set(
        model,
        test_dataset,
        test_entries,
        device,
        output_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
        amp_enabled=amp_enabled,
        thumbnail_size=args.thumbnail_size,
        robust_percentile=args.robust_percentile,
        metric_config=config["metrics"],
    )
    if not metrics_rows:
        _write_csv(output_dir / "failed_samples.csv", failures, FAILED_FIELDS)
        raise RuntimeError("Every test sample failed; no ranking can be generated")
    ranked = rank_samples(metrics_rows)
    # Persist Phase-1 evidence before the Top-K pass. If a later export fails,
    # the complete screening metrics and per-sample failure audit remain usable.
    _write_csv(output_dir / "all_samples_metrics.csv", metrics_rows, METRIC_FIELDS)
    _write_csv(output_dir / "ranking.csv", ranked, RANKING_FIELDS)
    _write_csv(output_dir / "failed_samples.csv", failures, FAILED_FIELDS)
    export_ranking_html(ranked, output_dir / "ranking.html")
    top_count = min(args.top_k, len(ranked))
    top_preliminary = ranked[:top_count]
    top_rows = export_top_k(
        model,
        test_dataset,
        top_preliminary,
        device,
        output_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_selector=selector,
        checkpoint_epoch=int(checkpoint["epoch"]),
        amp_enabled=amp_enabled,
        robust_percentile=args.robust_percentile,
    )
    top_folder_by_index = {
        int(row["sample_index"]): row["top_folder"] for row in top_rows
    }
    for row in ranked:
        row["top_folder"] = top_folder_by_index.get(int(row["sample_index"]), "")

    top_label = f"top{top_count}"
    _write_csv(output_dir / "ranking.csv", ranked, RANKING_FIELDS)
    _write_csv(output_dir / f"{top_label}.csv", top_rows, RANKING_FIELDS)
    _write_top_summary(
        top_rows,
        output_dir / f"{top_label}_summary.csv",
        output_dir / f"{top_label}_summary.txt",
    )
    build_contact_sheet(
        top_rows,
        output_dir,
        output_dir / f"{top_label}_contact_sheet.png",
    )

    processed_count = len(metrics_rows) + len(failures)
    if processed_count != len(test_entries):
        raise RuntimeError(
            f"Final processed count {processed_count} differs from test manifest {len(test_entries)}"
        )
    screening_config = {
        "script": "tools/visualize_uicf_field.py",
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_selector": selector,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_manifest": str(manifests["test"]),
        "test_manifest_sha256": sha256_file(manifests["test"]),
        "data_root": str(data_root),
        "ranking_weights": DEFAULT_VIZ_WEIGHTS,
        "seed": int(config["experiment"]["seed"]),
        "robust_percentile": float(args.robust_percentile),
        "top_k_requested": int(args.top_k),
        "top_k_exported": top_count,
        "thumbnail_size": int(args.thumbnail_size),
        "batch_size": int(args.batch_size),
        "num_workers": num_workers,
        "amp_requested": bool(amp_requested),
        "amp_enabled": bool(amp_enabled),
        "total_test_samples": len(test_entries),
        "processed_sample_count": processed_count,
        "successful_sample_count": len(metrics_rows),
        "failed_sample_count": len(failures),
        "formula_sanity_check": "PASS for every successful Phase-1 and Top-K inference",
        "normal_forward_equivalence": "PASS for first successful sample",
        "deterministic_ranking": True,
        "score_uses_psnr_or_ssim": False,
    }
    atomic_json(output_dir / "screening_config.json", screening_config)
    best = ranked[0]
    print(
        "\nUICF screening completed\n"
        f"Total samples: {len(test_entries)}\n"
        f"Successful samples: {len(metrics_rows)}\n"
        f"Failed samples: {len(failures)}\n"
        f"Top-K: {top_count}\n"
        f"Output directory: {output_dir}\n"
        f"Best candidate sample id: {best['sample_id']}\n"
        f"Best viz score: {float(best['viz_score']):.9f}"
    )


if __name__ == "__main__":
    main()
