#!/usr/bin/env python3
"""Visualize the internal UICF response of an existing v16 test checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.uicf_inr import UICFINROutput
from src.shared.uicf_models import UICFPreBackbone
from src.v16.dataset import LSUIDataset, ManifestEntry, validate_split_protocol
from src.v16.models import build_model
from src.v16.test import _checkpoint_path, _torch_load
from src.v16.utils import atomic_json, load_yaml, project_path, select_device, tensor_to_image


EXPECTED_VERSION = "v16"
CHANNEL_KEYS = ("R_r", "R_g", "R_b")
CHANNEL_TITLES = ("R_r(x)", "R_g(x)", "R_b(x)")
RGB_TITLES = ("Input", "Corrected I^c")
HEATMAP_DPI = (300, 300)


@dataclass(frozen=True)
class CapturedSample:
    index: int
    sample_id: str
    filename: str
    input_tensor: Tensor
    target_tensor: Tensor
    prediction: Tensor
    enhanced: Tensor
    correction_field: Tensor
    chromatic_anchor: Tensor
    global_feature: Tensor


def _comma_separated_indices(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--indices must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("--indices must contain at least one index")
    return values


def _comma_separated_ids(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--sample-ids must contain at least one sample id")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize UICF correction fields from an existing v16 checkpoint"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-samples", type=int, default=10)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--indices", type=_comma_separated_indices)
    selection.add_argument("--sample-ids", type=_comma_separated_ids)
    parser.add_argument(
        "--robust-percentile",
        type=float,
        default=None,
        help="Optional percentile of |R| used only for the shared symmetric display range",
    )
    return parser.parse_args(argv)


def select_sample_indices(
    entries: Sequence[ManifestEntry],
    *,
    indices: Sequence[int] | None,
    sample_ids: Sequence[str] | None,
    num_samples: int,
    random_seed: int,
) -> list[int]:
    """Resolve exactly one deterministic sample-selection mode."""
    if indices is not None and sample_ids is not None:
        raise ValueError("Specify either indices or sample_ids, not both")
    if indices is not None:
        selected = list(indices)
        if len(selected) != len(set(selected)):
            raise ValueError("--indices contains duplicates")
        invalid = [index for index in selected if index < 0 or index >= len(entries)]
        if invalid:
            raise IndexError(f"Test dataset indices out of range: {invalid}")
        return selected
    if sample_ids is not None:
        requested = list(sample_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("--sample-ids contains duplicates")
        by_id = {entry.sample_id: index for index, entry in enumerate(entries)}
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise KeyError(f"Sample ids are absent from test split: {missing}")
        return [by_id[sample_id] for sample_id in requested]
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    count = min(num_samples, len(entries))
    return random.Random(random_seed).sample(range(len(entries)), count)


def split_correction_channels(correction_field: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return R_r, R_g, and R_b without rescaling or modifying their values."""
    field = np.asarray(correction_field)
    if field.ndim != 3 or field.shape[0] != 3:
        raise ValueError(f"Expected correction field [3,H,W], got {field.shape}")
    if not np.isfinite(field).all():
        raise FloatingPointError("Correction field contains non-finite values")
    return field[0], field[1], field[2]


def symmetric_heatmap_range(
    correction_fields: Sequence[np.ndarray], robust_percentile: float | None = None
) -> tuple[float, float]:
    """Compute one zero-centered display range across every selected sample/channel."""
    if not correction_fields:
        raise ValueError("At least one correction field is required")
    if robust_percentile is not None and not 0.0 < robust_percentile <= 100.0:
        raise ValueError("--robust-percentile must be in (0, 100]")
    absolute_values: list[np.ndarray] = []
    for correction_field in correction_fields:
        channels = split_correction_channels(correction_field)
        absolute_values.extend(np.abs(channel).reshape(-1) for channel in channels)
    combined = np.concatenate(absolute_values)
    limit = (
        float(combined.max())
        if robust_percentile is None
        else float(np.percentile(combined, robust_percentile))
    )
    # A non-zero range is required by a diverging color scale even for an identity UICF.
    limit = max(limit, float(np.finfo(np.float32).eps))
    return -limit, limit


def validate_uicf_details(inputs: Tensor, prediction: Tensor, details: UICFINROutput) -> None:
    if not isinstance(details, UICFINROutput):
        raise TypeError(f"Expected UICFINROutput, got {type(details).__name__}")
    if inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError(f"Expected RGB BCHW inputs, got {tuple(inputs.shape)}")
    expected_image_shape = tuple(inputs.shape)
    shapes = {
        "prediction": tuple(prediction.shape),
        "details.enhanced": tuple(details.enhanced.shape),
        "details.correction_field": tuple(details.correction_field.shape),
    }
    for name, shape in shapes.items():
        if shape != expected_image_shape:
            raise ValueError(f"{name} shape is {shape}, expected {expected_image_shape}")
    batch = inputs.shape[0]
    if tuple(details.chromatic_anchor.shape) != (batch, 3):
        raise ValueError(
            f"details.chromatic_anchor shape is {tuple(details.chromatic_anchor.shape)}, "
            f"expected {(batch, 3)}"
        )
    if details.global_feature.ndim != 2 or details.global_feature.shape[0] != batch:
        raise ValueError(
            "details.global_feature must have shape [B,C], got "
            f"{tuple(details.global_feature.shape)}"
        )
    tensors = {
        "inputs": inputs,
        "prediction": prediction,
        "details.enhanced": details.enhanced,
        "details.correction_field": details.correction_field,
        "details.chromatic_anchor": details.chromatic_anchor,
        "details.global_feature": details.global_feature,
    }
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains non-finite values")


def reconstruct_uicf_corrected(inputs: Tensor, details: UICFINROutput) -> Tensor:
    return inputs + details.correction_field * (
        inputs - details.chromatic_anchor[:, :, None, None]
    )


def assert_uicf_consistency(
    inputs: Tensor,
    prediction: Tensor,
    details: UICFINROutput,
    prediction_normal: Tensor | None = None,
) -> Tensor:
    """Verify the paper formula and, when supplied, diagnostics-forward equivalence."""
    validate_uicf_details(inputs, prediction, details)
    manual = reconstruct_uicf_corrected(inputs, details)
    if not torch.allclose(manual, details.enhanced, atol=1e-6, rtol=1e-5):
        difference = float((manual - details.enhanced).abs().max().detach().cpu())
        raise RuntimeError(
            "UICF reconstruction check failed for Ic = I + R * (I - b); "
            f"max_abs_diff={difference:.9g}"
        )
    if prediction_normal is not None:
        if tuple(prediction_normal.shape) != tuple(prediction.shape):
            raise ValueError("Normal and diagnostics forward outputs have different shapes")
        if not torch.allclose(prediction, prediction_normal, atol=1e-6, rtol=1e-5):
            difference = float((prediction - prediction_normal).abs().max().detach().cpu())
            raise RuntimeError(
                "Diagnostics forward changed the v16 prediction; "
                f"max_abs_diff={difference:.9g}"
            )
    return manual


def correction_statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError("Cannot summarize a non-finite correction channel")
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "mean_abs": float(np.abs(array).mean()),
    }


def build_metadata(
    sample: CapturedSample,
    *,
    checkpoint_path: Path,
    checkpoint_selector: str,
    checkpoint_epoch: int,
    heatmap_vmin: float,
    heatmap_vmax: float,
    robust_percentile: float | None,
) -> dict[str, Any]:
    field = sample.correction_field.detach().float().cpu().numpy()
    channels = split_correction_channels(field)
    anchor = sample.chromatic_anchor.detach().float().cpu().reshape(-1).tolist()
    return {
        "sample_index": int(sample.index),
        "sample_id": sample.sample_id,
        "filename": sample.filename,
        "checkpoint": str(checkpoint_path),
        "checkpoint_selector": checkpoint_selector,
        "checkpoint_epoch": int(checkpoint_epoch),
        "chromatic_anchor_b": [float(value) for value in anchor],
        "correction_field": {
            key: correction_statistics(channel)
            for key, channel in zip(CHANNEL_KEYS, channels, strict=True)
        },
        "global_feature_shape": list(sample.global_feature.shape),
        "heatmap_vmin": float(heatmap_vmin),
        "heatmap_vmax": float(heatmap_vmax),
        "heatmap_colormap": "zero-centered blue-white-red",
        "heatmap_range_scope": "all selected samples and all RGB correction channels",
        "robust_percentile": (
            None if robust_percentile is None else float(robust_percentile)
        ),
        "uicf_formula": "Ic = I + R * (I - b)",
        "formula_sanity_check": True,
    }


def _diverging_rgb(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if not vmin < 0.0 < vmax or not np.isclose(-vmin, vmax):
        raise ValueError(f"Expected a symmetric zero-centered range, got [{vmin}, {vmax}]")
    normalized = np.clip(np.asarray(values, dtype=np.float64) / vmax, -1.0, 1.0)
    negative = np.array([59.0, 76.0, 192.0])
    center = np.array([247.0, 247.0, 247.0])
    positive = np.array([180.0, 4.0, 38.0])
    rgb = np.empty((*normalized.shape, 3), dtype=np.float64)
    below = normalized <= 0.0
    negative_weight = (-normalized[below])[:, None]
    rgb[below] = center * (1.0 - negative_weight) + negative * negative_weight
    positive_weight = normalized[~below][:, None]
    rgb[~below] = center * (1.0 - positive_weight) + positive * positive_weight
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def save_heatmaps(
    channels: Sequence[np.ndarray], directory: Path, vmin: float, vmax: float
) -> None:
    if len(channels) != 3:
        raise ValueError("Exactly three correction channels are required")
    directory.mkdir(parents=True, exist_ok=True)
    for key, channel in zip(CHANNEL_KEYS, channels, strict=True):
        image = Image.fromarray(_diverging_rgb(channel, vmin, vmax), mode="RGB")
        image.save(directory / f"{key}_heatmap.png", dpi=HEATMAP_DPI)


def _rgb_image(tensor: Tensor) -> Image.Image:
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected RGB CHW tensor, got {tuple(tensor.shape)}")
    return tensor_to_image(tensor)


def _centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text)
    width, height = right - left, bottom - top
    draw.text((xy[0] - width // 2, xy[1] - height // 2), text, fill="black")


def save_panel(
    input_tensor: Tensor,
    channels: Sequence[np.ndarray],
    enhanced: Tensor,
    path: Path,
    vmin: float,
    vmax: float,
) -> None:
    """Save Input | R_r | R_g | R_b | Corrected Ic with one shared colorbar."""
    if len(channels) != 3:
        raise ValueError("Exactly three correction channels are required")
    input_image, enhanced_image = _rgb_image(input_tensor), _rgb_image(enhanced)
    width, height = input_image.size
    if enhanced_image.size != (width, height):
        raise ValueError("Input and corrected image dimensions differ")
    heatmaps = [Image.fromarray(_diverging_rgb(channel, vmin, vmax), mode="RGB") for channel in channels]
    if any(image.size != (width, height) for image in heatmaps):
        raise ValueError("Correction heatmap dimensions differ from the input")

    # Only enlarge the display copies so tiny diagnostic tensors still make a
    # readable horizontal panel. Saved tensors and individual maps stay native.
    display_scale = max(1.0, 128.0 / min(width, height))
    display_size = (
        int(round(width * display_scale)),
        int(round(height * display_scale)),
    )
    if display_size != (width, height):
        input_image = input_image.resize(display_size, Image.Resampling.BILINEAR)
        enhanced_image = enhanced_image.resize(display_size, Image.Resampling.BILINEAR)
        heatmaps = [
            image.resize(display_size, Image.Resampling.BILINEAR) for image in heatmaps
        ]
    width, height = display_size

    title_height = 28
    colorbar_height = 44
    gap = 6
    panel = Image.new(
        "RGB", (5 * width + 4 * gap, title_height + height + colorbar_height), "white"
    )
    images = [input_image, *heatmaps, enhanced_image]
    titles = [RGB_TITLES[0], *CHANNEL_TITLES, RGB_TITLES[1]]
    draw = ImageDraw.Draw(panel)
    for column, (image, title) in enumerate(zip(images, titles, strict=True)):
        x = column * (width + gap)
        panel.paste(image, (x, title_height))
        _centered_text(draw, (x + width // 2, title_height // 2), title)

    colorbar_left = width + gap
    colorbar_right = 4 * width + 3 * gap
    bar_top = title_height + height + 7
    bar_height = 12
    scale = np.linspace(vmin, vmax, colorbar_right - colorbar_left, dtype=np.float64)[None, :]
    bar = Image.fromarray(_diverging_rgb(scale, vmin, vmax), mode="RGB").resize(
        (colorbar_right - colorbar_left, bar_height), Image.Resampling.NEAREST
    )
    panel.paste(bar, (colorbar_left, bar_top))
    draw.rectangle(
        (colorbar_left, bar_top, colorbar_right - 1, bar_top + bar_height - 1),
        outline="black",
    )
    label_y = bar_top + bar_height + 10
    _centered_text(draw, (colorbar_left, label_y), f"{vmin:.4g}")
    _centered_text(draw, ((colorbar_left + colorbar_right) // 2, label_y), "0")
    _centered_text(draw, (colorbar_right, label_y), f"{vmax:.4g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path, dpi=HEATMAP_DPI)


def _safe_sample_id(sample_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    return sanitized or "sample"


def save_visualization_sample(
    sample: CapturedSample,
    directory: Path,
    *,
    checkpoint_path: Path,
    checkpoint_selector: str,
    checkpoint_epoch: int,
    heatmap_vmin: float,
    heatmap_vmax: float,
    robust_percentile: float | None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    field = sample.correction_field.detach().float().cpu().numpy()
    channels = split_correction_channels(field)
    _rgb_image(sample.input_tensor).save(directory / "input.png")
    _rgb_image(sample.enhanced).save(directory / "uicf_corrected_Ic.png")
    _rgb_image(sample.prediction).save(directory / "v16_final.png")
    _rgb_image(sample.target_tensor).save(directory / "gt.png")
    for key, channel in zip(CHANNEL_KEYS, channels, strict=True):
        np.save(directory / f"{key}.npy", channel, allow_pickle=False)
    np.save(directory / "correction_field_rgb.npy", field, allow_pickle=False)
    np.save(
        directory / "global_feature.npy",
        sample.global_feature.detach().float().cpu().numpy(),
        allow_pickle=False,
    )
    save_heatmaps(channels, directory, heatmap_vmin, heatmap_vmax)
    save_panel(
        sample.input_tensor,
        channels,
        sample.enhanced,
        directory / "uicf_panel_1x5.png",
        heatmap_vmin,
        heatmap_vmax,
    )
    metadata = build_metadata(
        sample,
        checkpoint_path=checkpoint_path,
        checkpoint_selector=checkpoint_selector,
        checkpoint_epoch=checkpoint_epoch,
        heatmap_vmin=heatmap_vmin,
        heatmap_vmax=heatmap_vmax,
        robust_percentile=robust_percentile,
    )
    atomic_json(directory / "uicf_metadata.json", metadata)
    return metadata


def build_and_load_v16_model(
    config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device
) -> UICFPreBackbone:
    if checkpoint.get("version") != EXPECTED_VERSION:
        raise ValueError("Checkpoint version is incompatible")
    if checkpoint.get("resolved_config", {}).get("model") != config["model"]:
        raise ValueError("Checkpoint architecture differs from run config_resolved.yaml")
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not isinstance(model, UICFPreBackbone):
        raise TypeError(f"v16 diagnostics expected UICFPreBackbone, got {type(model).__name__}")
    model.eval()
    return model


def capture_sample(
    model: nn.Module,
    dataset: LSUIDataset,
    index: int,
    device: torch.device,
    *,
    amp_enabled: bool,
    compare_normal_forward: bool,
) -> CapturedSample:
    item = dataset[index]
    input_tensor, target_tensor = item["input"], item["target"]
    if not isinstance(input_tensor, Tensor) or not isinstance(target_tensor, Tensor):
        raise TypeError("Dataset input/target must be tensors")
    sample_id, filename = item["id"], item["filename"]
    if not isinstance(sample_id, str) or not isinstance(filename, str):
        raise TypeError("Dataset id/filename must be strings")
    inputs = input_tensor.unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, enabled=amp_enabled
    ):
        forward = getattr(model, "forward_with_uicf_details", None)
        if not callable(forward):
            raise TypeError("v16 model lacks forward_with_uicf_details")
        prediction, details = forward(inputs)
        prediction_normal = model(inputs) if compare_normal_forward else None
    if not isinstance(prediction, Tensor):
        raise TypeError("v16 diagnostics forward did not return a prediction tensor")
    assert_uicf_consistency(inputs, prediction, details, prediction_normal)
    return CapturedSample(
        index=index,
        sample_id=sample_id,
        filename=filename,
        input_tensor=inputs[0].detach().float().cpu(),
        target_tensor=target_tensor.detach().float().cpu(),
        prediction=prediction[0].detach().float().cpu(),
        enhanced=details.enhanced[0].detach().float().cpu(),
        correction_field=details.correction_field[0].detach().float().cpu(),
        chromatic_anchor=details.chromatic_anchor[0].detach().float().cpu(),
        global_feature=details.global_feature[0].detach().float().cpu(),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = project_path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run lacks config_resolved.yaml: {run_dir}")
    config = load_yaml(config_path)
    if config["experiment"]["version"] != EXPECTED_VERSION:
        raise ValueError(
            f"{EXPECTED_VERSION} visualization refuses run version "
            f"{config['experiment']['version']!r}"
        )
    if args.data_root is not None:
        config["data"]["root"] = str(Path(args.data_root).expanduser().resolve())
    selector = args.checkpoint or config["test"]["checkpoint"]
    checkpoint_path = _checkpoint_path(run_dir, selector)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    snapshot = run_dir / "split_snapshot"
    manifests = {
        name: snapshot / f"{name}.tsv" for name in ("train", "validation", "test")
    }
    entries = validate_split_protocol(
        manifests, config["data"].get("expected_counts")
    )
    data_root = Path(config["data"]["root"]).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data.root is unavailable: {data_root}")
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
        verify_files=True,
    )
    if test_dataset.entries != entries["test"]:
        raise RuntimeError("Validated test manifest order differs from Dataset order")
    random_seed = int(config["test"]["visualization"]["random_seed"])
    selected_indices = select_sample_indices(
        entries["test"],
        indices=args.indices,
        sample_ids=args.sample_ids,
        num_samples=args.num_samples,
        random_seed=random_seed,
    )

    device = select_device(args.gpu)
    checkpoint = _torch_load(checkpoint_path, device)
    model = build_and_load_v16_model(config, checkpoint, device)
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    captured = [
        capture_sample(
            model,
            test_dataset,
            index,
            device,
            amp_enabled=amp_enabled,
            compare_normal_forward=position == 0,
        )
        for position, index in enumerate(selected_indices)
    ]
    fields = [sample.correction_field.numpy() for sample in captured]
    heatmap_vmin, heatmap_vmax = symmetric_heatmap_range(
        fields, args.robust_percentile
    )

    output_dir = run_dir / "result" / "uicf_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_summaries: list[dict[str, Any]] = []
    for position, sample in enumerate(captured):
        sample_dir = output_dir / f"{position:03d}_{_safe_sample_id(sample.sample_id)}"
        save_visualization_sample(
            sample,
            sample_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_selector=selector,
            checkpoint_epoch=int(checkpoint["epoch"]),
            heatmap_vmin=heatmap_vmin,
            heatmap_vmax=heatmap_vmax,
            robust_percentile=args.robust_percentile,
        )
        sample_summaries.append(
            {
                "sample_index": sample.index,
                "sample_id": sample.sample_id,
                "filename": sample.filename,
                "directory": sample_dir.name,
            }
        )
    atomic_json(
        output_dir / "visualization_summary.json",
        {
            "version": EXPECTED_VERSION,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_selector": selector,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "data_root": str(data_root),
            "test_manifest": str(manifests["test"]),
            "sample_count": len(captured),
            "default_random_seed": random_seed,
            "robust_percentile": args.robust_percentile,
            "heatmap_vmin": heatmap_vmin,
            "heatmap_vmax": heatmap_vmax,
            "heatmap_colormap": "zero-centered blue-white-red",
            "shared_zero_centered_range": True,
            "raw_correction_fields_preserved": True,
            "formula_sanity_check": "PASS for every selected sample",
            "normal_forward_equivalence": "PASS for first selected sample",
            "samples": sample_summaries,
        },
    )
    print(
        f"Saved {len(captured)} v16 UICF visualizations to {output_dir}\n"
        f"Shared heatmap range: [{heatmap_vmin:.9g}, {heatmap_vmax:.9g}]"
    )


if __name__ == "__main__":
    main()
