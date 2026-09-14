#!/usr/bin/env python3
"""Create paper-grade CIELAB chroma and spatial CIEDE2000 analysis figures.

This is a post-processing tool. It consumes the frozen v16 test manifest,
official per-image metrics, and already exported 8-bit enhanced PNGs. It never
loads a model or checkpoint and never performs inference.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.e00 import delta_e00_from_lab
from src.v16.dataset import LSUIDataset, ManifestEntry, read_manifest
from src.v16.utils import atomic_json, load_yaml, sha256_file


SCRIPT_VERSION = "1.0"
EXPECTED_VERSION = "v16"
DEFAULT_OUTPUT_NAME = "chromatic_restoration_analysis"
ANALYSIS_FILENAMES = (
    "chromatic_restoration_analysis.png",
    "chromatic_restoration_analysis.pdf",
    "chromatic_restoration_analysis.svg",
    "chromatic_per_sample.csv",
    "selected_samples.csv",
    "analysis_protocol.json",
    "candidate_contact_sheet.png",
)
PER_SAMPLE_FIELDS = (
    "sample_id",
    "filename",
    "official_output_e00",
    "input_e00",
    "output_png_e00",
    "e00_reduction",
    "e00_reduction_ratio",
    "png_quantization_delta",
    "psnr",
    "ssim",
)
CANDIDATE_SELECTION_RULE = (
    "Deterministic representative selection: retain samples with input_e00 at or "
    "above the dataset median and positive PNG-derived E00 reduction; target evenly "
    "spaced input_e00 quantiles while preferring reductions near the eligible median, "
    "with manifest index as the final tie-breaker. If the primary pool is empty, "
    "fall back to positive-reduction samples, then to the full test set."
)
PNG_QUANTIZATION_NOTE = (
    "official_output_e00 comes from the pre-PNG float network prediction in the "
    "official v16 test. output_png_e00 and spatial output error maps are recomputed "
    "from the saved 8-bit enhanced PNG and may differ slightly due to quantization."
)


@dataclass(frozen=True)
class SampleAnalysis:
    index: int
    entry: ManifestEntry
    enhanced_filename: str
    input_rgb: np.ndarray
    output_rgb: np.ndarray
    reference_rgb: np.ndarray
    input_lab: np.ndarray
    output_lab: np.ndarray
    reference_lab: np.ndarray
    input_e00_map: np.ndarray
    output_e00_map: np.ndarray
    metrics: dict[str, Any]


def resolve_project_path(value: str | Path) -> Path:
    """Resolve paths independently of cwd for repo-root and workspace-root CLIs."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0] == PROJECT_ROOT.name:
        path = Path(*parts[1:]) if len(parts) > 1 else Path()
    return (PROJECT_ROOT / path).resolve()


def resolve_dataset_label(data_config: Mapping[str, Any]) -> str:
    """Label modern and legacy runs without restricting the paired loader."""
    configured = str(data_config.get("dataset", "")).strip()
    if configured:
        return configured

    normalized_metadata = [
        str(data_config.get(key, "")).replace("\\", "/").lower()
        for key in (
            "root",
            "train_manifest",
            "validation_manifest",
            "test_manifest",
        )
    ]
    evidence: set[str] = set()
    for value in normalized_metadata:
        path_tokens = {token for token in value.split("/") if token}
        if "lsui19" in value or "lsui" in path_tokens or "lsui" in value:
            evidence.add("LSUI19")
        if "uieb" in value:
            evidence.add("UIEB")
    if len(evidence) == 1:
        inferred = next(iter(evidence))
        print(
            "[WARNING] config_resolved.yaml has no data.dataset; inferred legacy "
            f"dataset label as {inferred!r} from data root/manifest metadata.",
            file=sys.stderr,
        )
        return inferred
    reason = "conflicting metadata" if evidence else "unrecognized metadata"
    print(
        "[WARNING] config_resolved.yaml has no data.dataset and dataset identity "
        f"has {reason}; recording dataset as 'unspecified'.",
        file=sys.stderr,
    )
    return "unspecified"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot chromatic restoration analysis from official v16 test outputs"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-ids", nargs="+", default=None)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--ab-bins", type=int, default=96)
    parser.add_argument("--ab-min", type=float, default=-128.0)
    parser.add_argument("--ab-max", type=float, default=127.0)
    parser.add_argument("--de-vmax", type=float, default=None)
    parser.add_argument("--de-percentile", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--show-sample-id", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be positive")
    if args.ab_bins < 2:
        raise ValueError("--ab-bins must be at least 2")
    if not math.isfinite(args.ab_min) or not math.isfinite(args.ab_max):
        raise ValueError("--ab-min and --ab-max must be finite")
    if args.ab_min >= args.ab_max:
        raise ValueError("--ab-min must be smaller than --ab-max")
    if args.de_vmax is not None and (
        not math.isfinite(args.de_vmax) or args.de_vmax <= 0.0
    ):
        raise ValueError("--de-vmax must be finite and positive")
    if not math.isfinite(args.de_percentile) or not 0.0 < args.de_percentile <= 100.0:
        raise ValueError("--de-percentile must be in (0,100]")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    if args.sample_ids is not None and len(args.sample_ids) != len(set(args.sample_ids)):
        raise ValueError("--sample-ids must not contain duplicates")


def _official_test_error(run_dir: Path, detail: str) -> FileNotFoundError:
    command = (
        "python -m src.v16.test "
        f"--run-dir {run_dir} --checkpoint best_psnr --gpu 0"
    )
    return FileNotFoundError(
        f"{detail}. Run the official v16 test first:\n{command}"
    )


def enhanced_filename_mapping(
    entries: Sequence[ManifestEntry],
) -> dict[str, str]:
    """Reproduce src.v16.test's exact ordered collision-handling algorithm."""
    mapping: dict[str, str] = {}
    used_names: set[str] = set()
    for entry in entries:
        if entry.sample_id in mapping:
            raise ValueError(f"Duplicate sample_id in test manifest: {entry.sample_id}")
        stem = Path(entry.input_relative).stem
        output_name = f"{stem}_enhanced.png"
        if output_name in used_names:
            output_name = f"{stem}_{entry.sample_id}_enhanced.png"
        if output_name in used_names:
            raise ValueError(
                "Enhanced filename collision remains after official fallback naming: "
                f"sample_id={entry.sample_id} filename={output_name}"
            )
        used_names.add(output_name)
        mapping[entry.sample_id] = output_name
    return mapping


def resolve_enhanced_paths(
    entries: Sequence[ManifestEntry], enhanced_dir: Path, run_dir: Path
) -> dict[str, Path]:
    if not enhanced_dir.is_dir():
        raise _official_test_error(
            run_dir, f"Official enhanced output directory is missing: {enhanced_dir}"
        )
    names = enhanced_filename_mapping(entries)
    paths = {sample_id: enhanced_dir / name for sample_id, name in names.items()}
    missing = [
        f"{sample_id}: {path.name}"
        for sample_id, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise _official_test_error(
            run_dir,
            f"Official enhanced outputs are incomplete ({len(missing)} missing; {preview})",
        )
    return paths


def read_official_metrics(
    path: Path, entries: Sequence[ManifestEntry], run_dir: Path
) -> dict[str, dict[str, float | str]]:
    if not path.is_file():
        raise _official_test_error(run_dir, f"Official test metrics are missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "sample_id", "psnr", "ssim", "e00"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain fields {sorted(required)}, got {reader.fieldnames}"
            )
        rows = list(reader)
    by_id: dict[str, dict[str, float | str]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"Duplicate sample_id in official test metrics: {sample_id}")
        try:
            psnr = float(row["psnr"])
            ssim = float(row["ssim"])
            e00 = float(row["e00"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid official metric value for sample_id={sample_id}"
            ) from error
        if math.isnan(psnr) or not math.isfinite(ssim) or not math.isfinite(e00):
            raise FloatingPointError(
                f"Non-finite official metric for sample_id={sample_id}"
            )
        if e00 < 0.0:
            raise ValueError(f"Negative official E00 for sample_id={sample_id}")
        by_id[sample_id] = {
            "filename": str(row["filename"]),
            "psnr": psnr,
            "ssim": ssim,
            "e00": e00,
        }
    expected_ids = [entry.sample_id for entry in entries]
    missing = sorted(set(expected_ids) - set(by_id))
    extra = sorted(set(by_id) - set(expected_ids))
    if missing or extra:
        raise ValueError(
            "Official test metrics and frozen test manifest differ: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    for entry in entries:
        expected_filename = Path(entry.input_relative).name
        actual_filename = str(by_id[entry.sample_id]["filename"])
        if actual_filename != expected_filename:
            raise ValueError(
                f"Official filename mismatch for sample_id={entry.sample_id}: "
                f"metrics={actual_filename!r} manifest={expected_filename!r}"
            )
    return by_id


def _tensor_rgb_to_numpy(tensor: Any, *, name: str) -> np.ndarray:
    array = np.asarray(tensor.permute(1, 2, 0).contiguous().numpy(), dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must be HWC RGB, got {array.shape}")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must be finite standard sRGB in [0,1]")
    return array


def _load_enhanced_rgb(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise RuntimeError(f"Official enhanced PNG must be RGB: {path}")
        array = np.asarray(image, dtype=np.float32) / 255.0
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"Enhanced PNG shape {tuple(array.shape)} does not match official evaluation "
            f"shape {expected_shape}: {path}"
        )
    return np.asarray(array, dtype=np.float64)


def rgb_to_project_lab(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Lab conversion expects HWC RGB, got {array.shape}")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("Lab conversion expects finite standard sRGB in [0,1]")
    lab = np.asarray(
        rgb2lab(array, illuminant="D65", observer="2", channel_axis=-1),
        dtype=np.float64,
    )
    if lab.shape != array.shape or not np.isfinite(lab).all():
        raise FloatingPointError("Project Lab conversion produced invalid values")
    return lab


def delta_e00_map_from_rgb(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"DeltaE00 map expects matching RGB shapes, got "
            f"{first_array.shape} and {second_array.shape}"
        )
    first_lab = rgb_to_project_lab(first_array)
    second_lab = rgb_to_project_lab(second_array)
    values = delta_e00_from_lab(first_lab, second_lab)
    if values.shape != first_array.shape[:2]:
        raise RuntimeError(
            f"DeltaE00 map shape {values.shape} differs from RGB {first_array.shape[:2]}"
        )
    return values


def ab_bin_edges(bins: int, minimum: float, maximum: float) -> np.ndarray:
    if bins < 2 or not minimum < maximum:
        raise ValueError("Invalid a*b* histogram specification")
    return np.linspace(minimum, maximum, bins + 1, dtype=np.float64)


def ab_probability_histogram(
    lab: np.ndarray, a_edges: np.ndarray, b_edges: np.ndarray
) -> np.ndarray:
    array = np.asarray(lab, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError("a*b* histogram expects finite HWC Lab")
    counts, returned_a, returned_b = np.histogram2d(
        array[..., 1].reshape(-1),
        array[..., 2].reshape(-1),
        bins=(a_edges, b_edges),
    )
    if not np.array_equal(returned_a, a_edges) or not np.array_equal(returned_b, b_edges):
        raise RuntimeError("np.histogram2d changed the requested common bin edges")
    total_pixels = array.shape[0] * array.shape[1]
    if int(counts.sum()) != total_pixels:
        raise ValueError(
            "a*/b* values fall outside --ab-min/--ab-max; widen the common range "
            "so probability mass remains normalized by all pixels"
        )
    probability = counts.astype(np.float64) / total_pixels
    if not math.isclose(float(probability.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("a*b* histogram probability mass does not sum to one")
    return probability


def shared_delta_e_vmax(
    maps: Sequence[np.ndarray], *, percentile: float, explicit_vmax: float | None
) -> float:
    if explicit_vmax is not None:
        if not math.isfinite(explicit_vmax) or explicit_vmax <= 0.0:
            raise ValueError("Explicit DeltaE00 vmax must be finite and positive")
        return float(explicit_vmax)
    if not maps:
        raise ValueError("At least one DeltaE00 map is required")
    flattened = []
    for values in maps:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0 or not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError("DeltaE00 visualization maps must be finite and non-negative")
        flattened.append(array.reshape(-1))
    pooled = np.concatenate(flattened)
    value = float(np.percentile(pooled, percentile))
    return max(value, float(np.finfo(np.float64).eps))


def _metric_crop(values: np.ndarray, border: int) -> np.ndarray:
    if border < 0:
        raise ValueError("metrics.crop_border must be non-negative")
    if border == 0:
        return values
    if values.shape[0] <= 2 * border or values.shape[1] <= 2 * border:
        raise ValueError("metrics.crop_border removes the entire DeltaE00 map")
    return values[border:-border, border:-border]


def analyze_sample(
    dataset: LSUIDataset,
    index: int,
    enhanced_path: Path,
    official: Mapping[str, float | str],
    *,
    crop_border: int,
) -> SampleAnalysis:
    entry = dataset.entries[index]
    item = dataset[index]
    if item["id"] != entry.sample_id:
        raise RuntimeError("Dataset item order differs from frozen test manifest")
    input_rgb = _tensor_rgb_to_numpy(item["input"], name="Input")
    reference_rgb = _tensor_rgb_to_numpy(item["target"], name="Reference")
    output_rgb = _load_enhanced_rgb(enhanced_path, tuple(input_rgb.shape))
    input_lab = rgb_to_project_lab(input_rgb)
    output_lab = rgb_to_project_lab(output_rgb)
    reference_lab = rgb_to_project_lab(reference_rgb)
    input_map = _metric_crop(
        delta_e00_from_lab(input_lab, reference_lab), crop_border
    )
    output_map = _metric_crop(
        delta_e00_from_lab(output_lab, reference_lab), crop_border
    )
    input_e00 = float(input_map.mean(dtype=np.float64))
    output_e00 = float(output_map.mean(dtype=np.float64))
    official_e00 = float(official["e00"])
    reduction = input_e00 - output_e00
    quantization_delta = output_e00 - official_e00
    metrics = {
        "_index": index,
        "sample_id": entry.sample_id,
        "filename": Path(entry.input_relative).name,
        "official_output_e00": official_e00,
        "input_e00": input_e00,
        "output_png_e00": output_e00,
        "e00_reduction": reduction,
        "e00_reduction_ratio": reduction / max(input_e00, 1e-12),
        "png_quantization_delta": quantization_delta,
        "psnr": float(official["psnr"]),
        "ssim": float(official["ssim"]),
    }
    if abs(quantization_delta) > 0.1:
        print(
            f"[WARNING] sample_id={entry.sample_id}: abs(PNG-derived E00 - official "
            f"E00)={abs(quantization_delta):.6f} exceeds 0.1",
            file=sys.stderr,
        )
    return SampleAnalysis(
        index=index,
        entry=entry,
        enhanced_filename=enhanced_path.name,
        input_rgb=input_rgb,
        output_rgb=output_rgb,
        reference_rgb=reference_rgb,
        input_lab=input_lab,
        output_lab=output_lab,
        reference_lab=reference_lab,
        input_e00_map=input_map,
        output_e00_map=output_map,
        metrics=metrics,
    )


def select_representative_candidates(
    rows: Sequence[Mapping[str, Any]], count: int
) -> list[dict[str, Any]]:
    if not rows or count < 1:
        raise ValueError("Representative selection requires rows and positive count")
    input_values = np.asarray([float(row["input_e00"]) for row in rows], dtype=np.float64)
    dataset_median = float(np.median(input_values))
    primary = [
        dict(row)
        for row in rows
        if float(row["input_e00"]) >= dataset_median
        and float(row["e00_reduction"]) > 0.0
    ]
    if primary:
        pool = primary
    else:
        positive = [dict(row) for row in rows if float(row["e00_reduction"]) > 0.0]
        pool = positive if positive else [dict(row) for row in rows]
    target_count = min(count, len(pool))
    reductions = np.asarray([float(row["e00_reduction"]) for row in pool])
    demands = np.asarray([float(row["input_e00"]) for row in pool])
    median_reduction = float(np.median(reductions))
    reduction_scale = max(float(np.ptp(reductions)), 1e-12)
    demand_scale = max(float(np.ptp(demands)), 1e-12)
    targets = np.quantile(demands, np.linspace(0.0, 1.0, target_count))
    selected: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for target in targets:
        candidates = [row for row in pool if int(row["_index"]) not in used_indices]
        chosen = min(
            candidates,
            key=lambda row: (
                abs(float(row["input_e00"]) - float(target)) / demand_scale
                + abs(float(row["e00_reduction"]) - median_reduction)
                / reduction_scale,
                abs(float(row["e00_reduction"]) - median_reduction),
                int(row["_index"]),
            ),
        )
        selected.append(chosen)
        used_indices.add(int(chosen["_index"]))
    return selected


def _shared_hist_norm(histograms: Sequence[np.ndarray]) -> LogNorm:
    positives = np.concatenate(
        [histogram[histogram > 0.0].reshape(-1) for histogram in histograms]
    )
    if positives.size == 0:
        raise RuntimeError("a*b* histograms contain no positive probability mass")
    minimum = float(positives.min())
    maximum = float(positives.max())
    if minimum == maximum:
        minimum = maximum / 10.0
    return LogNorm(vmin=minimum, vmax=maximum)


def _row_label(index: int, sample_id: str, show_sample_id: bool) -> str:
    letter = chr(ord("a") + index) if index < 26 else str(index + 1)
    label = f"({letter})"
    return f"{label}\n{sample_id}" if show_sample_id else label


def _paper_rc() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def save_final_figure(
    samples: Sequence[SampleAnalysis],
    output_dir: Path,
    *,
    a_edges: np.ndarray,
    b_edges: np.ndarray,
    de_vmax: float,
    dpi: int,
    show_sample_id: bool,
) -> None:
    if not samples:
        raise ValueError("Final figure requires at least one sample")
    histograms = []
    sample_histograms = []
    for sample in samples:
        current = tuple(
            ab_probability_histogram(lab, a_edges, b_edges)
            for lab in (sample.input_lab, sample.output_lab, sample.reference_lab)
        )
        sample_histograms.append(current)
        histograms.extend(current)
    histogram_norm = _shared_hist_norm(histograms)
    de_norm = Normalize(vmin=0.0, vmax=de_vmax, clip=True)
    extent = [a_edges[0], a_edges[-1], b_edges[0], b_edges[-1]]
    columns = (
        "Input",
        "ICAR-UIE",
        "Reference",
        "Input vs Ref.",
        "ICAR-UIE vs Ref.",
    )

    with plt.rc_context(_paper_rc()):
        figure, axes = plt.subplots(
            len(samples),
            5,
            squeeze=False,
            figsize=(11.6, max(2.15 * len(samples), 2.5)),
            facecolor="white",
        )
        de_image = None
        for row, (sample, distributions) in enumerate(
            zip(samples, sample_histograms, strict=True)
        ):
            for column in range(3):
                axis = axes[row, column]
                axis.imshow(
                    distributions[column].T,
                    origin="lower",
                    extent=extent,
                    cmap="magma",
                    norm=histogram_norm,
                    interpolation="nearest",
                    aspect="equal",
                )
                axis.set_xlim(a_edges[0], a_edges[-1])
                axis.set_ylim(b_edges[0], b_edges[-1])
                axis.set_aspect("equal", adjustable="box")
                axis.xaxis.set_major_locator(MaxNLocator(5))
                axis.yaxis.set_major_locator(MaxNLocator(5))
                if row == len(samples) - 1:
                    axis.set_xlabel("a*")
                if column == 0:
                    axis.set_ylabel("b*")
                else:
                    axis.tick_params(labelleft=False)
            for column, values in enumerate(
                (sample.input_e00_map, sample.output_e00_map), start=3
            ):
                axis = axes[row, column]
                de_image = axis.imshow(
                    values,
                    cmap="viridis",
                    norm=de_norm,
                    interpolation="nearest",
                )
                axis.set_xticks([])
                axis.set_yticks([])
            axes[row, 0].text(
                -0.35,
                0.5,
                _row_label(row, sample.entry.sample_id, show_sample_id),
                transform=axes[row, 0].transAxes,
                ha="right",
                va="center",
                fontsize=9,
            )
        for column, title in enumerate(columns):
            axes[0, column].set_title(title, pad=13)
        figure.text(
            0.315,
            0.99,
            "Chromatic distribution in CIELAB a*-b* plane",
            ha="center",
            va="top",
            fontsize=9.5,
            weight="semibold",
        )
        figure.text(
            0.792,
            0.99,
            "Spatial color error",
            ha="center",
            va="top",
            fontsize=9.5,
            weight="semibold",
        )
        figure.subplots_adjust(
            left=0.065, right=0.91, bottom=0.12, top=0.89, wspace=0.22, hspace=0.25
        )
        if de_image is None:
            raise RuntimeError("DeltaE00 image artist was not created")
        colorbar = figure.colorbar(
            de_image,
            ax=axes[:, 3:].reshape(-1).tolist(),
            fraction=0.035,
            pad=0.025,
        )
        colorbar.set_label("Delta E00")
        for extension in ("png", "pdf", "svg"):
            figure.savefig(
                output_dir / f"chromatic_restoration_analysis.{extension}",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                transparent=False,
            )
        plt.close(figure)


def save_candidate_contact_sheet(
    samples: Sequence[SampleAnalysis],
    path: Path,
    *,
    de_percentile: float,
    de_vmax_override: float | None,
    dpi: int,
) -> float:
    if not samples:
        raise ValueError("Candidate contact sheet requires at least one sample")
    maps = [
        values
        for sample in samples
        for values in (sample.input_e00_map, sample.output_e00_map)
    ]
    de_vmax = shared_delta_e_vmax(
        maps, percentile=de_percentile, explicit_vmax=de_vmax_override
    )
    norm = Normalize(vmin=0.0, vmax=de_vmax, clip=True)
    titles = (
        "Input",
        "Enhanced",
        "Reference",
        "Input vs Ref.",
        "Enhanced vs Ref.",
    )
    with plt.rc_context(_paper_rc()):
        figure, axes = plt.subplots(
            len(samples),
            5,
            squeeze=False,
            figsize=(10.8, max(1.85 * len(samples), 2.3)),
            facecolor="white",
        )
        image_artist = None
        for row, sample in enumerate(samples):
            for column, rgb in enumerate(
                (sample.input_rgb, sample.output_rgb, sample.reference_rgb)
            ):
                axes[row, column].imshow(rgb, interpolation="nearest")
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
            for column, values in enumerate(
                (sample.input_e00_map, sample.output_e00_map), start=3
            ):
                image_artist = axes[row, column].imshow(
                    values,
                    cmap="viridis",
                    norm=norm,
                    interpolation="nearest",
                )
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
            metrics = sample.metrics
            axes[row, 0].set_ylabel(
                f"{sample.entry.sample_id}\n"
                f"in={float(metrics['input_e00']):.2f}  "
                f"out={float(metrics['output_png_e00']):.2f}\n"
                f"reduction={float(metrics['e00_reduction']):.2f}",
                rotation=0,
                ha="right",
                va="center",
                labelpad=8,
            )
        for column, title in enumerate(titles):
            axes[0, column].set_title(title)
        figure.subplots_adjust(
            left=0.17, right=0.91, bottom=0.025, top=0.965, wspace=0.05, hspace=0.08
        )
        if image_artist is None:
            raise RuntimeError("Candidate DeltaE00 artist was not created")
        colorbar = figure.colorbar(
            image_artist,
            ax=axes[:, 3:].reshape(-1).tolist(),
            fraction=0.025,
            pad=0.02,
        )
        colorbar.set_label("Delta E00")
        figure.savefig(
            path,
            dpi=min(dpi, 200),
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        plt.close(figure)
    return de_vmax


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    existing = [name for name in ANALYSIS_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Analysis outputs already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace these dedicated analysis files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_visuals_for_rows(
    rows: Sequence[Mapping[str, Any]],
    dataset: LSUIDataset,
    enhanced_paths: Mapping[str, Path],
    official_metrics: Mapping[str, Mapping[str, float | str]],
    *,
    crop_border: int,
) -> list[SampleAnalysis]:
    return [
        analyze_sample(
            dataset,
            int(row["_index"]),
            enhanced_paths[str(row["sample_id"])],
            official_metrics[str(row["sample_id"])],
            crop_border=crop_border,
        )
        for row in rows
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    run_dir = resolve_project_path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run lacks config_resolved.yaml: {run_dir}")
    config = load_yaml(config_path)
    version = config.get("experiment", {}).get("version")
    if version != EXPECTED_VERSION:
        raise ValueError(
            f"Chromatic restoration analysis requires a v16 run, got {version!r}"
        )
    data = config["data"]
    dataset_name = resolve_dataset_label(data)
    data_root = (
        resolve_project_path(args.data_root)
        if args.data_root is not None
        else resolve_project_path(data["root"])
    )
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"data.root is unavailable: {data_root}; use --data-root after moving data"
        )
    test_manifest = (run_dir / "split_snapshot" / "test.tsv").resolve()
    if not test_manifest.is_file():
        raise FileNotFoundError(f"Frozen test split is missing: {test_manifest}")
    entries = read_manifest(test_manifest)
    expected_counts = data.get("expected_counts")
    if expected_counts is not None and "test" in expected_counts:
        expected_test = int(expected_counts["test"])
        if len(entries) != expected_test:
            raise ValueError(
                f"Frozen test manifest has {len(entries)} rows, expected {expected_test}"
            )
    sample_ids = [entry.sample_id for entry in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Frozen test manifest contains duplicate sample_id values")

    result_dir = run_dir / "result"
    enhanced_paths = resolve_enhanced_paths(
        entries, result_dir / "test_all_enhanced", run_dir
    )
    official_metrics = read_official_metrics(
        result_dir / "test_metrics.csv", entries, run_dir
    )
    evaluation = config["evaluation"]
    dataset = LSUIDataset(
        test_manifest,
        data_root,
        "test",
        int(data["patch_size"]),
        data["augmentation"],
        bool(data["pad_if_smaller"]),
        str(data["pad_mode"]),
        evaluation,
        verify_files=True,
    )
    if dataset.entries != entries:
        raise RuntimeError("Dataset order differs from the frozen test manifest")
    crop_border = int(config.get("metrics", {}).get("crop_border", 0))

    rows: list[dict[str, Any]] = []
    if args.sample_ids is None:
        analysis_mode = "screening"
        indices = list(range(len(entries)))
    else:
        analysis_mode = "final_figure"
        index_by_id = {entry.sample_id: index for index, entry in enumerate(entries)}
        missing_ids = [sample_id for sample_id in args.sample_ids if sample_id not in index_by_id]
        if missing_ids:
            raise ValueError(f"Requested sample IDs are not in frozen test split: {missing_ids}")
        indices = [index_by_id[sample_id] for sample_id in args.sample_ids]
    for index in indices:
        entry = entries[index]
        analysis = analyze_sample(
            dataset,
            index,
            enhanced_paths[entry.sample_id],
            official_metrics[entry.sample_id],
            crop_border=crop_border,
        )
        rows.append(analysis.metrics)

    if analysis_mode == "screening":
        candidate_rows = select_representative_candidates(rows, args.num_candidates)
        figure_rows = candidate_rows[: min(2, len(candidate_rows))]
    else:
        candidate_rows = rows
        figure_rows = rows
    candidate_samples = _load_visuals_for_rows(
        candidate_rows,
        dataset,
        enhanced_paths,
        official_metrics,
        crop_border=crop_border,
    )
    candidate_by_index = {sample.index: sample for sample in candidate_samples}
    figure_samples = [candidate_by_index[int(row["_index"])] for row in figure_rows]

    a_edges = ab_bin_edges(args.ab_bins, args.ab_min, args.ab_max)
    b_edges = a_edges.copy()
    figure_maps = [
        values
        for sample in figure_samples
        for values in (sample.input_e00_map, sample.output_e00_map)
    ]
    figure_de_vmax = shared_delta_e_vmax(
        figure_maps,
        percentile=args.de_percentile,
        explicit_vmax=args.de_vmax,
    )
    output_dir = (
        run_dir / "result" / DEFAULT_OUTPUT_NAME
        if args.output_dir is None
        else resolve_project_path(args.output_dir)
    ).resolve()
    _prepare_output(output_dir, overwrite=args.overwrite)

    _write_csv(output_dir / "chromatic_per_sample.csv", rows, PER_SAMPLE_FIELDS)
    selected_rows = []
    figure_indices = {int(row["_index"]) for row in figure_rows}
    for rank, row in enumerate(candidate_rows, start=1):
        selected_rows.append(
            {
                "selection_rank": rank,
                "selection_mode": analysis_mode,
                "candidate_selection_rule": (
                    CANDIDATE_SELECTION_RULE
                    if analysis_mode == "screening"
                    else "Explicit --sample-ids in user-specified order."
                ),
                "used_in_final_figure": int(row["_index"]) in figure_indices,
                **row,
            }
        )
    _write_csv(
        output_dir / "selected_samples.csv",
        selected_rows,
        (
            "selection_rank",
            "selection_mode",
            "candidate_selection_rule",
            "used_in_final_figure",
            *PER_SAMPLE_FIELDS,
        ),
    )
    contact_de_vmax = save_candidate_contact_sheet(
        candidate_samples,
        output_dir / "candidate_contact_sheet.png",
        de_percentile=args.de_percentile,
        de_vmax_override=args.de_vmax,
        dpi=args.dpi,
    )
    save_final_figure(
        figure_samples,
        output_dir,
        a_edges=a_edges,
        b_edges=b_edges,
        de_vmax=figure_de_vmax,
        dpi=args.dpi,
        show_sample_id=args.show_sample_id,
    )

    protocol = {
        "script": "tools/plot_chromatic_restoration_analysis.py",
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "run_dir": str(run_dir),
        "dataset": dataset_name,
        "data_root": str(data_root),
        "test_manifest": str(test_manifest),
        "config_resolved_sha256": sha256_file(config_path),
        "test_manifest_sha256": sha256_file(test_manifest),
        "test_manifest_sample_count": len(entries),
        "analysis_mode": analysis_mode,
        "analyzed_sample_count": len(rows),
        "sample_count": len(figure_samples),
        "selected_sample_ids": [sample.entry.sample_id for sample in figure_samples],
        "candidate_sample_ids": [sample.entry.sample_id for sample in candidate_samples],
        "candidate_selection_rule": CANDIDATE_SELECTION_RULE,
        "evaluation_resize": bool(evaluation.get("resize", True)),
        "evaluation_size": evaluation.get("size"),
        "evaluation_resize_method": "PIL.Image.Resampling.BILINEAR for Input and GT",
        "enhanced_resolution_policy": (
            "saved official PNG must already match the evaluation-transformed Input/GT; "
            "the analysis does not resize it"
        ),
        "metrics_crop_border": crop_border,
        "lab_conversion_protocol": {
            "input": "standard sRGB float in [0,1]",
            "implementation": "skimage.color.rgb2lab",
            "illuminant": "D65",
            "observer": "2",
            "channel_axis": -1,
        },
        "delta_e00_implementation": (
            "src.shared.e00.delta_e00_from_lab; CIEDE2000 kL=kC=kH=1"
        ),
        "spatial_error_map_source": (
            "Input and Reference use official evaluation tensors; ICAR-UIE output uses "
            "the official saved 8-bit enhanced PNG"
        ),
        "ab_bins": args.ab_bins,
        "ab_range": [args.ab_min, args.ab_max],
        "ab_histogram_normalization": "probability mass divided by total image pixels",
        "ab_panel_normalization": "one shared LogNorm across every a*b* panel",
        "de_visualization_vmin": 0.0,
        "de_visualization_vmax": figure_de_vmax,
        "candidate_contact_sheet_de_vmax": contact_de_vmax,
        "de_percentile": args.de_percentile,
        "de_percentile_scope": (
            "pooled unmodified Input-vs-Reference and Output-vs-Reference pixels from "
            "the samples shown in each figure; clipping affects display only"
        ),
        "png_quantization_note": PNG_QUANTIZATION_NOTE,
        "png_quantization_warning_threshold": 0.1,
        "official_metrics_path": str(result_dir / "test_metrics.csv"),
        "enhanced_image_directory": str(result_dir / "test_all_enhanced"),
        "output_directory": str(output_dir),
        "dpi": args.dpi,
        "contact_sheet_dpi": min(args.dpi, 200),
        "show_sample_id": bool(args.show_sample_id),
        "model_or_checkpoint_loaded": False,
        "inference_performed": False,
    }
    atomic_json(output_dir / "analysis_protocol.json", protocol)
    print(
        f"Chromatic restoration analysis completed\n"
        f"Mode: {analysis_mode}\n"
        f"Dataset: {dataset_name}\n"
        f"Analyzed samples: {len(rows)}\n"
        f"Figure samples: {', '.join(protocol['selected_sample_ids'])}\n"
        f"Output directory: {output_dir}"
    )


if __name__ == "__main__":
    main()
