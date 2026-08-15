"""Streaming Input-to-GT difficulty and image-distribution diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .common import ManifestEntry, SPLIT_ORDER, descriptive_statistics, source_folder, write_csv
from .duplicates import ImageFingerprint, fingerprint_image
from .metrics import image_statistics, paired_metrics, resize_for_evaluation


METRIC_NAMES = (
    "psnr_256", "ssim_256", "mae_256", "mse_256",
    "psnr_native", "ssim_native", "mae_native", "mse_native",
)
SUMMARY_STAT_NAMES = (
    "count", "finite_count", "infinite_count", "mean", "median", "std", "min", "max",
    "p05", "p10", "p25", "p50", "p75", "p90", "p95",
)


@dataclass
class SampleDiagnostic:
    metrics: dict[str, Any]
    image_statistics: dict[str, Any]
    resolution: dict[str, Any]
    fingerprints: tuple[ImageFingerprint, ImageFingerprint]


def analyze_pair(
    *,
    split: str,
    entry: ManifestEntry,
    data_root: Path,
    evaluation_size: int,
    metric_config: Mapping[str, Any],
) -> SampleDiagnostic:
    input_path = data_root / entry.input_relative
    gt_path = data_root / entry.gt_relative
    missing = [str(path) for path in (input_path, gt_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing image file(s): " + ", ".join(missing))

    with Image.open(input_path) as input_handle, Image.open(gt_path) as gt_handle:
        input_handle.load()
        gt_handle.load()
        input_mode, gt_mode = input_handle.mode, gt_handle.mode
        input_image = input_handle.convert("RGB")
        gt_image = gt_handle.convert("RGB")

    input_width, input_height = input_image.size
    gt_width, gt_height = gt_image.size
    native_shape_match = input_image.size == gt_image.size
    metric_kwargs = {
        "data_range": float(metric_config.get("data_range", 1.0)),
        "window_size": int(metric_config.get("ssim_window_size", 11)),
        "sigma": float(metric_config.get("ssim_sigma", 1.5)),
        "crop_border": int(metric_config.get("crop_border", 0)),
    }
    resized_metrics = paired_metrics(
        resize_for_evaluation(input_image, evaluation_size),
        resize_for_evaluation(gt_image, evaluation_size),
        **metric_kwargs,
    )
    native_metrics = paired_metrics(input_image, gt_image, **metric_kwargs) if native_shape_match else None

    metrics_row: dict[str, Any] = {
        "sample_id": entry.sample_id,
        "split": split,
        "input_path": entry.input_relative,
        "gt_path": entry.gt_relative,
        "input_width": input_width,
        "input_height": input_height,
        "gt_width": gt_width,
        "gt_height": gt_height,
        "native_shape_match": native_shape_match,
        **{f"{name}_256": value for name, value in resized_metrics.items()},
        **{
            f"{name}_native": native_metrics[name] if native_metrics is not None else None
            for name in ("psnr", "ssim", "mae", "mse")
        },
    }

    input_stats = image_statistics(input_image)
    gt_stats = image_statistics(gt_image)
    mean_rgb_delta = np.asarray(
        [gt_stats[f"mean_{channel}"] - input_stats[f"mean_{channel}"] for channel in "rgb"]
    )
    image_row: dict[str, Any] = {
        "sample_id": entry.sample_id,
        "split": split,
        "input_path": entry.input_relative,
        "gt_path": entry.gt_relative,
        **{f"input_{name}": value for name, value in input_stats.items()},
        **{f"gt_{name}": value for name, value in gt_stats.items()},
        "abs_mean_rgb_difference": float(np.abs(mean_rgb_delta).mean()),
        "luminance_difference": gt_stats["mean_luminance"] - input_stats["mean_luminance"],
        "saturation_difference": gt_stats["mean_saturation"] - input_stats["mean_saturation"],
        "abs_luminance_difference": abs(gt_stats["mean_luminance"] - input_stats["mean_luminance"]),
        "abs_saturation_difference": abs(gt_stats["mean_saturation"] - input_stats["mean_saturation"]),
    }
    resolution_row = {
        "sample_id": entry.sample_id,
        "split": split,
        "input_path": entry.input_relative,
        "gt_path": entry.gt_relative,
        "input_width": input_width,
        "input_height": input_height,
        "input_aspect_ratio": input_width / input_height,
        "gt_width": gt_width,
        "gt_height": gt_height,
        "gt_aspect_ratio": gt_width / gt_height,
        "native_shape_match": native_shape_match,
        "input_original_mode": input_mode,
        "gt_original_mode": gt_mode,
    }
    fingerprints = (
        fingerprint_image(
            split=split,
            sample_id=entry.sample_id,
            image_type="input",
            relative_path=entry.input_relative,
            absolute_path=input_path,
            image=input_image,
        ),
        fingerprint_image(
            split=split,
            sample_id=entry.sample_id,
            image_type="gt",
            relative_path=entry.gt_relative,
            absolute_path=gt_path,
            image=gt_image,
        ),
    )
    return SampleDiagnostic(metrics_row, image_row, resolution_row, fingerprints)


def summarize_difficulty(metrics_by_split: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    return {
        split: {
            metric: descriptive_statistics(row[metric] for row in metrics_by_split[split])
            for metric in METRIC_NAMES
        }
        for split in SPLIT_ORDER
    }


def split_summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"split": split, "metric": metric, **summary[split][metric]}
        for split in SPLIT_ORDER
        for metric in METRIC_NAMES
    ]


def summarize_resize_effect(metrics_by_split: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        rows = [row for row in metrics_by_split[split] if row["native_shape_match"]]
        delta_psnr = [row["psnr_256"] - row["psnr_native"] for row in rows]
        delta_ssim = [row["ssim_256"] - row["ssim_native"] for row in rows]
        psnr_stats = descriptive_statistics(delta_psnr)
        ssim_stats = descriptive_statistics(delta_ssim)
        result[split] = {
            "count": min(psnr_stats["count"], ssim_stats["count"]),
            "native_shape_match_count": len(rows),
            "mean_delta_psnr": psnr_stats["mean"],
            "median_delta_psnr": psnr_stats["median"],
            "mean_delta_ssim": ssim_stats["mean"],
            "median_delta_ssim": ssim_stats["median"],
        }
    return result


def summarize_resolution(resolution_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        rows = [row for row in resolution_rows if row["split"] == split]
        input_resolutions = sorted({f"{row['input_width']}x{row['input_height']}" for row in rows})
        gt_resolutions = sorted({f"{row['gt_width']}x{row['gt_height']}" for row in rows})
        result[split] = {
            "count": len(rows),
            "native_shape_match_count": sum(bool(row["native_shape_match"]) for row in rows),
            "native_shape_mismatch_count": sum(not bool(row["native_shape_match"]) for row in rows),
            "unique_input_resolution_count": len(input_resolutions),
            "unique_input_resolutions": input_resolutions,
            "unique_gt_resolution_count": len(gt_resolutions),
            "unique_gt_resolutions": gt_resolutions,
            "input_width": descriptive_statistics(row["input_width"] for row in rows),
            "input_height": descriptive_statistics(row["input_height"] for row in rows),
            "gt_width": descriptive_statistics(row["gt_width"] for row in rows),
            "gt_height": descriptive_statistics(row["gt_height"] for row in rows),
            "input_aspect_ratio": descriptive_statistics(row["input_aspect_ratio"] for row in rows),
            "gt_aspect_ratio": descriptive_statistics(row["gt_aspect_ratio"] for row in rows),
        }
    return result


def summarize_image_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = tuple(
        field
        for field in rows[0].keys()
        if field.startswith(("input_", "gt_")) and not field.endswith("_path")
    ) + (
        "abs_mean_rgb_difference", "luminance_difference", "saturation_difference",
        "abs_luminance_difference", "abs_saturation_difference",
    ) if rows else ()
    return {
        split: {
            field: descriptive_statistics(row[field] for row in rows if row["split"] == split)
            for field in fields
        }
        for split in SPLIT_ORDER
    }


def difficulty_extremes(metrics_by_split: Mapping[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        rows = list(metrics_by_split[split])
        selections = (
            ("highest", sorted(rows, key=lambda row: row["psnr_256"], reverse=True)[:20]),
            ("lowest", sorted(rows, key=lambda row: row["psnr_256"])[:20]),
        )
        for rank_type, selected in selections:
            for rank, row in enumerate(selected, start=1):
                output.append(
                    {
                        "split": split,
                        "rank_type": rank_type,
                        "rank": rank,
                        "sample_id": row["sample_id"],
                        "input_path": row["input_path"],
                        "gt_path": row["gt_path"],
                        "psnr_256": row["psnr_256"],
                        "ssim_256": row["ssim_256"],
                        "mae_256": row["mae_256"],
                    }
                )
    return output


def source_distribution(entries: Mapping[str, Sequence[ManifestEntry]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        for image_type, field in (("input", "input_relative"), ("gt", "gt_relative")):
            counts = Counter(source_folder(getattr(entry, field)) for entry in entries[split])
            for folder, count in sorted(counts.items()):
                rows.append(
                    {"split": split, "image_type": image_type, "source_folder": folder, "count": count}
                )
    return rows


def write_histograms(directory: Path, metrics_by_split: Mapping[str, Sequence[dict[str, Any]]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate diagnostic histograms") from error

    plots = (
        ("psnr_256", "Input→GT PSNR (Current-256)", "PSNR (dB)", "input_gt_psnr_hist.png"),
        ("ssim_256", "Input→GT SSIM (Current-256)", "SSIM", "input_gt_ssim_hist.png"),
        ("mae_256", "Input→GT MAE (Current-256)", "MAE", "input_gt_mae_hist.png"),
    )
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    for metric, title, xlabel, filename in plots:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        finite_by_split: dict[str, np.ndarray] = {}
        for split in SPLIT_ORDER:
            raw = np.asarray([row[metric] for row in metrics_by_split[split]], dtype=np.float64)
            finite_by_split[split] = raw[np.isfinite(raw)]
        finite_groups = [values for values in finite_by_split.values() if values.size]
        combined = np.concatenate(finite_groups) if finite_groups else np.asarray([], dtype=np.float64)
        bins: int | np.ndarray = 40
        if combined.size and float(combined.min()) != float(combined.max()):
            bins = np.histogram_bin_edges(combined, bins=40)
        for split in SPLIT_ORDER:
            raw = np.asarray([row[metric] for row in metrics_by_split[split]], dtype=np.float64)
            finite = finite_by_split[split]
            omitted = int(raw.size - finite.size)
            label = split if omitted == 0 else f"{split} ({omitted} non-finite omitted)"
            if finite.size:
                axis.hist(finite, bins=bins, alpha=0.42, label=label, color=colors[split])
            else:
                axis.plot([], [], color=colors[split], label=label)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Image count")
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / filename, dpi=160)
        plt.close(figure)


METRICS_FIELDS = (
    "sample_id", "split", "input_path", "gt_path", "input_width", "input_height",
    "gt_width", "gt_height", "native_shape_match", *METRIC_NAMES,
)
IMAGE_STAT_FIELDS = (
    "sample_id", "split", "input_path", "gt_path",
    *(f"input_{name}" for name in (
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
        "mean_luminance", "luminance_std", "mean_saturation",
    )),
    *(f"gt_{name}" for name in (
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
        "mean_luminance", "luminance_std", "mean_saturation",
    )),
    "abs_mean_rgb_difference", "luminance_difference", "saturation_difference",
    "abs_luminance_difference", "abs_saturation_difference",
)
RESOLUTION_FIELDS = (
    "sample_id", "split", "input_path", "gt_path", "input_width", "input_height",
    "input_aspect_ratio", "gt_width", "gt_height", "gt_aspect_ratio",
    "native_shape_match", "input_original_mode", "gt_original_mode",
)


def write_difficulty_outputs(
    directory: Path,
    *,
    metrics_by_split: Mapping[str, Sequence[dict[str, Any]]],
    image_rows: Sequence[dict[str, Any]],
    resolution_rows: Sequence[dict[str, Any]],
    split_summary: Mapping[str, Any],
    resize_summary: Mapping[str, Any],
    source_rows: Sequence[dict[str, Any]],
    generate_plots: bool = True,
) -> None:
    for split in SPLIT_ORDER:
        write_csv(directory / f"{split}_metrics.csv", metrics_by_split[split], METRICS_FIELDS)
    write_csv(
        directory / "split_summary.csv",
        split_summary_rows(split_summary),
        ("split", "metric", *SUMMARY_STAT_NAMES),
    )
    write_csv(directory / "image_statistics.csv", image_rows, IMAGE_STAT_FIELDS)
    write_csv(directory / "resolution_statistics.csv", resolution_rows, RESOLUTION_FIELDS)
    write_csv(
        directory / "resize_effect_summary.csv",
        [{"split": split, **resize_summary[split]} for split in SPLIT_ORDER],
        (
            "split", "count", "mean_delta_psnr", "median_delta_psnr",
            "mean_delta_ssim", "median_delta_ssim",
        ),
    )
    write_csv(
        directory / "difficulty_extremes.csv",
        difficulty_extremes(metrics_by_split),
        ("split", "rank_type", "rank", "sample_id", "input_path", "gt_path", "psnr_256", "ssim_256", "mae_256"),
    )
    write_csv(
        directory / "source_folder_distribution.csv",
        source_rows,
        ("split", "image_type", "source_folder", "count"),
    )
    if generate_plots:
        write_histograms(directory, metrics_by_split)
