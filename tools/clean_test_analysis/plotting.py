"""Matplotlib-only figures for clean-test sensitivity analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .statistics import CORE_SUBSETS, MODEL_ORDER, paired_delta_values


COLORS = {"Identity": "#4C78A8", "Point-INR": "#F58518", "GL-INR": "#54A24B"}


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for clean-test analysis plots") from error
    return plt


def _grouped_metric_bar(
    output: Path,
    subset_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> None:
    plt = _pyplot()
    lookup = {(row["subset"], row["model"]): row for row in subset_rows}
    x = np.arange(len(CORE_SUBSETS), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.6))
    for offset, model in zip((-width, 0.0, width), MODEL_ORDER):
        values = [lookup[(subset, model)][f"{metric}_mean"] for subset in CORE_SUBSETS]
        axis.bar(x + offset, values, width, label=model, color=COLORS[model])
    axis.set_xticks(x, CORE_SUBSETS)
    axis.set_ylabel(f"Mean {metric.upper()}")
    axis.set_title(f"Model {metric.upper()} across fixed test subsets")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _gain_bar(output: Path, gain_rows: Sequence[Mapping[str, Any]], metric: str) -> None:
    plt = _pyplot()
    lookup = {row["subset"]: row for row in gain_rows}
    x = np.arange(len(CORE_SUBSETS), dtype=np.float64)
    width = 0.34
    identity_key = f"glinr_minus_identity_{metric}"
    point_key = f"glinr_minus_point_{metric}"
    figure, axis = plt.subplots(figsize=(10, 5.6))
    axis.bar(
        x - width / 2,
        [lookup[subset][identity_key] for subset in CORE_SUBSETS],
        width,
        label="GL-INR - Identity",
        color="#54A24B",
    )
    axis.bar(
        x + width / 2,
        [lookup[subset][point_key] for subset in CORE_SUBSETS],
        width,
        label="GL-INR - Point-INR",
        color="#B279A2",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, CORE_SUBSETS)
    axis.set_ylabel(f"Mean paired Δ{metric.upper()}")
    axis.set_title(f"GL-INR paired {metric.upper()} gain sensitivity")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _paired_psnr_boxplot(output: Path, subsets, model_metrics) -> None:
    plt = _pyplot()
    data, labels, colors = [], [], []
    comparisons = (
        ("Point-INR", "Identity", "P-I", "#F58518"),
        ("GL-INR", "Identity", "G-I", "#54A24B"),
        ("GL-INR", "Point-INR", "G-P", "#B279A2"),
    )
    for subset in CORE_SUBSETS:
        for left, right, short, color in comparisons:
            values = paired_delta_values(
                subsets[subset], model_metrics[left], model_metrics[right], "psnr"
            )
            data.append(values if values.size else np.asarray([np.nan]))
            labels.append(f"{subset}\n{short}")
            colors.append(color)
    figure, axis = plt.subplots(figsize=(15, 6))
    boxes = axis.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.55)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("Per-image paired ΔPSNR (dB)")
    axis.set_title("Paired PSNR differences without outlier removal")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _full_vs_clean_c_scatter(output: Path, subsets, model_metrics) -> None:
    plt = _pyplot()
    retained = subsets["Clean-C"]
    excluded = subsets["Full"] - retained
    figure, axis = plt.subplots(figsize=(7, 7))
    all_values: list[float] = []
    for sample_ids, label, marker, color in (
        (retained, "Clean-C retained", "o", "#4C78A8"),
        (excluded, "Clean-C excluded suspect", "x", "#E45756"),
    ):
        ordered = sorted(sample_ids)
        x = [model_metrics["Identity"][sample_id].psnr for sample_id in ordered]
        y = [model_metrics["GL-INR"][sample_id].psnr for sample_id in ordered]
        all_values.extend(x)
        all_values.extend(y)
        axis.scatter(x, y, label=label, marker=marker, color=color, alpha=0.7)
    if all_values:
        lower, upper = min(all_values), max(all_values)
        axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", label="y = x")
    axis.set_xlabel("Identity per-image PSNR (dB)")
    axis.set_ylabel("GL-INR per-image PSNR (dB)")
    axis.set_title("Full test: Identity versus GL-INR")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def generate_plots(
    directory: Path,
    *,
    subset_rows: Sequence[Mapping[str, Any]],
    gain_rows: Sequence[Mapping[str, Any]],
    subsets,
    model_metrics,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _grouped_metric_bar(directory / "subset_psnr_bar.png", subset_rows, "psnr")
    _grouped_metric_bar(directory / "subset_ssim_bar.png", subset_rows, "ssim")
    _gain_bar(directory / "glinr_psnr_gain.png", gain_rows, "psnr")
    _gain_bar(directory / "glinr_ssim_gain.png", gain_rows, "ssim")
    _paired_psnr_boxplot(directory / "paired_psnr_delta_boxplot.png", subsets, model_metrics)
    _full_vs_clean_c_scatter(directory / "full_vs_clean_c_scatter.png", subsets, model_metrics)
