"""Human-readable Markdown reporting for LSUI diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import SPLIT_ORDER


def _format(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "∞" if number > 0 else "−∞"
    return f"{number:.{digits}f}"


def _difference(first: Any, second: Any) -> float | None:
    if first is None or second is None:
        return None
    first, second = float(first), float(second)
    return first - second if math.isfinite(first) and math.isfinite(second) else None


def _metric_table(difficulty: Mapping[str, Any], statistic: str) -> list[str]:
    lines = [
        f"| Metric ({statistic}) | Train | Validation | Test |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("psnr_256", "ssim_256", "mae_256", "psnr_native", "ssim_native", "mae_native"):
        lines.append(
            "| " + metric + " | "
            + " | ".join(_format(difficulty[split][metric][statistic]) for split in SPLIT_ORDER)
            + " |"
        )
    return lines


def _comparison_table(difficulty: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Difference | Mean | Median |",
        "|---|---:|---:|",
    ]
    for metric in ("psnr_256", "ssim_256", "mae_256"):
        for reference in ("train", "validation"):
            mean_delta = _difference(
                difficulty["test"][metric]["mean"], difficulty[reference][metric]["mean"]
            )
            median_delta = _difference(
                difficulty["test"][metric]["median"], difficulty[reference][metric]["median"]
            )
            lines.append(
                f"| Test {metric} − {reference.title()} | {_format(mean_delta)} | {_format(median_delta)} |"
            )
    return lines


def _extreme_table(rows: Sequence[Mapping[str, Any]], rank_type: str) -> list[str]:
    title = "Highest" if rank_type == "highest" else "Lowest"
    lines = [
        f"| Split | Rank | Sample | PSNR_256 | SSIM_256 | MAE_256 | Input path |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for split in SPLIT_ORDER:
        selected = [row for row in rows if row["split"] == split and row["rank_type"] == rank_type][:10]
        for row in selected:
            lines.append(
                f"| {split} | {row['rank']} | {row['sample_id']} | "
                f"{_format(row['psnr_256'])} | {_format(row['ssim_256'])} | "
                f"{_format(row['mae_256'])} | `{row['input_path']}` |"
            )
    if len(lines) == 2:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    return lines


def render_report(
    *,
    run_info: Mapping[str, Any],
    summary: Mapping[str, Any],
    extremes: Sequence[Mapping[str, Any]],
) -> str:
    integrity = summary["split_counts"]
    difficulty = summary["difficulty"]
    resize = summary["resize_effect"]
    duplicates = summary["exact_duplicates"]
    near = summary["near_duplicate_candidates"]
    resolution = summary["resolution"]
    image_statistics = summary["image_statistics"]
    source_rows = summary["source_folder_distribution"]

    lines = [
        "# LSUI Dataset Diagnostic Report",
        "",
        "## 1. Dataset configuration",
        "",
        f"- Timestamp: `{run_info['timestamp']}`",
        f"- Config: `{run_info['config_path']}`",
        f"- Data root: `{run_info['data_root']}`",
        f"- dHash threshold: `{run_info['dhash_threshold']}`",
        "- Diagnostic PSNR/SSIM protocol copied from current UIE4 evaluation implementation.",
        "- Current-256 protocol: RGB float `[0,1]`, paired 256×256 PIL bilinear resize, "
        "PSNR data range 1, and Gaussian-window SSIM (11×11, sigma 1.5 under the current config).",
        "- Native metrics are computed only when the decoded Input and GT dimensions match.",
        "",
        "## 2. Split integrity",
        "",
        "| Split | Actual | Expected |",
        "|---|---:|---:|",
    ]
    for split in SPLIT_ORDER:
        lines.append(
            f"| {split} | {integrity['counts'][split]} | {integrity['expected_counts'][split]} |"
        )
    lines.extend(
        [
            f"| **total** | **{integrity['total']}** | **{sum(integrity['expected_counts'].values())}** |",
            "",
            f"Counts match the requested fixed protocol: **{_format(integrity['counts_match_expected'])}**.",
            "",
            "Path-level overlap and within-split duplicate counts:",
            "",
            "| Check | Input paths | GT paths |",
            "|---|---:|---:|",
        ]
    )
    for split in SPLIT_ORDER:
        within = integrity["within_split_duplicates"][split]
        lines.append(
            f"| Within {split} | {within['input_path']['duplicate_occurrences']} | "
            f"{within['gt_path']['duplicate_occurrences']} |"
        )
    for name, overlap in integrity["cross_split_path_overlaps"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {overlap['input_path']['count']} | {overlap['gt_path']['count']} |"
        )
    lines.extend(
        [
            "",
            "Manifest source-folder distribution:",
            "",
            "| Split | Image type | Source folder | Count |",
            "|---|---|---|---:|",
        ]
    )
    for row in source_rows:
        lines.append(
            f"| {row['split']} | {row['image_type']} | `{row['source_folder']}` | {row['count']} |"
        )

    lines.extend(["", "## 3. Raw Input→GT difficulty", "", *_metric_table(difficulty, "mean")])
    lines.extend(["", "Median values:", "", *_metric_table(difficulty, "median")])
    lines.extend(
        [
            "",
            "Full count/mean/median/std/min/max and P05–P95 statistics are in "
            "`difficulty/split_summary.csv`.",
            "",
            "## 4. Train / Validation / Test comparison",
            "",
            *_comparison_table(difficulty),
            "",
            "Positive values mean the test statistic is numerically higher; negative values mean it is lower. "
            "These differences describe the observed distributions and do not by themselves establish dataset validity.",
            "",
            "## 5. Effect of 256×256 resizing",
            "",
            "| Split | Native-shape pairs | Mean ΔPSNR | Median ΔPSNR | Mean ΔSSIM | Median ΔSSIM |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLIT_ORDER:
        row = resize[split]
        lines.append(
            f"| {split} | {row['count']} | {_format(row['mean_delta_psnr'])} | "
            f"{_format(row['median_delta_psnr'])} | {_format(row['mean_delta_ssim'])} | "
            f"{_format(row['median_delta_ssim'])} |"
        )
    lines.extend(
        [
            "",
            "Here Δ is `Current-256 − native`; no native value is synthesized for shape-mismatched pairs.",
            "",
            "## 6. Resolution distribution",
            "",
            "| Split | Shape matches | Shape mismatches | Unique input resolutions | Median input W×H | Range W×H |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLIT_ORDER:
        row = resolution[split]
        lines.append(
            f"| {split} | {row['native_shape_match_count']} | {row['native_shape_mismatch_count']} | "
            f"{row['unique_input_resolution_count']} | {_format(row['input_width']['median'], 1)}×"
            f"{_format(row['input_height']['median'], 1)} | "
            f"{_format(row['input_width']['min'], 0)}–{_format(row['input_width']['max'], 0)} × "
            f"{_format(row['input_height']['min'], 0)}–{_format(row['input_height']['max'], 0)} |"
        )

    lines.extend(
        [
            "",
            "## 7. Basic color / luminance distribution",
            "",
            "| Split | Input mean luminance | GT mean luminance | Input mean saturation | GT mean saturation | Mean absolute RGB-mean change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLIT_ORDER:
        row = image_statistics[split]
        lines.append(
            f"| {split} | {_format(row['input_mean_luminance']['mean'])} | "
            f"{_format(row['gt_mean_luminance']['mean'])} | "
            f"{_format(row['input_mean_saturation']['mean'])} | "
            f"{_format(row['gt_mean_saturation']['mean'])} | "
            f"{_format(row['abs_mean_rgb_difference']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "Luminance uses fixed BT.709 RGB coefficients. Saturation is HSV saturation. These are dataset "
            "statistics, not perceptual-quality metrics. Signed luminance/saturation differences are `GT − Input`.",
            "",
            "## 8. Exact duplicate audit",
            "",
            f"- Raw-file duplicate groups: **{duplicates['file_duplicate_groups']}** "
            f"({duplicates['file_duplicate_pairs']} pairs).",
            f"- Decoded-pixel duplicate groups: **{duplicates['pixel_duplicate_groups']}** "
            f"({duplicates['pixel_duplicate_pairs']} pairs).",
            f"- Exact cross-split decoded-pixel duplicate pairs: "
            f"**{duplicates['cross_split_exact_pixel_duplicate_pairs']}**.",
            "",
            "An exact cross-split duplicate is reported only from matching decoded RGB pixel hashes "
            "(including width and height), not merely from filenames or dHash.",
            "",
            "## 9. Near-duplicate candidates",
            "",
            f"- Threshold: Hamming distance ≤ **{near['dhash_threshold']}**.",
            f"- Input candidates: **{near['near_duplicate_candidate_pairs_by_type']['input']}**.",
            f"- GT candidates: **{near['near_duplicate_candidate_pairs_by_type']['gt']}**.",
            "",
            "> Near-duplicate matches are candidates for manual inspection and are not automatically classified as data leakage.",
            "",
            "Candidate rows include auxiliary 128×128 RGB PSNR and MAE values.",
            "",
            "## 10. Most similar Input→GT samples",
            "",
            *_extreme_table(extremes, "highest"),
            "",
            "The CSV contains the top 20 per split; this report displays the first 10 per split.",
            "",
            "## 11. Hardest Input→GT samples",
            "",
            *_extreme_table(extremes, "lowest"),
            "",
            "The CSV contains the bottom 20 per split; this report displays the first 10 per split.",
            "",
            "## 12. Observations",
            "",
        ]
    )
    for metric in ("psnr_256", "ssim_256", "mae_256"):
        for reference in ("train", "validation"):
            delta = _difference(difficulty["test"][metric]["mean"], difficulty[reference][metric]["mean"])
            if delta is not None:
                if delta == 0:
                    lines.append(f"- The test mean {metric} is equal to the {reference} mean.")
                else:
                    direction = "higher than" if delta > 0 else "lower than"
                    lines.append(
                        f"- The test mean {metric} is {_format(abs(delta))} {direction} the {reference} mean."
                    )
    cross_count = duplicates["cross_split_exact_pixel_duplicate_pairs"]
    if cross_count:
        lines.append(f"- The audit found {cross_count} exact cross-split decoded-pixel duplicate pair(s).")
    else:
        lines.append("- The audit found no exact cross-split decoded-pixel duplicate pairs.")
    lines.extend(
        [
            "- Histogram plotting uses every finite value. Any non-finite values (for example infinite PSNR "
            "from identical pairs) are counted in the legend rather than silently clipped.",
            "- Interpretations beyond these numerical observations are intentionally left to the researcher.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    run_info: Mapping[str, Any],
    summary: Mapping[str, Any],
    extremes: Sequence[Mapping[str, Any]],
) -> None:
    path.write_text(render_report(run_info=run_info, summary=summary, extremes=extremes), encoding="utf-8")
