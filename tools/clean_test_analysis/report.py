"""Fact-only Markdown report for clean-test sensitivity results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .statistics import CORE_SUBSETS, MODEL_ORDER, SUSPECT_SUBSETS


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if math.isnan(number):
        return "N/A"
    if math.isinf(number):
        return "∞" if number > 0 else "−∞"
    return f"{number:.{digits}f}"


def _model_table(subset_metrics: Sequence[Mapping[str, Any]], metric: str) -> list[str]:
    lookup = {(row["subset"], row["model"]): row for row in subset_metrics}
    lines = [
        f"| Subset | N | Identity | Point-INR | GL-INR |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset in CORE_SUBSETS:
        rows = [lookup[(subset, model)] for model in MODEL_ORDER]
        lines.append(
            f"| {subset} | {rows[0]['n']} | "
            + " | ".join(_fmt(row[f"{metric}_mean"]) for row in rows)
            + " |"
        )
    return lines


def _paired_table(pairwise: Sequence[Mapping[str, Any]], metric: str) -> list[str]:
    lines = [
        "| Subset | Comparison | Mean Δ | Median Δ | Wins/Losses/Ties | Win rate | Bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for subset in CORE_SUBSETS:
        for row in pairwise:
            if row["subset"] == subset and row["metric"] == metric:
                lines.append(
                    f"| {subset} | {row['comparison']} | {_fmt(row['mean_delta'])} | "
                    f"{_fmt(row['median_delta'])} | {row['positive_count']}/{row['negative_count']}/"
                    f"{row['tie_count']} | {_fmt(row['win_rate'])} | "
                    f"[{_fmt(row['bootstrap_ci_low'])}, {_fmt(row['bootstrap_ci_high'])}] |"
                )
    return lines


def render_report(
    *,
    provenance: Mapping[str, Any],
    alignment: Mapping[str, Any],
    candidate_statistics: Mapping[str, Any],
    subsets: Mapping[str, set[str]],
    raw_rows: Sequence[Mapping[str, Any]],
    subset_metrics: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
    difficulty_correlations: Sequence[Mapping[str, Any]],
    near_correlations: Sequence[Mapping[str, Any]],
    strong_psnr_threshold: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> str:
    raw_lookup = {row["subset"]: row for row in raw_rows}
    metric_lookup = {(row["subset"], row["model"]): row for row in subset_metrics}
    delta_lookup = {
        (row["subset"], row["comparison"], row["metric"]): row for row in pairwise
    }
    lines = [
        "# Clean-Test Sensitivity Analysis",
        "",
        "## 1. Inputs and provenance",
        "",
        f"- Diagnostic directory: `{provenance['diagnostic_dir']}`",
        f"- Identity run: `{provenance['v1_run']}`",
        f"- Point-INR run: `{provenance['v2_run']}`",
        f"- GL-INR run: `{provenance['v3_run']}`",
        f"- Test manifest: `{provenance['test_manifest']}`",
        "- Analysis mode: post-hoc per-image CSV analysis only; no checkpoint, model inference, "
        "enhanced image, or recomputed PNG metric was used.",
        "",
        "Detected input schemas:",
        "",
        "| Input | Fields |",
        "|---|---|",
    ]
    for label, fields in provenance["schemas"].items():
        lines.append(f"| {label} | `{', '.join(fields)}` |")
    lines.extend(
        [
            "",
            "## 2. Sample alignment",
            "",
            f"- Sample alignment: **{alignment['status']}**",
            f"- Aligned test samples: **{alignment['aligned_test_samples']}**",
            f"- Canonical ID sets identical: **{alignment['canonical_id_sets_identical']}**",
            "",
            "## 3. Near-duplicate candidate statistics",
            "",
            f"- Normalized cross-split counterpart pairs: **{candidate_statistics['normalized_pair_count']}**",
            f"- Candidate test samples: **{candidate_statistics['candidate_test_sample_count']}**",
            f"- Train↔test pairs: **{candidate_statistics['train_test_pair_count']}**",
            f"- Validation↔test pairs: **{candidate_statistics['validation_test_pair_count']}**",
            f"- Diagnostic dHash threshold: **{_fmt(candidate_statistics['diagnostic_dhash_threshold'])}**.",
            f"- Strong Input threshold: **{strong_psnr_threshold:g} dB** at 128×128.",
            "",
            "Near-duplicate candidates are audit candidates, not automatically classified as a leaked set.",
            "",
            "## 4. Subset definitions",
            "",
            "- **Full:** all aligned test samples.",
            "- **Clean-A:** removes every test sample appearing in any cross-split Input or GT dHash candidate. "
            "This is an aggressive candidate filter because dHash similarity is not confirmation of leakage.",
            f"- **Clean-B:** removes test samples having Input candidate PSNR_128 ≥ {strong_psnr_threshold:g} dB.",
            f"- **Clean-C:** removes a test sample only when the same non-test counterpart pair has both "
            f"Input and GT candidate PSNR_128 ≥ {strong_psnr_threshold:g} dB.",
            "- **Hard-Half:** the lower-PSNR half ranked only by raw Input→GT PSNR_256, with sample ID as tie-breaker.",
            "",
            "## 5. Subset sizes",
            "",
            "| Subset | N | Removed from Full |",
            "|---|---:|---:|",
        ]
    )
    full_n = len(subsets["Full"])
    for subset in CORE_SUBSETS:
        lines.append(f"| {subset} | {len(subsets[subset])} | {full_n - len(subsets[subset])} |")
    lines.extend(
        [
            "",
            "## 6. Raw Input→GT difficulty",
            "",
            "| Subset | N | Mean raw PSNR_256 | Median raw PSNR_256 | Mean raw SSIM_256 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for subset in CORE_SUBSETS:
        row = raw_lookup[subset]
        lines.append(
            f"| {subset} | {row['n']} | {_fmt(row['raw_psnr_mean'])} | "
            f"{_fmt(row['raw_psnr_median'])} | {_fmt(row['raw_ssim_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 7. Identity / Point-INR / GL-INR results",
            "",
            "Mean PSNR:",
            "",
            *_model_table(subset_metrics, "psnr"),
            "",
            "Mean SSIM:",
            "",
            *_model_table(subset_metrics, "ssim"),
            "",
            "## 8. Paired PSNR differences",
            "",
            *_paired_table(pairwise, "PSNR"),
            "",
            "## 9. Paired SSIM differences",
            "",
            *_paired_table(pairwise, "SSIM"),
            "",
            "## 10. Bootstrap confidence intervals",
            "",
            f"Each interval above is the 2.5th–97.5th percentile interval of "
            f"**{bootstrap_iterations}** paired bootstrap mean resamples (base seed **{bootstrap_seed}**). "
            "These are exploratory 95% bootstrap confidence intervals for paired mean differences; "
            "the report does not automatically label results statistically significant.",
            "",
            "## 11. Near-duplicate suspect subsets",
            "",
            "| Subset | N | Identity PSNR/SSIM | Point-INR PSNR/SSIM | GL-INR PSNR/SSIM |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for subset in SUSPECT_SUBSETS:
        rows = [metric_lookup[(subset, model)] for model in MODEL_ORDER]
        lines.append(
            f"| {subset} | {rows[0]['n']} | "
            + " | ".join(
                f"{_fmt(row['psnr_mean'])} / {_fmt(row['ssim_mean'])}" for row in rows
            )
            + " |"
        )
    hard = raw_lookup["Hard-Half"]
    lines.extend(
        [
            "",
            "These are near-duplicate suspect subsets (`Full − Clean-X`), not sets automatically called leaked.",
            "",
            "## 12. Hard-Half results",
            "",
            f"Hard-Half contains **{hard['n']}** samples. Its mean/median raw Input→GT PSNR_256 is "
            f"**{_fmt(hard['raw_psnr_mean'])} / {_fmt(hard['raw_psnr_median'])} dB**. "
            "Its model results are included in the main tables above.",
            "",
            "## 13. Exploratory correlations",
            "",
            "Raw difficulty versus model-output PSNR:",
            "",
            "| Model | N | Pearson | Spearman |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in difficulty_correlations:
        lines.append(
            f"| {row['model']} | {row['n']} | {_fmt(row['pearson'])} | {_fmt(row['spearman'])} |"
        )
    lines.extend(
        [
            "",
            "Near-duplicate strength correlations are listed in `near_duplicate_correlation.csv`; "
            "they are exploratory associations and are not causal estimates.",
            "",
            "## 14. Observations",
            "",
            f"- Clean-C contains {len(subsets['Clean-C'])} of the {full_n} aligned Full samples.",
        ]
    )
    for subset in ("Full", "Clean-C"):
        gl_i = delta_lookup[(subset, "GL-INR - Identity", "PSNR")]
        gl_p = delta_lookup[(subset, "GL-INR - Point-INR", "PSNR")]
        lines.append(
            f"- On {subset}, mean paired GL-INR − Identity PSNR is {_fmt(gl_i['mean_delta'])} dB "
            f"and GL-INR − Point-INR PSNR is {_fmt(gl_p['mean_delta'])} dB."
        )
    if subsets["Suspect-C"]:
        suspect_identity = metric_lookup[("Suspect-C", "Identity")]["psnr_mean"]
        clean_identity = metric_lookup[("Clean-C", "Identity")]["psnr_mean"]
        relation = "higher than" if suspect_identity > clean_identity else "lower than" if suspect_identity < clean_identity else "equal to"
        lines.append(
            f"- Identity mean PSNR on Suspect-C is {_fmt(suspect_identity)} dB, which is {relation} "
            f"its Clean-C mean of {_fmt(clean_identity)} dB."
        )
    lines.extend(
        [
            "- All statements above describe the supplied CSV values. No causal or benchmark-validity conclusion is made.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, **kwargs: Any) -> None:
    path.write_text(render_report(**kwargs), encoding="utf-8")
