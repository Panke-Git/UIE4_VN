#!/usr/bin/env python3
"""Post-hoc clean-test sensitivity analysis from existing per-image CSV metrics."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.clean_test_analysis.io import (
    AnalysisInputError,
    read_csv_rows,
    read_difficulty_metrics,
    read_json,
    read_model_metrics,
    read_test_manifest,
    sha256_file,
    validate_run_version,
    write_csv,
    write_json,
)
from tools.clean_test_analysis.plotting import generate_plots
from tools.clean_test_analysis.report import write_report
from tools.clean_test_analysis.statistics import (
    CORE_SUBSETS,
    MODEL_ORDER,
    SUSPECT_SUBSETS,
    compute_pairwise_differences,
    compute_subset_metrics,
    difficulty_correlations,
    glinr_gain_summary,
    near_duplicate_correlations,
    per_sample_rows,
    raw_difficulty_summary,
)
from tools.clean_test_analysis.subsets import (
    build_subsets,
    candidate_pair_rows,
    normalize_candidate_pairs,
)


SUBSET_FIELDS = (
    "sample_id", "input_path", "gt_path", "raw_input_gt_psnr_256", "raw_input_gt_ssim_256"
)
EXCLUDED_FIELDS = (
    "sample_id", "excluded_from_clean_a", "excluded_from_clean_b", "excluded_from_clean_c",
    "reason_clean_a", "reason_clean_b", "reason_clean_c", "best_input_candidate_psnr_128",
    "best_gt_candidate_psnr_128", "matched_non_test_split", "matched_non_test_sample_id",
    "candidate_count",
)
CANDIDATE_PAIR_FIELDS = (
    "non_test_split", "non_test_sample_id", "test_sample_id", "input_dhash_distance",
    "input_candidate_psnr_128", "input_candidate_mae_128", "gt_dhash_distance",
    "gt_candidate_psnr_128", "gt_candidate_mae_128",
)
SUBSET_METRIC_FIELDS = (
    "subset", "model", "n", "psnr_mean", "psnr_median", "psnr_std",
    "ssim_mean", "ssim_median", "ssim_std",
)
PAIRWISE_FIELDS = (
    "subset", "comparison", "left_model", "right_model", "metric", "n", "mean_delta",
    "median_delta", "std_delta", "min_delta", "max_delta", "positive_count", "negative_count",
    "tie_count", "win_rate", "bootstrap_ci_low", "bootstrap_ci_high",
)
GAIN_FIELDS = (
    "subset", "n", "identity_psnr", "point_inr_psnr", "glinr_psnr",
    "glinr_minus_identity_psnr", "glinr_minus_point_psnr", "identity_ssim",
    "point_inr_ssim", "glinr_ssim", "glinr_minus_identity_ssim", "glinr_minus_point_ssim",
)
PER_SAMPLE_FIELDS = (
    "sample_id", "input_path", "gt_path", "raw_input_gt_psnr", "raw_input_gt_ssim",
    "identity_psnr", "identity_ssim", "point_psnr", "point_ssim", "glinr_psnr",
    "glinr_ssim", "point_minus_identity_psnr", "glinr_minus_identity_psnr",
    "glinr_minus_point_psnr", "point_minus_identity_ssim", "glinr_minus_identity_ssim",
    "glinr_minus_point_ssim", "near_duplicate_candidate", "strong_input_candidate",
    "strong_paired_candidate", "best_input_candidate_psnr_128", "best_gt_candidate_psnr_128",
    "in_clean_a", "in_clean_b", "in_clean_c", "in_hard_half",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _select_output(path: Path | None) -> Path:
    output = path or PROJECT_ROOT / "analysis" / f"clean_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Analysis output directory already exists: {output}")
    return output


def _create_output(output: Path) -> None:
    (output / "subsets").mkdir(parents=True)
    (output / "plots").mkdir()


def _assert_output_is_read_only_safe(output: Path, inputs: Sequence[Path]) -> None:
    for protected_name in ("src", "configs", "split", "experiments", "diagnostics"):
        protected = (PROJECT_ROOT / protected_name).resolve()
        if output == protected or protected in output.parents:
            raise ValueError(f"Analysis output must not be written beneath project {protected_name}/")
    for source in inputs:
        if output == source or source in output.parents:
            raise ValueError(f"Output directory must not be inside a read-only input directory: {source}")


def _membership_rows(sample_ids, samples, difficulty) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "input_path": sample.input_path,
            "gt_path": sample.gt_path,
            "raw_input_gt_psnr_256": difficulty[sample.sample_id].psnr_256,
            "raw_input_gt_ssim_256": difficulty[sample.sample_id].ssim_256,
        }
        for sample in sorted(samples.values(), key=lambda item: item.order)
        if sample.sample_id in sample_ids
    ]


def _nested_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        result.setdefault(row["subset"], {})[row["model"]] = {
            key: value for key, value in row.items() if key not in {"subset", "model"}
        }
    return result


def _nested_pairwise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        result.setdefault(row["subset"], {}).setdefault(row["comparison"], {})[row["metric"]] = {
            key: value
            for key, value in row.items()
            if key not in {"subset", "comparison", "left_model", "right_model", "metric"}
        }
    return result


def _print_primary_tables(subset_metrics, pairwise) -> None:
    lookup = {(row["subset"], row["model"]): row for row in subset_metrics}
    print("\nPSNR")
    print("Subset\tN\tIdentity\tPoint-INR\tGL-INR")
    for subset in CORE_SUBSETS:
        rows = [lookup[(subset, model)] for model in MODEL_ORDER]
        print(
            f"{subset}\t{rows[0]['n']}\t{rows[0]['psnr_mean']:.6f}\t"
            f"{rows[1]['psnr_mean']:.6f}\t{rows[2]['psnr_mean']:.6f}"
        )
    print("\nSSIM")
    print("Subset\tN\tIdentity\tPoint-INR\tGL-INR")
    for subset in CORE_SUBSETS:
        rows = [lookup[(subset, model)] for model in MODEL_ORDER]
        print(
            f"{subset}\t{rows[0]['n']}\t{rows[0]['ssim_mean']:.6f}\t"
            f"{rows[1]['ssim_mean']:.6f}\t{rows[2]['ssim_mean']:.6f}"
        )
    print("\nGL-INR paired PSNR gains")
    print("Subset\tGL-I mean [95% CI]\tGL-P mean [95% CI]")
    lookup_delta = {
        (row["subset"], row["comparison"]): row
        for row in pairwise
        if row["metric"] == "PSNR"
    }
    for subset in CORE_SUBSETS:
        gl_i = lookup_delta[(subset, "GL-INR - Identity")]
        gl_p = lookup_delta[(subset, "GL-INR - Point-INR")]
        print(
            f"{subset}\t{gl_i['mean_delta']:.6f} [{gl_i['bootstrap_ci_low']:.6f}, "
            f"{gl_i['bootstrap_ci_high']:.6f}]\t{gl_p['mean_delta']:.6f} "
            f"[{gl_p['bootstrap_ci_low']:.6f}, {gl_p['bootstrap_ci_high']:.6f}]"
        )


def run_analysis(
    *,
    diagnostic_dir: Path,
    v1_run: Path,
    v2_run: Path,
    v3_run: Path,
    output_dir: Path | None,
    strong_psnr_threshold: float = 35.0,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 3407,
    expected_test_count: int = 428,
    generate_figures: bool = True,
    test_manifest_path: Path | None = None,
) -> Path:
    diagnostic_dir, v1_run, v2_run, v3_run = map(
        lambda path: path.expanduser().resolve(), (diagnostic_dir, v1_run, v2_run, v3_run)
    )
    output = _select_output(output_dir)
    _assert_output_is_read_only_safe(output, (diagnostic_dir, v1_run, v2_run, v3_run))
    if generate_figures and importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("matplotlib is required; install requirements.txt before analysis")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")

    manifest_path = (test_manifest_path or PROJECT_ROOT / "split/lsui19/test.tsv").resolve()
    samples = read_test_manifest(manifest_path, expected_count=expected_test_count)
    difficulty_path = diagnostic_dir / "difficulty/test_metrics.csv"
    candidate_path = diagnostic_dir / "duplicates/near_duplicate_candidates.csv"
    difficulty, difficulty_fields = read_difficulty_metrics(difficulty_path, samples)
    candidate_rows, candidate_fields = read_csv_rows(candidate_path, allow_empty=True)

    run_specs = (
        ("Identity", "v1", v1_run),
        ("Point-INR", "v2", v2_run),
        ("GL-INR", "v3", v3_run),
    )
    model_metrics = {}
    model_fields = {}
    for model, version, run_dir in run_specs:
        validate_run_version(run_dir, version)
        metrics, fields = read_model_metrics(
            run_dir / "result/test_metrics.csv", samples, model_label=model
        )
        model_metrics[model] = metrics
        model_fields[model] = fields

    schemas = {
        "diagnostic difficulty": difficulty_fields,
        "near-duplicate candidates": candidate_fields,
        **{f"{model} test metrics": fields for model, fields in model_fields.items()},
    }
    for label, fields in schemas.items():
        print(f"schema [{label}]: {fields}")

    pairs = normalize_candidate_pairs(candidate_rows, candidate_fields, set(samples))
    subsets, evidence = build_subsets(samples, difficulty, pairs, strong_psnr_threshold)
    subset_rows = compute_subset_metrics(subsets, model_metrics)
    pairwise_rows = compute_pairwise_differences(
        subsets,
        model_metrics,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    gain_rows = glinr_gain_summary(subset_rows, pairwise_rows)
    raw_rows = raw_difficulty_summary(subsets, difficulty)
    difficulty_corr = difficulty_correlations(subsets["Full"], difficulty, model_metrics)
    near_corr = near_duplicate_correlations(evidence, model_metrics)
    sample_rows = per_sample_rows(samples, difficulty, model_metrics, subsets, evidence)

    _create_output(output)
    normalized_rows = candidate_pair_rows(pairs)
    write_csv(output / "candidate_pairs_normalized.csv", normalized_rows, CANDIDATE_PAIR_FIELDS)
    subset_filenames = {
        "Full": "full.csv",
        "Clean-A": "clean_a.csv",
        "Clean-B": "clean_b.csv",
        "Clean-C": "clean_c.csv",
        "Hard-Half": "hard_half.csv",
    }
    for subset, filename in subset_filenames.items():
        write_csv(
            output / "subsets" / filename,
            _membership_rows(subsets[subset], samples, difficulty),
            SUBSET_FIELDS,
        )
    excluded_ids = subsets["Suspect-A"] | subsets["Suspect-B"] | subsets["Suspect-C"]
    excluded_rows = [
        {"sample_id": sample.sample_id, **evidence[sample.sample_id]}
        for sample in sorted(samples.values(), key=lambda item: item.order)
        if sample.sample_id in excluded_ids
    ]
    write_csv(output / "subsets/excluded_samples.csv", excluded_rows, EXCLUDED_FIELDS)
    write_csv(output / "subset_metrics.csv", subset_rows, SUBSET_METRIC_FIELDS)
    write_csv(
        output / "suspect_subset_metrics.csv",
        [row for row in subset_rows if row["subset"] in SUSPECT_SUBSETS],
        SUBSET_METRIC_FIELDS,
    )
    write_csv(output / "pairwise_differences.csv", pairwise_rows, PAIRWISE_FIELDS)
    write_csv(output / "glinr_gain_summary.csv", gain_rows, GAIN_FIELDS)
    write_csv(
        output / "raw_difficulty_summary.csv",
        raw_rows,
        ("subset", "n", "raw_psnr_mean", "raw_psnr_median", "raw_ssim_mean"),
    )
    write_csv(
        output / "difficulty_correlation.csv",
        difficulty_corr,
        ("model", "n", "raw_metric", "output_metric", "pearson", "spearman"),
    )
    write_csv(
        output / "near_duplicate_correlation.csv",
        near_corr,
        ("candidate_strength", "outcome", "n", "pearson", "spearman"),
    )
    write_csv(output / "per_sample_analysis.csv", sample_rows, PER_SAMPLE_FIELDS)

    duplicate_summary = read_json(diagnostic_dir / "duplicates/duplicate_summary.json")
    candidate_statistics = {
        "diagnostic_dhash_threshold": duplicate_summary.get("dhash_threshold"),
        "normalized_pair_count": len(pairs),
        "candidate_test_sample_count": len(subsets["Suspect-A"]),
        "train_test_pair_count": sum(pair.non_test_split == "train" for pair in pairs),
        "validation_test_pair_count": sum(pair.non_test_split == "validation" for pair in pairs),
        "input_candidate_pair_count": sum(pair.input_candidate_psnr_128 is not None for pair in pairs),
        "gt_candidate_pair_count": sum(pair.gt_candidate_psnr_128 is not None for pair in pairs),
        "strong_input_test_sample_count": len(subsets["Suspect-B"]),
        "strong_paired_test_sample_count": len(subsets["Suspect-C"]),
        "strong_psnr_threshold": strong_psnr_threshold,
    }
    alignment = {
        "status": "PASS",
        "aligned_test_samples": len(samples),
        "expected_test_samples": expected_test_count,
        "canonical_id_sets_identical": True,
        "counts": {
            "manifest": len(samples),
            "diagnostic": len(difficulty),
            **{model: len(metrics) for model, metrics in model_metrics.items()},
        },
    }
    provenance = {
        "diagnostic_dir": str(diagnostic_dir),
        "v1_run": str(v1_run),
        "v2_run": str(v2_run),
        "v3_run": str(v3_run),
        "test_manifest": str(manifest_path),
        "schemas": schemas,
        "sha256": {
            "test_manifest": sha256_file(manifest_path),
            "diagnostic_test_metrics": sha256_file(difficulty_path),
            "near_duplicate_candidates": sha256_file(candidate_path),
            **{
                f"{version}_test_metrics": sha256_file(run_dir / "result/test_metrics.csv")
                for _, version, run_dir in run_specs
            },
        },
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "strong_psnr_threshold": strong_psnr_threshold,
    }
    bootstrap_rows = [
        {
            "subset": row["subset"],
            "comparison": row["comparison"],
            "metric": row["metric"],
            "mean_delta": row["mean_delta"],
            "bootstrap_ci_low": row["bootstrap_ci_low"],
            "bootstrap_ci_high": row["bootstrap_ci_high"],
        }
        for row in pairwise_rows
    ]
    summary = {
        "sample_alignment": alignment,
        "subset_sizes": {name: len(subsets[name]) for name in (*CORE_SUBSETS, *SUSPECT_SUBSETS)},
        "raw_difficulty": {row["subset"]: dict(row) for row in raw_rows},
        "model_metrics": _nested_metrics(subset_rows),
        "pairwise_deltas": _nested_pairwise(pairwise_rows),
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "intervals": bootstrap_rows,
        },
        "candidate_statistics": candidate_statistics,
        "difficulty_correlation": difficulty_corr,
        "near_duplicate_correlation": near_corr,
        "provenance": provenance,
    }
    write_json(output / "summary.json", summary)
    if generate_figures:
        generate_plots(
            output / "plots",
            subset_rows=subset_rows,
            gain_rows=gain_rows,
            subsets=subsets,
            model_metrics=model_metrics,
        )
    write_report(
        output / "report.md",
        provenance=provenance,
        alignment=alignment,
        candidate_statistics=candidate_statistics,
        subsets=subsets,
        raw_rows=raw_rows,
        subset_metrics=subset_rows,
        pairwise=pairwise_rows,
        difficulty_correlations=difficulty_corr,
        near_correlations=near_corr,
        strong_psnr_threshold=strong_psnr_threshold,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    _print_primary_tables(subset_rows, pairwise_rows)
    print(f"\nsample alignment: PASS\naligned test samples: {len(samples)}")
    print(f"analysis output: {output}")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", required=True)
    parser.add_argument("--v1-run", required=True)
    parser.add_argument("--v2-run", required=True)
    parser.add_argument("--v3-run", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--strong-psnr-threshold", type=float, default=35.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_analysis(
        diagnostic_dir=_resolve(args.diagnostic_dir),
        v1_run=_resolve(args.v1_run),
        v2_run=_resolve(args.v2_run),
        v3_run=_resolve(args.v3_run),
        output_dir=_resolve(args.output_dir) if args.output_dir else None,
        strong_psnr_threshold=args.strong_psnr_threshold,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
