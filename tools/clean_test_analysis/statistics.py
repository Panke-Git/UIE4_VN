"""Subset metrics, paired deltas, bootstrap intervals, and exploratory correlations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .io import DifficultyMetric, ModelMetric, TestSample


CORE_SUBSETS = ("Full", "Clean-A", "Clean-B", "Clean-C", "Hard-Half")
SUSPECT_SUBSETS = ("Suspect-A", "Suspect-B", "Suspect-C")
MODEL_ORDER = ("Identity", "Point-INR", "GL-INR")
PAIR_ORDER = (
    ("Point-INR", "Identity", "Point-INR - Identity"),
    ("GL-INR", "Identity", "GL-INR - Identity"),
    ("GL-INR", "Point-INR", "GL-INR - Point-INR"),
)


def _values(sample_ids: set[str], metrics: Mapping[str, ModelMetric], field: str) -> np.ndarray:
    return np.asarray([getattr(metrics[sample_id], field) for sample_id in sorted(sample_ids)], dtype=np.float64)


def describe(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "std": math.nan}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def compute_subset_metrics(
    subsets: Mapping[str, set[str]],
    model_metrics: Mapping[str, Mapping[str, ModelMetric]],
    subset_order: Sequence[str] = (*CORE_SUBSETS, *SUSPECT_SUBSETS),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset in subset_order:
        sample_ids = subsets[subset]
        for model in MODEL_ORDER:
            psnr = describe(_values(sample_ids, model_metrics[model], "psnr"))
            ssim = describe(_values(sample_ids, model_metrics[model], "ssim"))
            rows.append(
                {
                    "subset": subset,
                    "model": model,
                    "n": psnr["n"],
                    "psnr_mean": psnr["mean"],
                    "psnr_median": psnr["median"],
                    "psnr_std": psnr["std"],
                    "ssim_mean": ssim["mean"],
                    "ssim_median": ssim["median"],
                    "ssim_std": ssim["std"],
                }
            )
    return rows


def paired_delta_values(
    sample_ids: set[str],
    left: Mapping[str, ModelMetric],
    right: Mapping[str, ModelMetric],
    metric: str,
) -> np.ndarray:
    ordered = sorted(sample_ids)
    return np.asarray(
        [getattr(left[sample_id], metric) - getattr(right[sample_id], metric) for sample_id in ordered],
        dtype=np.float64,
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
    batch_size: int = 1000,
) -> tuple[float, float]:
    if iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if values.size == 0:
        return math.nan, math.nan
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    offset = 0
    while offset < iterations:
        count = min(batch_size, iterations - offset)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[offset : offset + count] = values[indices].mean(axis=1)
        offset += count
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def compute_pairwise_differences(
    subsets: Mapping[str, set[str]],
    model_metrics: Mapping[str, Mapping[str, ModelMetric]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    subset_order: Sequence[str] = CORE_SUBSETS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stream = 0
    for subset in subset_order:
        for left_name, right_name, comparison in PAIR_ORDER:
            for metric in ("psnr", "ssim"):
                deltas = paired_delta_values(
                    subsets[subset], model_metrics[left_name], model_metrics[right_name], metric
                )
                if deltas.size:
                    ties = np.isclose(deltas, 0.0, rtol=0.0, atol=1e-12)
                    positive = int(np.count_nonzero((deltas > 0) & ~ties))
                    negative = int(np.count_nonzero((deltas < 0) & ~ties))
                    tie_count = int(np.count_nonzero(ties))
                    low, high = bootstrap_mean_ci(
                        deltas,
                        iterations=bootstrap_iterations,
                        seed=bootstrap_seed + stream,
                    )
                    row = {
                        "subset": subset,
                        "comparison": comparison,
                        "left_model": left_name,
                        "right_model": right_name,
                        "metric": metric.upper(),
                        "n": int(deltas.size),
                        "mean_delta": float(np.mean(deltas)),
                        "median_delta": float(np.median(deltas)),
                        "std_delta": float(np.std(deltas)),
                        "min_delta": float(np.min(deltas)),
                        "max_delta": float(np.max(deltas)),
                        "positive_count": positive,
                        "negative_count": negative,
                        "tie_count": tie_count,
                        "win_rate": positive / deltas.size,
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                    }
                else:
                    row = {
                        "subset": subset,
                        "comparison": comparison,
                        "left_model": left_name,
                        "right_model": right_name,
                        "metric": metric.upper(),
                        "n": 0,
                        **{
                            field: math.nan
                            for field in (
                                "mean_delta", "median_delta", "std_delta", "min_delta", "max_delta",
                                "win_rate", "bootstrap_ci_low", "bootstrap_ci_high",
                            )
                        },
                        "positive_count": 0,
                        "negative_count": 0,
                        "tie_count": 0,
                    }
                rows.append(row)
                stream += 1
    return rows


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    array_x, array_y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(array_x) & np.isfinite(array_y)
    array_x, array_y = array_x[mask], array_y[mask]
    if array_x.size < 2 or np.all(array_x == array_x[0]) or np.all(array_y == array_y[0]):
        return math.nan
    return float(np.corrcoef(array_x, array_y)[0, 1])


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    array_x, array_y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(array_x) & np.isfinite(array_y)
    array_x, array_y = array_x[mask], array_y[mask]
    if array_x.size < 2:
        return math.nan
    return pearson_correlation(_average_ranks(array_x), _average_ranks(array_y))


def difficulty_correlations(
    sample_ids: set[str],
    difficulty: Mapping[str, DifficultyMetric],
    model_metrics: Mapping[str, Mapping[str, ModelMetric]],
) -> list[dict[str, Any]]:
    ordered = sorted(sample_ids)
    raw = [difficulty[sample_id].psnr_256 for sample_id in ordered]
    rows = []
    for model in MODEL_ORDER:
        output = [model_metrics[model][sample_id].psnr for sample_id in ordered]
        rows.append(
            {
                "model": model,
                "n": len(ordered),
                "raw_metric": "Input->GT PSNR_256",
                "output_metric": "model PSNR",
                "pearson": pearson_correlation(raw, output),
                "spearman": spearman_correlation(raw, output),
            }
        )
    return rows


def near_duplicate_correlations(
    evidence: Mapping[str, Mapping[str, Any]],
    model_metrics: Mapping[str, Mapping[str, ModelMetric]],
) -> list[dict[str, Any]]:
    candidate_ids = sorted(
        sample_id for sample_id, item in evidence.items() if item["near_duplicate_candidate"]
    )
    outcomes: dict[str, list[float]] = {
        "Identity output PSNR": [model_metrics["Identity"][sample_id].psnr for sample_id in candidate_ids],
        "Point-INR output PSNR": [model_metrics["Point-INR"][sample_id].psnr for sample_id in candidate_ids],
        "GL-INR output PSNR": [model_metrics["GL-INR"][sample_id].psnr for sample_id in candidate_ids],
        "GL-INR - Identity PSNR": [
            model_metrics["GL-INR"][sample_id].psnr - model_metrics["Identity"][sample_id].psnr
            for sample_id in candidate_ids
        ],
        "GL-INR - Point-INR PSNR": [
            model_metrics["GL-INR"][sample_id].psnr - model_metrics["Point-INR"][sample_id].psnr
            for sample_id in candidate_ids
        ],
    }
    rows: list[dict[str, Any]] = []
    for evidence_key, strength_name in (
        ("best_input_candidate_psnr_128", "max input candidate PSNR_128"),
        ("best_gt_candidate_psnr_128", "max GT candidate PSNR_128"),
    ):
        strength = [
            math.nan if evidence[sample_id][evidence_key] is None else evidence[sample_id][evidence_key]
            for sample_id in candidate_ids
        ]
        for outcome_name, outcome in outcomes.items():
            valid_n = int(np.count_nonzero(np.isfinite(strength) & np.isfinite(outcome)))
            rows.append(
                {
                    "candidate_strength": strength_name,
                    "outcome": outcome_name,
                    "n": valid_n,
                    "pearson": pearson_correlation(strength, outcome),
                    "spearman": spearman_correlation(strength, outcome),
                }
            )
    return rows


def raw_difficulty_summary(
    subsets: Mapping[str, set[str]], difficulty: Mapping[str, DifficultyMetric]
) -> list[dict[str, Any]]:
    rows = []
    for subset in (*CORE_SUBSETS, *SUSPECT_SUBSETS):
        ordered = sorted(subsets[subset])
        psnr = np.asarray([difficulty[sample_id].psnr_256 for sample_id in ordered], dtype=np.float64)
        ssim = np.asarray([difficulty[sample_id].ssim_256 for sample_id in ordered], dtype=np.float64)
        rows.append(
            {
                "subset": subset,
                "n": len(ordered),
                "raw_psnr_mean": float(np.mean(psnr)) if psnr.size else math.nan,
                "raw_psnr_median": float(np.median(psnr)) if psnr.size else math.nan,
                "raw_ssim_mean": float(np.mean(ssim)) if ssim.size else math.nan,
            }
        )
    return rows


def glinr_gain_summary(
    subset_metrics: Sequence[Mapping[str, Any]], pairwise_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    metric_lookup = {
        (row["subset"], row["model"]): row for row in subset_metrics if row["subset"] in CORE_SUBSETS
    }
    delta_lookup = {
        (row["subset"], row["comparison"], row["metric"]): row
        for row in pairwise_rows
    }
    rows = []
    for subset in CORE_SUBSETS:
        identity, point, glinr = (
            metric_lookup[(subset, model)] for model in MODEL_ORDER
        )
        rows.append(
            {
                "subset": subset,
                "n": identity["n"],
                "identity_psnr": identity["psnr_mean"],
                "point_inr_psnr": point["psnr_mean"],
                "glinr_psnr": glinr["psnr_mean"],
                "glinr_minus_identity_psnr": delta_lookup[
                    (subset, "GL-INR - Identity", "PSNR")
                ]["mean_delta"],
                "glinr_minus_point_psnr": delta_lookup[
                    (subset, "GL-INR - Point-INR", "PSNR")
                ]["mean_delta"],
                "identity_ssim": identity["ssim_mean"],
                "point_inr_ssim": point["ssim_mean"],
                "glinr_ssim": glinr["ssim_mean"],
                "glinr_minus_identity_ssim": delta_lookup[
                    (subset, "GL-INR - Identity", "SSIM")
                ]["mean_delta"],
                "glinr_minus_point_ssim": delta_lookup[
                    (subset, "GL-INR - Point-INR", "SSIM")
                ]["mean_delta"],
            }
        )
    return rows


def per_sample_rows(
    samples: Mapping[str, TestSample],
    difficulty: Mapping[str, DifficultyMetric],
    model_metrics: Mapping[str, Mapping[str, ModelMetric]],
    subsets: Mapping[str, set[str]],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for sample in sorted(samples.values(), key=lambda item: item.order):
        sample_id = sample.sample_id
        identity = model_metrics["Identity"][sample_id]
        point = model_metrics["Point-INR"][sample_id]
        glinr = model_metrics["GL-INR"][sample_id]
        item = evidence[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "input_path": sample.input_path,
                "gt_path": sample.gt_path,
                "raw_input_gt_psnr": difficulty[sample_id].psnr_256,
                "raw_input_gt_ssim": difficulty[sample_id].ssim_256,
                "identity_psnr": identity.psnr,
                "identity_ssim": identity.ssim,
                "point_psnr": point.psnr,
                "point_ssim": point.ssim,
                "glinr_psnr": glinr.psnr,
                "glinr_ssim": glinr.ssim,
                "point_minus_identity_psnr": point.psnr - identity.psnr,
                "glinr_minus_identity_psnr": glinr.psnr - identity.psnr,
                "glinr_minus_point_psnr": glinr.psnr - point.psnr,
                "point_minus_identity_ssim": point.ssim - identity.ssim,
                "glinr_minus_identity_ssim": glinr.ssim - identity.ssim,
                "glinr_minus_point_ssim": glinr.ssim - point.ssim,
                "near_duplicate_candidate": item["near_duplicate_candidate"],
                "strong_input_candidate": item["strong_input_candidate"],
                "strong_paired_candidate": item["strong_paired_candidate"],
                "best_input_candidate_psnr_128": item["best_input_candidate_psnr_128"],
                "best_gt_candidate_psnr_128": item["best_gt_candidate_psnr_128"],
                "in_clean_a": sample_id in subsets["Clean-A"],
                "in_clean_b": sample_id in subsets["Clean-B"],
                "in_clean_c": sample_id in subsets["Clean-C"],
                "in_hard_half": sample_id in subsets["Hard-Half"],
            }
        )
    return rows
