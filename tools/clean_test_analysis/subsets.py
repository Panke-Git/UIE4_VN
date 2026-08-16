"""Near-duplicate normalization and deterministic clean/hard subset definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .io import AnalysisInputError, DifficultyMetric, TestSample, parse_float, resolve_field


@dataclass
class CandidatePair:
    non_test_split: str
    non_test_sample_id: str
    test_sample_id: str
    input_dhash_distance: int | None = None
    input_candidate_psnr_128: float | None = None
    input_candidate_mae_128: float | None = None
    gt_dhash_distance: int | None = None
    gt_candidate_psnr_128: float | None = None
    gt_candidate_mae_128: float | None = None


def _normalize_split(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {"val": "validation", "valid": "validation", "training": "train", "testing": "test"}
    return aliases.get(normalized, normalized)


def _optional_float(row: Mapping[str, str], field: str | None, context: str) -> float | None:
    if field is None:
        return None
    return parse_float(row[field], context=context, allow_blank=True)


def _prefer_candidate(
    current_psnr: float | None,
    current_distance: int | None,
    new_psnr: float | None,
    new_distance: int | None,
) -> bool:
    current_score = -math.inf if current_psnr is None else current_psnr
    new_score = -math.inf if new_psnr is None else new_psnr
    if new_score != current_score:
        return new_score > current_score
    current_hamming = 65 if current_distance is None else current_distance
    new_hamming = 65 if new_distance is None else new_distance
    return new_hamming < current_hamming


def normalize_candidate_pairs(
    rows: Sequence[Mapping[str, str]],
    fields: Sequence[str],
    test_sample_ids: set[str],
) -> list[CandidatePair]:
    split_a_field = resolve_field(fields, ("split_a",), label="candidate split_a")
    split_b_field = resolve_field(fields, ("split_b",), label="candidate split_b")
    sample_a_field = resolve_field(fields, ("sample_a", "sample_id_a"), label="candidate sample_a")
    sample_b_field = resolve_field(fields, ("sample_b", "sample_id_b"), label="candidate sample_b")
    type_field = resolve_field(fields, ("image_type", "type"), label="candidate image_type")
    distance_field = resolve_field(
        fields, ("hamming_distance", "dhash_distance"), required=False, label="dHash distance"
    )
    psnr_field = resolve_field(
        fields, ("candidate_psnr_128", "psnr_128"), required=False, label="candidate PSNR_128"
    )
    mae_field = resolve_field(
        fields, ("candidate_mae_128", "mae_128"), required=False, label="candidate MAE_128"
    )

    pairs: dict[tuple[str, str, str], CandidatePair] = {}
    for line, row in enumerate(rows, start=2):
        split_a, split_b = _normalize_split(row[split_a_field]), _normalize_split(row[split_b_field])
        if split_a == split_b or (split_a != "test" and split_b != "test"):
            continue
        if split_a == "test":
            test_id, non_test_id, non_test_split = row[sample_a_field], row[sample_b_field], split_b
        else:
            test_id, non_test_id, non_test_split = row[sample_b_field], row[sample_a_field], split_a
        if non_test_split not in {"train", "validation"}:
            raise AnalysisInputError(
                f"Candidate row {line} pairs test with unsupported split {non_test_split!r}"
            )
        if test_id not in test_sample_ids:
            raise AnalysisInputError(f"Candidate row {line} has unknown test sample_id {test_id!r}")
        if not non_test_id:
            raise AnalysisInputError(f"Candidate row {line} has empty non-test sample ID")
        image_type = row[type_field].strip().lower()
        if image_type not in {"input", "gt"}:
            raise AnalysisInputError(f"Candidate row {line} has invalid image_type {image_type!r}")
        distance_value = _optional_float(row, distance_field, f"candidate row {line} distance")
        distance = int(distance_value) if distance_value is not None else None
        if distance_value is not None and distance_value != distance:
            raise AnalysisInputError(f"Candidate row {line} has non-integral dHash distance")
        psnr = _optional_float(row, psnr_field, f"candidate row {line} PSNR")
        mae = _optional_float(row, mae_field, f"candidate row {line} MAE")
        key = (non_test_split, non_test_id, test_id)
        pair = pairs.setdefault(key, CandidatePair(*key))
        if image_type == "input" and _prefer_candidate(
            pair.input_candidate_psnr_128, pair.input_dhash_distance, psnr, distance
        ):
            pair.input_dhash_distance = distance
            pair.input_candidate_psnr_128 = psnr
            pair.input_candidate_mae_128 = mae
        elif image_type == "gt" and _prefer_candidate(
            pair.gt_candidate_psnr_128, pair.gt_dhash_distance, psnr, distance
        ):
            pair.gt_dhash_distance = distance
            pair.gt_candidate_psnr_128 = psnr
            pair.gt_candidate_mae_128 = mae
    return sorted(
        pairs.values(),
        key=lambda pair: (pair.non_test_split, pair.non_test_sample_id, pair.test_sample_id),
    )


def _sample_sort_key(sample_id: str) -> tuple[int, int | str, str]:
    return (0, int(sample_id), sample_id) if sample_id.isdigit() else (1, sample_id, sample_id)


def _pair_strength(pair: CandidatePair) -> tuple[int, float, float, int, str, str]:
    input_score = -math.inf if pair.input_candidate_psnr_128 is None else pair.input_candidate_psnr_128
    gt_score = -math.inf if pair.gt_candidate_psnr_128 is None else pair.gt_candidate_psnr_128
    paired = int(
        pair.input_candidate_psnr_128 is not None
        and pair.gt_candidate_psnr_128 is not None
    )
    return (
        paired,
        min(input_score, gt_score),
        max(input_score, gt_score),
        -({"train": 0, "validation": 1}[pair.non_test_split]),
        pair.non_test_sample_id,
        pair.test_sample_id,
    )


def build_subsets(
    samples: Mapping[str, TestSample],
    difficulty: Mapping[str, DifficultyMetric],
    pairs: Sequence[CandidatePair],
    strong_psnr_threshold: float = 35.0,
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    if not math.isfinite(strong_psnr_threshold):
        raise ValueError("strong_psnr_threshold must be finite")
    full = set(samples)
    pairs_by_test: dict[str, list[CandidatePair]] = {sample_id: [] for sample_id in full}
    for pair in pairs:
        if pair.test_sample_id not in full:
            raise AnalysisInputError(f"Candidate pair has unknown test sample {pair.test_sample_id!r}")
        pairs_by_test[pair.test_sample_id].append(pair)

    excluded_a = {sample_id for sample_id, values in pairs_by_test.items() if values}
    excluded_b = {
        sample_id
        for sample_id, values in pairs_by_test.items()
        if any(
            pair.input_candidate_psnr_128 is not None
            and pair.input_candidate_psnr_128 >= strong_psnr_threshold
            for pair in values
        )
    }
    excluded_c = {
        sample_id
        for sample_id, values in pairs_by_test.items()
        if any(
            pair.input_candidate_psnr_128 is not None
            and pair.gt_candidate_psnr_128 is not None
            and pair.input_candidate_psnr_128 >= strong_psnr_threshold
            and pair.gt_candidate_psnr_128 >= strong_psnr_threshold
            for pair in values
        )
    }
    hard_order = sorted(
        full,
        key=lambda sample_id: (
            difficulty[sample_id].psnr_256,
            _sample_sort_key(sample_id),
        ),
    )
    subsets = {
        "Full": full,
        "Clean-A": full - excluded_a,
        "Clean-B": full - excluded_b,
        "Clean-C": full - excluded_c,
        "Hard-Half": set(hard_order[: len(full) // 2]),
        "Suspect-A": excluded_a,
        "Suspect-B": excluded_b,
        "Suspect-C": excluded_c,
    }

    evidence: dict[str, dict[str, Any]] = {}
    for sample_id, candidates in pairs_by_test.items():
        best_pair = max(candidates, key=_pair_strength) if candidates else None
        input_scores = [
            pair.input_candidate_psnr_128
            for pair in candidates
            if pair.input_candidate_psnr_128 is not None
        ]
        gt_scores = [
            pair.gt_candidate_psnr_128
            for pair in candidates
            if pair.gt_candidate_psnr_128 is not None
        ]
        evidence[sample_id] = {
            "candidate_count": len(candidates),
            "near_duplicate_candidate": sample_id in excluded_a,
            "strong_input_candidate": sample_id in excluded_b,
            "strong_paired_candidate": sample_id in excluded_c,
            "best_input_candidate_psnr_128": max(input_scores) if input_scores else None,
            "best_gt_candidate_psnr_128": max(gt_scores) if gt_scores else None,
            "matched_non_test_split": best_pair.non_test_split if best_pair else None,
            "matched_non_test_sample_id": best_pair.non_test_sample_id if best_pair else None,
            "excluded_from_clean_a": sample_id in excluded_a,
            "excluded_from_clean_b": sample_id in excluded_b,
            "excluded_from_clean_c": sample_id in excluded_c,
            "reason_clean_a": (
                "test sample has at least one cross-split input or GT dHash candidate"
                if sample_id in excluded_a else ""
            ),
            "reason_clean_b": (
                f"same test sample has input candidate PSNR_128 >= {strong_psnr_threshold:g} dB"
                if sample_id in excluded_b else ""
            ),
            "reason_clean_c": (
                f"one identical counterpart pair has input and GT candidate PSNR_128 >= {strong_psnr_threshold:g} dB"
                if sample_id in excluded_c else ""
            ),
        }
    return subsets, evidence


def candidate_pair_rows(pairs: Sequence[CandidatePair]) -> list[dict[str, Any]]:
    return [pair.__dict__.copy() for pair in pairs]
