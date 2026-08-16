from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.clean_test_analysis.io import (
    AnalysisInputError,
    DifficultyMetric,
    ModelMetric,
    TestSample,
    read_model_metrics,
)
from tools.clean_test_analysis.statistics import compute_pairwise_differences
from tools.clean_test_analysis.subsets import build_subsets, normalize_candidate_pairs
from tools.analyze_clean_test import run_analysis


CANDIDATE_FIELDS = [
    "split_a",
    "sample_a",
    "split_b",
    "sample_b",
    "image_type",
    "hamming_distance",
    "candidate_psnr_128",
    "candidate_mae_128",
]


def _samples(count: int = 4) -> dict[str, TestSample]:
    return {
        str(index): TestSample(
            str(index), f"Val/input/{index}.jpg", f"Val/GT/{index}.jpg", index - 1
        )
        for index in range(1, count + 1)
    }


def _difficulty(values: dict[str, float]) -> dict[str, DifficultyMetric]:
    return {
        sample_id: DifficultyMetric(sample_id, psnr, 0.5 + int(sample_id) * 0.01)
        for sample_id, psnr in values.items()
    }


def _candidate(
    test_id: str,
    non_test_id: str,
    image_type: str,
    psnr: float,
    *,
    non_test_split: str = "train",
    reverse: bool = False,
) -> dict[str, str]:
    row = {
        "split_a": non_test_split,
        "sample_a": non_test_id,
        "split_b": "test",
        "sample_b": test_id,
        "image_type": image_type,
        "hamming_distance": "1",
        "candidate_psnr_128": str(psnr),
        "candidate_mae_128": "0.01",
    }
    if reverse:
        row["split_a"], row["split_b"] = row["split_b"], row["split_a"]
        row["sample_a"], row["sample_b"] = row["sample_b"], row["sample_a"]
    return row


def _model_csv(path: Path, sample_ids: list[str], *, use_filename: bool = False) -> None:
    identity = "filename" if use_filename else "sample_id"
    lines = [f"{identity},psnr,ssim\n"]
    for index, sample_id in enumerate(sample_ids):
        key = f"{sample_id}_enhanced.png" if use_filename else sample_id
        lines.append(f"{key},{20 + index * 0.1},{0.8 + index * 0.001}\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_three_model_style_metrics_align_by_canonical_sample_id_or_filename(tmp_path: Path) -> None:
    samples = _samples(4)
    aligned = []
    for index, label in enumerate(("Identity", "Point-INR", "GL-INR")):
        path = tmp_path / f"v{index + 1}.csv"
        _model_csv(path, ["1", "2", "3", "4"], use_filename=index == 1)
        metrics, fields = read_model_metrics(path, samples, model_label=label)
        assert fields
        aligned.append(set(metrics))
    assert aligned[0] == aligned[1] == aligned[2] == set(samples)


def test_clean_a_removes_any_cross_split_candidate_sample() -> None:
    samples = _samples()
    rows = [_candidate("1", "101", "input", 20.0, reverse=True)]
    pairs = normalize_candidate_pairs(rows, CANDIDATE_FIELDS, set(samples))
    subsets, _ = build_subsets(
        samples, _difficulty({"1": 10, "2": 20, "3": 30, "4": 40}), pairs
    )
    assert subsets["Suspect-A"] == {"1"}
    assert subsets["Clean-A"] == {"2", "3", "4"}


def test_clean_b_uses_only_strong_input_threshold() -> None:
    samples = _samples()
    rows = [
        _candidate("1", "101", "input", 34.999),
        _candidate("2", "102", "input", 35.0),
        _candidate("3", "103", "gt", 50.0),
    ]
    pairs = normalize_candidate_pairs(rows, CANDIDATE_FIELDS, set(samples))
    subsets, _ = build_subsets(
        samples,
        _difficulty({"1": 10, "2": 20, "3": 30, "4": 40}),
        pairs,
        strong_psnr_threshold=35.0,
    )
    assert subsets["Suspect-B"] == {"2"}
    assert subsets["Clean-B"] == {"1", "3", "4"}


def test_clean_c_requires_input_and_gt_from_same_counterpart_pair() -> None:
    samples = _samples()
    rows = [
        _candidate("1", "train-a", "input", 40.0),
        _candidate("1", "train-b", "gt", 42.0),
        _candidate("2", "train-c", "input", 41.0),
        _candidate("2", "train-c", "gt", 43.0),
    ]
    pairs = normalize_candidate_pairs(rows, CANDIDATE_FIELDS, set(samples))
    subsets, evidence = build_subsets(
        samples, _difficulty({"1": 10, "2": 20, "3": 30, "4": 40}), pairs
    )
    assert "1" in subsets["Clean-C"]
    assert "2" not in subsets["Clean-C"]
    assert evidence["1"]["strong_paired_candidate"] is False
    assert evidence["2"]["strong_paired_candidate"] is True


def test_hard_half_is_ranked_by_raw_input_gt_psnr_with_id_tie_break() -> None:
    samples = _samples()
    difficulty = _difficulty({"1": 40.0, "2": 10.0, "3": 20.0, "4": 20.0})
    subsets, _ = build_subsets(samples, difficulty, [])
    assert subsets["Hard-Half"] == {"2", "3"}


def test_paired_delta_statistics_use_per_sample_differences() -> None:
    ids = {"1", "2"}
    model_metrics = {
        "Identity": {
            "1": ModelMetric("1", 20.0, 0.80),
            "2": ModelMetric("2", 30.0, 0.90),
        },
        "Point-INR": {
            "1": ModelMetric("1", 21.0, 0.82),
            "2": ModelMetric("2", 29.0, 0.89),
        },
        "GL-INR": {
            "1": ModelMetric("1", 23.0, 0.83),
            "2": ModelMetric("2", 31.0, 0.92),
        },
    }
    rows = compute_pairwise_differences(
        {"Full": ids},
        model_metrics,
        bootstrap_iterations=100,
        bootstrap_seed=3407,
        subset_order=("Full",),
    )
    gl_identity_psnr = next(
        row
        for row in rows
        if row["comparison"] == "GL-INR - Identity" and row["metric"] == "PSNR"
    )
    assert gl_identity_psnr["mean_delta"] == pytest.approx(2.0)
    assert gl_identity_psnr["median_delta"] == pytest.approx(2.0)
    assert gl_identity_psnr["positive_count"] == 2
    assert gl_identity_psnr["negative_count"] == 0
    assert gl_identity_psnr["win_rate"] == 1.0


def test_missing_model_sample_fails_instead_of_inner_join(tmp_path: Path) -> None:
    samples = _samples(4)
    path = tmp_path / "missing.csv"
    _model_csv(path, ["1", "2", "3"])
    with pytest.raises(AnalysisInputError, match="sample alignment failed"):
        read_model_metrics(path, samples, model_label="Identity")


def test_synthetic_end_to_end_analysis_writes_machine_and_human_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "test.tsv"
    manifest.write_text(
        "".join(
            f"{sample_id}\tVal/input/{sample_id}.jpg\tVal/GT/{sample_id}.jpg\n"
            for sample_id in ("1", "2", "3", "4")
        ),
        encoding="utf-8",
    )
    diagnostic = tmp_path / "diagnostic"
    (diagnostic / "difficulty").mkdir(parents=True)
    (diagnostic / "duplicates").mkdir()
    (diagnostic / "difficulty/test_metrics.csv").write_text(
        "sample_id,input_path,gt_path,psnr_256,ssim_256\n"
        "1,Val/input/1.jpg,Val/GT/1.jpg,10,0.5\n"
        "2,Val/input/2.jpg,Val/GT/2.jpg,20,0.6\n"
        "3,Val/input/3.jpg,Val/GT/3.jpg,30,0.7\n"
        "4,Val/input/4.jpg,Val/GT/4.jpg,40,0.8\n",
        encoding="utf-8",
    )
    candidate_lines = [",".join(CANDIDATE_FIELDS) + "\n"]
    for row in (
        _candidate("2", "train-x", "input", 40),
        _candidate("2", "train-x", "gt", 41),
    ):
        candidate_lines.append(",".join(row[field] for field in CANDIDATE_FIELDS) + "\n")
    (diagnostic / "duplicates/near_duplicate_candidates.csv").write_text(
        "".join(candidate_lines), encoding="utf-8"
    )
    (diagnostic / "duplicates/duplicate_summary.json").write_text(
        json.dumps({"dhash_threshold": 4}), encoding="utf-8"
    )

    runs = []
    for index, version in enumerate(("v1", "v2", "v3")):
        run = tmp_path / version
        (run / "result").mkdir(parents=True)
        (run / "run_info.json").write_text(
            json.dumps({"version": version}), encoding="utf-8"
        )
        lines = ["filename,sample_id,psnr,ssim\n"]
        for sample_id in ("1", "2", "3", "4"):
            lines.append(
                f"{sample_id}.jpg,{sample_id},{20 + int(sample_id) + index * 0.25},"
                f"{0.8 + int(sample_id) * 0.01 + index * 0.001}\n"
            )
        (run / "result/test_metrics.csv").write_text("".join(lines), encoding="utf-8")
        runs.append(run)

    output = run_analysis(
        diagnostic_dir=diagnostic,
        v1_run=runs[0],
        v2_run=runs[1],
        v3_run=runs[2],
        output_dir=tmp_path / "output",
        bootstrap_iterations=20,
        expected_test_count=4,
        generate_figures=False,
        test_manifest_path=manifest,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["sample_alignment"]["status"] == "PASS"
    assert summary["subset_sizes"]["Full"] == 4
    assert summary["subset_sizes"]["Clean-C"] == 3
    assert summary["subset_sizes"]["Hard-Half"] == 2
    assert (output / "report.md").is_file()
    assert (output / "per_sample_analysis.csv").is_file()
    assert (output / "subsets/clean_c.csv").is_file()
    assert (output / "pairwise_differences.csv").is_file()
