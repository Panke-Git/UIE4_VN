from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.v16.models import build_model
from tools.analyze_uicf_alignment import (
    _write_csv,
    average_rank,
    direction_cosine_top_demand,
    exact_top_fraction_mask,
    generate_null_shifts,
    main as alignment_main,
    pearson_correlation,
    per_pixel_delta_e00_map,
    resolve_dataset_name,
    save_metric_plots,
    spearman_correlation,
    top_fraction_overlap,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_model_config() -> dict:
    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text(encoding="utf-8"))
    model = deepcopy(config["model"])
    model["base_channels"] = 4
    model["color_query"] = {
        **model["color_query"],
        "token_dim": 16,
        "num_heads": 4,
    }
    model["uicf"] = {
        "feat_dim": 4,
        "num_frequencies": 2,
        "mlp_hidden_dim": 8,
        "mlp_hidden_layers": 1,
        "anchor_hidden_dim": 4,
        "query_chunk_size": 31,
    }
    return model


def test_actual_uicf_correction_formula_is_not_the_raw_field() -> None:
    inputs = torch.linspace(0.05, 0.95, 3 * 5 * 7).reshape(1, 3, 5, 7)
    field = torch.linspace(-0.4, 0.6, inputs.numel()).reshape_as(inputs)
    anchor = torch.tensor([[0.2, 0.4, 0.7]])
    enhanced = inputs + field * (inputs - anchor[:, :, None, None])
    actual_delta = enhanced - inputs
    expected_delta = field * (inputs - anchor[:, :, None, None])
    torch.testing.assert_close(actual_delta, expected_delta)
    assert not torch.allclose(actual_delta, field)


def test_perfect_magnitude_alignment_has_unit_correlations_and_overlap() -> None:
    delta_gt = np.zeros((3, 4, 4), dtype=np.float64)
    delta_gt[0] = np.arange(1.0, 17.0).reshape(4, 4)
    delta_uicf = 2.0 * delta_gt
    gt_magnitude = np.linalg.norm(delta_gt, axis=0)
    uicf_magnitude = np.linalg.norm(delta_uicf, axis=0)
    spearman = spearman_correlation(uicf_magnitude, gt_magnitude)
    pearson = pearson_correlation(uicf_magnitude, gt_magnitude)
    iou, _, _ = top_fraction_overlap(uicf_magnitude, gt_magnitude, 0.20)
    assert spearman.valid and spearman.value == pytest.approx(1.0)
    assert pearson.valid and pearson.value == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)


def test_perfect_direction_alignment_has_unit_cosine() -> None:
    delta_gt = np.arange(1.0, 3 * 4 * 4 + 1.0).reshape(3, 4, 4)
    delta_uicf = 3.5 * delta_gt
    result = direction_cosine_top_demand(
        delta_uicf, delta_gt, np.linalg.norm(delta_gt, axis=0), 0.20
    )
    assert result.valid and result.value == pytest.approx(1.0)


def test_opposite_direction_alignment_has_negative_unit_cosine() -> None:
    delta_gt = np.arange(1.0, 3 * 4 * 4 + 1.0).reshape(3, 4, 4)
    result = direction_cosine_top_demand(
        -delta_gt, delta_gt, np.linalg.norm(delta_gt, axis=0), 0.20
    )
    assert result.valid and result.value == pytest.approx(-1.0)


def test_average_ranks_and_spearman_handle_ties_deterministically() -> None:
    values = np.array([10.0, 20.0, 20.0, 40.0])
    np.testing.assert_array_equal(average_rank(values), [1.0, 2.5, 2.5, 4.0])
    result = spearman_correlation(values, np.array([1.0, 2.0, 2.0, 4.0]))
    assert result.valid and result.value == pytest.approx(1.0)


def test_constant_map_correlations_are_explicitly_undefined() -> None:
    for function in (pearson_correlation, spearman_correlation):
        result = function(np.ones(16), np.arange(16, dtype=np.float64))
        assert result.valid is False
        assert result.value is None


def test_exact_top_fraction_selects_exact_ceil_count_with_stable_ties() -> None:
    values = np.ones((3, 3), dtype=np.float64)
    mask = exact_top_fraction_mask(values, 0.20)
    assert int(mask.sum()) == math.ceil(values.size * 0.20) == 2
    expected = np.zeros((3, 3), dtype=bool)
    expected.reshape(-1)[:2] = True
    np.testing.assert_array_equal(mask, expected)


def test_large_spatial_shift_null_is_worse_than_exact_alignment() -> None:
    demand = np.zeros((8, 8), dtype=np.float64)
    demand[1:3, 1:3] = np.array([[10.0, 11.0], [12.0, 13.0]])
    real_spearman = spearman_correlation(demand, demand)
    real_iou, _, _ = top_fraction_overlap(demand, demand, 0.20)
    shifts = generate_null_shifts(8, 8, 20, seed=3520, sample_index=7)
    assert len(shifts) == 20
    assert all((dy != 0 or dx != 0) and (abs(dy) >= 2 or abs(dx) >= 2) for dy, dx in shifts)
    shifted_spearman = []
    shifted_iou = []
    for dy, dx in shifts:
        shifted = np.roll(demand, (dy, dx), axis=(0, 1))
        result = spearman_correlation(shifted, demand)
        assert result.valid
        shifted_spearman.append(float(result.value))
        shifted_iou.append(top_fraction_overlap(shifted, demand, 0.20)[0])
    assert real_spearman.value == pytest.approx(1.0)
    assert real_spearman.value > np.mean(shifted_spearman)
    assert real_iou > np.mean(shifted_iou)


def test_per_pixel_e00_identical_rgb_is_zero_finite_and_correct_shape() -> None:
    image = torch.rand(3, 9, 11)
    values = per_pixel_delta_e00_map(image, image.clone())
    assert values.shape == (9, 11)
    assert values.dtype == np.float64
    assert np.isfinite(values).all()
    assert np.count_nonzero(values) == 0


def test_csv_serialization_uses_blank_for_none_and_rejects_nonfinite(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write_csv(path, [{"sample_id": "ok", "score": None}], ["sample_id", "score"])
    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [{"sample_id": "ok", "score": ""}]
    lowered = path.read_text(encoding="utf-8").lower()
    assert "nan" not in lowered
    with pytest.raises(FloatingPointError, match="non-finite CSV"):
        _write_csv(path, [{"sample_id": "bad", "score": float("inf")}], ["sample_id", "score"])


@pytest.mark.parametrize(
    "values",
    (
        [0.80, 0.81, 0.82, 0.83],
        [0.82, 0.82, 0.82, 0.82],
        [0.82],
        [-1.0, 1.0],
    ),
)
def test_metric_plots_handle_empty_histogram_bins(
    tmp_path: Path, values: list[float]
) -> None:
    rows = [
        {
            "spearman_rgb": value,
            "null_spearman_rgb_mean": 0.0,
            "raw_field_spearman_rgb": value,
            "raw_field_null_spearman_rgb_mean": 0.0,
            "anchor_spearman_rgb": value,
            "gradient_spearman_rgb": value,
        }
        for value in values
    ]
    save_metric_plots(rows, tmp_path)
    expected = {
        "metric_histogram_spearman.png",
        "real_vs_null_spearman.png",
        "raw_field_spearman_histogram.png",
        "raw_field_real_vs_null_spearman.png",
        "representation_vs_controls_spearman.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    for filename in expected:
        path = tmp_path / filename
        assert path.stat().st_size > 0
        with Image.open(path) as image:
            image.verify()


def test_dataset_resolver_accepts_modern_lsui() -> None:
    assert resolve_dataset_name(
        {"dataset": "LSUI19", "root": "/unrelated"}, context="test config"
    ) == "LSUI19"


@pytest.mark.parametrize(
    ("data_config", "expected"),
    (
        (
            {
                "root": "/datasets/LSUI19_dup_train",
                "test_manifest": "split/lsui19/test.tsv",
            },
            "LSUI19",
        ),
        (
            {"root": "/datasets/UIEB19", "test_manifest": "split/uieb/test.tsv"},
            "UIEB",
        ),
    ),
)
def test_dataset_resolver_infers_unambiguous_legacy_metadata(
    data_config: dict, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert resolve_dataset_name(data_config, context="legacy test config") == expected
    warning = capsys.readouterr().err
    assert "[WARNING] legacy test config has no data.dataset" in warning
    assert f"inferred legacy dataset as '{expected}'" in warning


def test_dataset_resolver_rejects_conflicting_legacy_metadata() -> None:
    with pytest.raises(ValueError, match="conflicting legacy dataset evidence"):
        resolve_dataset_name(
            {"root": "/datasets/LSUI19", "test_manifest": "split/uieb/test.tsv"},
            context="legacy test config",
        )


def test_dataset_resolver_rejects_unknown_legacy_metadata() -> None:
    with pytest.raises(ValueError, match="no unambiguous LSUI19 or UIEB evidence"):
        resolve_dataset_name(
            {"root": "/datasets/paired", "test_manifest": "split/test.tsv"},
            context="legacy test config",
        )


@pytest.mark.parametrize(
    ("dataset_name", "legacy"),
    (("LSUI19", False), ("UIEB", False), ("LSUI19", True)),
)
def test_tiny_zero_uicf_cli_smoke_run_writes_complete_audit(
    tmp_path: Path,
    dataset_name: str,
    legacy: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "v16_alignment_run"
    data_root = tmp_path / dataset_name
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "split_snapshot").mkdir()
    for directory in (data_root / "input", data_root / "gt"):
        directory.mkdir(parents=True)
    split_ids = {"train": ["train_id"], "validation": ["validation_id"], "test": ["test_id"]}
    for offset, sample_id in enumerate(sum(split_ids.values(), [])):
        height, width = 16, 16
        yy, xx = np.mgrid[:height, :width]
        base = 30 + offset * 25 + xx * 2 + yy
        input_array = np.stack((base, base + 4, base + 8), axis=-1).clip(0, 255).astype(np.uint8)
        gt_array = input_array.copy()
        gt_array[4:12, 4:12] = np.clip(gt_array[4:12, 4:12].astype(np.int16) + 18, 0, 255)
        Image.fromarray(input_array, mode="RGB").save(data_root / "input" / f"{sample_id}.png")
        Image.fromarray(gt_array, mode="RGB").save(data_root / "gt" / f"{sample_id}.png")
    for split, sample_ids in split_ids.items():
        (run_dir / "split_snapshot" / f"{split}.tsv").write_text(
            "\n".join(
                f"{sample_id}\tinput/{sample_id}.png\tgt/{sample_id}.png"
                for sample_id in sample_ids
            )
            + "\n",
            encoding="utf-8",
        )

    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text(encoding="utf-8"))
    config["model"] = _small_model_config()
    config["data"] = {
        **config["data"],
        "dataset": dataset_name,
        "root": str(data_root),
        "expected_counts": {split: len(ids) for split, ids in split_ids.items()},
        "num_workers": 0,
    }
    if legacy:
        del config["data"]["dataset"]
    config["evaluation"] = {"resize": True, "size": 16}
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    model = build_model(config["model"])
    # The canonical UICF output layer is zero-initialized.  This intentionally
    # exercises undefined correlation and direction handling without NaNs.
    torch.save(
        {
            "version": "v16",
            "epoch": 5,
            "resolved_config": {"model": config["model"]},
            "model_state_dict": model.state_dict(),
        },
        run_dir / "best" / "best_psnr.pt",
    )

    alignment_main(
        [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best_psnr",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--patch-size",
            "16",
            "--num-null-shifts",
            "4",
            "--bootstrap-samples",
            "20",
            "--viz-k",
            "1",
        ]
    )
    output = run_dir / "result" / "uicf_representation_alignment"
    required = {
        "per_sample_metrics.csv",
        "summary.json",
        "summary.txt",
        "protocol.json",
        "ranking_by_spearman.csv",
        "null_control_summary.csv",
        "failed_samples.csv",
        "top_alignment",
        "representative",
        "top_alignment_contact_sheet.png",
        "representative_contact_sheet.png",
        "metric_histogram_spearman.png",
        "real_vs_null_spearman.png",
        "ranking_by_raw_field_spearman.csv",
        "raw_field_null_control_summary.csv",
        "representation_baseline_summary.csv",
        "representation_bootstrap_summary.json",
        "top_raw_field_alignment",
        "representative_raw_field",
        "top_raw_field_alignment_contact_sheet.png",
        "representative_raw_field_contact_sheet.png",
        "raw_field_spearman_histogram.png",
        "raw_field_real_vs_null_spearman.png",
        "representation_vs_controls_spearman.png",
    }
    assert required <= {path.name for path in output.iterdir()}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset"] == dataset_name
    assert (summary["train_count"], summary["validation_count"], summary["test_count"]) == (
        1,
        1,
        1,
    )
    assert summary["data_root"] == str(data_root)
    assert summary["test_manifest"] == str(run_dir / "split_snapshot" / "test.tsv")
    assert summary["total_test_samples"] == 1
    assert summary["processed_sample_count"] == 1
    assert summary["successful_sample_count"] == 1
    assert summary["failed_sample_count"] == 0
    assert summary["evaluation_mode"] == "in_domain"
    assert summary["checkpoint_dataset"] == dataset_name
    assert summary["evaluation_dataset"] == dataset_name
    assert "raw_field_representation" in summary
    assert "baseline_controls" in summary
    assert "paired_representation_comparisons" in summary
    assert summary["valid_sample_counts"]["spearman_rgb"] == 0
    assert summary["metrics"]["spearman_rgb"]["invalid_count"] == 1
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["dataset"] == dataset_name
    assert protocol["checkpoint_dataset"] == dataset_name
    assert protocol["evaluation_dataset"] == dataset_name
    assert (protocol["train_count"], protocol["validation_count"], protocol["test_count"]) == (
        1,
        1,
        1,
    )
    assert protocol["data_root"] == str(data_root)
    assert protocol["test_manifest"] == str(run_dir / "split_snapshot" / "test.tsv")
    assert protocol["checkpoint_selector"] == "best_psnr"
    assert protocol["script_version"] == "2.0"
    assert protocol["analysis_target"] == "UICF implicit coefficient representation R(x)"
    assert protocol["raw_field_interpretation"] == (
        "R(x) is not interpreted as the RGB target residual"
    )
    assert protocol["test_manifest_sample_count"] == 1
    assert f"{dataset_name} test set" in (output / "summary.txt").read_text(encoding="utf-8")
    csv_text = (output / "per_sample_metrics.csv").read_text(encoding="utf-8").lower()
    assert "nan" not in csv_text
    if legacy:
        captured = capsys.readouterr()
        assert captured.err.count("inferred legacy dataset as 'LSUI19'") == 1
