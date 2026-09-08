from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from src.shared.uicf_controls import (
    ConvolutionalCorrectionField,
    correction_parameter_report,
    trainable_parameter_count,
)
from src.shared.uicf_inr import UnderwaterImplicitCorrectionField
from src.v16.models import build_model
from tools.analyze_uicf_alignment import (
    analyze_spatial_maps,
    anchor_deviation_map,
    bootstrap_mean_difference,
    main as alignment_main,
    rgb_sobel_gradient,
)
from tools.visualize_v16_uicf import build_and_load_v16_model
from tools.summarize_uicf_representation_runs import main as summarize_main


ROOT = Path(__file__).resolve().parents[1]


def _small_model_config() -> dict:
    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text())
    model = deepcopy(config["model"])
    model["base_channels"] = 4
    model["color_query"] = {
        **model["color_query"], "token_dim": 16, "num_heads": 4
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


def test_raw_field_perfect_rgb_alignment_and_null_fields() -> None:
    yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    demand = (1.0 + xx + 2.0 * yy).float() / 300.0
    inputs = torch.full((3, 8, 8), 0.25)
    target = inputs.clone()
    target[0] += demand
    field = torch.zeros_like(inputs)
    field[0] = demand * 2.5
    row, maps = analyze_spatial_maps(
        inputs, target, inputs, field, torch.tensor([0.2, 0.4, 0.6]),
        sample_id="perfect", sample_index=3, patch_size=1, top_fraction=0.20,
        num_null_shifts=8, null_seed=3520,
    )
    assert row["raw_field_spearman_rgb"] == pytest.approx(1.0)
    assert row["raw_field_pearson_rgb"] == pytest.approx(1.0)
    assert row["raw_field_top20_iou_rgb"] == pytest.approx(1.0)
    assert row["raw_field_null_spearman_valid_shift_count"] == 8
    assert row["raw_field_null_pearson_valid_shift_count"] == 8
    assert row["raw_field_null_top20_valid_shift_count"] == 8
    assert row["raw_field_spearman_gain_over_null"] > 0.0
    assert row["raw_field_spearman_e00"] is not None
    assert np.isfinite(maps["raw_field_magnitude"]).all()


def test_anchor_deviation_is_exact() -> None:
    image = torch.tensor(
        [[[0.0, 1.0]], [[0.5, 0.5]], [[1.0, 0.0]]], dtype=torch.float32
    )
    anchor = torch.tensor([0.0, 0.5, 0.0])
    expected = np.linalg.norm(
        image.numpy().astype(np.float64) - anchor.numpy()[:, None, None], axis=0
    )
    np.testing.assert_allclose(anchor_deviation_map(image, anchor), expected, atol=1e-12)


def test_rgb_sobel_constant_and_ramps() -> None:
    constant = rgb_sobel_gradient(torch.ones(3, 9, 11))
    assert constant.shape == (9, 11)
    assert np.count_nonzero(constant) == 0
    horizontal = torch.linspace(0.0, 1.0, 11).repeat(9, 1)
    vertical = torch.linspace(0.0, 1.0, 9)[:, None].repeat(1, 11)
    for ramp in (horizontal, vertical):
        values = rgb_sobel_gradient(ramp.repeat(3, 1, 1))
        assert values.shape == (9, 11)
        assert np.isfinite(values).all() and np.all(values >= 0.0)
        assert float(values[2:-2, 2:-2].mean()) > 0.0


def test_positive_paired_bootstrap_has_positive_ci() -> None:
    result = bootstrap_mean_difference(
        [0.2, 0.3, 0.4, 0.5], samples=1000, seed=3520
    )
    assert result["mean_difference"] == pytest.approx(0.35)
    assert result["ci95_low"] > 0.0
    assert result["ci95_high"] > result["ci95_low"]


def test_anchor_and_gradient_shift_controls_are_deterministic() -> None:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, 8), torch.linspace(0.0, 1.0, 8), indexing="ij"
    )
    inputs = torch.stack((xx, yy, (xx + yy) / 2.0)) * 0.5
    target = (inputs + torch.stack((yy, xx, xx * yy)) * 0.1).clamp(0.0, 1.0)
    field = torch.stack((xx, yy, xx + yy)) * 0.2
    arguments = dict(
        sample_id="controls", sample_index=11, patch_size=2, top_fraction=0.20,
        num_null_shifts=7, null_seed=3520,
    )
    first, _ = analyze_spatial_maps(
        inputs, target, inputs, field, torch.tensor([0.2, 0.4, 0.6]), **arguments
    )
    second, _ = analyze_spatial_maps(
        inputs, target, inputs, field, torch.tensor([0.2, 0.4, 0.6]), **arguments
    )
    for prefix in ("anchor", "gradient"):
        for suffix in (
            "null_spearman_rgb_mean", "null_top20_iou_rgb_mean",
            "null_spearman_e00_mean",
        ):
            assert first[f"{prefix}_{suffix}"] == second[f"{prefix}_{suffix}"]


def test_conditioning_ablations_preserve_capacity_and_change_signal() -> None:
    kwargs = dict(
        feat_dim=4, num_frequencies=2, mlp_hidden_dim=8,
        mlp_hidden_layers=2, anchor_hidden_dim=4, query_chunk_size=None,
    )
    torch.manual_seed(3520)
    full = UnderwaterImplicitCorrectionField(**kwargs)
    with torch.no_grad():
        for layer in full.field_mlp.net:
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(0.02)
                layer.bias.fill_(0.01)
    no_coordinate = UnderwaterImplicitCorrectionField(
        **kwargs, use_spatial_conditioning=False
    )
    no_global = UnderwaterImplicitCorrectionField(
        **kwargs, use_global_field_conditioning=False
    )
    no_coordinate.load_state_dict(full.state_dict(), strict=True)
    no_global.load_state_dict(full.state_dict(), strict=True)
    assert trainable_parameter_count(full) == trainable_parameter_count(no_coordinate)
    assert trainable_parameter_count(full) == trainable_parameter_count(no_global)
    assert {
        key: tuple(value.shape) for key, value in full.state_dict().items()
    } == {key: tuple(value.shape) for key, value in no_coordinate.state_dict().items()}
    assert full.state_dict().keys() == no_global.state_dict().keys()
    inputs = torch.linspace(0.05, 0.95, 3 * 9 * 11).reshape(1, 3, 9, 11)
    with torch.inference_mode():
        full_output = full(inputs)
        coordinate_output = no_coordinate(inputs)
        global_output = no_global(inputs)
    assert not torch.equal(full_output, coordinate_output)
    assert not torch.equal(full_output, global_output)


def test_legacy_v16_config_and_strict_checkpoint_remain_compatible() -> None:
    model_config = _small_model_config()
    assert "field_variant" not in model_config["uicf"]
    assert "use_spatial_conditioning" not in model_config["uicf"]
    source = build_model(model_config)
    assert source.uicf.use_spatial_conditioning is True
    assert source.uicf.use_global_field_conditioning is True
    checkpoint = {
        "version": "v16", "epoch": 1,
        "resolved_config": {"model": model_config},
        "model_state_dict": source.state_dict(),
    }
    loaded = build_and_load_v16_model(
        {"model": model_config}, checkpoint, torch.device("cpu")
    )
    assert loaded.state_dict().keys() == source.state_dict().keys()


def test_full_and_conditioning_ablation_models_have_identical_state_shapes() -> None:
    full_config = _small_model_config()
    coordinate_config = deepcopy(full_config)
    global_config = deepcopy(full_config)
    coordinate_config["uicf"]["use_spatial_conditioning"] = False
    global_config["uicf"]["use_global_field_conditioning"] = False
    models = [build_model(item) for item in (full_config, coordinate_config, global_config)]
    counts = [trainable_parameter_count(model) for model in models]
    assert counts[0] == counts[1] == counts[2]
    backbone_shapes = [
        {key: tuple(value.shape) for key, value in model.backbone.state_dict().items()}
        for model in models
    ]
    uicf_shapes = [
        {key: tuple(value.shape) for key, value in model.uicf.state_dict().items()}
        for model in models
    ]
    assert backbone_shapes[0] == backbone_shapes[1] == backbone_shapes[2]
    assert uicf_shapes[0] == uicf_shapes[1] == uicf_shapes[2]


def test_conv_control_identity_shapes_and_parameter_match() -> None:
    implicit = UnderwaterImplicitCorrectionField()
    control = ConvolutionalCorrectionField()
    assert control.field_variant == "conv_control"
    assert control.use_spatial_conditioning is False
    inputs = torch.rand(2, 3, 15, 17)
    with torch.inference_mode():
        details = control(inputs, return_details=True)
    assert details.enhanced.shape == details.correction_field.shape == inputs.shape
    assert details.chromatic_anchor.shape == (2, 3)
    assert details.global_feature.shape == (2, 48)
    assert torch.count_nonzero(details.correction_field) == 0
    torch.testing.assert_close(details.enhanced, inputs, rtol=0, atol=0)
    report = correction_parameter_report(implicit, control)
    assert report["relative_difference"] <= 0.05
    assert report["implicit_trainable_parameters"] == 137_734
    assert report["convolutional_trainable_parameters"] == 136_918


@pytest.mark.parametrize("dataset", ("lsui", "uieb"))
def test_ablation_configs_change_only_allowed_fields(dataset: str) -> None:
    if dataset == "lsui":
        base_path = ROOT / "configs/config_v16.yaml"
        names = (
            "config_v16_uicf_no_coordinate.yaml",
            "config_v16_uicf_no_global_field.yaml",
            "config_v16_uicf_conv_control.yaml",
        )
    else:
        base_path = ROOT / "configs/uieb/config_v16_uieb.yaml"
        names = (
            "config_v16_uicf_no_coordinate_uieb.yaml",
            "config_v16_uicf_no_global_field_uieb.yaml",
            "config_v16_uicf_conv_control_uieb.yaml",
        )
    base = yaml.safe_load(base_path.read_text())
    for name in names:
        candidate = yaml.safe_load((ROOT / "configs/ablations" / dataset / name).read_text())
        for section in (
            "data", "loss", "optimizer", "scheduler", "training", "checkpoint",
            "evaluation", "metrics", "test", "logging",
        ):
            assert candidate[section] == base[section], (name, section)
        assert {k: v for k, v in candidate["model"].items() if k != "uicf"} == {
            k: v for k, v in base["model"].items() if k != "uicf"
        }


def test_cross_dataset_cli_uses_external_test_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "lsui_run"
    lsui_root, uieb_root = tmp_path / "lsui", tmp_path / "uieb"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "split_snapshot").mkdir()
    for root in (lsui_root, uieb_root):
        (root / "input").mkdir(parents=True)
        (root / "gt").mkdir()
    for split in ("train", "validation", "test"):
        sample_id = f"lsui_{split}"
        array = np.full((16, 16, 3), 50, dtype=np.uint8)
        Image.fromarray(array).save(lsui_root / "input" / f"{sample_id}.png")
        Image.fromarray(array + 5).save(lsui_root / "gt" / f"{sample_id}.png")
        (run_dir / "split_snapshot" / f"{split}.tsv").write_text(
            f"{sample_id}\tinput/{sample_id}.png\tgt/{sample_id}.png\n"
        )
    external_id = "uieb_test"
    yy, xx = np.mgrid[:16, :16]
    array = np.stack((40 + xx, 45 + yy, 50 + xx + yy), axis=-1).astype(np.uint8)
    Image.fromarray(array).save(uieb_root / "input" / f"{external_id}.png")
    Image.fromarray(array + 8).save(uieb_root / "gt" / f"{external_id}.png")
    external_manifest = tmp_path / "uieb_test.tsv"
    external_manifest.write_text(
        f"{external_id}\tinput/{external_id}.png\tgt/{external_id}.png\n"
    )
    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text())
    config["model"] = _small_model_config()
    config["data"] = {
        **config["data"], "root": str(lsui_root),
        "expected_counts": {"train": 1, "validation": 1, "test": 1},
        "num_workers": 0,
    }
    config["evaluation"] = {"resize": True, "size": 16}
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    model = build_model(config["model"])
    torch.save(
        {
            "version": "v16", "epoch": 2,
            "resolved_config": {"model": config["model"]},
            "model_state_dict": model.state_dict(),
        },
        run_dir / "best/best_psnr.pt",
    )
    evaluation_config = deepcopy(config)
    evaluation_config["data"] = {
        **evaluation_config["data"], "dataset": "UIEB", "root": str(uieb_root),
        "test_manifest": str(external_manifest), "expected_counts": {"test": 1},
    }
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(yaml.safe_dump(evaluation_config, sort_keys=False))
    alignment_main(
        [
            "--run-dir", str(run_dir), "--checkpoint", "best_psnr",
            "--evaluation-config", str(evaluation_path), "--batch-size", "1",
            "--num-workers", "0", "--patch-size", "16", "--num-null-shifts", "2",
            "--bootstrap-samples", "20", "--viz-k", "1",
        ]
    )
    output = run_dir / "result/uicf_representation_alignment"
    protocol = json.loads((output / "protocol.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert protocol["checkpoint_dataset"] == "LSUI19"
    assert protocol["evaluation_dataset"] == "UIEB"
    assert protocol["evaluation_mode"] == "cross_dataset"
    assert protocol["train_count"] is None and protocol["validation_count"] is None
    assert summary["evaluation_mode"] == "cross_dataset"
    assert summary["test_count"] == 1


def test_run_summarizer_reads_existing_results_without_recomputation(tmp_path: Path) -> None:
    run_dir = tmp_path / "full_run"
    output = run_dir / "result/uicf_representation_alignment"
    output.mkdir(parents=True)
    metric = lambda value: {"mean": value, "std": 0.0, "median": value, "valid_count": 1, "invalid_count": 0}
    summary = {
        "dataset": "LSUI19",
        "quality_metrics": {"psnr": metric(25.0), "ssim": metric(0.8), "e00": metric(4.0)},
        "raw_field_representation": {
            "metrics": {
                "raw_field_spearman_rgb": metric(0.5),
                "raw_field_top20_iou_rgb": metric(0.3),
                "raw_field_spearman_e00": metric(0.4),
            },
            "null_controls": {
                "spearman_rgb": {"mean_null": 0.1, "mean_real_minus_null": 0.4},
                "top20_iou_rgb": {"mean_null": 0.2},
                "spearman_e00": {"mean_null": 0.05},
            },
        },
        "bootstrap": {
            "raw_field_spearman_gain": {"ci95_low": 0.2, "ci95_high": 0.6},
            "raw_minus_anchor_spearman_rgb": {"ci95_low": 0.1, "ci95_high": 0.3},
            "raw_minus_gradient_spearman_rgb": {"ci95_low": 0.05, "ci95_high": 0.25},
        },
    }
    protocol = {
        "field_variant": "implicit", "use_spatial_conditioning": True,
        "use_global_field_conditioning": True, "use_learned_anchor": True,
    }
    (output / "summary.json").write_text(json.dumps(summary))
    (output / "protocol.json").write_text(json.dumps(protocol))
    destination = tmp_path / "comparison"
    summarize_main(
        ["--run-dir", str(run_dir), "--output-dir", str(destination)]
    )
    csv_text = (destination / "representation_run_comparison.csv").read_text()
    result = json.loads((destination / "representation_run_comparison.json").read_text())
    assert "raw_field_spearman_rgb" in csv_text
    assert result["run_count"] == 1
    assert result["runs"][0]["psnr"] == 25.0
