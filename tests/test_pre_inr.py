from __future__ import annotations

import gc
import hashlib
import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from src.shared.pre_inr import PreINRModel
from src.v1.models import build_model as build_v1
from src.v1.models.nafnet import NAFNet
from src.v2.models.point_inr import PointINR
from src.v3.models.glinr import GLINR
from src.v4.models import build_model as build_v4
from src.v4.models.unet import PlainUNet


ROOT = Path(__file__).resolve().parents[1]
PRE_VERSIONS = ("v7", "v8", "v9", "v10")
SPLIT_HASHES = {
    "train": "5cf9be63b7ed565ad3190936c61efe56c7b27c1e0cb7d8b0c9266ef62f87c6ab",
    "validation": "e81c35ae694ce9c0e2ba656ad5dece093ae7804d4ffd711eeb357103686f9c18",
    "test": "cee6a22aeb2903f1cd053f641eab3aa1733f55a394682e257cb4ab4b27b0373c",
}


def _load(version: str) -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / f"config_{version}.yaml").read_text(encoding="utf-8")
    )


def _build(version: str) -> PreINRModel:
    builder = importlib.import_module(f"src.{version}.models").build_model
    model = builder(_load(version)["model"])
    assert isinstance(model, PreINRModel)
    return model


def _small_config(version: str) -> dict:
    config = dict(_load(version)["model"])
    if version in {"v7", "v8"}:
        config["base_channels"] = 8
    else:
        config["width"] = 8
        config["enc_blk_nums"] = [1]
        config["dec_blk_nums"] = [1]
    return config


def _build_small(version: str) -> PreINRModel:
    builder = importlib.import_module(f"src.{version}.models").build_model
    return builder(_small_config(version))


def test_pre_inr_configs_preserve_their_baseline_protocols() -> None:
    v1, v2, v3, v4 = (_load(f"v{index}") for index in (1, 2, 3, 4))
    configs = {version: _load(version) for version in PRE_VERSIONS}
    protocol_sections = (
        "data",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "checkpoint",
        "evaluation",
        "metrics",
        "test",
        "logging",
    )
    for section in protocol_sections:
        if section == "data":
            v4_data = {
                key: value
                for key, value in v4[section].items()
                if key not in {"dataset", "expected_counts"}
            }
            assert configs["v7"][section] == configs["v8"][section] == v4_data
        else:
            assert configs["v7"][section] == configs["v8"][section] == v4[section]
        assert configs["v9"][section] == configs["v10"][section] == v1[section]

    unet_keys = ("in_channels", "out_channels", "base_channels", "use_batch_norm", "output_activation")
    naf_keys = ("img_channel", "width", "enc_blk_nums", "middle_blk_num", "dec_blk_nums")
    for key in unet_keys:
        assert configs["v7"]["model"][key] == configs["v8"]["model"][key] == v4["model"][key]
    for key in naf_keys:
        assert configs["v9"]["model"][key] == configs["v10"]["model"][key] == v1["model"][key]
    assert configs["v7"]["model"]["point_inr"] == configs["v9"]["model"]["point_inr"] == v2["model"]["point_inr"]
    assert configs["v8"]["model"]["glinr"] == configs["v10"]["model"]["glinr"] == v3["model"]["glinr"]


def test_pre_inr_versions_keep_fixed_split_hashes() -> None:
    for version in PRE_VERSIONS:
        config = _load(version)
        for split, expected in SPLIT_HASHES.items():
            manifest = ROOT / config["data"][f"{split}_manifest"]
            assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected


def test_pre_inr_reuses_exact_existing_classes_without_model_copies() -> None:
    models = {version: _build(version) for version in PRE_VERSIONS}
    assert type(models["v7"].pre_inr) is type(models["v9"].pre_inr) is PointINR
    assert type(models["v8"].pre_inr) is type(models["v10"].pre_inr) is GLINR
    assert type(models["v7"].backbone) is type(models["v8"].backbone) is PlainUNet
    assert type(models["v9"].backbone) is type(models["v10"].backbone) is NAFNet
    for version in PRE_VERSIONS:
        assert not (ROOT / "src" / version / "models" / "point_inr.py").exists()
        assert not (ROOT / "src" / version / "models" / "glinr.py").exists()
        assert not (ROOT / "src" / version / "models" / "unet.py").exists()
        assert not (ROOT / "src" / version / "models" / "nafnet.py").exists()


def test_pre_inr_is_rgb_only_and_no_adapter_or_second_inr_exists() -> None:
    for version in PRE_VERSIONS:
        model = _build(version)
        assert model.pre_inr.channels == 3
        assert list(dict(model.named_children())) == ["pre_inr", "backbone"]
        if version in {"v7", "v9"}:
            assert isinstance(model.pre_inr, PointINR)
            assert not any(isinstance(module, GLINR) for module in model.modules())
        else:
            assert isinstance(model.pre_inr, GLINR)
            assert not any(isinstance(module, PointINR) for module in model.modules())
        if isinstance(model.backbone, NAFNet):
            assert isinstance(model.backbone.bottleneck_module, nn.Identity)
            assert len(model.backbone.middle_blks) == 0

    wrapper_source = (ROOT / "src/shared/pre_inr.py").read_text(encoding="utf-8")
    assert "clamp(" not in wrapper_source
    assert "Sigmoid" not in wrapper_source


def test_protocol_implementation_matches_the_requested_baseline() -> None:
    protocol_files = ("dataset.py", "engine.py", "experiment.py", "losses.py", "metrics.py", "utils.py")
    for version, baseline in (("v7", "v4"), ("v8", "v4"), ("v9", "v1"), ("v10", "v1")):
        for filename in protocol_files:
            reference = "v1" if filename == "dataset.py" else baseline
            assert (ROOT / "src" / version / filename).read_bytes() == (
                ROOT / "src" / reference / filename
            ).read_bytes()


@pytest.mark.parametrize("version", PRE_VERSIONS)
def test_formal_pre_inr_and_final_forward_shapes_are_finite(version: str) -> None:
    model = _build(version).eval()
    inputs = torch.rand(2, 3, 256, 256)
    with torch.inference_mode():
        pre_output = model.forward_pre_inr(inputs)
        output, shapes = model.forward_with_shapes(inputs)
    assert pre_output.shape == inputs.shape == output.shape
    assert shapes["input"] == shapes["pre_inr_output"] == shapes["final_output"] == tuple(inputs.shape)
    assert torch.isfinite(pre_output).all() and torch.isfinite(output).all()
    assert torch.equal(pre_output, inputs)
    del model, inputs, pre_output, output
    gc.collect()


@pytest.mark.parametrize(
    ("version", "baseline_version"),
    (("v7", "v4"), ("v8", "v4"), ("v9", "v1"), ("v10", "v1")),
)
def test_formal_initialization_equivalence_after_copying_backbone_state(
    version: str, baseline_version: str
) -> None:
    baseline_config = _load(baseline_version)["model"]
    baseline = (build_v4 if baseline_version == "v4" else build_v1)(baseline_config).eval()
    model = _build(version).eval()
    model.backbone.load_state_dict(baseline.state_dict(), strict=True)
    inputs = torch.rand(2, 3, 256, 256)
    with torch.inference_mode():
        expected = baseline(inputs)
        actual = model(inputs)
    max_abs_diff = float((actual - expected).abs().max())
    assert max_abs_diff == 0.0
    del baseline, model, inputs, expected, actual
    gc.collect()


def test_equal_seed_also_preserves_common_backbone_initialization() -> None:
    for version, baseline_version in (("v7", "v4"), ("v8", "v4"), ("v9", "v1"), ("v10", "v1")):
        torch.manual_seed(3520)
        baseline_config = _load(baseline_version)["model"]
        baseline = (build_v4 if baseline_version == "v4" else build_v1)(baseline_config)
        torch.manual_seed(3520)
        model = _build(version)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(model.backbone.state_dict()[name], value, rtol=0, atol=0)


def test_cross_backbone_inr_parameter_deltas_are_identical() -> None:
    baselines = {
        "v1": build_v1(_load("v1")["model"]),
        "v4": build_v4(_load("v4")["model"]),
    }
    models = {version: _build(version) for version in PRE_VERSIONS}
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    assert count(models["v7"].pre_inr) == count(models["v9"].pre_inr) == 20_739
    assert count(models["v8"].pre_inr) == count(models["v10"].pre_inr) == 139_651
    assert count(models["v7"]) - count(baselines["v4"]) == 20_739
    assert count(models["v9"]) - count(baselines["v1"]) == 20_739
    assert count(models["v8"]) - count(baselines["v4"]) == 139_651
    assert count(models["v10"]) - count(baselines["v1"]) == 139_651


@pytest.mark.parametrize("version", PRE_VERSIONS)
def test_pre_inr_statistics_match_input_at_zero_initialized_start(version: str) -> None:
    model = _build_small(version).eval()
    inputs = torch.rand(2, 3, 32, 32)
    with torch.inference_mode():
        pre_output = model.forward_pre_inr(inputs)
    assert torch.equal(pre_output, inputs)
    input_stats = torch.stack((inputs.min(), inputs.max(), inputs.mean(), inputs.std()))
    pre_stats = torch.stack((pre_output.min(), pre_output.max(), pre_output.mean(), pre_output.std()))
    torch.testing.assert_close(pre_stats, input_stats, rtol=0, atol=0)


def _nonzero_gradient(parameters) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in parameters
    )


@pytest.mark.parametrize("version", PRE_VERSIONS)
def test_two_steps_connect_all_inr_branches_to_optimization(version: str) -> None:
    model = _build_small(version).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    inputs = torch.rand(2, 3, 32, 32)
    targets = torch.rand_like(inputs)

    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    inr_parameters = list(model.pre_inr.parameters())
    assert inr_parameters and all(parameter.requires_grad for parameter in inr_parameters)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in inr_parameters)
    if isinstance(model.pre_inr, PointINR):
        assert _nonzero_gradient(model.pre_inr.mlp.net[-1].parameters())
    else:
        assert _nonzero_gradient(model.pre_inr.fusion.net[-1].parameters())

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_loss = (model(inputs) - targets).square().mean()
    second_loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(second_loss)
    if isinstance(model.pre_inr, PointINR):
        linear_layers = [module for module in model.pre_inr.mlp.net if isinstance(module, nn.Linear)]
        assert all(_nonzero_gradient(layer.parameters()) for layer in linear_layers)
    else:
        assert _nonzero_gradient(model.pre_inr.latent_projection.parameters())
        assert _nonzero_gradient(model.pre_inr.local_branch.parameters())
        assert _nonzero_gradient(model.pre_inr.global_branch.parameters())
        assert _nonzero_gradient(model.pre_inr.fusion.parameters())


@pytest.mark.parametrize("version", PRE_VERSIONS)
def test_autocast_forward_backward_and_optimizer_step(version: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_small(version).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    inputs = torch.rand(2, 3, 32, 32, device=device)
    targets = torch.rand_like(inputs)
    optimizer.zero_grad(set_to_none=True)
    autocast_kwargs = {"device_type": device.type, "enabled": True}
    if device.type == "cpu":
        autocast_kwargs["dtype"] = torch.bfloat16
    with torch.autocast(**autocast_kwargs):
        output = model(inputs)
        loss = (output - targets).square().mean()
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all() and torch.isfinite(loss)
