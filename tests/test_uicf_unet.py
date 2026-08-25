from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from src.shared.uicf_inr import UICFINROutput, UnderwaterImplicitCorrectionField
from src.shared.uicf_models import UICFParallelBranch, UICFPreBackbone
from src.v4.models import PlainUNet, build_model as build_v4


ROOT = Path(__file__).resolve().parents[1]
UICF_CONFIG = {
    "feat_dim": 48,
    "num_frequencies": 8,
    "mlp_hidden_dim": 128,
    "mlp_hidden_layers": 3,
    "anchor_hidden_dim": 64,
    "query_chunk_size": 65536,
}


def _load(version: str) -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / f"config_{version}.yaml").read_text(encoding="utf-8")
    )


def _model_config(version: str, *, base_channels: int | None = None) -> dict:
    config = dict(_load(version)["model"])
    config["uicf"] = dict(config.get("uicf", {}))
    if base_channels is not None:
        config["base_channels"] = base_channels
    return config


def _build(version: str, *, base_channels: int | None = None) -> nn.Module:
    return importlib.import_module(f"src.{version}.models").build_model(
        _model_config(version, base_channels=base_channels)
    )


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _has_nonzero_gradient(parameters) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in parameters
    )


def test_v13_v14_configs_preserve_v4_protocol_and_v11_v12_uicf() -> None:
    v4, v11, v12, v13, v14 = (_load(version) for version in ("v4", "v11", "v12", "v13", "v14"))
    for section in (
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
    ):
        assert v13[section] == v14[section] == v4[section]
        if section == "data":
            legacy_data = {
                key: value
                for key, value in v13[section].items()
                if key not in {"dataset", "expected_counts"}
            }
            assert legacy_data == v11[section] == v12[section]
        else:
            assert v13[section] == v11[section] == v12[section]

    assert v13["experiment"] == {
        "version": "v13",
        "name": "PlainUNet_UICF_PreBackbone",
        "seed": 3520,
        "output_root": "experiments",
    }
    assert v14["experiment"] == {
        "version": "v14",
        "name": "PlainUNet_UICF_ParallelBranch",
        "seed": 3520,
        "output_root": "experiments",
    }
    assert v13["model"]["type"] == "plain_unet_uicf_pre_backbone"
    assert v14["model"]["type"] == "plain_unet_uicf_parallel_branch"
    for key in (
        "in_channels",
        "out_channels",
        "base_channels",
        "use_batch_norm",
        "output_activation",
    ):
        assert v13["model"][key] == v14["model"][key] == v4["model"][key]
    assert v13["model"]["uicf"] == v14["model"]["uicf"] == UICF_CONFIG
    assert v13["model"]["uicf"] == v11["model"]["uicf"] == v12["model"]["uicf"]


def test_v13_v14_reuse_isolated_pipeline_and_one_canonical_uicf() -> None:
    for version, reference in (("v13", "v11"), ("v14", "v12")):
        for filename in ("engine.py", "experiment.py", "losses.py", "metrics.py", "utils.py"):
            assert (ROOT / "src" / version / filename).read_bytes() == (
                ROOT / "src" / reference / filename
            ).read_bytes()
        assert not (ROOT / "src" / version / "models" / "unet.py").exists()
        assert not (ROOT / "src" / version / "models" / "uicf_inr.py").exists()
    assert (ROOT / "src/v13/dataset.py").read_bytes() == (
        ROOT / "src/v14/dataset.py"
    ).read_bytes()
    assert (ROOT / "src/v11/dataset.py").read_bytes() == (
        ROOT / "src/v12/dataset.py"
    ).read_bytes()

    v11 = importlib.import_module("src.v11.models").UnderwaterImplicitCorrectionField
    v12 = importlib.import_module("src.v12.models").UnderwaterImplicitCorrectionField
    v13 = importlib.import_module("src.v13.models").UnderwaterImplicitCorrectionField
    v14 = importlib.import_module("src.v14.models").UnderwaterImplicitCorrectionField
    assert v11 is v12 is v13 is v14 is UnderwaterImplicitCorrectionField


@pytest.mark.parametrize(
    ("version", "wrapper_type"),
    (("v13", UICFPreBackbone), ("v14", UICFParallelBranch)),
)
def test_model_type_backbone_diagnostics_identity_and_odd_resolution(
    version: str, wrapper_type: type[nn.Module]
) -> None:
    model = _build(version, base_channels=8).eval()
    assert isinstance(model, wrapper_type)
    assert isinstance(model.backbone, PlainUNet)
    assert type(model.uicf) is UnderwaterImplicitCorrectionField
    assert list(dict(model.named_children())) == ["backbone", "uicf"]
    assert model.backbone.in_channels == model.backbone.out_channels == 3
    assert model.backbone.base_channels == 8
    assert model.backbone.output_activation_name == "sigmoid"

    inputs = torch.rand(1, 3, 63, 79)
    with torch.inference_mode():
        output, details = model.forward_with_uicf_details(inputs)
        shape_output, shapes = model.forward_with_shapes(inputs)
    assert output.shape == shape_output.shape == inputs.shape
    assert details.enhanced.shape == details.correction_field.shape == inputs.shape
    assert details.chromatic_anchor.shape == (1, 3)
    assert details.global_feature.shape == (1, 48)
    assert shapes["uicf_output"] == shapes["correction_field"] == tuple(inputs.shape)
    assert shapes["final_output"] == tuple(inputs.shape)
    assert torch.count_nonzero(details.correction_field) == 0
    torch.testing.assert_close(details.enhanced, inputs, rtol=0, atol=0)
    torch.testing.assert_close(output, shape_output, rtol=0, atol=0)


@pytest.mark.parametrize("version", ("v13", "v14"))
def test_same_seed_backbone_state_and_initial_v4_output_are_exact(version: str) -> None:
    v4_config = _load("v4")["model"]
    torch.manual_seed(3520)
    baseline = build_v4(v4_config).eval()
    torch.manual_seed(3520)
    model = _build(version).eval()

    baseline_state = baseline.state_dict()
    backbone_state = model.backbone.state_dict()
    assert baseline_state.keys() == backbone_state.keys()
    for key in baseline_state:
        assert torch.equal(baseline_state[key], backbone_state[key]), key

    inputs = torch.rand(1, 3, 32, 48)
    with torch.inference_mode():
        expected = baseline(inputs)
        actual, details = model.forward_with_uicf_details(inputs)
    torch.testing.assert_close(details.enhanced, inputs, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _AddTwoUICF(nn.Module):
    def forward(self, inputs: torch.Tensor, return_details: bool = False):
        enhanced = inputs + 2.0
        if not return_details:
            return enhanced
        return UICFINROutput(
            enhanced=enhanced,
            correction_field=torch.full_like(inputs, 2.0),
            chromatic_anchor=inputs.new_zeros((inputs.shape[0], 3)),
            global_feature=inputs.new_zeros((inputs.shape[0], 48)),
        )


class _TimesThreeBackbone(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 3.0


def test_v13_v14_topology_semantics_are_backbone_independent() -> None:
    inputs = torch.rand(1, 3, 5, 7)
    v13 = UICFPreBackbone(_TimesThreeBackbone(), _AddTwoUICF())
    v14 = UICFParallelBranch(_TimesThreeBackbone(), _AddTwoUICF())
    torch.testing.assert_close(v13(inputs), 3.0 * (inputs + 2.0), rtol=0, atol=0)
    expected_v14 = 3.0 * inputs + ((inputs + 2.0) - inputs)
    torch.testing.assert_close(v14(inputs), expected_v14, rtol=0, atol=0)


@pytest.mark.parametrize("version", ("v13", "v14"))
def test_backward_reaches_plain_unet_and_uicf_field(version: str) -> None:
    torch.manual_seed(3520)
    model = _build(version, base_channels=4).train()
    inputs = torch.rand(1, 3, 32, 32)
    targets = torch.rand_like(inputs)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert _has_nonzero_gradient(model.backbone.parameters())
    assert _has_nonzero_gradient(model.uicf.field_mlp.parameters())
    assert all(parameter.grad is not None for parameter in model.uicf.parameters())


def test_v13_v14_exact_parameter_counts_and_no_fusion_parameters() -> None:
    baseline = build_v4(_load("v4")["model"])
    canonical_uicf = UnderwaterImplicitCorrectionField()
    v13, v14 = _build("v13"), _build("v14")
    assert _count(baseline) == 31_037_763
    assert _count(canonical_uicf) == _count(v13.uicf) == _count(v14.uicf) == 137_734
    assert _count(v13) == _count(v14) == 31_175_497
    assert _count(v13) - _count(baseline) == 137_734
    assert _count(v14) - _count(baseline) == 137_734
    assert list(dict(v14.named_children())) == ["backbone", "uicf"]
