from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from src.v4.models import build_model as build_v4
from src.v5.models import PlainUNetPointINR, PointINR, build_model as build_v5
from src.v6.models import GLINR, PlainUNetGLINR, build_model as build_v6


ROOT = Path(__file__).resolve().parents[1]
BUILDERS = {"v4": build_v4, "v5": build_v5, "v6": build_v6}
SPLIT_HASHES = {
    "train": "5cf9be63b7ed565ad3190936c61efe56c7b27c1e0cb7d8b0c9266ef62f87c6ab",
    "validation": "e81c35ae694ce9c0e2ba656ad5dece093ae7804d4ffd711eeb357103686f9c18",
    "test": "cee6a22aeb2903f1cd053f641eab3aa1733f55a394682e257cb4ab4b27b0373c",
}


def _load(version: str) -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / f"config_{version}.yaml").read_text(encoding="utf-8")
    )


def _small_model_config(version: str) -> dict:
    config = dict(_load(version)["model"])
    config["base_channels"] = 8
    return config


def test_v5_v6_formal_configs_are_strict_controlled_variants() -> None:
    v2, v3, v4, v5, v6 = (_load(f"v{index}") for index in range(2, 7))
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
            assert v4_data == v5[section] == v6[section]
        else:
            assert v4[section] == v5[section] == v6[section]

    common_model = (
        "in_channels",
        "out_channels",
        "base_channels",
        "use_batch_norm",
        "output_activation",
    )
    for key in common_model:
        assert v4["model"][key] == v5["model"][key] == v6["model"][key]
    assert v5["model"]["point_inr"] == v2["model"]["point_inr"]
    assert v6["model"]["glinr"] == v3["model"]["glinr"]
    assert v5["experiment"]["name"] == "PlainUNet_PointINR"
    assert v6["experiment"]["name"] == "PlainUNet_GLINR"


def test_v5_v6_keep_exact_split_manifests_and_hashes() -> None:
    for version in ("v4", "v5", "v6"):
        config = _load(version)
        for split, expected_hash in SPLIT_HASHES.items():
            relative = config["data"][f"{split}_manifest"]
            payload = (ROOT / relative).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == expected_hash


def test_backbone_and_inr_sources_are_byte_identical_to_their_controls() -> None:
    unet_sources = [
        (ROOT / "src" / version / "models" / "unet.py").read_bytes()
        for version in ("v4", "v5", "v6")
    ]
    assert unet_sources[0] == unet_sources[1] == unet_sources[2]
    assert (ROOT / "src/v5/models/point_inr.py").read_bytes() == (
        ROOT / "src/v2/models/point_inr.py"
    ).read_bytes()
    assert (ROOT / "src/v6/models/glinr.py").read_bytes() == (
        ROOT / "src/v3/models/glinr.py"
    ).read_bytes()


def test_v5_v6_are_import_isolated_and_protocol_code_matches_v4() -> None:
    protocol_files = ("engine.py", "experiment.py", "losses.py", "metrics.py", "utils.py")
    for version in ("v5", "v6"):
        for filename in protocol_files:
            assert (ROOT / "src" / version / filename).read_bytes() == (
                ROOT / "src/v4" / filename
            ).read_bytes()
        for path in (ROOT / "src" / version).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert not any(f"src.v{other}" in source for other in range(1, 7) if f"v{other}" != version)
    assert (ROOT / "src/v5/dataset.py").read_bytes() == (
        ROOT / "src/v6/dataset.py"
    ).read_bytes() == (ROOT / "src/v1/dataset.py").read_bytes()


def test_formal_builders_create_only_the_requested_inr() -> None:
    v5 = build_v5(_load("v5")["model"])
    v6 = build_v6(_load("v6")["model"])
    assert isinstance(v5, PlainUNetPointINR)
    assert isinstance(v5.bottleneck_module, PointINR)
    assert not any(isinstance(module, GLINR) for module in v5.modules())
    assert isinstance(v6, PlainUNetGLINR)
    assert isinstance(v6.bottleneck_module, GLINR)
    assert not any(isinstance(module, PointINR) for module in v6.modules())


def test_common_unet_parameters_and_initial_predictions_match_v4() -> None:
    models = {}
    for version, builder in BUILDERS.items():
        torch.manual_seed(3520)
        models[version] = builder(_small_model_config(version)).eval()
    v4_state = models["v4"].state_dict()
    for version in ("v5", "v6"):
        state = models[version].state_dict()
        for name, value in v4_state.items():
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)

    inputs = torch.rand(2, 3, 32, 32)
    with torch.inference_mode():
        outputs = {version: model(inputs) for version, model in models.items()}
    torch.testing.assert_close(outputs["v5"], outputs["v4"], rtol=0, atol=0)
    torch.testing.assert_close(outputs["v6"], outputs["v4"], rtol=0, atol=0)


@pytest.mark.parametrize("version", ("v4", "v5", "v6"))
def test_formal_256_forward_shape_and_finite(version: str) -> None:
    model = BUILDERS[version](_load(version)["model"]).eval()
    inputs = torch.rand(2, 3, 256, 256)
    with torch.inference_mode():
        output, shapes = model.forward_with_shapes(inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert shapes["bottleneck"] == (2, 1024, 16, 16)
    if version != "v4":
        assert shapes["module_input"] == shapes["module_output"] == shapes["bottleneck"]
    del output, inputs, model
    gc.collect()


@pytest.mark.parametrize("version", ("v5", "v6"))
def test_inr_forward_backward_is_finite_and_connected(version: str) -> None:
    model = BUILDERS[version](_small_model_config(version)).train()
    inputs = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand_like(inputs)
    output = model(inputs)
    loss = (output - target).square().mean()
    loss.backward()

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all() and torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    inr_parameters = list(model.bottleneck_module.parameters())
    assert inr_parameters and all(parameter.requires_grad for parameter in inr_parameters)
    assert all(parameter.grad is not None for parameter in inr_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in inr_parameters)


@pytest.mark.parametrize("version", ("v5", "v6"))
def test_training_amp_code_path_has_no_dtype_or_autograd_error(version: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BUILDERS[version](_small_model_config(version)).to(device).train()
    inputs = torch.rand(2, 3, 32, 32, device=device)
    targets = torch.rand_like(inputs)
    amp_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        loss = (model(inputs) - targets).square().mean()
    loss.backward()
    assert torch.isfinite(loss)


class _ZeroINR(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(features)


@pytest.mark.parametrize("version", ("v5", "v6"))
def test_decoder_consumes_inr_output_without_an_outer_double_residual(version: str) -> None:
    model = BUILDERS[version](_small_model_config(version)).eval()
    model.bottleneck_module = _ZeroINR()
    inputs = torch.rand(1, 3, 33, 35)
    with torch.inference_mode():
        padded, height, width = model._pad(inputs)
        e1 = model.encoder1(padded)
        e2 = model.encoder2(model.pool(e1))
        e3 = model.encoder3(model.pool(e2))
        e4 = model.encoder4(model.pool(e3))
        bottleneck = model.bottleneck(model.pool(e4))
        d4 = model.decoder4(model._concat(model.upconv4(torch.zeros_like(bottleneck)), e4))
        d3 = model.decoder3(model._concat(model.upconv3(d4), e3))
        d2 = model.decoder2(model._concat(model.upconv2(d3), e2))
        d1 = model.decoder1(model._concat(model.upconv1(d2), e1))
        expected = model.output_activation(model.output_conv(d1))[..., :height, :width]
        actual = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_formal_parameter_counts_and_deltas() -> None:
    expected = {"v4": 31_037_763, "v5": 31_320_899, "v6": 32_485_315}
    for version, count in expected.items():
        model = BUILDERS[version](_load(version)["model"])
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        assert total == trainable == count
    assert expected["v5"] - expected["v4"] == 283_136
    assert expected["v6"] - expected["v4"] == 1_447_552
