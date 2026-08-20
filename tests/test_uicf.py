from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from src.shared.uicf_inr import (
    ConvBlock,
    CorrectionFieldMLP,
    GlobalChromaticAnchor,
    ImageEncoder,
    PeriodicSpatialEncoding,
    UICFINROutput,
    UnderwaterImplicitCorrectionField,
)
from src.shared.uicf_models import UICFParallelBranch, UICFPreBackbone
from src.v1.models import build_model as build_v1


ROOT = Path(__file__).resolve().parents[1]
SPLIT_HASHES = {
    "train": "5cf9be63b7ed565ad3190936c61efe56c7b27c1e0cb7d8b0c9266ef62f87c6ab",
    "validation": "e81c35ae694ce9c0e2ba656ad5dece093ae7804d4ffd711eeb357103686f9c18",
    "test": "cee6a22aeb2903f1cd053f641eab3aa1733f55a394682e257cb4ab4b27b0373c",
}


def _load(version: str) -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / f"config_{version}.yaml").read_text(encoding="utf-8")
    )


def _build(version: str):
    return importlib.import_module(f"src.{version}.models").build_model(
        _load(version)["model"]
    )


def _small_config(version: str) -> dict:
    config = dict(_load(version)["model"])
    config["width"] = 8
    config["enc_blk_nums"] = [1]
    config["dec_blk_nums"] = [1]
    return config


def _build_small(version: str):
    return importlib.import_module(f"src.{version}.models").build_model(
        _small_config(version)
    )


def test_uicf_encoder_anchor_and_field_mlp_are_exact() -> None:
    module = UnderwaterImplicitCorrectionField()
    assert module.feat_dim == 48
    assert isinstance(module.encoder, ImageEncoder)
    assert [type(layer) for layer in module.encoder.intro] == [nn.Conv2d, nn.GELU]
    intro = module.encoder.intro[0]
    assert intro.in_channels == 3 and intro.out_channels == 48
    assert intro.kernel_size == (3, 3) and intro.stride == (1, 1) and intro.padding == (1, 1)
    assert len(module.encoder.blocks) == 2
    for block in module.encoder.blocks:
        assert isinstance(block, ConvBlock)
        assert isinstance(block.skip, nn.Identity)
        assert [type(layer) for layer in block.block] == [nn.Conv2d, nn.GELU, nn.Conv2d, nn.GELU]
        for convolution in (block.block[0], block.block[2]):
            assert convolution.in_channels == convolution.out_channels == 48
            assert convolution.kernel_size == (3, 3)
            assert convolution.stride == (1, 1) and convolution.padding == (1, 1)

    assert isinstance(module.chromatic_anchor, GlobalChromaticAnchor)
    anchor_layers = list(module.chromatic_anchor.anchor_mlp)
    assert [type(layer) for layer in anchor_layers] == [nn.Linear, nn.GELU, nn.Linear, nn.Sigmoid]
    assert (anchor_layers[0].in_features, anchor_layers[0].out_features) == (48, 64)
    assert (anchor_layers[2].in_features, anchor_layers[2].out_features) == (64, 3)

    assert isinstance(module.field_mlp, CorrectionFieldMLP)
    linear_layers = [layer for layer in module.field_mlp.net if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (128, 128), (128, 128), (128, 128), (128, 3)
    ]
    assert sum(isinstance(layer, nn.GELU) for layer in module.field_mlp.net) == 3
    assert torch.count_nonzero(linear_layers[-1].weight) == 0
    assert torch.count_nonzero(linear_layers[-1].bias) == 0
    assert module.field_mlp.query_chunk_size == 65536


def test_periodic_spatial_encoding_uses_fixed_xy_coordinates_without_pi() -> None:
    encoding = PeriodicSpatialEncoding(num_frequencies=8)
    actual = encoding(2, 2, 2, device=torch.device("cpu"), dtype=torch.float32)
    assert actual.shape == (2, 4, 32)
    bands = 2.0 ** torch.arange(8)
    coordinates = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    expected_rows = []
    for x, y in coordinates:
        expected_rows.append(
            torch.cat(((x * bands).sin(), (x * bands).cos(), (y * bands).sin(), (y * bands).cos()))
        )
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected, rtol=0, atol=0)
    torch.testing.assert_close(encoding.frequency_bands, bands, rtol=0, atol=0)


def test_uicf_details_shapes_and_initial_identity() -> None:
    module = UnderwaterImplicitCorrectionField().eval()
    inputs = torch.rand(2, 3, 64, 80)
    with torch.inference_mode():
        details = module(inputs, return_details=True)
        tensor_output = module(inputs, return_details=False)
    assert isinstance(details, UICFINROutput)
    assert isinstance(tensor_output, torch.Tensor)
    assert details.enhanced.shape == details.correction_field.shape == (2, 3, 64, 80)
    assert details.chromatic_anchor.shape == (2, 3)
    assert details.global_feature.shape == (2, 48)
    assert details.chromatic_anchor.min() >= 0 and details.chromatic_anchor.max() <= 1
    torch.testing.assert_close(details.enhanced, inputs, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(tensor_output, inputs, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(details.correction_field) == 0


def test_reconstruction_is_exact_formula_and_has_no_output_clamp() -> None:
    module = UnderwaterImplicitCorrectionField().eval()
    output_layer = module.field_mlp.net[-1]
    assert isinstance(output_layer, nn.Linear)
    with torch.no_grad():
        output_layer.bias.fill_(4.0)
    inputs = torch.ones(1, 3, 7, 9)
    with torch.inference_mode():
        details = module(inputs, return_details=True)
    expected = inputs + details.correction_field * (
        inputs - details.chromatic_anchor[:, :, None, None]
    )
    torch.testing.assert_close(details.enhanced, expected, rtol=0, atol=0)
    assert details.enhanced.max() > 1.0
    source = (ROOT / "src/shared/uicf_inr.py").read_text(encoding="utf-8")
    assert ".clamp(" not in source and "torch.clamp(" not in source


def test_chunked_and_unchunked_queries_are_equivalent() -> None:
    torch.manual_seed(3520)
    unchunked = UnderwaterImplicitCorrectionField(query_chunk_size=None).eval()
    output_layer = unchunked.field_mlp.net[-1]
    assert isinstance(output_layer, nn.Linear)
    with torch.no_grad():
        nn.init.normal_(output_layer.weight, std=0.01)
        nn.init.normal_(output_layer.bias, std=0.01)
    chunked = UnderwaterImplicitCorrectionField(query_chunk_size=17).eval()
    chunked.load_state_dict(unchunked.state_dict(), strict=True)
    inputs = torch.rand(2, 3, 9, 11)
    with torch.inference_mode():
        expected = unchunked(inputs, return_details=True)
        actual = chunked(inputs, return_details=True)
    assert isinstance(expected, UICFINROutput) and isinstance(actual, UICFINROutput)
    torch.testing.assert_close(actual.correction_field, expected.correction_field, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.enhanced, expected.enhanced, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual.chromatic_anchor, expected.chromatic_anchor, rtol=0, atol=0)
    torch.testing.assert_close(actual.global_feature, expected.global_feature, rtol=0, atol=0)


def test_v11_v12_configs_and_protocol_match_v1() -> None:
    baseline, v11, v12 = _load("v1"), _load("v11"), _load("v12")
    for section in (
        "data", "loss", "optimizer", "scheduler", "training", "checkpoint",
        "evaluation", "metrics", "test", "logging",
    ):
        assert v11[section] == v12[section] == baseline[section]
    backbone_keys = ("img_channel", "width", "enc_blk_nums", "middle_blk_num", "dec_blk_nums")
    for key in backbone_keys:
        assert v11["model"][key] == v12["model"][key] == baseline["model"][key]
    assert v11["model"]["uicf"] == v12["model"]["uicf"] == {
        "feat_dim": 48,
        "num_frequencies": 8,
        "mlp_hidden_dim": 128,
        "mlp_hidden_layers": 3,
        "anchor_hidden_dim": 64,
        "query_chunk_size": 65536,
    }
    for config in (v11, v12):
        for split, expected_hash in SPLIT_HASHES.items():
            manifest = ROOT / config["data"][f"{split}_manifest"]
            assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected_hash


def test_v11_v12_share_one_canonical_uicf_and_exact_v1_backbone() -> None:
    v11, v12 = _build("v11"), _build("v12")
    assert isinstance(v11, UICFPreBackbone)
    assert isinstance(v12, UICFParallelBranch)
    assert type(v11.uicf) is type(v12.uicf) is UnderwaterImplicitCorrectionField
    assert type(v11.backbone) is type(v12.backbone) is type(build_v1(_load("v1")["model"]))
    assert not (ROOT / "src/v11/models/uicf_inr.py").exists()
    assert not (ROOT / "src/v12/models/uicf_inr.py").exists()
    for version in ("v11", "v12"):
        for filename in ("dataset.py", "engine.py", "experiment.py", "losses.py", "metrics.py", "utils.py"):
            assert (ROOT / "src" / version / filename).read_bytes() == (
                ROOT / "src/v1" / filename
            ).read_bytes()


@pytest.mark.parametrize(("version", "wrapper_type"), (("v11", UICFPreBackbone), ("v12", UICFParallelBranch)))
def test_odd_resolution_and_initial_baseline_equivalence(version: str, wrapper_type) -> None:
    baseline = build_v1(_load("v1")["model"]).eval()
    model = _build(version).eval()
    assert isinstance(model, wrapper_type)
    model.backbone.load_state_dict(baseline.state_dict(), strict=True)
    inputs = torch.rand(2, 3, 63, 79)
    with torch.inference_mode():
        expected = baseline(inputs)
        actual, details = model.forward_with_uicf_details(inputs)
    assert actual.shape == details.enhanced.shape == details.correction_field.shape == inputs.shape
    assert details.chromatic_anchor.shape == (2, 3)
    assert details.global_feature.shape == (2, 48)
    assert torch.isfinite(actual).all() and torch.isfinite(details.enhanced).all()
    assert float((details.enhanced - inputs).abs().max()) == 0.0
    assert float((actual - expected).abs().max()) == 0.0


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


def test_v11_v12_topologies_are_the_exact_prescribed_formulas() -> None:
    inputs = torch.rand(1, 3, 5, 7)
    v11 = UICFPreBackbone(_TimesThreeBackbone(), _AddTwoUICF())
    v12 = UICFParallelBranch(_TimesThreeBackbone(), _AddTwoUICF())
    torch.testing.assert_close(v11(inputs), (inputs + 2.0) * 3.0, rtol=0, atol=0)
    expected_parallel = inputs * 3.0 + ((inputs + 2.0) - inputs)
    torch.testing.assert_close(v12(inputs), expected_parallel, rtol=0, atol=0)


def _has_nonzero_gradient(parameters) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in parameters
    )


@pytest.mark.parametrize("version", ("v11", "v12"))
def test_joint_backward_and_second_step_reach_uicf_and_backbone(version: str) -> None:
    model = _build_small(version).train()
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.uicf.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    inputs = torch.rand(2, 3, 16, 16)
    targets = torch.rand_like(inputs)

    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert _has_nonzero_gradient(model.backbone.parameters())
    assert _has_nonzero_gradient(model.uicf.field_mlp.net[-1].parameters())
    assert all(parameter.grad is not None for parameter in model.uicf.parameters())

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_loss = (model(inputs) - targets).square().mean()
    second_loss.backward()
    assert torch.isfinite(second_loss)
    assert _has_nonzero_gradient(model.uicf.encoder.parameters())
    assert _has_nonzero_gradient(model.uicf.chromatic_anchor.parameters())
    assert _has_nonzero_gradient(model.uicf.field_mlp.parameters())


@pytest.mark.parametrize("version", ("v11", "v12"))
def test_autocast_forward_backward_optimizer_step(version: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_small(version).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    inputs = torch.rand(2, 3, 16, 16, device=device)
    targets = torch.rand_like(inputs)
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


def test_v12_has_no_learnable_fusion_and_parameter_deltas_match() -> None:
    baseline = build_v1(_load("v1")["model"])
    v11, v12 = _build("v11"), _build("v12")
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    assert list(dict(v12.named_children())) == ["backbone", "uicf"]
    assert count(v11.uicf) == count(v12.uicf) == 137_734
    assert count(v11) == count(v12) == 1_115_689
    assert count(v11) - count(baseline) == count(v12) - count(baseline) == 137_734
