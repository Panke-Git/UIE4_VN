from __future__ import annotations

import gc
import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from src.shared.color_query_unet import (
    ColorTokenRefinementBlock,
    PlainUNetColorQuery,
    ResidualLocalRefine,
    SpatialTokenGuidance,
)
from src.shared.uicf_inr import UnderwaterImplicitCorrectionField
from src.shared.uicf_models import UICFParallelBranch, UICFPreBackbone
from src.v4.models import DoubleConv, PlainUNet, build_model as build_v4


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SECTIONS = (
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
PLAIN_UNET_KEYS = (
    "in_channels",
    "out_channels",
    "base_channels",
    "use_batch_norm",
    "output_activation",
)
COLOR_QUERY_CONFIG = {
    "num_color_queries": 8,
    "token_dim": 128,
    "num_heads": 4,
    "ffn_expansion": 2.0,
    "dropout": 0.0,
}
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


def _model_config(
    version: str,
    *,
    base_channels: int | None = None,
    token_dim: int | None = None,
) -> dict:
    config = dict(_load(version)["model"])
    if "color_query" in config:
        config["color_query"] = dict(config["color_query"])
        if token_dim is not None:
            config["color_query"]["token_dim"] = token_dim
    if "uicf" in config:
        config["uicf"] = dict(config["uicf"])
    if base_channels is not None:
        config["base_channels"] = base_channels
    return config


def _build(
    version: str,
    *,
    base_channels: int | None = None,
    token_dim: int | None = None,
) -> nn.Module:
    return importlib.import_module(f"src.{version}.models").build_model(
        _model_config(version, base_channels=base_channels, token_dim=token_dim)
    )


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _has_nonzero_gradient(parameters) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in parameters
    )


def _assert_module_has_nonzero_gradient(module: nn.Module) -> None:
    assert _has_nonzero_gradient(module.parameters()), type(module).__name__


def test_v15_protocol_and_plain_unet_config_match_v4() -> None:
    v4, v15 = _load("v4"), _load("v15")
    for section in PROTOCOL_SECTIONS:
        assert v15[section] == v4[section]
    for key in PLAIN_UNET_KEYS:
        assert v15["model"][key] == v4["model"][key]
    assert v15["experiment"] == {
        "version": "v15",
        "name": "PlainUNet_ColorQuery",
        "seed": 3520,
        "output_root": "experiments",
    }
    assert v15["model"]["type"] == "plain_unet_color_query"
    assert v15["model"]["color_query"] == COLOR_QUERY_CONFIG


def test_v16_v17_protocol_uicf_and_color_query_configs_are_fair() -> None:
    v13, v14, v15, v16, v17 = (
        _load(version) for version in ("v13", "v14", "v15", "v16", "v17")
    )
    for section in PROTOCOL_SECTIONS:
        assert v16[section] == v13[section]
        assert v17[section] == v14[section]
    for key in PLAIN_UNET_KEYS:
        assert v16["model"][key] == v17["model"][key] == v15["model"][key]
    assert v16["model"]["color_query"] == v17["model"]["color_query"] == COLOR_QUERY_CONFIG
    assert (
        v16["model"]["uicf"]
        == v17["model"]["uicf"]
        == v13["model"]["uicf"]
        == v14["model"]["uicf"]
        == UICF_CONFIG
    )


def test_v15_v17_reuse_stable_pipelines_and_one_shared_cq_implementation() -> None:
    for version, reference in (("v15", "v4"), ("v16", "v13"), ("v17", "v14")):
        for filename in ("dataset.py", "engine.py", "experiment.py", "losses.py", "metrics.py", "utils.py"):
            assert (ROOT / "src" / version / filename).read_bytes() == (
                ROOT / "src" / reference / filename
            ).read_bytes()
        assert not (ROOT / "src" / version / "models" / "color_query_unet.py").exists()
    implementations = [
        importlib.import_module(f"src.{version}.models").PlainUNetColorQuery
        for version in ("v15", "v16", "v17")
    ]
    assert implementations[0] is implementations[1] is implementations[2] is PlainUNetColorQuery


def test_same_seed_v4_v15_common_modules_are_tensor_exact() -> None:
    torch.manual_seed(3520)
    baseline = build_v4(_load("v4")["model"])
    torch.manual_seed(3520)
    color_query = _build("v15")
    assert isinstance(color_query, PlainUNetColorQuery)
    for module_name in (
        "encoder1",
        "encoder2",
        "encoder3",
        "encoder4",
        "bottleneck",
        "upconv4",
        "decoder4",
        "upconv3",
        "decoder3",
        "upconv2",
        "decoder2",
        "upconv1",
        "decoder1",
        "output_conv",
    ):
        expected = getattr(baseline, module_name).state_dict()
        actual = getattr(color_query, module_name).state_dict()
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.equal(expected[key], actual[key]), f"{module_name}.{key}"


def test_color_query_modules_have_the_prescribed_structure_and_validation() -> None:
    model = _build("v15", base_channels=8, token_dim=32)
    assert isinstance(model, PlainUNet)
    assert isinstance(model, PlainUNetColorQuery)
    assert model.base_queries.shape == (1, 8, 32)
    assert isinstance(model.upconv4, nn.ConvTranspose2d)
    assert isinstance(model.upconv3, nn.ConvTranspose2d)
    assert isinstance(model.upconv2, nn.ConvTranspose2d)
    assert isinstance(model.upconv1, nn.ConvTranspose2d)
    for stage, channels in (("1", 8), ("2", 16), ("3", 32), ("4", 64)):
        projection = model.feature_to_token[stage]
        assert projection.in_channels == channels and projection.out_channels == 32
        assert projection.kernel_size == (1, 1)
        block = model.token_refinement[stage]
        assert isinstance(block, ColorTokenRefinementBlock)
        assert block.cross_attn.batch_first and block.self_attn.batch_first
        assert block.cross_attn.num_heads == block.self_attn.num_heads == 4
        linear_layers = [layer for layer in block.ffn if isinstance(layer, nn.Linear)]
        assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
            (32, 64),
            (64, 32),
        ]
    for stage, channels in ((4, 64), (3, 32), (2, 16), (1, 8)):
        guide = getattr(model, f"guide{stage}")
        refine = getattr(model, f"refine{stage}")
        assert isinstance(guide, SpatialTokenGuidance)
        assert guide.spatial_channels == channels and guide.token_dim == 32
        assert isinstance(refine, ResidualLocalRefine)
        assert isinstance(refine.refine, DoubleConv)
    with pytest.raises(ValueError, match="divisible"):
        PlainUNetColorQuery(base_channels=4, token_dim=30, num_heads=4)


def test_odd_resolution_token_decoder_shapes_and_sigmoid_output() -> None:
    model = _build("v15", base_channels=8, token_dim=32).eval()
    inputs = torch.rand(1, 3, 63, 79)
    with torch.inference_mode():
        output, shapes = model.forward_with_shapes(inputs)
    assert output.shape == inputs.shape
    for key in ("t0", "t4", "t3", "t2", "t1"):
        assert shapes[key] == (1, 8, 32)
    assert shapes["e1"] == (1, 8, 64, 80)
    assert shapes["e2"] == (1, 16, 32, 40)
    assert shapes["e3"] == (1, 32, 16, 20)
    assert shapes["e4"] == (1, 64, 8, 10)
    assert shapes["bottleneck"] == (1, 128, 4, 5)
    assert shapes["d4"] == (1, 64, 8, 10)
    assert shapes["d3"] == (1, 32, 16, 20)
    assert shapes["d2"] == (1, 16, 32, 40)
    assert shapes["d1"] == shapes["decoder_output"] == (1, 8, 64, 80)
    assert shapes["final_output"] == tuple(inputs.shape)
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0


def test_deep_to_shallow_refinement_and_matching_decoder_guidance() -> None:
    model = _build("v15", base_channels=4, token_dim=32).eval()
    execution_order: list[str] = []
    refined_tokens: dict[str, torch.Tensor] = {}
    guided_tokens: dict[str, torch.Tensor] = {}
    handles = []

    def record_projection(stage: str):
        def hook(_module, _inputs, _output) -> None:
            execution_order.append(f"project{stage}")

        return hook

    def record_refinement(stage: str):
        def hook(_module, _inputs, output) -> None:
            execution_order.append(f"token{stage}")
            refined_tokens[stage] = output

        return hook

    def record_guidance(stage: str):
        def hook(_module, inputs) -> None:
            execution_order.append(f"guide{stage}")
            guided_tokens[stage] = inputs[1]

        return hook

    for stage in ("4", "3", "2", "1"):
        handles.append(
            model.feature_to_token[stage].register_forward_hook(record_projection(stage))
        )
        handles.append(
            model.token_refinement[stage].register_forward_hook(record_refinement(stage))
        )
        guide = getattr(model, f"guide{stage}")
        handles.append(
            guide.register_forward_pre_hook(record_guidance(stage))
        )
    with torch.inference_mode():
        model(torch.rand(1, 3, 32, 32))
    for handle in handles:
        handle.remove()
    assert execution_order == [
        "project4",
        "token4",
        "project3",
        "token3",
        "project2",
        "token2",
        "project1",
        "token1",
        "guide4",
        "guide3",
        "guide2",
        "guide1",
    ]
    for stage in ("4", "3", "2", "1"):
        assert guided_tokens[stage] is refined_tokens[stage]


def test_all_attention_calls_disable_weight_materialization(monkeypatch) -> None:
    model = _build("v15", base_channels=4, token_dim=32).eval()
    need_weights_values: list[object] = []
    for module in model.modules():
        if not isinstance(module, nn.MultiheadAttention):
            continue
        original_forward = module.forward

        def wrapped_forward(*args, _original=original_forward, **kwargs):
            need_weights_values.append(kwargs.get("need_weights"))
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, "forward", wrapped_forward)
    with torch.inference_mode():
        model(torch.rand(1, 3, 32, 32))
    assert len(need_weights_values) == 12
    assert all(value is False for value in need_weights_values)


def test_backward_reaches_every_color_query_and_original_unet_family() -> None:
    torch.manual_seed(3520)
    model = _build("v15", base_channels=4, token_dim=32).train()
    inputs = torch.rand(1, 3, 32, 32)
    targets = torch.rand_like(inputs)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.base_queries.grad is not None
    assert bool(torch.count_nonzero(model.base_queries.grad))
    for stage in ("4", "3", "2", "1"):
        _assert_module_has_nonzero_gradient(model.feature_to_token[stage])
        block = model.token_refinement[stage]
        _assert_module_has_nonzero_gradient(block.cross_attn)
        _assert_module_has_nonzero_gradient(block.self_attn)
        _assert_module_has_nonzero_gradient(block.ffn)
        _assert_module_has_nonzero_gradient(getattr(model, f"guide{stage}"))
        _assert_module_has_nonzero_gradient(getattr(model, f"refine{stage}"))
    for module in (
        model.encoder1,
        model.encoder4,
        model.bottleneck,
        model.decoder4,
        model.decoder1,
        model.output_conv,
    ):
        _assert_module_has_nonzero_gradient(module)


@pytest.mark.parametrize(
    ("version", "wrapper_type"),
    (("v16", UICFPreBackbone), ("v17", UICFParallelBranch)),
)
def test_v16_v17_wrapper_types_and_independent_modules(
    version: str, wrapper_type: type[nn.Module]
) -> None:
    model = _build(version, base_channels=4, token_dim=32)
    assert isinstance(model, wrapper_type)
    assert isinstance(model.backbone, PlainUNetColorQuery)
    assert type(model.uicf) is UnderwaterImplicitCorrectionField
    assert list(dict(model.named_children())) == ["backbone", "uicf"]
    assert not hasattr(model.backbone, "uicf")
    assert model.backbone.base_queries.shape == (1, 8, 32)


@pytest.mark.parametrize("version", ("v16", "v17"))
def test_same_seed_v15_wrapper_backbone_and_initial_outputs_are_exact(version: str) -> None:
    torch.manual_seed(3520)
    baseline = _build("v15").eval()
    torch.manual_seed(3520)
    model = _build(version).eval()
    baseline_state = baseline.state_dict()
    backbone_state = model.backbone.state_dict()
    assert baseline_state.keys() == backbone_state.keys()
    for key in baseline_state:
        assert torch.equal(baseline_state[key], backbone_state[key]), key

    inputs = torch.rand(1, 3, 16, 16)
    with torch.inference_mode():
        expected = baseline(inputs)
        actual, details = model.forward_with_uicf_details(inputs)
    assert torch.count_nonzero(details.correction_field) == 0
    torch.testing.assert_close(details.enhanced, inputs, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v15_v17_parameter_relations_are_exact() -> None:
    counts: dict[str, int] = {}
    for version in ("v15", "v16", "v17"):
        model = _build(version)
        counts[version] = _count(model)
        if version != "v15":
            assert _count(model.uicf) == 137_734
        del model
        gc.collect()
    assert counts == {
        "v15": 38_740_483,
        "v16": 38_878_217,
        "v17": 38_878_217,
    }
    assert counts["v16"] - counts["v15"] == 137_734
    assert counts["v17"] - counts["v15"] == 137_734
