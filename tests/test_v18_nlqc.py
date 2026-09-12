from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from src.shared.color_query_unet import PlainUNetColorQuery, SpatialTokenGuidance
from src.shared.uicf_models import UICFPreBackbone
from src.v16.models import build_model as build_v16
from src.v18.models import NLQCSpatialTokenGuidance, build_model as build_v18
from src.v18.models.nlqc_guidance import require_finite_tensor
from tools.diagnose_v18_numerics import main as diagnose_numerics


ROOT = Path(__file__).resolve().parents[1]
UNCHANGED_PROTOCOL_SECTIONS = (
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


def _config(version: str, *, small: bool = True) -> dict:
    path = ROOT / "configs" / f"config_{version}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
    if small:
        config["base_channels"] = 8
        config["color_query"] = dict(config["color_query"], token_dim=32)
        config["uicf"] = {
            "feat_dim": 8,
            "num_frequencies": 2,
            "mlp_hidden_dim": 16,
            "mlp_hidden_layers": 1,
            "anchor_hidden_dim": 8,
            "query_chunk_size": 256,
        }
        if version == "v18":
            config["nlqc"] = dict(config["nlqc"], kernel_chunk_size=7)
    return config


def _count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _paired_models() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(3520)
    v16 = build_v16(_config("v16"))
    torch.manual_seed(3520)
    v18 = build_v18(_config("v18"))
    return v16, v18


def test_v18_config_changes_only_the_controlled_model_fields() -> None:
    v16 = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text(encoding="utf-8"))
    v18 = yaml.safe_load((ROOT / "configs/config_v18.yaml").read_text(encoding="utf-8"))
    for section in UNCHANGED_PROTOCOL_SECTIONS:
        assert v18[section] == v16[section]
    assert v18["experiment"] == {
        "version": "v18",
        "name": "PlainUNet_ColorQuery_UICF_PreBackbone_NLQC",
        "seed": 3520,
        "output_root": "experiments",
    }
    common_model = copy.deepcopy(v18["model"])
    assert common_model.pop("nlqc") == {
        "kernel_lambda": 4.0,
        "epsilon": 1e-6,
        "kernel_chunk_size": 1024,
        "alpha_init": 0.0,
    }
    common_model["type"] = "plain_unet_color_query_uicf_pre_backbone"
    assert common_model == v16["model"]


def test_v18_builds_canonical_wrapper_and_four_nlqc_guides() -> None:
    model = build_v18(_config("v18"))
    assert isinstance(model, UICFPreBackbone)
    assert isinstance(model.backbone, PlainUNetColorQuery)
    for stage in (4, 3, 2, 1):
        guide = getattr(model.backbone, f"guide{stage}")
        assert isinstance(guide, NLQCSpatialTokenGuidance)
        assert guide.kernel_lambda == 4.0
        assert guide.epsilon == 1e-6
        assert guide.kernel_chunk_size == 7


def test_only_four_scalar_parameters_are_added() -> None:
    v16, v18 = _paired_models()
    state16, state18 = v16.state_dict(), v18.state_dict()
    extras = sorted(set(state18) - set(state16))
    assert extras == [
        "backbone.guide1.nlqc_alpha",
        "backbone.guide2.nlqc_alpha",
        "backbone.guide3.nlqc_alpha",
        "backbone.guide4.nlqc_alpha",
    ]
    assert all(state18[key].shape == torch.Size([]) for key in extras)
    assert _count(v18) - _count(v16) == 4


def test_alpha_initializes_to_zero_at_every_decoder_stage() -> None:
    model = build_v18(_config("v18"))
    for stage in (4, 3, 2, 1):
        alpha = getattr(model.backbone, f"guide{stage}").nlqc_alpha
        assert alpha.requires_grad
        assert alpha.shape == torch.Size([])
        assert float(alpha.detach()) == 0.0


def test_wrapper_reuses_exact_guidance_modules_without_new_qkv_projections() -> None:
    base = SpatialTokenGuidance(
        spatial_channels=16, token_dim=32, num_heads=4, dropout=0.0
    )
    query_projection = base.query_projection
    attention = base.attention
    output_projection = base.output_projection
    wrapped = NLQCSpatialTokenGuidance(base, kernel_chunk_size=3)
    assert wrapped.query_projection is query_projection
    assert wrapped.attention is attention
    assert wrapped.output_projection is output_projection
    assert not hasattr(wrapped, "q_proj")
    assert not hasattr(wrapped, "k_proj")
    assert not hasattr(wrapped, "v_proj")
    assert [name for name, _ in wrapped.named_parameters() if name == "nlqc_alpha"] == [
        "nlqc_alpha"
    ]


def test_same_seed_common_v16_v18_state_is_tensor_exact() -> None:
    v16, v18 = _paired_models()
    state16, state18 = v16.state_dict(), v18.state_dict()
    assert set(state16).issubset(state18)
    for key, expected in state16.items():
        assert torch.equal(expected, state18[key]), key


def test_alpha_zero_v18_output_is_tensor_exact_to_v16() -> None:
    v16, v18 = _paired_models()
    v16.eval()
    v18.eval()
    inputs = torch.rand(1, 3, 32, 40)
    with torch.inference_mode():
        expected = v16(inputs)
        actual = v18(inputs)
    assert torch.equal(actual, expected)


def test_odd_resolution_forward_shape_and_finite_output() -> None:
    model = build_v18(_config("v18")).eval()
    inputs = torch.rand(1, 3, 63, 79)
    with torch.inference_mode():
        output = model(inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()


def test_normalized_response_shapes_statistics_finiteness_and_chunk_equivalence() -> None:
    torch.manual_seed(3520)
    guide = NLQCSpatialTokenGuidance(
        SpatialTokenGuidance(16, 32, 4, 0.0),
        kernel_lambda=4.0,
        epsilon=1e-6,
        kernel_chunk_size=3,
    )
    spatial_feature = torch.rand(2, 16, 5, 7)
    tokens = torch.rand(2, 8, 32)
    spatial_queries = guide.query_projection(spatial_feature).flatten(2).transpose(1, 2)
    chunked = guide.normalized_laplacian_response(spatial_queries, tokens)
    assert chunked.shape == (2, 4, 35, 8)
    assert chunked.dtype == torch.float32
    assert torch.isfinite(chunked).all()
    torch.testing.assert_close(
        chunked.mean(dim=2), torch.zeros(2, 4, 8), atol=2e-5, rtol=0
    )
    guide.kernel_chunk_size = 10_000
    unchunked = guide.normalized_laplacian_response(spatial_queries, tokens)
    torch.testing.assert_close(chunked, unchunked, atol=0, rtol=0)


def test_nlqc_is_passed_as_floating_additive_attention_mask(monkeypatch) -> None:
    guide = NLQCSpatialTokenGuidance(
        SpatialTokenGuidance(16, 32, 4, 0.0), kernel_chunk_size=3, alpha_init=0.25
    )
    captured: dict[str, object] = {}
    original_forward = guide.attention.forward

    def capture(*args, **kwargs):
        captured["mask"] = kwargs.get("attn_mask")
        captured["need_weights"] = kwargs.get("need_weights")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(guide.attention, "forward", capture)
    output = guide(torch.rand(2, 16, 5, 7), torch.rand(2, 8, 32))
    mask = captured["mask"]
    assert output.shape == (2, 16, 5, 7)
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (2 * 4, 5 * 7, 8)
    assert mask.is_floating_point() and torch.isfinite(mask).all()
    assert captured["need_weights"] is False


def test_normalized_laplacian_branch_keeps_direct_qk_gradients() -> None:
    torch.manual_seed(3520)
    guide = NLQCSpatialTokenGuidance(
        SpatialTokenGuidance(16, 32, 4, 0.0), kernel_chunk_size=3
    )
    spatial_feature = torch.rand(2, 16, 5, 7)
    tokens = torch.rand(2, 8, 32)
    spatial_queries = guide.query_projection(spatial_feature).flatten(2).transpose(1, 2)
    normalized = guide.normalized_laplacian_response(spatial_queries, tokens)
    assert normalized.requires_grad
    normalized.square().mean().backward()
    gradient = guide.attention.in_proj_weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert bool(torch.count_nonzero(gradient[: guide.token_dim]))
    assert bool(torch.count_nonzero(gradient[guide.token_dim : 2 * guide.token_dim]))
    assert torch.count_nonzero(gradient[2 * guide.token_dim :]) == 0


def test_backward_reaches_all_alphas_and_existing_qk_projection() -> None:
    torch.manual_seed(3520)
    model = build_v18(_config("v18")).train()
    output = model(torch.rand(1, 3, 32, 32))
    loss = output.mean()
    loss.backward()
    assert torch.isfinite(loss)
    alpha_gradients = []
    for stage in (4, 3, 2, 1):
        guide = getattr(model.backbone, f"guide{stage}")
        alpha_gradient = guide.nlqc_alpha.grad
        qk_gradient = guide.attention.in_proj_weight.grad
        assert alpha_gradient is not None and torch.isfinite(alpha_gradient)
        assert qk_gradient is not None and torch.isfinite(qk_gradient).all()
        assert bool(torch.count_nonzero(qk_gradient[: 2 * guide.token_dim]))
        alpha_gradients.append(alpha_gradient)
    assert all(bool(torch.count_nonzero(gradient)) for gradient in alpha_gradients)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"kernel_lambda": 0.0}, "kernel_lambda"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"kernel_chunk_size": 0}, "kernel_chunk_size"),
        ({"kernel_chunk_size": 1.5}, "kernel_chunk_size"),
        ({"alpha_init": float("nan")}, "alpha_init"),
    ),
)
def test_invalid_nlqc_configuration_fails_clearly(overrides: dict, message: str) -> None:
    config = _config("v18")
    config["nlqc"].update(overrides)
    with pytest.raises(ValueError, match=message):
        build_v18(config)


def test_finite_diagnostic_reports_nan_inf_and_finite_statistics() -> None:
    injected = torch.tensor([float("nan"), float("inf"), -float("inf"), -3.0, 1.0])
    with pytest.raises(FloatingPointError) as caught:
        require_finite_tensor("injected_tensor", injected, context="unit-test")
    message = str(caught.value)
    for expected in (
        "tensor_name=injected_tensor",
        "shape=(5,)",
        "dtype=torch.float32",
        "device=cpu",
        "num_nan=1",
        "num_posinf=1",
        "num_neginf=1",
        "finite_min=-3",
        "finite_max=1",
        "finite_abs_max=3",
        "finite_mean=-1",
        "finite_std=2",
        "context=unit-test",
    ):
        assert expected in message


def test_finite_diagnostic_handles_no_finite_values_safely() -> None:
    injected = torch.tensor([float("nan"), float("inf"), -float("inf")])
    with pytest.raises(FloatingPointError) as caught:
        require_finite_tensor("all_nonfinite", injected, context="unit-test")
    message = str(caught.value)
    assert "finite_count=0" in message
    for field in (
        "finite_min",
        "finite_max",
        "finite_abs_max",
        "finite_mean",
        "finite_std",
    ):
        assert f"{field}=N/A" in message


def test_runtime_diagnostic_distinguishes_upstream_and_token_failures() -> None:
    guide = NLQCSpatialTokenGuidance(SpatialTokenGuidance(16, 32, 4, 0.0))
    spatial_feature = torch.rand(1, 16, 4, 4)
    tokens = torch.rand(1, 8, 32)

    bad_spatial = spatial_feature.clone()
    bad_spatial[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="tensor_name=spatial_feature"):
        guide(bad_spatial, tokens)

    bad_tokens = tokens.clone()
    bad_tokens[0, 0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="tensor_name=tokens"):
        guide(spatial_feature, bad_tokens)


def test_runtime_diagnostic_identifies_amp_boundary_spatial_queries(monkeypatch) -> None:
    guide = NLQCSpatialTokenGuidance(SpatialTokenGuidance(16, 32, 4, 0.0))
    spatial_feature = torch.rand(1, 16, 4, 4)
    tokens = torch.rand(1, 8, 32)

    def nonfinite_projection(value: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (value.shape[0], 32, value.shape[2], value.shape[3]),
            float("inf"),
            dtype=value.dtype,
            device=value.device,
        )

    monkeypatch.setattr(guide.query_projection, "forward", nonfinite_projection)
    with pytest.raises(FloatingPointError) as caught:
        guide(spatial_feature, tokens)
    message = str(caught.value)
    assert "tensor_name=spatial_queries" in message
    assert "spatial_feature_dtype=torch.float32" in message
    assert "spatial_queries_dtype=torch.float32" in message
    assert "tokens_dtype=torch.float32" in message


def test_runtime_diagnostic_identifies_packed_q_projection_overflow() -> None:
    guide = NLQCSpatialTokenGuidance(SpatialTokenGuidance(16, 32, 4, 0.0))
    with torch.no_grad():
        guide.attention.in_proj_weight[:32].fill_(3.0e38)
    spatial_queries = torch.full((1, 5, 32), 2.0)
    tokens = torch.zeros(1, 8, 32)
    with pytest.raises(FloatingPointError) as caught:
        guide._project_qk_fp32(spatial_queries, tokens)
    message = str(caught.value)
    assert "tensor_name=projected_q" in message
    assert "num_posinf=" in message


def test_checkpoint_numerics_tool_reports_model_and_optimizer_nonfinite_state(
    tmp_path: Path, capsys
) -> None:
    run_dir = tmp_path / "v18_run"
    (run_dir / "checkpoint").mkdir(parents=True)
    config = {
        "experiment": {"version": "v18"},
        "model": _config("v18"),
        "test": {"checkpoint": "last"},
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    model = build_v18(config["model"])
    model_state = model.state_dict()
    model_state["backbone.guide3.nlqc_alpha"] = torch.tensor(float("nan"))
    checkpoint = {
        "epoch": 193,
        "version": "v18",
        "model_state_dict": model_state,
        "optimizer_state_dict": {
            "state": {0: {"exp_avg": torch.tensor([float("inf")])}},
            "param_groups": [],
        },
    }
    torch.save(checkpoint, run_dir / "checkpoint" / "last.pt")

    diagnose_numerics(["--run-dir", str(run_dir), "--checkpoint", "last"])
    output = capsys.readouterr().out
    assert "checkpoint epoch: 193" in output
    assert "model_state_dict: status=NON_FINITE" in output
    assert "optimizer_state_dict: status=NON_FINITE" in output
    assert "model_state_dict.backbone.guide3.nlqc_alpha" in output
    assert "optimizer_state_dict.state.0.exp_avg" in output
    assert "strict model_state_dict load: passed" in output
    assert "model parameter/floating-buffer max_abs Top-30" in output
    assert "BatchNorm modules:" in output
    assert "overall checkpoint numerical status: NON_FINITE" in output
