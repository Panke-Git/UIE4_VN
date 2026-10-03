from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from src.shared.color_query_unet import PlainUNetColorQuery, SpatialTokenGuidance
from src.shared.paper_evaluation import SUPPORTED_PAPER_VERSIONS
from src.shared.uicf_models import UICFPreBackbone
from src.v16.models import build_model as build_v16
from src.v19.models import SDQRPlainUNetColorQuery, _route_queries, build_model as build_v19
from src.v19.engine import train_model
from src.v19.experiment import update_status
from src.v19.losses import CharbonnierLoss
from tools.inspect_v19_sdqr import main as inspect_sdqr


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
SDQR_KEYS = tuple(f"backbone.sdqr_beta{stage}" for stage in (1, 2, 3, 4))


def _load(version: str) -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / f"config_{version}.yaml").read_text(encoding="utf-8")
    )


def _model_config(version: str, *, small: bool = True) -> dict:
    config = copy.deepcopy(_load(version)["model"])
    if small:
        config["base_channels"] = 4
        config["color_query"]["token_dim"] = 16
        config["uicf"] = {
            "feat_dim": 4,
            "num_frequencies": 2,
            "mlp_hidden_dim": 8,
            "mlp_hidden_layers": 1,
            "anchor_hidden_dim": 4,
            "query_chunk_size": 128,
        }
    return config


def _paired_models(*, small: bool = True) -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(3520)
    v16 = build_v16(_model_config("v16", small=small))
    torch.manual_seed(3520)
    v19 = build_v19(_model_config("v19", small=small))
    return v16, v19


def _count(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def test_v19_config_is_a_strict_v16_controlled_extension() -> None:
    v16, v19 = _load("v16"), _load("v19")
    for section in PROTOCOL_SECTIONS:
        assert v19[section] == v16[section], section
    assert v19["experiment"] == {
        "version": "v19",
        "name": "PlainUNet_ColorQuery_UICF_PreBackbone_SDQR",
        "seed": 3520,
        "output_root": "experiments",
    }
    common = copy.deepcopy(v19["model"])
    assert common.pop("sdqr") == {"beta_init": 0.0}
    common["type"] = "plain_unet_color_query_uicf_pre_backbone"
    assert common == v16["model"]
    assert "v19" not in SUPPORTED_PAPER_VERSIONS


def test_v19_protocol_modules_are_byte_identical_to_v16() -> None:
    for filename in (
        "dataset.py",
        "engine.py",
        "experiment.py",
        "losses.py",
        "metrics.py",
        "utils.py",
    ):
        assert (ROOT / "src/v19" / filename).read_bytes() == (
            ROOT / "src/v16" / filename
        ).read_bytes()


def test_v19_builds_expected_wrapper_backbone_and_unchanged_guidance() -> None:
    model = build_v19(_model_config("v19"))
    assert isinstance(model, UICFPreBackbone)
    assert isinstance(model.backbone, SDQRPlainUNetColorQuery)
    assert isinstance(model.backbone, PlainUNetColorQuery)
    for stage in (4, 3, 2, 1):
        assert type(getattr(model.backbone, f"guide{stage}")) is SpatialTokenGuidance


def test_only_four_per_query_zero_initialized_parameters_are_added() -> None:
    v16, v19 = _paired_models()
    state16, state19 = v16.state_dict(), v19.state_dict()
    assert sorted(set(state19) - set(state16)) == list(SDQR_KEYS)
    assert set(state16).issubset(state19)
    for key in SDQR_KEYS:
        beta = state19[key]
        assert beta.shape == (1, 8, 1)
        assert torch.count_nonzero(beta) == 0
    assert _count(v19) - _count(v16) == 32
    assert _count(v19, trainable_only=True) - _count(v16, trainable_only=True) == 32


def test_standard_parameter_counts_are_v16_plus_exactly_32() -> None:
    v16, v19 = _paired_models(small=False)
    assert _count(v16) == _count(v16, trainable_only=True) == 38_878_217
    assert _count(v19) == _count(v19, trainable_only=True) == 38_878_249


def test_same_seed_all_common_state_including_buffers_and_uicf_is_tensor_exact() -> None:
    v16, v19 = _paired_models()
    state16, state19 = v16.state_dict(), v19.state_dict()
    assert set(state19) == set(state16) | set(SDQR_KEYS)
    for key, expected in state16.items():
        assert torch.equal(expected, state19[key]), key
    for key, expected in v16.uicf.state_dict().items():
        assert torch.equal(expected, v19.uicf.state_dict()[key]), f"uicf.{key}"


def test_route_queries_exact_math_and_per_query_channel_broadcasting() -> None:
    previous = torch.tensor(
        [[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]]]
    )
    current = previous + torch.tensor(
        [[[2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [4.0, 8.0, 12.0]]]
    )
    beta_zero = torch.zeros(1, 3, 1)
    assert torch.equal(_route_queries(current, previous, beta_zero), current)

    beta_positive = torch.full((1, 3, 1), torch.atanh(torch.tensor(0.5)))
    torch.testing.assert_close(
        _route_queries(current, previous, beta_positive),
        current + 0.5 * (current - previous),
        rtol=0,
        atol=0,
    )
    beta_negative = torch.full((1, 3, 1), torch.atanh(torch.tensor(-0.5)))
    torch.testing.assert_close(
        _route_queries(current, previous, beta_negative),
        current - 0.5 * (current - previous),
        rtol=0,
        atol=0,
    )

    per_query_eta = torch.tensor([-0.5, 0.0, 0.5]).atanh().reshape(1, 3, 1)
    routed = _route_queries(current, previous, per_query_eta)
    multiplier = (routed - current) / (current - previous)
    expected = torch.tensor([-0.5, 0.0, 0.5]).reshape(1, 3, 1).expand_as(current)
    torch.testing.assert_close(multiplier, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("current_shape", "previous_shape", "beta_shape", "message"),
    (
        ((1, 3, 4, 1), (1, 3, 4, 1), (1, 3, 1), r"\[B,M,C\]"),
        ((1, 3, 4), (1, 2, 4), (1, 3, 1), "must match"),
        ((1, 3, 4), (1, 3, 4), (3, 1), "beta"),
    ),
)
def test_route_queries_rejects_invalid_shapes(
    current_shape: tuple[int, ...],
    previous_shape: tuple[int, ...],
    beta_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _route_queries(
            torch.zeros(current_shape),
            torch.zeros(previous_shape),
            torch.zeros(beta_shape),
        )


def test_route_queries_reports_nonfinite_values_without_masking() -> None:
    current = torch.zeros(1, 2, 3)
    previous = torch.zeros_like(current)
    beta = torch.zeros(1, 2, 1)
    current[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="current"):
        _route_queries(current, previous, beta)
    current.zero_()
    beta[0, 1, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="beta"):
        _route_queries(current, previous, beta)


def test_beta_zero_output_is_tensor_exact_to_v16_and_odd_shape_is_preserved() -> None:
    v16, v19 = _paired_models()
    v16.eval()
    v19.eval()
    inputs = torch.rand(1, 3, 63, 79)
    with torch.inference_mode():
        expected = v16(inputs)
        actual, shapes = v19(inputs), v19.backbone.forward_with_shapes(inputs)[1]
    assert torch.equal(actual, expected)
    assert actual.shape == inputs.shape
    assert torch.isfinite(actual).all()
    for key in ("t4", "t3", "t2", "t1", "rt4", "rt3", "rt2", "rt1"):
        assert shapes[key] == (1, 8, 16)


def test_beta_zero_common_parameter_gradients_are_tensor_exact_and_betas_learn() -> None:
    v16, v19 = _paired_models()
    v16.eval()
    v19.eval()
    inputs = torch.rand(1, 3, 16, 16)
    targets = torch.rand_like(inputs)

    loss16 = (v16(inputs) - targets).square().mean()
    loss19 = (v19(inputs) - targets).square().mean()
    assert torch.equal(loss19, loss16)
    loss16.backward()
    loss19.backward()

    gradients16 = {name: parameter.grad for name, parameter in v16.named_parameters()}
    gradients19 = {name: parameter.grad for name, parameter in v19.named_parameters()}
    assert set(gradients19) == set(gradients16) | set(SDQR_KEYS)
    for name, expected in gradients16.items():
        actual = gradients19[name]
        if expected is None or actual is None:
            assert expected is None and actual is None, name
        else:
            assert torch.equal(actual, expected), name
            assert torch.isfinite(actual).all(), name

    for key in SDQR_KEYS:
        gradient = gradients19[key]
        assert gradient is not None, key
        assert torch.isfinite(gradient).all(), key
        assert bool(torch.count_nonzero(gradient)), key


def test_invalid_sdqr_config_is_rejected() -> None:
    config = _model_config("v19")
    config["sdqr"]["beta_init"] = 0.1
    with pytest.raises(ValueError, match="exactly 0.0"):
        build_v19(config)
    config = _model_config("v19")
    config["sdqr"]["extra"] = 1
    with pytest.raises(ValueError, match="only beta_init"):
        build_v19(config)


def test_one_epoch_v19_engine_checkpoint_and_strict_reload(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(3520)

    def sample(index: int) -> dict[str, torch.Tensor | str]:
        local = torch.Generator().manual_seed(3520 + index)
        return {
            "id": str(index),
            "filename": f"{index}.png",
            "input": torch.rand(3, 16, 16, generator=local),
            "target": torch.rand(3, 16, 16, generator=local),
        }

    train_loader = DataLoader(
        [sample(0), sample(1)], batch_size=2, shuffle=True, generator=generator
    )
    validation_loader = DataLoader([sample(2)], batch_size=1, shuffle=False)
    model_config = _model_config("v19")
    config = {
        "experiment": {"version": "v19", "seed": 3520},
        "training": {
            "epochs": 1,
            "amp": False,
            "validate_every": 1,
            "save_every": 1,
            "gradient_clip_norm": None,
            "fail_on_nonfinite": True,
        },
        "checkpoint": {
            "save_best_psnr": True,
            "save_best_ssim": True,
            "save_best_val_loss": True,
            "save_last": True,
            "save_periodic": True,
        },
        "metrics": {
            "data_range": 1.0,
            "crop_border": 0,
            "ssim_window_size": 11,
            "ssim_sigma": 1.5,
        },
        "logging": {"log_every_steps": 0},
        "model": model_config,
    }
    torch.manual_seed(3520)
    model = build_v19(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    logger = logging.getLogger("v19.sdqr.integration")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    run_dir = tmp_path / "run"
    for directory in ("best", "checkpoint", "log", "result"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    status = update_status(
        run_dir, {"status": "running", "last_epoch": 0}, status="running"
    )
    train_model(
        model,
        train_loader,
        validation_loader,
        CharbonnierLoss(1e-3),
        optimizer,
        None,
        scaler,
        torch.device("cpu"),
        config,
        run_dir,
        logger,
        logger,
        status,
    )

    checkpoint_path = run_dir / "checkpoint/last.pt"
    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["version"] == "v19" and checkpoint["epoch"] == 1
    reloaded = build_v19(model_config)
    reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert (run_dir / "checkpoint/epoch_0001.pt").is_file()
    assert (run_dir / "result/validation_summary.json").is_file()
    history = json.loads((run_dir / "log/metrics_history.json").read_text())
    assert history["epochs"][0]["val_e00"] >= 0.0


def test_inspection_tool_reads_checkpoint_and_writes_optional_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "v19_run"
    (run_dir / "best").mkdir(parents=True)
    config = {
        "experiment": {"version": "v19"},
        "model": _model_config("v19"),
        "test": {"checkpoint": "best_psnr"},
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    model = build_v19(config["model"])
    with torch.no_grad():
        model.backbone.sdqr_beta2[0, 0, 0] = torch.atanh(torch.tensor(0.5))
    checkpoint = {
        "epoch": 17,
        "version": "v19",
        "resolved_config": config,
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, run_dir / "best/best_psnr.pt")
    output_json = tmp_path / "sdqr.json"

    assert inspect_sdqr(
        [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best_psnr",
            "--output-json",
            str(output_json),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "checkpoint epoch: 17" in output
    assert "stage4:" in output and "stage1:" in output
    assert "query0" in output and "all 32 beta finite: True" in output
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["epoch"] == 17
    assert report["all_beta_finite"] is True
    assert report["stages"]["2"]["eta"][0] == pytest.approx(0.5)
