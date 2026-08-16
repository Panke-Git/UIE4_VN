from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import yaml

from src.v4.models import DoubleConv, PlainUNet, build_model
from src.v4.train import parse_args
from src.v4.utils import apply_overrides


ROOT = Path(__file__).resolve().parents[1]


def _load(version: str) -> dict:
    return yaml.safe_load((ROOT / f"configs/config_{version}.yaml").read_text(encoding="utf-8"))


def test_v4_formal_config_and_256_shapes() -> None:
    config = _load("v4")
    assert config["experiment"] == {
        "version": "v4",
        "name": "PlainUNet_CurrentSplit",
        "seed": 3520,
        "output_root": "experiments",
    }
    assert config["model"] == {
        "type": "plain_unet",
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 64,
        "use_batch_norm": True,
        "output_activation": "sigmoid",
    }
    model = build_model(config["model"]).eval()
    assert isinstance(model, PlainUNet)
    inputs = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        output, shapes = model.forward_with_shapes(inputs)
    assert shapes == {
        "input": (1, 3, 256, 256),
        "e1": (1, 64, 256, 256),
        "e2": (1, 128, 128, 128),
        "e3": (1, 256, 64, 64),
        "e4": (1, 512, 32, 32),
        "bottleneck": (1, 1024, 16, 16),
        "decoder_output": (1, 64, 256, 256),
        "final_output": (1, 3, 256, 256),
    }
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert output.min() >= 0 and output.max() <= 1


def test_v4_training_protocol_matches_current_uie4_protocol() -> None:
    v1, v4 = _load("v1"), _load("v4")
    for section in (
        "data", "loss", "optimizer", "scheduler", "training", "checkpoint",
        "evaluation", "metrics", "test", "logging",
    ):
        assert v4[section] == v1[section]


def test_v4_protocol_code_is_copied_but_version_is_import_isolated() -> None:
    for filename in ("dataset.py", "engine.py", "experiment.py", "losses.py", "metrics.py"):
        assert (ROOT / "src/v4" / filename).read_bytes() == (ROOT / "src/v1" / filename).read_bytes()
    for path in (ROOT / "src/v4").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "src.v1" not in source
        assert "src.v2" not in source
        assert "src.v3" not in source


def test_v4_formal_model_contains_only_plain_unet_layers() -> None:
    model = build_model(_load("v4")["model"])
    allowed = {
        PlainUNet,
        DoubleConv,
        nn.Sequential,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.ConvTranspose2d,
        nn.Sigmoid,
    }
    assert all(type(module) in allowed for module in model.modules())


def test_v4_batch_size_cli_override_is_isolated() -> None:
    args = parse_args(["--batch-size", "4"])
    original = _load("v4")
    resolved = apply_overrides(original, batch_size=args.batch_size)
    assert original["data"]["batch_size"] == 16
    assert resolved["data"]["batch_size"] == 4
