from pathlib import Path

import torch
import yaml

from src.v3.models import build_model


ROOT = Path(__file__).resolve().parents[1]


def test_v3_forward_backward_and_padding() -> None:
    config = yaml.safe_load((ROOT / "configs/config_v3.yaml").read_text())["model"]
    model = build_model(config)
    inputs = torch.randn(1, 3, 17, 19, requires_grad=True)
    output = model(inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    output.mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()

