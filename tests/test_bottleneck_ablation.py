from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from src.v1.models import build_model as build_v1
from src.v2.models import build_model as build_v2
from src.v3.models import build_model as build_v3


ROOT = Path(__file__).resolve().parents[1]
BUILDERS = (("v1", build_v1), ("v2", build_v2), ("v3", build_v3))


def _config(version: str) -> dict:
    path = ROOT / "configs" / f"config_{version}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["model"]


def test_initial_bottlenecks_are_identical_on_formal_feature_shape() -> None:
    torch.manual_seed(3407)
    encoded = torch.randn(2, 256, 32, 32)
    outputs = []
    for version, builder in BUILDERS:
        torch.manual_seed(3520)
        model = builder(_config(version))
        output = model.bottleneck_module(encoded)
        assert output.shape == encoded.shape
        torch.testing.assert_close(output, encoded, rtol=0, atol=0)
        outputs.append(output)

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)


class ZeroBottleneck(nn.Module):
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(encoded)


@pytest.mark.parametrize(("version", "builder"), BUILDERS)
def test_network_uses_bottleneck_output_without_double_residual(version: str, builder) -> None:
    torch.manual_seed(3520)
    model = builder(_config(version)).eval()
    model.bottleneck_module = ZeroBottleneck()
    inputs = torch.randn(1, 3, 17, 19)

    with torch.no_grad():
        padded, height, width = model._pad(inputs)
        encoded, skips = model.encode(padded)
        expected = model.decode(torch.zeros_like(encoded), skips) + padded
        actual = model(inputs)

    torch.testing.assert_close(actual, expected[..., :height, :width], rtol=0, atol=0)
