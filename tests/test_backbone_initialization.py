from pathlib import Path

import torch
import yaml

from src.v1.models import build_model as build_v1
from src.v2.models import build_model as build_v2
from src.v3.models import build_model as build_v3


ROOT = Path(__file__).resolve().parents[1]
COMMON_PREFIXES = (
    "intro.",
    "encoders.",
    "downs.",
    "middle_blks.",
    "ups.",
    "decoders.",
    "ending.",
)


def _config(version: str) -> dict:
    path = ROOT / "configs" / f"config_{version}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["model"]


def _common_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name.startswith(COMMON_PREFIXES)
    }


def test_same_seed_gives_exactly_identical_common_backbone_initialization() -> None:
    models = []
    for version, builder in (("v1", build_v1), ("v2", build_v2), ("v3", build_v3)):
        torch.manual_seed(3520)
        models.append(_common_state(builder(_config(version))))

    assert models[0].keys() == models[1].keys() == models[2].keys()
    for name in models[0]:
        assert torch.equal(models[0][name], models[1][name]), f"v1/v2 differ at {name}"
        assert torch.equal(models[0][name], models[2][name]), f"v1/v3 differ at {name}"

