from pathlib import Path

import torch
import yaml

from src.v1.models import build_model as build_v1
from src.v2.models import build_model as build_v2
from src.v3.models import build_model as build_v3


ROOT = Path(__file__).resolve().parents[1]
ENCODER_PREFIXES = (
    "intro.",
    "encoders.",
    "downs.",
)
DECODER_PREFIXES = (
    "ups.",
    "decoders.",
    "ending.",
)


def _config(version: str) -> dict:
    path = ROOT / "configs" / f"config_{version}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["model"]


def _state(model: torch.nn.Module, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name.startswith(prefixes)
    }


def _models_with_same_seed() -> list[torch.nn.Module]:
    models = []
    for version, builder in (("v1", build_v1), ("v2", build_v2), ("v3", build_v3)):
        torch.manual_seed(3520)
        models.append(builder(_config(version)))
    return models


def _assert_identical(states: list[dict[str, torch.Tensor]]) -> None:
    assert states[0].keys() == states[1].keys() == states[2].keys()
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name]), f"v1/v2 differ at {name}"
        assert torch.equal(states[0][name], states[2][name]), f"v1/v3 differ at {name}"


def test_same_seed_gives_exactly_identical_encoder_initialization() -> None:
    _assert_identical([_state(model, ENCODER_PREFIXES) for model in _models_with_same_seed()])


def test_same_seed_gives_exactly_identical_decoder_initialization() -> None:
    _assert_identical([_state(model, DECODER_PREFIXES) for model in _models_with_same_seed()])


def test_phase_one_models_have_no_middle_naf_blocks() -> None:
    models = _models_with_same_seed()
    assert all(model.middle_blk_num == 0 for model in models)
    assert all(len(model.middle_blks) == 0 for model in models)
