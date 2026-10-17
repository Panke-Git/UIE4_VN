"""Only invoke canonical V16/V15 factories; no new network or architectural overrides."""
from copy import deepcopy
from pathlib import Path

import torch
import yaml

CANONICAL = Path(__file__).resolve().parents[2] / "configs/config_v16.yaml"
BACKBONE_KEYS = ("in_channels", "out_channels", "base_channels", "use_batch_norm", "output_activation", "color_query")


def canonical_model_config():
    return yaml.safe_load(CANONICAL.read_text())["model"]


def backbone_config(full):
    return {**{key: deepcopy(full[key]) for key in BACKBONE_KEYS}, "type": "plain_unet_color_query"}


def build_model(variant, model_config):
    if model_config != canonical_model_config():
        raise ValueError("Haze4K model config must exactly equal configs/config_v16.yaml:model")
    if variant == "full":
        from src.v16.models.network import build_model as factory
        return factory(deepcopy(model_config))
    if variant == "wo_uicf":
        from src.v15.models.network import build_model as factory
        return factory(backbone_config(model_config))
    raise ValueError(f"Unknown variant: {variant}")


def assert_initial_backbone_equivalence(seed=3520):
    """Explicit diagnostic, not called during training (would consume extra RNG)."""
    torch.manual_seed(seed)
    full = build_model("full", canonical_model_config())
    torch.manual_seed(seed)
    wo = build_model("wo_uicf", canonical_model_config())
    a, b = full.backbone.state_dict(), wo.state_dict()
    if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
        raise AssertionError("Initial V16 backbone and independently constructed V15 differ")
    return full, wo
