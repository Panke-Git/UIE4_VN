"""v11 UICF pre-backbone model factory."""

from __future__ import annotations

from torch import nn

from src.shared.uicf_models import build_uicf_experiment_model


def build_model(config: dict) -> nn.Module:
    return build_uicf_experiment_model(
        config, expected_type="nafnet_uicf_pre_backbone", placement="pre"
    )
