"""v12 UICF parallel-branch model factory."""

from __future__ import annotations

from torch import nn

from src.shared.uicf_models import build_uicf_experiment_model


def build_model(config: dict) -> nn.Module:
    return build_uicf_experiment_model(
        config, expected_type="nafnet_uicf_parallel_branch", placement="parallel"
    )
