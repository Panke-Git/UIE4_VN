"""v14 Plain U-Net + independent UICF correction-branch model factory."""

from __future__ import annotations

from torch import nn

from src.shared.uicf_models import build_uicf_unet_experiment_model


def build_model(config: dict) -> nn.Module:
    return build_uicf_unet_experiment_model(
        config,
        expected_type="plain_unet_uicf_parallel_branch",
        placement="parallel",
    )
