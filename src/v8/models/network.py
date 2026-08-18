"""v8 Plain U-Net + pre-GL-INR factory."""

from __future__ import annotations

from torch import nn

from src.shared.pre_inr import build_pre_inr_model


def build_model(config: dict) -> nn.Module:
    return build_pre_inr_model(
        config,
        expected_type="plain_unet_pre_glinr",
        backbone_family="plain_unet",
        inr_kind="gl",
    )
