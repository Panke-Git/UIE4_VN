"""v9 NAF encoder/decoder + pre-Point-INR factory."""

from __future__ import annotations

from torch import nn

from src.shared.pre_inr import build_pre_inr_model


def build_model(config: dict) -> nn.Module:
    return build_pre_inr_model(
        config,
        expected_type="nafnet_pre_point_inr",
        backbone_family="nafnet",
        inr_kind="point",
    )
