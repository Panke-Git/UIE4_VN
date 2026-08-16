"""v4 Plain U-Net model factory."""

from __future__ import annotations

from torch import nn

from .unet import PlainUNet


def build_model(config: dict) -> nn.Module:
    if config.get("type") != "plain_unet":
        raise ValueError(f"v4 requires model.type=plain_unet, got {config.get('type')!r}")
    return PlainUNet(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        base_channels=int(config["base_channels"]),
        use_batch_norm=bool(config["use_batch_norm"]),
        output_activation=str(config["output_activation"]),
    )
