"""v1 baseline model factory."""

from __future__ import annotations

from torch import nn

from .nafnet import NAFNet


def build_model(config: dict) -> nn.Module:
    if config.get("type") != "nafnet_small":
        raise ValueError(f"Unsupported model type: {config.get('type')!r}")
    return NAFNet(
        img_channel=int(config["img_channel"]),
        width=int(config["width"]),
        enc_blk_nums=tuple(config["enc_blk_nums"]),
        middle_blk_num=int(config["middle_blk_num"]),
        dec_blk_nums=tuple(config["dec_blk_nums"]),
    )
