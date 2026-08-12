"""v2 Point-INR model factory."""

from __future__ import annotations

from torch import nn

from .nafnet import NAFNet
from .point_inr import PointINR


def build_model(config: dict) -> nn.Module:
    if config.get("type") != "nafnet_small":
        raise ValueError(f"Unsupported model type: {config.get('type')!r}")
    inr = config["point_inr"]
    model = NAFNet(
        img_channel=int(config["img_channel"]),
        width=int(config["width"]),
        enc_blk_nums=tuple(config["enc_blk_nums"]),
        middle_blk_num=int(config["middle_blk_num"]),
        dec_blk_nums=tuple(config["dec_blk_nums"]),
    )
    model.bottleneck_module = PointINR(
        channels=model.bottleneck_channels,
        hidden_dim=int(inr["hidden_dim"]),
        num_frequencies=int(inr["num_frequencies"]),
        depth=int(inr["depth"]),
        include_raw_coordinate=bool(inr["include_raw_coordinate"]),
        query_chunk=int(inr["query_chunk"]),
        residual=bool(inr["residual"]),
    )
    return model
