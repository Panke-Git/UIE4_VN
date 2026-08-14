"""v3 GL-INR model factory."""

from __future__ import annotations

from torch import nn

from .glinr import GLINR
from .nafnet import NAFNet


def build_model(config: dict) -> nn.Module:
    if config.get("type") != "nafnet_small":
        raise ValueError(f"Unsupported model type: {config.get('type')!r}")
    middle_blk_num = int(config["middle_blk_num"])
    if middle_blk_num != 0:
        raise ValueError("Phase-one bottleneck replacement requires middle_blk_num=0")
    glinr = config["glinr"]
    model = NAFNet(
        img_channel=int(config["img_channel"]),
        width=int(config["width"]),
        enc_blk_nums=tuple(config["enc_blk_nums"]),
        middle_blk_num=middle_blk_num,
        dec_blk_nums=tuple(config["dec_blk_nums"]),
    )
    model.bottleneck_module = GLINR(
        channels=model.bottleneck_channels,
        latent_dim=int(glinr["latent_dim"]),
        hidden_dim=int(glinr["hidden_dim"]),
        latent_stride=int(glinr["latent_stride"]),
        global_num_frequencies=int(glinr["global_num_frequencies"]),
        local_num_frequencies=int(glinr["local_num_frequencies"]),
        include_raw_absolute_coordinate=bool(glinr["include_raw_absolute_coordinate"]),
        include_raw_relative_coordinate=bool(glinr["include_raw_relative_coordinate"]),
        local_depth=int(glinr["local_depth"]),
        global_depth=int(glinr["global_depth"]),
        fusion_depth=int(glinr["fusion_depth"]),
        query_chunk=int(glinr["query_chunk"]),
        residual=bool(glinr["residual"]),
    )
    return model
