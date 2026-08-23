"""v15 shared Color-Query Plain U-Net model factory."""

from __future__ import annotations

from torch import nn

from src.shared.color_query_unet import PlainUNetColorQuery


def build_model(config: dict) -> nn.Module:
    if config.get("type") != "plain_unet_color_query":
        raise ValueError(
            "v15 requires model.type=plain_unet_color_query, "
            f"got {config.get('type')!r}"
        )
    color_query = config["color_query"]
    return PlainUNetColorQuery(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        base_channels=int(config["base_channels"]),
        use_batch_norm=bool(config["use_batch_norm"]),
        output_activation=str(config["output_activation"]),
        num_color_queries=int(color_query["num_color_queries"]),
        token_dim=int(color_query["token_dim"]),
        num_heads=int(color_query["num_heads"]),
        ffn_expansion=float(color_query["ffn_expansion"]),
        dropout=float(color_query["dropout"]),
    )
