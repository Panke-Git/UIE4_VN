"""V19 factory: V16 topology plus Stage-Differential Query Routing."""

from __future__ import annotations

from torch import nn

from src.shared.uicf_models import UICFPreBackbone, _build_correction_module

from .sdqr_unet import SDQRPlainUNetColorQuery


EXPECTED_TYPE = "plain_unet_color_query_uicf_pre_backbone_sdqr"


def build_model(config: dict) -> nn.Module:
    if config.get("type") != EXPECTED_TYPE:
        raise ValueError(f"Expected model.type={EXPECTED_TYPE}, got {config.get('type')!r}")
    color_query = config.get("color_query")
    if not isinstance(color_query, dict):
        raise ValueError("V19 model config requires a color_query mapping")
    sdqr = config.get("sdqr")
    if not isinstance(sdqr, dict) or set(sdqr) != {"beta_init"}:
        raise ValueError("V19 model.sdqr must contain only beta_init")

    # Keep V16 construction order exact: full backbone, then canonical UICF.
    backbone = SDQRPlainUNetColorQuery(
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
        beta_init=float(sdqr["beta_init"]),
    )
    uicf = _build_correction_module(config["uicf"])
    return UICFPreBackbone(backbone, uicf)
