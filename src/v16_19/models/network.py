"""V16 architecture assembled exclusively from this version's frozen modules."""
from torch import nn

from .color_query_unet import PlainUNetColorQuery
from .topology import UICFPreBackbone
from .uicf_inr import UnderwaterImplicitCorrectionField


def build_model(config: dict) -> nn.Module:
    if config.get("type") != "plain_unet_color_query_uicf_pre_backbone":
        raise ValueError("V16-19 requires the original V16 model type")
    cq = config["color_query"]
    # Preserve V16 construction order, initialization, and state_dict keys.
    backbone = PlainUNetColorQuery(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        base_channels=int(config["base_channels"]),
        use_batch_norm=bool(config["use_batch_norm"]),
        output_activation=str(config["output_activation"]),
        num_color_queries=int(cq["num_color_queries"]),
        token_dim=int(cq["token_dim"]),
        num_heads=int(cq["num_heads"]),
        ffn_expansion=float(cq["ffn_expansion"]),
        dropout=float(cq["dropout"]),
    )
    field = config["uicf"]
    if field.get("field_variant", "implicit") not in {"implicit", "implicit_inr"}:
        raise ValueError("V16-19 snapshots the implicit V16 field, not a field ablation")
    uicf = UnderwaterImplicitCorrectionField(
        feat_dim=int(field["feat_dim"]),
        num_frequencies=int(field["num_frequencies"]),
        mlp_hidden_dim=int(field["mlp_hidden_dim"]),
        mlp_hidden_layers=int(field["mlp_hidden_layers"]),
        anchor_hidden_dim=int(field["anchor_hidden_dim"]),
        query_chunk_size=None if field["query_chunk_size"] is None else int(field["query_chunk_size"]),
        use_global_field_conditioning=bool(field.get("use_global_field_conditioning", True)),
        use_learned_anchor=bool(field.get("use_learned_anchor", True)),
        use_spatial_conditioning=bool(field.get("use_spatial_conditioning", True)),
    )
    return UICFPreBackbone(backbone, uicf)
