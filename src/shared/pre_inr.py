"""Compose unchanged project backbones and INRs at the RGB input boundary."""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from src.v1.models.nafnet import NAFNet
from src.v2.models.point_inr import PointINR
from src.v3.models.glinr import GLINR
from src.v4.models.unet import PlainUNet


BackboneFamily = Literal["plain_unet", "nafnet"]
INRKind = Literal["point", "gl"]


class PreINRModel(nn.Module):
    """Apply one RGB INR exactly once, then pass its output to one backbone."""

    def __init__(self, pre_inr: nn.Module, backbone: nn.Module) -> None:
        super().__init__()
        self.pre_inr = pre_inr
        self.backbone = backbone

    def forward_pre_inr(self, inputs: Tensor) -> Tensor:
        return self.pre_inr(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.backbone(self.forward_pre_inr(inputs))

    def forward_with_shapes(self, inputs: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        pre_inr_output = self.forward_pre_inr(inputs)
        forward_with_shapes = getattr(self.backbone, "forward_with_shapes", None)
        if forward_with_shapes is None:
            output = self.backbone(pre_inr_output)
            backbone_shapes: dict[str, tuple[int, ...]] = {}
        else:
            output, backbone_shapes = forward_with_shapes(pre_inr_output)
        shapes = {
            "input": tuple(inputs.shape),
            "pre_inr_output": tuple(pre_inr_output.shape),
            **{f"backbone_{key}": value for key, value in backbone_shapes.items()},
            "final_output": tuple(output.shape),
        }
        return output, shapes


def _build_backbone(config: dict, family: BackboneFamily) -> nn.Module:
    if family == "plain_unet":
        if int(config["in_channels"]) != 3 or int(config["out_channels"]) != 3:
            raise ValueError("Pre-INR Plain U-Net experiments require three-channel RGB I/O")
        return PlainUNet(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            base_channels=int(config["base_channels"]),
            use_batch_norm=bool(config["use_batch_norm"]),
            output_activation=str(config["output_activation"]),
        )
    if int(config["img_channel"]) != 3:
        raise ValueError("Pre-INR NAF experiments require three-channel RGB I/O")
    middle_blk_num = int(config["middle_blk_num"])
    if middle_blk_num != 0:
        raise ValueError("Pre-INR NAF controls must preserve v1 middle_blk_num=0")
    return NAFNet(
        img_channel=int(config["img_channel"]),
        width=int(config["width"]),
        enc_blk_nums=tuple(config["enc_blk_nums"]),
        middle_blk_num=middle_blk_num,
        dec_blk_nums=tuple(config["dec_blk_nums"]),
    )


def _build_pre_inr(config: dict, kind: INRKind) -> nn.Module:
    if kind == "point":
        inr = config["point_inr"]
        return PointINR(
            channels=3,
            hidden_dim=int(inr["hidden_dim"]),
            num_frequencies=int(inr["num_frequencies"]),
            depth=int(inr["depth"]),
            include_raw_coordinate=bool(inr["include_raw_coordinate"]),
            query_chunk=int(inr["query_chunk"]),
            residual=bool(inr["residual"]),
        )
    inr = config["glinr"]
    return GLINR(
        channels=3,
        latent_dim=int(inr["latent_dim"]),
        hidden_dim=int(inr["hidden_dim"]),
        latent_stride=int(inr["latent_stride"]),
        global_num_frequencies=int(inr["global_num_frequencies"]),
        local_num_frequencies=int(inr["local_num_frequencies"]),
        include_raw_absolute_coordinate=bool(inr["include_raw_absolute_coordinate"]),
        include_raw_relative_coordinate=bool(inr["include_raw_relative_coordinate"]),
        local_depth=int(inr["local_depth"]),
        global_depth=int(inr["global_depth"]),
        fusion_depth=int(inr["fusion_depth"]),
        query_chunk=int(inr["query_chunk"]),
        residual=bool(inr["residual"]),
    )


def build_pre_inr_model(
    config: dict,
    *,
    expected_type: str,
    backbone_family: BackboneFamily,
    inr_kind: INRKind,
) -> PreINRModel:
    if config.get("type") != expected_type:
        raise ValueError(f"Expected model.type={expected_type}, got {config.get('type')!r}")
    # Building the backbone first preserves the baseline initialization RNG
    # sequence under an equal seed; the additional INR consumes RNG afterward.
    backbone = _build_backbone(config, backbone_family)
    pre_inr = _build_pre_inr(config, inr_kind)
    return PreINRModel(pre_inr=pre_inr, backbone=backbone)
