"""Backbone-agnostic UICF topology wrappers for the controlled experiments."""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from src.v1.models.network import build_model as build_v1_model
from src.v4.models.network import build_model as build_v4_model

from .uicf_inr import UICFINROutput, UnderwaterImplicitCorrectionField


class UICFPreBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, uicf: UnderwaterImplicitCorrectionField) -> None:
        super().__init__()
        self.backbone = backbone
        self.uicf = uicf

    def forward(self, image: Tensor) -> Tensor:
        uicf_output = self.uicf(image, return_details=False)
        if not isinstance(uicf_output, Tensor):
            raise TypeError("UICF tensor forward returned diagnostics unexpectedly")
        return self.backbone(uicf_output)

    def forward_with_uicf_details(self, image: Tensor) -> tuple[Tensor, UICFINROutput]:
        details = self.uicf(image, return_details=True)
        if not isinstance(details, UICFINROutput):
            raise TypeError("UICF diagnostics forward returned a tensor unexpectedly")
        return self.backbone(details.enhanced), details

    def forward_with_shapes(self, image: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        output, details = self.forward_with_uicf_details(image)
        return output, {
            "input": tuple(image.shape),
            "uicf_output": tuple(details.enhanced.shape),
            "correction_field": tuple(details.correction_field.shape),
            "chromatic_anchor": tuple(details.chromatic_anchor.shape),
            "global_feature": tuple(details.global_feature.shape),
            "final_output": tuple(output.shape),
        }


class UICFParallelBranch(nn.Module):
    def __init__(self, backbone: nn.Module, uicf: UnderwaterImplicitCorrectionField) -> None:
        super().__init__()
        self.backbone = backbone
        self.uicf = uicf

    def forward(self, image: Tensor) -> Tensor:
        backbone_output = self.backbone(image)
        uicf_output = self.uicf(image, return_details=False)
        if not isinstance(uicf_output, Tensor):
            raise TypeError("UICF tensor forward returned diagnostics unexpectedly")
        return backbone_output + (uicf_output - image)

    def forward_with_uicf_details(self, image: Tensor) -> tuple[Tensor, UICFINROutput]:
        backbone_output = self.backbone(image)
        details = self.uicf(image, return_details=True)
        if not isinstance(details, UICFINROutput):
            raise TypeError("UICF diagnostics forward returned a tensor unexpectedly")
        return backbone_output + (details.enhanced - image), details

    def forward_with_shapes(self, image: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        backbone_output = self.backbone(image)
        details = self.uicf(image, return_details=True)
        if not isinstance(details, UICFINROutput):
            raise TypeError("UICF diagnostics forward returned a tensor unexpectedly")
        output = backbone_output + (details.enhanced - image)
        return output, {
            "input": tuple(image.shape),
            "backbone_output": tuple(backbone_output.shape),
            "uicf_output": tuple(details.enhanced.shape),
            "uicf_delta": tuple((details.enhanced - image).shape),
            "correction_field": tuple(details.correction_field.shape),
            "chromatic_anchor": tuple(details.chromatic_anchor.shape),
            "global_feature": tuple(details.global_feature.shape),
            "final_output": tuple(output.shape),
        }


Placement = Literal["pre", "parallel"]


def build_uicf_experiment_model(
    config: dict, *, expected_type: str, placement: Placement
) -> nn.Module:
    if config.get("type") != expected_type:
        raise ValueError(f"Expected model.type={expected_type}, got {config.get('type')!r}")
    backbone_config = {
        key: config[key]
        for key in (
            "img_channel",
            "width",
            "enc_blk_nums",
            "middle_blk_num",
            "dec_blk_nums",
        )
    }
    backbone_config["type"] = "nafnet_small"
    # Construct the exact baseline first so equal seeds preserve v1 weights.
    backbone = build_v1_model(backbone_config)
    uicf_config = config["uicf"]
    uicf = UnderwaterImplicitCorrectionField(
        feat_dim=int(uicf_config["feat_dim"]),
        num_frequencies=int(uicf_config["num_frequencies"]),
        mlp_hidden_dim=int(uicf_config["mlp_hidden_dim"]),
        mlp_hidden_layers=int(uicf_config["mlp_hidden_layers"]),
        anchor_hidden_dim=int(uicf_config["anchor_hidden_dim"]),
        query_chunk_size=(
            None
            if uicf_config["query_chunk_size"] is None
            else int(uicf_config["query_chunk_size"])
        ),
    )
    return (
        UICFPreBackbone(backbone, uicf)
        if placement == "pre"
        else UICFParallelBranch(backbone, uicf)
    )


def build_uicf_unet_experiment_model(
    config: dict, *, expected_type: str, placement: Placement
) -> nn.Module:
    """Compose the unchanged v4 Plain U-Net with the canonical UICF.

    The backbone is deliberately constructed first so a fixed seed produces
    exactly the same Plain U-Net state as the standalone v4 builder.
    """
    if config.get("type") != expected_type:
        raise ValueError(f"Expected model.type={expected_type}, got {config.get('type')!r}")
    backbone_config = {
        key: config[key]
        for key in (
            "in_channels",
            "out_channels",
            "base_channels",
            "use_batch_norm",
            "output_activation",
        )
    }
    backbone_config["type"] = "plain_unet"
    # Construct the exact baseline first so equal seeds preserve v4 weights.
    backbone = build_v4_model(backbone_config)
    uicf_config = config["uicf"]
    uicf = UnderwaterImplicitCorrectionField(
        feat_dim=int(uicf_config["feat_dim"]),
        num_frequencies=int(uicf_config["num_frequencies"]),
        mlp_hidden_dim=int(uicf_config["mlp_hidden_dim"]),
        mlp_hidden_layers=int(uicf_config["mlp_hidden_layers"]),
        anchor_hidden_dim=int(uicf_config["anchor_hidden_dim"]),
        query_chunk_size=(
            None
            if uicf_config["query_chunk_size"] is None
            else int(uicf_config["query_chunk_size"])
        ),
    )
    return (
        UICFPreBackbone(backbone, uicf)
        if placement == "pre"
        else UICFParallelBranch(backbone, uicf)
    )
