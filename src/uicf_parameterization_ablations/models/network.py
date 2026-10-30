"""Only the final UICF correction formula changes; all QGRD code is reused."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.shared.uicf_inr import UnderwaterImplicitCorrectionField
from src.v15.models import build_model as build_backbone
from .. import FORMULAS, VARIANTS


@dataclass
class ParameterizationOutput:
    enhanced: Tensor
    actual_correction: Tensor
    global_feature: Tensor
    variant: str
    # A direct RGB residual must never be exposed as a coefficient R(x).
    predicted_residual: Tensor | None = None
    correction_field: Tensor | None = None
    reconstruction_anchor: Tensor | None = None


class GlobalFeatureOnly(nn.Module):
    """Keep the exact global pooling; remove the unused learned anchor parameters."""

    def forward(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        global_feature = F.adaptive_avg_pool2d(encoded, 1).flatten(1)
        return global_feature.new_full((encoded.shape[0], 3), 0.5), global_feature


class ParameterizedUICF(UnderwaterImplicitCorrectionField):
    def __init__(self, variant: str, **kwargs) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        for flag in ("use_spatial_conditioning", "use_global_field_conditioning"):
            if kwargs.get(flag, True) is not True:
                raise ValueError(f"Parameterization controls require {flag}=True")
        if kwargs.get("use_learned_anchor", False) is not False:
            raise ValueError("These three controls do not use an image-adaptive anchor")
        kwargs["use_learned_anchor"] = False
        # Consume exactly the canonical RNG sequence (encoder -> anchor -> field
        # MLP), THEN physically remove the unused anchor MLP. No dummy parameters
        # or replacement layers are retained to claim a false capacity match.
        super().__init__(**kwargs)
        self.removed_anchor_parameters = sum(p.numel() for p in self.chromatic_anchor.parameters())
        self.chromatic_anchor = GlobalFeatureOnly()
        self.variant = variant

    def forward(self, image: Tensor, return_details: bool = False) -> Tensor | ParameterizationOutput:
        if self.variant == "fixed_anchor":
            # Reuse the original canonical use_learned_anchor=False / full_like
            # 0.5 calculation. The feature-only module preserves g exactly.
            canonical = super().forward(image, return_details=True)
            enhanced = canonical.enhanced
            field = canonical.correction_field
            global_feature = canonical.global_feature
            anchor = canonical.chromatic_anchor
            residual = None
        else:
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError("UICF expects a BCHW three-channel RGB tensor")
            batch, _, height, width = image.shape
            encoded = self.encoder(image)
            local = encoded.permute(0, 2, 3, 1).reshape(batch, height * width, self.feat_dim)
            positional = self.spatial_encoding(batch, height, width, device=image.device, dtype=image.dtype)
            _, global_feature = self.chromatic_anchor(encoded)
            global_map = global_feature.unsqueeze(1).expand(-1, height * width, -1)
            prediction = self.field_mlp(torch.cat((local, positional, global_map), dim=-1))
            prediction = prediction.reshape(batch, height, width, 3).permute(0, 3, 1, 2)
            if self.variant == "direct_residual":
                enhanced, residual, field, anchor = image + prediction, prediction, None, None
            else:
                enhanced, residual, field = image + prediction * image, None, prediction
                anchor = image.new_zeros(batch, 3)
        if not return_details:
            return enhanced
        return ParameterizationOutput(
            enhanced=enhanced, actual_correction=enhanced-image,
            global_feature=global_feature, variant=self.variant,
            predicted_residual=residual, correction_field=field, reconstruction_anchor=anchor,
        )


class ParameterizationModel(nn.Module):
    def __init__(self, backbone: nn.Module, uicf: ParameterizedUICF) -> None:
        super().__init__()
        self.backbone, self.uicf = backbone, uicf

    def forward(self, image: Tensor) -> Tensor:
        return self.backbone(self.uicf(image))

    def forward_with_uicf_details(self, image: Tensor) -> tuple[Tensor, ParameterizationOutput]:
        details = self.uicf(image, return_details=True)
        return self.backbone(details.enhanced), details


def build_model(config: dict) -> ParameterizationModel:
    if config.get("type") != "uicf_parameterization_pre_backbone":
        raise ValueError("Expected model.type=uicf_parameterization_pre_backbone")
    variant = config["parameterization"]
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    field_config = dict(config["uicf"])
    if field_config.pop("field_variant", "implicit") != "implicit":
        raise ValueError("Conv. Control is a different experiment")
    backbone_config = {key: config[key] for key in (
        "in_channels", "out_channels", "base_channels", "use_batch_norm", "output_activation", "color_query")}
    backbone_config["type"] = "plain_unet_color_query"
    # The original V16 factory builds its V15 backbone BEFORE its UICF.
    backbone = build_backbone(backbone_config)
    uicf = ParameterizedUICF(variant, **field_config)
    return ParameterizationModel(backbone, uicf)


def parameter_counts(model: ParameterizationModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    removed = model.uicf.removed_anchor_parameters
    return {
        "total": total, "trainable": trainable,
        "forward_participating": total, "output_connected": total, "not_output_connected": 0,
        "full_uicf_total": total+removed, "difference_from_full": -removed,
        "removed_anchor_predictor": removed,
        "method": "Unused anchor MLP physically removed; graph connectivity verified by local tests/capacity tool",
        "formula": FORMULAS[model.uicf.variant],
    }
