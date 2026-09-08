"""Parameter-matched convolutional control for UICF representation ablations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .uicf_inr import GlobalChromaticAnchor, ImageEncoder, UICFINROutput


class ConvolutionalFieldPredictor(nn.Module):
    """Full-resolution local convolutional predictor with an identity start."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 33,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or hidden_layers < 1:
            raise ValueError("Convolutional field dimensions must be positive")
        layers: list[nn.Module] = []
        channels = input_dim
        for _ in range(hidden_layers):
            layers.extend(
                (nn.Conv2d(channels, hidden_dim, 3, 1, 1), nn.GELU())
            )
            channels = hidden_dim
        layers.append(nn.Conv2d(channels, 3, 3, 1, 1))
        self.net = nn.Sequential(*layers)
        output_layer = self.net[-1]
        if not isinstance(output_layer, nn.Conv2d):
            raise TypeError("Convolutional correction predictor must end with Conv2d")
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class ConvolutionalCorrectionField(nn.Module):
    """UICF control that replaces only the implicit coordinate MLP predictor.

    The image encoder, global chromatic anchor, output interface and exact
    reconstruction equation are shared with the canonical UICF-INR.
    """

    def __init__(
        self,
        feat_dim: int = 48,
        anchor_hidden_dim: int = 64,
        conv_hidden_dim: int = 33,
        conv_hidden_layers: int = 3,
        use_global_field_conditioning: bool = True,
        use_learned_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.field_variant = "conv_control"
        self.use_spatial_conditioning = False
        self.use_global_field_conditioning = bool(use_global_field_conditioning)
        self.use_learned_anchor = bool(use_learned_anchor)
        self.encoder = ImageEncoder(feat_dim)
        self.chromatic_anchor = GlobalChromaticAnchor(feat_dim, anchor_hidden_dim)
        self.field_predictor = ConvolutionalFieldPredictor(
            input_dim=feat_dim * 2,
            hidden_dim=conv_hidden_dim,
            hidden_layers=conv_hidden_layers,
        )

    def forward(self, image: Tensor, return_details: bool = False) -> Tensor | UICFINROutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Convolutional UICF control expects BCHW three-channel RGB")
        batch, _, height, width = image.shape
        encoded = self.encoder(image)
        anchor, global_feature = self.chromatic_anchor(encoded)
        if self.use_global_field_conditioning:
            global_map = global_feature[:, :, None, None].expand(-1, -1, height, width)
        else:
            global_map = global_feature.new_zeros(batch, self.feat_dim, height, width)
        correction_field = self.field_predictor(torch.cat((encoded, global_map), dim=1))
        reconstruction_anchor = anchor if self.use_learned_anchor else anchor.new_full(anchor.shape, 0.5)
        enhanced = image + correction_field * (
            image - reconstruction_anchor[:, :, None, None]
        )
        if not return_details:
            return enhanced
        return UICFINROutput(
            enhanced=enhanced,
            correction_field=correction_field,
            chromatic_anchor=reconstruction_anchor,
            global_feature=global_feature,
        )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def correction_parameter_report(
    implicit_module: nn.Module, convolutional_module: nn.Module
) -> dict[str, float | int]:
    """Return deterministic trainable counts and symmetric relative mismatch."""

    implicit = trainable_parameter_count(implicit_module)
    convolutional = trainable_parameter_count(convolutional_module)
    denominator = max(implicit, 1)
    return {
        "implicit_trainable_parameters": implicit,
        "convolutional_trainable_parameters": convolutional,
        "absolute_difference": abs(implicit - convolutional),
        "relative_difference": abs(implicit - convolutional) / denominator,
    }
