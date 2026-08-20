"""Canonical Underwater Implicit Correction-Field INR implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class UICFINROutput:
    enhanced: Tensor
    correction_field: Tensor
    chromatic_anchor: Tensor
    global_feature: Tensor


class ConvBlock(nn.Module):
    """Two 3x3 Conv-GELU pairs plus an identity or 1x1 residual skip."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GELU(),
        )
        self.skip: nn.Module = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs) + self.skip(inputs)


class ImageEncoder(nn.Module):
    """Full-resolution RGB encoder prescribed by UICF-INR."""

    def __init__(self, feat_dim: int = 48) -> None:
        super().__init__()
        if feat_dim < 1:
            raise ValueError("feat_dim must be positive")
        self.feat_dim = feat_dim
        self.intro = nn.Sequential(
            nn.Conv2d(3, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(ConvBlock(feat_dim, feat_dim), ConvBlock(feat_dim, feat_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.blocks(self.intro(inputs))


class PeriodicSpatialEncoding(nn.Module):
    """Fixed Zhao-style encoding with 2**k bands and no pi multiplier."""

    def __init__(self, num_frequencies: int = 8) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        self.num_frequencies = num_frequencies
        self.output_dim = 4 * num_frequencies
        bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequency_bands", bands, persistent=False)

    def forward(
        self,
        batch_size: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(height * width, 2)
        bands = self.frequency_bands.to(device=device, dtype=dtype)
        x_angles = coordinates[:, 0:1] * bands
        y_angles = coordinates[:, 1:2] * bands
        encoded = torch.cat(
            (x_angles.sin(), x_angles.cos(), y_angles.sin(), y_angles.cos()), dim=-1
        )
        return encoded.unsqueeze(0).expand(batch_size, -1, -1)


class GlobalChromaticAnchor(nn.Module):
    """Global average feature and its learned three-channel chromatic anchor."""

    def __init__(self, feat_dim: int = 48, hidden_dim: int = 64) -> None:
        super().__init__()
        self.anchor_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        global_feature = F.adaptive_avg_pool2d(features, 1).flatten(1)
        chromatic_anchor = self.anchor_mlp(global_feature)
        return chromatic_anchor, global_feature


class CorrectionFieldMLP(nn.Module):
    """Shared per-pixel MLP that predicts an unconstrained RGB correction field."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        query_chunk_size: int | None = 65536,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        if query_chunk_size is not None and query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive or None")
        self.query_chunk_size = query_chunk_size
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.GELU()))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 3))
        self.net = nn.Sequential(*layers)
        output_layer = self.net[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("UICF correction MLP must end with Linear")
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        queries = inputs.shape[1]
        if self.query_chunk_size is None or queries <= self.query_chunk_size:
            return self.net(inputs)
        outputs = [
            self.net(inputs[:, start : start + self.query_chunk_size])
            for start in range(0, queries, self.query_chunk_size)
        ]
        return torch.cat(outputs, dim=1)


class UnderwaterImplicitCorrectionField(nn.Module):
    """Predict R(x) and reconstruct exactly I + R * (I - b)."""

    def __init__(
        self,
        feat_dim: int = 48,
        num_frequencies: int = 8,
        mlp_hidden_dim: int = 128,
        mlp_hidden_layers: int = 3,
        anchor_hidden_dim: int = 64,
        query_chunk_size: int | None = 65536,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.encoder = ImageEncoder(feat_dim)
        self.spatial_encoding = PeriodicSpatialEncoding(num_frequencies)
        self.chromatic_anchor = GlobalChromaticAnchor(feat_dim, anchor_hidden_dim)
        mlp_input_dim = feat_dim + self.spatial_encoding.output_dim + feat_dim
        self.field_mlp = CorrectionFieldMLP(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_layers=mlp_hidden_layers,
            query_chunk_size=query_chunk_size,
        )

    def forward(self, image: Tensor, return_details: bool = False) -> Tensor | UICFINROutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("UICF-INR expects a BCHW three-channel RGB tensor")
        batch, _, height, width = image.shape
        encoded = self.encoder(image)
        local_feature = encoded.permute(0, 2, 3, 1).reshape(batch, height * width, self.feat_dim)
        positional_feature = self.spatial_encoding(
            batch, height, width, device=image.device, dtype=image.dtype
        )
        anchor, global_feature = self.chromatic_anchor(encoded)
        expanded_global = global_feature.unsqueeze(1).expand(-1, height * width, -1)
        mlp_input = torch.cat((local_feature, positional_feature, expanded_global), dim=-1)
        field_flat = self.field_mlp(mlp_input)
        correction_field = field_flat.reshape(batch, height, width, 3).permute(0, 3, 1, 2)
        enhanced = image + correction_field * (image - anchor[:, :, None, None])
        if not return_details:
            return enhanced
        return UICFINROutput(
            enhanced=enhanced,
            correction_field=correction_field,
            chromatic_anchor=anchor,
            global_feature=global_feature,
        )
