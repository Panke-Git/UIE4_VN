"""Point-wise feature-conditioned absolute-coordinate INR baseline."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FourierEncoding(nn.Module):
    def __init__(self, num_frequencies: int, include_raw: bool = True) -> None:
        super().__init__()
        if num_frequencies < 0:
            raise ValueError("num_frequencies must be non-negative")
        self.num_frequencies = num_frequencies
        self.include_raw = include_raw
        frequencies = torch.tensor([2**k * math.pi for k in range(num_frequencies)])
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.output_dim = (2 if include_raw else 0) + 4 * num_frequencies

    def forward(self, coordinates: Tensor) -> Tensor:
        parts = [coordinates] if self.include_raw else []
        if self.num_frequencies:
            angles = coordinates.unsqueeze(-1) * self.frequencies.to(coordinates)
            parts.extend((angles.sin().flatten(-2), angles.cos().flatten(-2)))
        if not parts:
            return coordinates.new_empty((*coordinates.shape[:-1], 0))
        return torch.cat(parts, dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("MLP depth must be at least one")
        dimensions = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def pixel_center_coordinates(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return H*W (x, y) pixel-center coordinates in [-1, 1]."""
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


class PointINR(nn.Module):
    """Map [E(x), absolute Fourier coordinates] to a C-channel feature residual."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        num_frequencies: int,
        depth: int,
        include_raw_coordinate: bool,
        query_chunk: int,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        self.channels = channels
        self.query_chunk = query_chunk
        self.residual = residual
        self.encoding = FourierEncoding(num_frequencies, include_raw_coordinate)
        self.mlp = MLP(channels + self.encoding.output_dim, hidden_dim, channels, depth)
        output_layer = self.mlp.net[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("Point-INR correction MLP must end with a Linear layer")
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    def forward(self, features: Tensor) -> Tensor:
        # E: [B,C,H,W], queries: [B*H*W,C+PE]
        batch, channels, height, width = features.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        coordinates = pixel_center_coordinates(height, width, features.device, features.dtype)
        encoded = self.encoding(coordinates)
        encoded = encoded.unsqueeze(0).expand(batch, -1, -1).reshape(batch * height * width, -1)
        flattened = features.permute(0, 2, 3, 1).reshape(batch * height * width, channels)

        outputs = []
        for start in range(0, flattened.shape[0], self.query_chunk):
            stop = min(start + self.query_chunk, flattened.shape[0])
            outputs.append(self.mlp(torch.cat((flattened[start:stop], encoded[start:stop]), dim=-1)))
        correction = torch.cat(outputs, dim=0).reshape(batch, height, width, channels)
        correction = correction.permute(0, 3, 1, 2).contiguous()
        return features + correction if self.residual else correction
