"""Global-Local Implicit Neural Representation at NAFNet bottleneck features."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FourierEncoding(nn.Module):
    """Encode 2-D coordinates with frequencies 2^k*pi and optional raw coordinates."""

    def __init__(self, num_frequencies: int, include_raw: bool) -> None:
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
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


class GlobalCoordinateBranch(nn.Module):
    """Coordinate-only global branch h_G(x); it deliberately never reads E or Z."""

    def __init__(self, num_frequencies: int, include_raw: bool, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.encoding = FourierEncoding(num_frequencies, include_raw)
        if self.encoding.output_dim == 0:
            raise ValueError("Global coordinate encoding cannot be empty")
        self.mlp = MLP(self.encoding.output_dim, hidden_dim, hidden_dim, depth)

    def forward(self, coordinates: Tensor) -> Tensor:
        return self.mlp(self.encoding(coordinates))


class LocalImplicitBranch(nn.Module):
    """Shared local MLP plus feature-level four-neighbor bilinear ensemble."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_frequencies: int,
        include_raw: bool,
        depth: int,
    ) -> None:
        super().__init__()
        self.encoding = FourierEncoding(num_frequencies, include_raw)
        if self.encoding.output_dim == 0:
            raise ValueError("Local relative-coordinate encoding cannot be empty")
        # Exactly one MLP is shared by top-left, top-right, bottom-left, bottom-right.
        self.mlp = MLP(latent_dim + self.encoding.output_dim, hidden_dim, hidden_dim, depth)

    def forward(self, neighbor_codes: Tensor, relative_coordinates: Tensor, weights: Tensor) -> Tensor:
        # codes [B,Q,4,Cz], relative [Q,4,2], weights [Q,4]
        batch, queries, neighbors, latent_dim = neighbor_codes.shape
        encoded = self.encoding(relative_coordinates)
        encoded = encoded.unsqueeze(0).expand(batch, -1, -1, -1)
        implicit = self.mlp(
            torch.cat((neighbor_codes, encoded), dim=-1).reshape(batch * queries * neighbors, -1)
        ).reshape(batch, queries, neighbors, -1)
        return (implicit * weights[None, :, :, None]).sum(dim=2)


class GLINR(nn.Module):
    """Feature-level GL-INR: global coordinates + local latent interpolation + fusion."""

    def __init__(
        self,
        channels: int,
        latent_dim: int,
        hidden_dim: int,
        latent_stride: int,
        global_num_frequencies: int,
        local_num_frequencies: int,
        include_raw_absolute_coordinate: bool,
        include_raw_relative_coordinate: bool,
        local_depth: int,
        global_depth: int,
        fusion_depth: int,
        query_chunk: int,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if latent_stride < 1 or query_chunk < 1:
            raise ValueError("latent_stride and query_chunk must be positive")
        self.channels = channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.latent_stride = latent_stride
        self.query_chunk = query_chunk
        self.residual = residual
        self.latent_projection = nn.Conv2d(
            channels, latent_dim, kernel_size=3, stride=latent_stride, padding=1
        )
        self.global_branch = GlobalCoordinateBranch(
            global_num_frequencies,
            include_raw_absolute_coordinate,
            hidden_dim,
            global_depth,
        )
        self.local_branch = LocalImplicitBranch(
            latent_dim,
            hidden_dim,
            local_num_frequencies,
            include_raw_relative_coordinate,
            local_depth,
        )
        self.fusion = MLP(hidden_dim * 2, hidden_dim, channels, fusion_depth)
        output_layer = self.fusion.net[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("GL-INR fusion MLP must end with a Linear layer")
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    @staticmethod
    def query_geometry(
        query_height: int,
        query_width: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return query coords, [Q,4,(y,x)] indices, relative coords, and weights."""
        coordinates = pixel_center_coordinates(query_height, query_width, device, dtype)
        # align_corners=False mapping from normalized coordinates to latent index space.
        gx = (coordinates[:, 0] + 1.0) * latent_width / 2.0 - 0.5
        gy = (coordinates[:, 1] + 1.0) * latent_height / 2.0 - 0.5
        x0_raw, y0_raw = torch.floor(gx), torch.floor(gy)
        x1_raw, y1_raw = x0_raw + 1.0, y0_raw + 1.0
        fx, fy = gx - x0_raw, gy - y0_raw

        x0 = x0_raw.long().clamp(0, latent_width - 1)
        x1 = x1_raw.long().clamp(0, latent_width - 1)
        y0 = y0_raw.long().clamp(0, latent_height - 1)
        y1 = y1_raw.long().clamp(0, latent_height - 1)
        indices = torch.stack(
            (
                torch.stack((y0, x0), dim=-1),
                torch.stack((y0, x1), dim=-1),
                torch.stack((y1, x0), dim=-1),
                torch.stack((y1, x1), dim=-1),
            ),
            dim=1,
        )
        weights = torch.stack(
            ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy), dim=1
        )

        # Local geometry is expressed in latent-cell units. Feature indices are
        # clamped for safe gathering, while the unclamped geometric neighbors
        # preserve the correct relative offsets at borders.
        rel_tl = torch.stack((gx - x0_raw, gy - y0_raw), dim=-1)
        rel_tr = torch.stack((gx - x1_raw, gy - y0_raw), dim=-1)
        rel_bl = torch.stack((gx - x0_raw, gy - y1_raw), dim=-1)
        rel_br = torch.stack((gx - x1_raw, gy - y1_raw), dim=-1)
        relative = torch.stack((rel_tl, rel_tr, rel_bl, rel_br), dim=1)
        return coordinates, indices, relative, weights

    def forward(self, features: Tensor) -> Tensor:
        # E [B,C,Hf,Wf] -> Z [B,Cz,Hl,Wl]
        batch, channels, height, width = features.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        latent = self.latent_projection(features)
        _, latent_dim, latent_height, latent_width = latent.shape
        coordinates, indices, relative, weights = self.query_geometry(
            height, width, latent_height, latent_width, features.device, features.dtype
        )
        latent_flat = latent.permute(0, 2, 3, 1).reshape(batch, latent_height * latent_width, latent_dim)
        spatial_chunk = max(1, self.query_chunk // batch)
        fused_chunks: list[Tensor] = []

        for start in range(0, coordinates.shape[0], spatial_chunk):
            stop = min(start + spatial_chunk, coordinates.shape[0])
            chunk_indices = indices[start:stop]
            linear = chunk_indices[..., 0] * latent_width + chunk_indices[..., 1]
            gather_index = linear.reshape(1, -1, 1).expand(batch, -1, latent_dim)
            neighbor_codes = torch.gather(latent_flat, 1, gather_index)
            neighbor_codes = neighbor_codes.reshape(batch, stop - start, 4, latent_dim)

            local = self.local_branch(
                neighbor_codes, relative[start:stop], weights[start:stop]
            )  # [B,Qc,hidden]
            global_feature = self.global_branch(coordinates[start:stop])
            global_feature = global_feature.unsqueeze(0).expand(batch, -1, -1)
            fused = self.fusion(
                torch.cat((local, global_feature), dim=-1).reshape(batch * (stop - start), -1)
            ).reshape(batch, stop - start, channels)
            fused_chunks.append(fused)

        correction = torch.cat(fused_chunks, dim=1).reshape(batch, height, width, channels)
        correction = correction.permute(0, 3, 1, 2).contiguous()
        return features + correction if self.residual else correction
