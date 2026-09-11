"""Normalized Laplacian calibration for the existing decoder token guidance."""

from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.shared.color_query_unet import SpatialTokenGuidance


class NLQCSpatialTokenGuidance(nn.Module):
    """Add an image-local normalized Laplacian response to MHA logits.

    The projection, attention, and output modules are the exact instances from
    ``base_guidance``.  Consequently the only new learnable state is one scalar
    ``nlqc_alpha`` for this decoder stage.
    """

    def __init__(
        self,
        base_guidance: nn.Module,
        kernel_lambda: float = 4.0,
        epsilon: float = 1e-6,
        kernel_chunk_size: int = 1024,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_guidance, SpatialTokenGuidance):
            raise TypeError(
                "base_guidance must be the existing SpatialTokenGuidance instance, "
                f"got {type(base_guidance).__name__}"
            )
        if not math.isfinite(kernel_lambda) or kernel_lambda <= 0.0:
            raise ValueError("nlqc.kernel_lambda must be finite and positive")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("nlqc.epsilon must be finite and positive")
        if isinstance(kernel_chunk_size, bool) or int(kernel_chunk_size) <= 0:
            raise ValueError("nlqc.kernel_chunk_size must be a positive integer")
        if int(kernel_chunk_size) != kernel_chunk_size:
            raise ValueError("nlqc.kernel_chunk_size must be an integer")
        if not math.isfinite(alpha_init):
            raise ValueError("nlqc.alpha_init must be finite")

        attention = base_guidance.attention
        if not isinstance(attention, nn.MultiheadAttention):
            raise TypeError("base_guidance.attention must be nn.MultiheadAttention")
        if not attention.batch_first:
            raise ValueError("NLQC requires batch_first=True attention")
        if not getattr(attention, "_qkv_same_embed_dim", False):
            raise ValueError("NLQC requires equal Q/K/V embedding dimensions")
        if attention.in_proj_weight is None:
            raise ValueError("NLQC requires packed attention.in_proj_weight")
        embed_dim = int(attention.embed_dim)
        if attention.in_proj_weight.shape != (3 * embed_dim, embed_dim):
            raise ValueError(
                "Unexpected packed Q/K/V projection shape: "
                f"{tuple(attention.in_proj_weight.shape)}"
            )
        if attention.in_proj_bias is not None and attention.in_proj_bias.shape != (3 * embed_dim,):
            raise ValueError(
                "Unexpected packed Q/K/V bias shape: "
                f"{tuple(attention.in_proj_bias.shape)}"
            )
        if embed_dim % int(attention.num_heads) != 0:
            raise ValueError("attention.embed_dim must be divisible by num_heads")
        if int(base_guidance.token_dim) != embed_dim:
            raise ValueError("base guidance token_dim differs from attention embed_dim")

        self.spatial_channels = int(base_guidance.spatial_channels)
        self.token_dim = int(base_guidance.token_dim)
        self.kernel_lambda = float(kernel_lambda)
        self.epsilon = float(epsilon)
        self.kernel_chunk_size = int(kernel_chunk_size)

        # Reuse these exact initialized module instances. Do not reconstruct them.
        self.query_projection = base_guidance.query_projection
        self.attention = attention
        self.output_projection = base_guidance.output_projection
        self.nlqc_alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _project_qk_fp32(
        self, spatial_queries: Tensor, tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply the existing packed MHA Q/K projections in FP32."""
        if spatial_queries.ndim != 3 or tokens.ndim != 3:
            raise ValueError("NLQC expects batch-first [B,N,C] query and [B,M,C] tokens")
        if spatial_queries.shape[0] != tokens.shape[0]:
            raise ValueError("NLQC query and token batch sizes must match")
        if spatial_queries.shape[-1] != self.token_dim or tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"NLQC expects embedding dimension {self.token_dim}, got "
                f"{spatial_queries.shape[-1]} and {tokens.shape[-1]}"
            )
        if tokens.shape[1] <= 0 or spatial_queries.shape[1] <= 0:
            raise ValueError("NLQC query and token sequences must be non-empty")

        weight = self.attention.in_proj_weight
        if weight is None or weight.shape != (3 * self.token_dim, self.token_dim):
            raise RuntimeError("Packed attention Q/K/V projection changed after NLQC construction")
        bias = self.attention.in_proj_bias
        if bias is not None and bias.shape != (3 * self.token_dim,):
            raise RuntimeError("Packed attention Q/K/V bias changed after NLQC construction")
        q_bias = None if bias is None else bias[: self.token_dim].float()
        k_bias = None if bias is None else bias[self.token_dim : 2 * self.token_dim].float()
        projected_q = F.linear(
            spatial_queries.float(), weight[: self.token_dim].float(), q_bias
        )
        projected_k = F.linear(
            tokens.float(), weight[self.token_dim : 2 * self.token_dim].float(), k_bias
        )
        return projected_q, projected_k

    def normalized_laplacian_response(
        self, spatial_queries: Tensor, tokens: Tensor
    ) -> Tensor:
        """Return Z with shape [B, heads, spatial_queries, color_queries]."""
        device_type = spatial_queries.device.type
        autocast_disabled = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast_disabled:
            projected_q, projected_k = self._project_qk_fp32(spatial_queries, tokens)
            batch_size, spatial_count, _ = projected_q.shape
            color_count = projected_k.shape[1]
            num_heads = int(self.attention.num_heads)
            head_dim = self.token_dim // num_heads

            q_heads = projected_q.reshape(
                batch_size, spatial_count, num_heads, head_dim
            ).transpose(1, 2)
            k_heads = projected_k.reshape(
                batch_size, color_count, num_heads, head_dim
            ).transpose(1, 2)
            q_flat = q_heads.reshape(batch_size * num_heads, spatial_count, head_dim)
            k_flat = k_heads.reshape(batch_size * num_heads, color_count, head_dim)

            responses: list[Tensor] = []
            for start in range(0, spatial_count, self.kernel_chunk_size):
                stop = min(start + self.kernel_chunk_size, spatial_count)
                l1_distance = torch.cdist(q_flat[:, start:stop], k_flat, p=1.0)
                responses.append(torch.exp(-l1_distance / self.kernel_lambda))
            kernel = torch.cat(responses, dim=1).reshape(
                batch_size, num_heads, spatial_count, color_count
            )
            centered = kernel - kernel.mean(dim=-1, keepdim=True)
            spatial_mean = centered.mean(dim=2, keepdim=True)
            spatial_variance = (centered - spatial_mean).square().mean(
                dim=2, keepdim=True
            )
            normalized = (centered - spatial_mean) * torch.rsqrt(
                spatial_variance + self.epsilon
            )

        if normalized.dtype != torch.float32:
            raise RuntimeError("NLQC normalization must be computed in FP32")
        if not bool(torch.isfinite(normalized).all()):
            raise FloatingPointError("NLQC normalized Laplacian response contains NaN/Inf")
        return normalized

    def forward(self, spatial_feature: Tensor, tokens: Tensor) -> Tensor:
        if spatial_feature.ndim != 4:
            raise ValueError("NLQC spatial feature must be BCHW")
        batch_size, channels, height, width = spatial_feature.shape
        if channels != self.spatial_channels:
            raise ValueError(
                f"NLQC expected {self.spatial_channels} spatial channels, got {channels}"
            )
        spatial_queries = self.query_projection(spatial_feature).flatten(2).transpose(1, 2)
        normalized = self.normalized_laplacian_response(spatial_queries, tokens)
        additive_mask = (self.nlqc_alpha.float() * normalized).reshape(
            batch_size * int(self.attention.num_heads), height * width, tokens.shape[1]
        )
        additive_mask = additive_mask.to(dtype=spatial_queries.dtype)
        guided, _ = self.attention(
            spatial_queries,
            tokens,
            tokens,
            need_weights=False,
            attn_mask=additive_mask,
        )
        guided = self.output_projection(guided).transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        return spatial_feature + guided
