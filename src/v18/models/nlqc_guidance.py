"""Normalized Laplacian calibration for the existing decoder token guidance."""

from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.shared.color_query_unet import SpatialTokenGuidance


def tensor_finite_diagnostics(name: str, tensor: Tensor) -> dict[str, object]:
    """Return robust finite/non-finite statistics without changing autograd state."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor for finite diagnostics: {name}")
    if not tensor.is_floating_point():
        raise TypeError(
            f"Finite diagnostics require a floating tensor: {name} has dtype={tensor.dtype}"
        )

    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    report: dict[str, object] = {
        "tensor_name": name,
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": detached.numel(),
        "finite_count": finite_count,
        "num_nan": int(torch.isnan(detached).sum().item()),
        "num_posinf": int(torch.isposinf(detached).sum().item()),
        "num_neginf": int(torch.isneginf(detached).sum().item()),
        "finite_min": None,
        "finite_max": None,
        "finite_abs_max": None,
        "finite_mean": None,
        "finite_std": None,
    }
    if finite_count:
        # These copies happen only after a failure. CPU float64 reductions keep
        # the diagnostic itself safe from AMP and low-precision overflow.
        finite_values = detached[finite_mask].to(device="cpu", dtype=torch.float64)
        report.update(
            {
                "finite_min": float(finite_values.min().item()),
                "finite_max": float(finite_values.max().item()),
                "finite_abs_max": float(finite_values.abs().max().item()),
                "finite_mean": float(finite_values.mean().item()),
                "finite_std": float(finite_values.std(unbiased=False).item()),
            }
        )
    return report


def format_finite_diagnostics(report: dict[str, object]) -> str:
    """Format every required field, including safe N/A values when all are non-finite."""

    def value(key: str) -> str:
        item = report[key]
        if item is None:
            return "N/A"
        if isinstance(item, float):
            return f"{item:.17g}"
        return str(item)

    fields = (
        "tensor_name",
        "shape",
        "dtype",
        "device",
        "numel",
        "finite_count",
        "num_nan",
        "num_posinf",
        "num_neginf",
        "finite_min",
        "finite_max",
        "finite_abs_max",
        "finite_mean",
        "finite_std",
    )
    return "\n".join(f"{key}={value(key)}" for key in fields)


def require_finite_tensor(name: str, tensor: Tensor, *, context: str) -> None:
    """Fail at the first non-finite tensor and include actionable statistics."""
    if bool(torch.isfinite(tensor).all()):
        return
    report = tensor_finite_diagnostics(name, tensor)
    raise FloatingPointError(
        "NLQC finite diagnostic failure:\n"
        f"{format_finite_diagnostics(report)}\n"
        f"context={context}"
    )


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

    def _diagnostic_context(
        self,
        *,
        spatial_feature: Tensor | None = None,
        spatial_queries: Tensor | None = None,
        tokens: Tensor | None = None,
        extra: str | None = None,
    ) -> str:
        fields = [
            f"spatial_channels={self.spatial_channels}",
            f"token_dim={self.token_dim}",
            "spatial_feature_dtype="
            + (str(spatial_feature.dtype) if spatial_feature is not None else "not_available"),
            "spatial_queries_dtype="
            + (str(spatial_queries.dtype) if spatial_queries is not None else "not_computed"),
            "tokens_dtype=" + (str(tokens.dtype) if tokens is not None else "not_available"),
        ]
        if extra is not None:
            fields.append(extra)
        return ", ".join(fields)

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

        context = self._diagnostic_context(
            spatial_queries=spatial_queries, tokens=tokens
        )
        require_finite_tensor("spatial_queries", spatial_queries, context=context)
        require_finite_tensor("tokens", tokens, context=context)

        weight = self.attention.in_proj_weight
        if weight is None or weight.shape != (3 * self.token_dim, self.token_dim):
            raise RuntimeError("Packed attention Q/K/V projection changed after NLQC construction")
        bias = self.attention.in_proj_bias
        if bias is not None and bias.shape != (3 * self.token_dim,):
            raise RuntimeError("Packed attention Q/K/V bias changed after NLQC construction")
        require_finite_tensor("attention.in_proj_weight", weight, context=context)
        if bias is not None:
            require_finite_tensor("attention.in_proj_bias", bias, context=context)
        q_bias = None if bias is None else bias[: self.token_dim].float()
        k_bias = None if bias is None else bias[self.token_dim : 2 * self.token_dim].float()
        projected_q = F.linear(
            spatial_queries.float(), weight[: self.token_dim].float(), q_bias
        )
        projected_k = F.linear(
            tokens.float(), weight[self.token_dim : 2 * self.token_dim].float(), k_bias
        )
        require_finite_tensor("projected_q", projected_q, context=context)
        require_finite_tensor("projected_k", projected_k, context=context)
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
            context = self._diagnostic_context(
                spatial_queries=spatial_queries, tokens=tokens
            )
            require_finite_tensor("q_flat", q_flat, context=context)
            require_finite_tensor("k_flat", k_flat, context=context)

            responses: list[Tensor] = []
            for start in range(0, spatial_count, self.kernel_chunk_size):
                stop = min(start + self.kernel_chunk_size, spatial_count)
                l1_distance = torch.cdist(q_flat[:, start:stop], k_flat, p=1.0)
                chunk_context = self._diagnostic_context(
                    spatial_queries=spatial_queries,
                    tokens=tokens,
                    extra=f"spatial_chunk=[{start}:{stop}]",
                )
                require_finite_tensor(
                    "l1_distance", l1_distance, context=chunk_context
                )
                kernel_chunk = torch.exp(-l1_distance / self.kernel_lambda)
                require_finite_tensor("kernel", kernel_chunk, context=chunk_context)
                responses.append(kernel_chunk)
            kernel = torch.cat(responses, dim=1).reshape(
                batch_size, num_heads, spatial_count, color_count
            )
            require_finite_tensor("kernel", kernel, context=context)
            centered = kernel - kernel.mean(dim=-1, keepdim=True)
            require_finite_tensor("centered", centered, context=context)
            spatial_mean = centered.mean(dim=2, keepdim=True)
            require_finite_tensor("spatial_mean", spatial_mean, context=context)
            spatial_variance = (centered - spatial_mean).square().mean(
                dim=2, keepdim=True
            )
            require_finite_tensor(
                "spatial_variance", spatial_variance, context=context
            )
            normalized = (centered - spatial_mean) * torch.rsqrt(
                spatial_variance + self.epsilon
            )
            require_finite_tensor("normalized", normalized, context=context)

        if normalized.dtype != torch.float32:
            raise RuntimeError("NLQC normalization must be computed in FP32")
        return normalized

    def forward(self, spatial_feature: Tensor, tokens: Tensor) -> Tensor:
        if spatial_feature.ndim != 4:
            raise ValueError("NLQC spatial feature must be BCHW")
        batch_size, channels, height, width = spatial_feature.shape
        if channels != self.spatial_channels:
            raise ValueError(
                f"NLQC expected {self.spatial_channels} spatial channels, got {channels}"
            )
        input_context = self._diagnostic_context(
            spatial_feature=spatial_feature, tokens=tokens
        )
        # The order is intentional: it separates upstream decoder failures from
        # token-path failures before the AMP query projection is evaluated.
        require_finite_tensor("spatial_feature", spatial_feature, context=input_context)
        require_finite_tensor("tokens", tokens, context=input_context)
        require_finite_tensor(
            "query_projection.weight", self.query_projection.weight, context=input_context
        )
        if self.query_projection.bias is not None:
            require_finite_tensor(
                "query_projection.bias", self.query_projection.bias, context=input_context
            )
        spatial_queries = self.query_projection(spatial_feature).flatten(2).transpose(1, 2)
        projection_context = self._diagnostic_context(
            spatial_feature=spatial_feature,
            spatial_queries=spatial_queries,
            tokens=tokens,
        )
        require_finite_tensor(
            "spatial_queries", spatial_queries, context=projection_context
        )
        normalized = self.normalized_laplacian_response(spatial_queries, tokens)
        require_finite_tensor("nlqc_alpha", self.nlqc_alpha, context=projection_context)
        additive_mask_fp32 = (self.nlqc_alpha.float() * normalized).reshape(
            batch_size * int(self.attention.num_heads), height * width, tokens.shape[1]
        )
        require_finite_tensor(
            "additive_mask_fp32", additive_mask_fp32, context=projection_context
        )
        additive_mask = additive_mask_fp32.to(dtype=spatial_queries.dtype)
        require_finite_tensor("additive_mask", additive_mask, context=projection_context)
        guided, _ = self.attention(
            spatial_queries,
            tokens,
            tokens,
            need_weights=False,
            attn_mask=additive_mask,
        )
        require_finite_tensor("attention_output", guided, context=projection_context)
        require_finite_tensor(
            "output_projection.weight",
            self.output_projection.weight,
            context=projection_context,
        )
        if self.output_projection.bias is not None:
            require_finite_tensor(
                "output_projection.bias",
                self.output_projection.bias,
                context=projection_context,
            )
        projected_guidance = self.output_projection(guided)
        require_finite_tensor(
            "output_projection_output", projected_guidance, context=projection_context
        )
        guided = projected_guidance.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        output = spatial_feature + guided
        require_finite_tensor("guidance_residual_output", output, context=projection_context)
        return output
