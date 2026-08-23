"""Color-query augmentation of the unchanged v4 Plain U-Net decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.v4.models.unet import DoubleConv, PlainUNet


class ColorTokenRefinementBlock(nn.Module):
    """Refine color tokens with feature cross-attention, self-attention, and an FFN."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        ffn_expansion: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or num_heads <= 0:
            raise ValueError("token_dim and num_heads must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if ffn_expansion <= 0:
            raise ValueError("ffn_expansion must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        hidden_dim = int(token_dim * ffn_expansion)
        if hidden_dim <= 0:
            raise ValueError("expanded FFN dimension must be positive")

        self.cross_token_norm = nn.LayerNorm(token_dim)
        self.cross_feature_norm = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, feature_sequence: Tensor) -> Tensor:
        if tokens.ndim != 3 or feature_sequence.ndim != 3:
            raise ValueError("Color token refinement expects [B,N,C] tensors")
        if tokens.shape[0] != feature_sequence.shape[0]:
            raise ValueError("Token and feature batch sizes must match")
        normalized_features = self.cross_feature_norm(feature_sequence)
        cross_output, _ = self.cross_attn(
            self.cross_token_norm(tokens),
            normalized_features,
            normalized_features,
            need_weights=False,
        )
        tokens = tokens + cross_output
        normalized = self.self_norm(tokens)
        self_output, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + self_output
        return tokens + self.ffn(self.ffn_norm(tokens))


class SpatialTokenGuidance(nn.Module):
    """Guide decoder pixels as queries using color tokens as keys and values."""

    def __init__(
        self,
        spatial_channels: int,
        token_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if spatial_channels <= 0 or token_dim <= 0 or num_heads <= 0:
            raise ValueError("Guidance dimensions must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.spatial_channels = spatial_channels
        self.token_dim = token_dim
        self.query_projection = nn.Conv2d(spatial_channels, token_dim, kernel_size=1)
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(token_dim, spatial_channels)

    def forward(self, spatial_feature: Tensor, tokens: Tensor) -> Tensor:
        if spatial_feature.ndim != 4 or tokens.ndim != 3:
            raise ValueError("Guidance expects BCHW features and BMC tokens")
        if spatial_feature.shape[0] != tokens.shape[0]:
            raise ValueError("Spatial feature and token batch sizes must match")
        if spatial_feature.shape[1] != self.spatial_channels:
            raise ValueError(f"Expected {self.spatial_channels} spatial channels")
        if tokens.shape[2] != self.token_dim:
            raise ValueError(f"Expected token dimension {self.token_dim}")
        batch, _, height, width = spatial_feature.shape
        spatial_queries = self.query_projection(spatial_feature).flatten(2).transpose(1, 2)
        guided, _ = self.attention(
            spatial_queries,
            tokens,
            tokens,
            need_weights=False,
        )
        guided = self.output_projection(guided)
        guided = guided.transpose(1, 2).reshape(batch, self.spatial_channels, height, width)
        return spatial_feature + guided


class ResidualLocalRefine(nn.Module):
    """V4-style residual DoubleConv refinement after color-token guidance."""

    def __init__(self, channels: int, use_batch_norm: bool) -> None:
        super().__init__()
        self.refine = DoubleConv(channels, channels, use_batch_norm=use_batch_norm)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.refine(inputs)


class PlainUNetColorQuery(PlainUNet):
    """The exact V4 Plain U-Net with color-query guidance added after each decoder block."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        use_batch_norm: bool = True,
        output_activation: str = "sigmoid",
        num_color_queries: int = 8,
        token_dim: int = 128,
        num_heads: int = 4,
        ffn_expansion: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        # The full V4 model must consume RNG first so all common states match V4.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            use_batch_norm=use_batch_norm,
            output_activation=output_activation,
        )
        if num_color_queries <= 0:
            raise ValueError("num_color_queries must be positive")
        if token_dim <= 0 or num_heads <= 0:
            raise ValueError("token_dim and num_heads must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")

        self.num_color_queries = num_color_queries
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.ffn_expansion = ffn_expansion
        self.color_query_dropout = dropout
        self.base_queries = nn.Parameter(torch.empty(1, num_color_queries, token_dim))
        nn.init.trunc_normal_(self.base_queries, std=0.02)

        encoder_channels = {
            "1": base_channels,
            "2": base_channels * 2,
            "3": base_channels * 4,
            "4": base_channels * 8,
        }
        self.feature_to_token = nn.ModuleDict(
            {
                stage: nn.Conv2d(channels, token_dim, kernel_size=1)
                for stage, channels in encoder_channels.items()
            }
        )
        self.token_refinement = nn.ModuleDict(
            {
                stage: ColorTokenRefinementBlock(
                    token_dim=token_dim,
                    num_heads=num_heads,
                    ffn_expansion=ffn_expansion,
                    dropout=dropout,
                )
                for stage in ("4", "3", "2", "1")
            }
        )

        self.guide4 = SpatialTokenGuidance(base_channels * 8, token_dim, num_heads, dropout)
        self.guide3 = SpatialTokenGuidance(base_channels * 4, token_dim, num_heads, dropout)
        self.guide2 = SpatialTokenGuidance(base_channels * 2, token_dim, num_heads, dropout)
        self.guide1 = SpatialTokenGuidance(base_channels, token_dim, num_heads, dropout)
        self.refine4 = ResidualLocalRefine(base_channels * 8, use_batch_norm)
        self.refine3 = ResidualLocalRefine(base_channels * 4, use_batch_norm)
        self.refine2 = ResidualLocalRefine(base_channels * 2, use_batch_norm)
        self.refine1 = ResidualLocalRefine(base_channels, use_batch_norm)

    def _project_feature(self, stage: str, feature: Tensor) -> Tensor:
        return self.feature_to_token[stage](feature).flatten(2).transpose(1, 2)

    def _refine_color_queries(
        self, features: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        e1, e2, e3, e4 = features
        batch = e1.shape[0]
        t0 = self.base_queries.expand(batch, -1, -1)
        t4 = self.token_refinement["4"](t0, self._project_feature("4", e4))
        t3 = self.token_refinement["3"](t4, self._project_feature("3", e3))
        t2 = self.token_refinement["2"](t3, self._project_feature("2", e2))
        t1 = self.token_refinement["1"](t2, self._project_feature("1", e1))
        return t0, t4, t3, t2, t1

    def _forward_impl(self, inputs: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(f"Expected BCHW input with {self.in_channels} channels")
        padded, original_height, original_width = self._pad(inputs)

        # Exact inherited V4 encoder and bottleneck.
        e1 = self.encoder1(padded)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        # Learnable base queries are refined deep-to-shallow: E4 -> E3 -> E2 -> E1.
        t0, t4, t3, t2, t1 = self._refine_color_queries((e1, e2, e3, e4))

        # Each original V4 decoder block is followed by its matching guidance and refine stage.
        d4 = self.decoder4(self._concat(self.upconv4(bottleneck), e4))
        d4 = self.refine4(self.guide4(d4, t4))
        d3 = self.decoder3(self._concat(self.upconv3(d4), e3))
        d3 = self.refine3(self.guide3(d3, t3))
        d2 = self.decoder2(self._concat(self.upconv2(d3), e2))
        d2 = self.refine2(self.guide2(d2, t2))
        d1 = self.decoder1(self._concat(self.upconv1(d2), e1))
        d1 = self.refine1(self.guide1(d1, t1))

        # Exact V4 direct RGB formulation: Conv1x1 -> configured activation, then crop.
        output = self.output_activation(self.output_conv(d1))
        output = output[..., :original_height, :original_width]

        shapes = {
            "input": tuple(inputs.shape),
            "e1": tuple(e1.shape),
            "e2": tuple(e2.shape),
            "e3": tuple(e3.shape),
            "e4": tuple(e4.shape),
            "bottleneck": tuple(bottleneck.shape),
            "t0": tuple(t0.shape),
            "t4": tuple(t4.shape),
            "t3": tuple(t3.shape),
            "t2": tuple(t2.shape),
            "t1": tuple(t1.shape),
            "d4": tuple(d4.shape),
            "d3": tuple(d3.shape),
            "d2": tuple(d2.shape),
            "d1": tuple(d1.shape),
            "decoder_output": tuple(d1.shape),
            "final_output": tuple(output.shape),
        }
        return output, shapes


__all__ = [
    "ColorTokenRefinementBlock",
    "PlainUNetColorQuery",
    "ResidualLocalRefine",
    "SpatialTokenGuidance",
]
