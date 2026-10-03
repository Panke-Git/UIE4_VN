"""Stage-Differential Query Routing for the unchanged Color-Query Plain U-Net."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.shared.color_query_unet import PlainUNetColorQuery


def _route_queries(current: Tensor, previous: Tensor, beta: Tensor) -> Tensor:
    """Route one stage's query update with one scalar per color query."""
    if current.ndim != 3 or previous.ndim != 3:
        raise ValueError("SDQR current and previous tokens must have shape [B,M,C]")
    if current.shape != previous.shape:
        raise ValueError(
            "SDQR current and previous token shapes must match, got "
            f"{tuple(current.shape)} and {tuple(previous.shape)}"
        )
    expected_beta_shape = (1, current.shape[1], 1)
    if tuple(beta.shape) != expected_beta_shape:
        raise ValueError(
            f"SDQR beta must have shape {expected_beta_shape}, got {tuple(beta.shape)}"
        )
    if not current.is_floating_point() or not previous.is_floating_point():
        raise TypeError("SDQR tokens must be floating tensors")
    if not beta.is_floating_point():
        raise TypeError("SDQR beta must be a floating tensor")
    if current.device != previous.device or current.device != beta.device:
        raise ValueError("SDQR current, previous, and beta must be on the same device")
    for name, value in (("current", current), ("previous", previous), ("beta", beta)):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"SDQR {name} contains NaN/Inf")

    eta = torch.tanh(beta).to(dtype=current.dtype)
    if not bool(torch.isfinite(eta).all()):
        raise FloatingPointError("SDQR eta contains NaN/Inf")
    delta = current - previous
    if not bool(torch.isfinite(delta).all()):
        raise FloatingPointError("SDQR token delta contains NaN/Inf")
    routed = current + eta * delta
    if not bool(torch.isfinite(routed).all()):
        raise FloatingPointError("SDQR routed tokens contain NaN/Inf")
    return routed


class SDQRPlainUNetColorQuery(PlainUNetColorQuery):
    """V16 Color-Query backbone plus four per-query differential coefficients."""

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
        beta_init: float = 0.0,
    ) -> None:
        # Construct the complete V16 backbone first. The four zero tensors below
        # consume no RNG, preserving both backbone and subsequent UICF states.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            use_batch_norm=use_batch_norm,
            output_activation=output_activation,
            num_color_queries=num_color_queries,
            token_dim=token_dim,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
        )
        if not math.isfinite(beta_init) or beta_init != 0.0:
            raise ValueError("SDQR beta_init must be exactly 0.0 for V16-equivalent initialization")
        shape = (1, num_color_queries, 1)
        self.sdqr_beta4 = nn.Parameter(torch.zeros(shape))
        self.sdqr_beta3 = nn.Parameter(torch.zeros(shape))
        self.sdqr_beta2 = nn.Parameter(torch.zeros(shape))
        self.sdqr_beta1 = nn.Parameter(torch.zeros(shape))

    def _forward_impl(self, inputs: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(f"Expected BCHW input with {self.in_channels} channels")
        padded, original_height, original_width = self._pad(inputs)

        # Exact inherited V16 encoder, bottleneck, and query-refinement chain.
        e1 = self.encoder1(padded)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        t0, t4, t3, t2, t1 = self._refine_color_queries((e1, e2, e3, e4))

        # The only V19 mechanism: route each already-computed stage refinement.
        rt4 = _route_queries(t4, t0, self.sdqr_beta4)
        rt3 = _route_queries(t3, t4, self.sdqr_beta3)
        rt2 = _route_queries(t2, t3, self.sdqr_beta2)
        rt1 = _route_queries(t1, t2, self.sdqr_beta1)

        # Exact inherited V16 decoder guidance and local refinement modules.
        d4 = self.decoder4(self._concat(self.upconv4(bottleneck), e4))
        d4 = self.refine4(self.guide4(d4, rt4))
        d3 = self.decoder3(self._concat(self.upconv3(d4), e3))
        d3 = self.refine3(self.guide3(d3, rt3))
        d2 = self.decoder2(self._concat(self.upconv2(d3), e2))
        d2 = self.refine2(self.guide2(d2, rt2))
        d1 = self.decoder1(self._concat(self.upconv1(d2), e1))
        d1 = self.refine1(self.guide1(d1, rt1))

        output = self.output_activation(self.output_conv(d1))
        output = output[..., :original_height, :original_width]
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("V19 output contains NaN/Inf")

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
            "rt4": tuple(rt4.shape),
            "rt3": tuple(rt3.shape),
            "rt2": tuple(rt2.shape),
            "rt1": tuple(rt1.shape),
            "d4": tuple(d4.shape),
            "d3": tuple(d3.shape),
            "d2": tuple(d2.shape),
            "d1": tuple(d1.shape),
            "decoder_output": tuple(d1.shape),
            "final_output": tuple(output.shape),
        }
        return output, shapes


__all__ = ["SDQRPlainUNetColorQuery", "_route_queries"]
