"""Independent NAFNet-small-style encoder/decoder implementation."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Per-pixel layer normalization over channels for BCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Nonlinear-activation-free restoration block."""

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        depthwise_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, depthwise_channels, 1)
        self.conv2 = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            3,
            padding=1,
            groups=depthwise_channels,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(depthwise_channels // 2, depthwise_channels // 2, 1),
        )
        self.conv3 = nn.Conv2d(depthwise_channels // 2, channels, 1)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # The standard NAFNet residual scaling initialization starts blocks as identity.
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        branch = self.conv1(self.norm1(x))
        branch = self.conv2(branch)
        branch = self.sg(branch)
        branch = branch * self.sca(branch)
        branch = self.dropout1(self.conv3(branch))
        y = x + branch * self.beta

        branch = self.conv4(self.norm2(y))
        branch = self.sg(branch)
        branch = self.dropout2(self.conv5(branch))
        return y + branch * self.gamma


class NAFNet(nn.Module):
    """NAF-style encoder/decoder with a replaceable phase-one bottleneck."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 0,
        enc_blk_nums: Sequence[int] = (2, 2, 2),
        dec_blk_nums: Sequence[int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("Encoder and decoder stage counts must match")
        if middle_blk_num < 0:
            raise ValueError("middle_blk_num must be non-negative")
        self.img_channel = img_channel
        self.width = width
        self.enc_blk_nums = tuple(enc_blk_nums)
        self.dec_blk_nums = tuple(dec_blk_nums)
        self.middle_blk_num = middle_blk_num
        self.padder_size = 2 ** len(enc_blk_nums)

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = width
        for blocks in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(blocks)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.bottleneck_channels = channels
        # Phase-one configs deliberately set this to zero so the encoder output
        # goes directly through Identity, Point-INR, or GL-INR before decoding.
        self.middle_blks = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blk_num)])

        for blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(blocks)]))

        # Experiment-specific modules are installed by network.py only after the
        # complete common backbone has consumed its initialization RNG sequence.
        self.bottleneck_module = nn.Identity()

    def _pad(self, x: Tensor) -> tuple[Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h)), height, width

    def encode(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        features = self.intro(x)
        skips: list[Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        return self.middle_blks(features), skips

    def decode(self, features: Tensor, skips: list[Tensor]) -> Tensor:
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = up(features) + skip
            features = decoder(features)
        return self.ending(features)

    def forward_with_shapes(self, x: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        padded, height, width = self._pad(x)
        bottleneck, skips = self.encode(padded)
        transformed = self.bottleneck_module(bottleneck)
        decoded = self.decode(transformed, skips)
        output = decoded + padded
        shapes = {
            "input": tuple(x.shape),
            "encoder_output": tuple(bottleneck.shape),
            "bottleneck": tuple(bottleneck.shape),
            "module_input": tuple(bottleneck.shape),
            "module_output": tuple(transformed.shape),
            "decoder_output": tuple(decoded[..., :height, :width].shape),
            "output": tuple(output[..., :height, :width].shape),
        }
        return output[..., :height, :width], shapes

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self.forward_with_shapes(x)
        return output
