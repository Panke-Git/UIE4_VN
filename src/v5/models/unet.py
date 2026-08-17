"""Classic four-level Plain U-Net for the v4 sanity baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two plain 3x3 Conv-BatchNorm-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("DoubleConv channels must be positive")
        layers: list[nn.Module] = []
        for input_channels in (in_channels, out_channels):
            layers.append(
                nn.Conv2d(
                    input_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=not use_batch_norm,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class PlainUNet(nn.Module):
    """Standard U-Net with max pooling, transposed convolutions, and concat skips."""

    padding_factor = 16

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        use_batch_norm: bool = True,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels) <= 0:
            raise ValueError("U-Net channel counts must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.output_activation_name = output_activation.lower()

        channels = [base_channels * (2**index) for index in range(5)]
        self.encoder1 = DoubleConv(in_channels, channels[0], use_batch_norm)
        self.encoder2 = DoubleConv(channels[0], channels[1], use_batch_norm)
        self.encoder3 = DoubleConv(channels[1], channels[2], use_batch_norm)
        self.encoder4 = DoubleConv(channels[2], channels[3], use_batch_norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(channels[3], channels[4], use_batch_norm)

        self.upconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(channels[4], channels[3], use_batch_norm)
        self.upconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(channels[3], channels[2], use_batch_norm)
        self.upconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(channels[2], channels[1], use_batch_norm)
        self.upconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(channels[1], channels[0], use_batch_norm)
        self.output_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        if self.output_activation_name == "sigmoid":
            self.output_activation: nn.Module = nn.Sigmoid()
        elif self.output_activation_name in {"identity", "none"}:
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation!r}")

    def _pad(self, inputs: Tensor) -> tuple[Tensor, int, int]:
        height, width = inputs.shape[-2:]
        pad_height = (self.padding_factor - height % self.padding_factor) % self.padding_factor
        pad_width = (self.padding_factor - width % self.padding_factor) % self.padding_factor
        if not (pad_height or pad_width):
            return inputs, height, width
        mode = "reflect"
        if height < 2 or width < 2 or pad_height >= height or pad_width >= width:
            mode = "replicate"
        return F.pad(inputs, (0, pad_width, 0, pad_height), mode=mode), height, width

    @staticmethod
    def _concat(upsampled: Tensor, skip: Tensor) -> Tensor:
        if upsampled.shape[-2:] != skip.shape[-2:]:
            raise RuntimeError(
                f"U-Net skip shape mismatch: {tuple(upsampled.shape)} vs {tuple(skip.shape)}"
            )
        return torch.cat((skip, upsampled), dim=1)

    def _forward_impl(self, inputs: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(f"Expected BCHW input with {self.in_channels} channels")
        padded, original_height, original_width = self._pad(inputs)

        e1 = self.encoder1(padded)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(self._concat(self.upconv4(bottleneck), e4))
        d3 = self.decoder3(self._concat(self.upconv3(d4), e3))
        d2 = self.decoder2(self._concat(self.upconv2(d3), e2))
        d1 = self.decoder1(self._concat(self.upconv1(d2), e1))
        output = self.output_activation(self.output_conv(d1))
        output = output[..., :original_height, :original_width]

        shapes = {
            "input": tuple(inputs.shape),
            "e1": tuple(e1.shape),
            "e2": tuple(e2.shape),
            "e3": tuple(e3.shape),
            "e4": tuple(e4.shape),
            "bottleneck": tuple(bottleneck.shape),
            "decoder_output": tuple(d1.shape),
            "final_output": tuple(output.shape),
        }
        return output, shapes

    def forward(self, inputs: Tensor) -> Tensor:
        output, _ = self._forward_impl(inputs)
        return output

    def forward_with_shapes(self, inputs: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        return self._forward_impl(inputs)
