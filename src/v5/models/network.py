"""v5 Plain U-Net + Point-INR model factory."""

from __future__ import annotations

from torch import Tensor, nn

from .point_inr import PointINR
from .unet import PlainUNet


class PlainUNetPointINR(PlainUNet):
    """The exact v4 U-Net backbone with Point-INR before decoder upsampling."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        use_batch_norm: bool,
        output_activation: str,
        point_inr: dict,
    ) -> None:
        # Construct every v4 backbone layer first and in the same order. This
        # preserves common-backbone initialization under the same random seed.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            use_batch_norm=use_batch_norm,
            output_activation=output_activation,
        )
        self.bottleneck_channels = base_channels * 16
        self.bottleneck_module = PointINR(
            channels=self.bottleneck_channels,
            hidden_dim=int(point_inr["hidden_dim"]),
            num_frequencies=int(point_inr["num_frequencies"]),
            depth=int(point_inr["depth"]),
            include_raw_coordinate=bool(point_inr["include_raw_coordinate"]),
            query_chunk=int(point_inr["query_chunk"]),
            residual=bool(point_inr["residual"]),
        )

    def _forward_impl(self, inputs: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(f"Expected BCHW input with {self.in_channels} channels")
        padded, original_height, original_width = self._pad(inputs)

        e1 = self.encoder1(padded)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        transformed = self.bottleneck_module(bottleneck)

        d4 = self.decoder4(self._concat(self.upconv4(transformed), e4))
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
            "module_input": tuple(bottleneck.shape),
            "module_output": tuple(transformed.shape),
            "decoder_output": tuple(d1.shape),
            "final_output": tuple(output.shape),
        }
        return output, shapes


def build_model(config: dict) -> nn.Module:
    if config.get("type") != "plain_unet_point_inr":
        raise ValueError(
            f"v5 requires model.type=plain_unet_point_inr, got {config.get('type')!r}"
        )
    return PlainUNetPointINR(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        base_channels=int(config["base_channels"]),
        use_batch_norm=bool(config["use_batch_norm"]),
        output_activation=str(config["output_activation"]),
        point_inr=config["point_inr"],
    )
