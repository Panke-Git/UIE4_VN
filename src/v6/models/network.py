"""v6 Plain U-Net + GL-INR model factory."""

from __future__ import annotations

from torch import Tensor, nn

from .glinr import GLINR
from .unet import PlainUNet


class PlainUNetGLINR(PlainUNet):
    """The exact v4 U-Net backbone with GL-INR before decoder upsampling."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        use_batch_norm: bool,
        output_activation: str,
        glinr: dict,
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
        self.bottleneck_module = GLINR(
            channels=self.bottleneck_channels,
            latent_dim=int(glinr["latent_dim"]),
            hidden_dim=int(glinr["hidden_dim"]),
            latent_stride=int(glinr["latent_stride"]),
            global_num_frequencies=int(glinr["global_num_frequencies"]),
            local_num_frequencies=int(glinr["local_num_frequencies"]),
            include_raw_absolute_coordinate=bool(
                glinr["include_raw_absolute_coordinate"]
            ),
            include_raw_relative_coordinate=bool(
                glinr["include_raw_relative_coordinate"]
            ),
            local_depth=int(glinr["local_depth"]),
            global_depth=int(glinr["global_depth"]),
            fusion_depth=int(glinr["fusion_depth"]),
            query_chunk=int(glinr["query_chunk"]),
            residual=bool(glinr["residual"]),
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
    if config.get("type") != "plain_unet_glinr":
        raise ValueError(
            f"v6 requires model.type=plain_unet_glinr, got {config.get('type')!r}"
        )
    return PlainUNetGLINR(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        base_channels=int(config["base_channels"]),
        use_batch_norm=bool(config["use_batch_norm"]),
        output_activation=str(config["output_activation"]),
        glinr=config["glinr"],
    )
