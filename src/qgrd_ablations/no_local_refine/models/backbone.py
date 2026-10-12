"""Fixed no_local_refine architecture; canonical V16 modules are read-only dependencies."""

from torch import Tensor
from src.shared.color_query_unet import PlainUNetColorQuery
from src.v4.models.unet import PlainUNet


class NoLocalRefineBackbone(PlainUNetColorQuery):
    def __init__(self, **kwargs) -> None:
        # Consume exactly V16's complete initialization RNG before deleting modules.
        super().__init__(**kwargs)
        del self.refine4
        del self.refine3
        del self.refine2
        del self.refine1

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

        # no_local_refine: the sole, fixed decoder definition for this experiment.
        d4 = self.decoder4(self._concat(self.upconv4(bottleneck), e4))
        d4 = self.guide4(d4, t4)
        d3 = self.decoder3(self._concat(self.upconv3(d4), e3))
        d3 = self.guide3(d3, t3)
        d2 = self.decoder2(self._concat(self.upconv2(d3), e2))
        d2 = self.guide2(d2, t2)
        d1 = self.decoder1(self._concat(self.upconv1(d2), e1))
        d1 = self.guide1(d1, t1)

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
