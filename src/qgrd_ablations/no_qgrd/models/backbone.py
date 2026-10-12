"""Fixed no_qgrd architecture; canonical V16 modules are read-only dependencies."""

from torch import Tensor
from src.shared.color_query_unet import PlainUNetColorQuery
from src.v4.models.unet import PlainUNet


class NoQGRDBackbone(PlainUNetColorQuery):
    def __init__(self, **kwargs) -> None:
        # Consume exactly V16's complete initialization RNG before deleting modules.
        super().__init__(**kwargs)
        del self.base_queries
        del self.feature_to_token
        del self.token_refinement
        del self.guide4
        del self.guide3
        del self.guide2
        del self.guide1
        del self.refine4
        del self.refine3
        del self.refine2
        del self.refine1

    def _forward_impl(self, inputs: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        # Explicit standard U-Net path: no query, guidance, or local-refine calls.
        return PlainUNet._forward_impl(self, inputs)
