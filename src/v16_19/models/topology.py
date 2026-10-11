"""Frozen V16 pre-backbone wrapper."""
from torch import Tensor, nn
from .uicf_inr import UICFINROutput

class UICFPreBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, uicf: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.uicf = uicf

    def forward(self, image: Tensor) -> Tensor:
        uicf_output = self.uicf(image, return_details=False)
        if not isinstance(uicf_output, Tensor):
            raise TypeError("UICF tensor forward returned diagnostics unexpectedly")
        return self.backbone(uicf_output)

    def forward_with_uicf_details(self, image: Tensor) -> tuple[Tensor, UICFINROutput]:
        details = self.uicf(image, return_details=True)
        if not isinstance(details, UICFINROutput):
            raise TypeError("UICF diagnostics forward returned a tensor unexpectedly")
        return self.backbone(details.enhanced), details

    def forward_with_shapes(self, image: Tensor) -> tuple[Tensor, dict[str, tuple[int, ...]]]:
        output, details = self.forward_with_uicf_details(image)
        return output, {
            "input": tuple(image.shape),
            "uicf_output": tuple(details.enhanced.shape),
            "correction_field": tuple(details.correction_field.shape),
            "chromatic_anchor": tuple(details.chromatic_anchor.shape),
            "global_feature": tuple(details.global_feature.shape),
            "final_output": tuple(output.shape),
        }
