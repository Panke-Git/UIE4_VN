"""Independent shared_query model builder; backbone first, canonical UICF second."""
from torch import nn
from src.shared.uicf_models import UICFPreBackbone, _build_correction_module
from .backbone import SharedQueryBackbone

EXPECTED_TYPE = "plain_unet_color_query_uicf_pre_backbone_shared_query"


def build_model(config: dict) -> nn.Module:
    if config.get("type") != EXPECTED_TYPE:
        raise ValueError(f"Expected model.type={EXPECTED_TYPE}, got {config.get('type')!r}")
    keys = ("in_channels", "out_channels", "base_channels", "use_batch_norm", "output_activation")
    backbone = SharedQueryBackbone(**{key: config[key] for key in keys}, **config["color_query"])
    uicf = _build_correction_module(config["uicf"])
    return UICFPreBackbone(backbone, uicf)
