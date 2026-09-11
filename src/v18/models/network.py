"""V18 factory: V16 topology with NLQC decoder guidance wrappers."""

from __future__ import annotations

from torch import nn

from src.shared.color_query_unet import PlainUNetColorQuery
from src.shared.uicf_models import (
    UICFPreBackbone,
    build_uicf_color_query_unet_experiment_model,
)

from .nlqc_guidance import NLQCSpatialTokenGuidance


def build_model(config: dict) -> nn.Module:
    model = build_uicf_color_query_unet_experiment_model(
        config,
        expected_type="plain_unet_color_query_uicf_pre_backbone_nlqc",
        placement="pre",
    )
    if not isinstance(model, UICFPreBackbone):
        raise TypeError("V18 requires the canonical UICFPreBackbone wrapper")
    if not isinstance(model.backbone, PlainUNetColorQuery):
        raise TypeError("V18 requires the shared PlainUNetColorQuery backbone")
    nlqc = config.get("nlqc")
    if not isinstance(nlqc, dict):
        raise ValueError("V18 model config requires an nlqc mapping")
    required = {"kernel_lambda", "epsilon", "kernel_chunk_size", "alpha_init"}
    missing = sorted(required - set(nlqc))
    if missing:
        raise ValueError(f"V18 nlqc config is missing keys: {missing}")

    for stage in (4, 3, 2, 1):
        base_guidance = getattr(model.backbone, f"guide{stage}")
        setattr(
            model.backbone,
            f"guide{stage}",
            NLQCSpatialTokenGuidance(
                base_guidance,
                kernel_lambda=float(nlqc["kernel_lambda"]),
                epsilon=float(nlqc["epsilon"]),
                kernel_chunk_size=nlqc["kernel_chunk_size"],
                alpha_init=float(nlqc["alpha_init"]),
            ),
        )
    return model
