"""A2-only fixed FP32 evaluation policy; training precision is independent."""
from __future__ import annotations

import torch
from torch import Tensor


def evaluation_precision(config: dict) -> str:
    value = config.get("evaluation", {}).get("precision")
    if value != "fp32":
        raise ValueError(
            "New A2 runs require evaluation.precision=fp32. Legacy AMP-evaluated "
            "runs must remain unchanged; start a new run with the updated config, "
            "do not resume or relabel the old run."
        )
    return value


def evaluation_protocol(config: dict) -> dict:
    evaluation_precision(config)
    return {
        "precision": "fp32",
        "autocast_enabled": False,
        "input_dtype": "torch.float32",
        "prediction_dtype": "torch.float32",
        "loss_dtype": "torch.float32",
        "model_weights_changed_for_evaluation": False,
    }


def fp32_context(config: dict, device: torch.device):
    evaluation_precision(config)
    # Disable surrounding autocast as well. No model casts, parameter edits,
    # fallback-on-error, clamps, or hooks are used to implement this policy.
    return torch.autocast(device_type=device.type, enabled=False)


def require_fp32(value: Tensor, name: str) -> None:
    if value.dtype != torch.float32:
        raise TypeError(f"A2 FP32 evaluation expected {name} dtype=torch.float32, got {value.dtype}")
