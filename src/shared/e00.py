"""Shared float-sRGB CIEDE2000 evaluation for every experiment version."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

try:
    import skimage
    from skimage.color import deltaE_ciede2000, rgb2lab
except ImportError as error:  # pragma: no cover - exercised only in an incomplete environment.
    raise ImportError(
        "CIEDE2000 evaluation requires scikit-image; install requirements.txt"
    ) from error


def _validate_lab_pair(lab1: np.ndarray, lab2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(lab1, dtype=np.float64)
    second = np.asarray(lab2, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"CIEDE2000 expects matching Lab arrays, got {first.shape} and {second.shape}")
    if first.ndim < 1 or first.shape[-1] != 3:
        raise ValueError(f"CIEDE2000 expects Lab channels on the last axis, got {first.shape}")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise FloatingPointError("CIEDE2000 received non-finite Lab values")
    return first, second


def delta_e00_from_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Evaluate the shared CIEDE2000 core on last-axis CIE Lab arrays."""
    first, second = _validate_lab_pair(lab1, lab2)
    values = np.asarray(
        deltaE_ciede2000(
            first,
            second,
            kL=1.0,
            kC=1.0,
            kH=1.0,
            channel_axis=-1,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("CIEDE2000 produced non-finite values")
    if np.any(values < 0.0):
        raise FloatingPointError("CIEDE2000 produced a negative color difference")
    return values


def _validate_rgb_pair(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("E00 expects PyTorch tensors")
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "E00 expects matching BCHW tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape[1] != 3:
        raise ValueError(f"E00 expects RGB tensors with three channels, got C={prediction.shape[1]}")
    if prediction.shape[0] < 1 or prediction.shape[-2] < 1 or prediction.shape[-1] < 1:
        raise ValueError("E00 expects non-empty batch and spatial dimensions")
    prediction_float = prediction.detach().to(device="cpu", dtype=torch.float32)
    target_float = target.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(prediction_float).all() or not torch.isfinite(target_float).all():
        raise FloatingPointError("E00 received non-finite RGB values")
    if bool((target_float < 0.0).any()) or bool((target_float > 1.0).any()):
        raise ValueError("E00 target must be standard sRGB in [0,1]")
    return prediction_float.clamp(0.0, 1.0), target_float


def _crop_once(tensor: Tensor, border: int) -> Tensor:
    if border < 0:
        raise ValueError("crop_border must be non-negative")
    if border == 0:
        return tensor
    if tensor.shape[-2] <= 2 * border or tensor.shape[-1] <= 2 * border:
        raise ValueError("crop_border removes the entire image")
    return tensor[..., border:-border, border:-border]


def batch_delta_e00(prediction: Tensor, target: Tensor, config: dict[str, Any]) -> Tensor:
    """Return one mean per-pixel ΔE00 value per image as a CPU float64 tensor.

    Prediction is converted to float32 and clamped to [0,1] before a single
    configured border crop. Both RGB tensors are converted from BCHW to BHWC
    float64 sRGB, transformed to CIE Lab (D65/2°), and evaluated pixelwise.
    """
    prediction_float, target_float = _validate_rgb_pair(prediction, target)
    border = int(config.get("crop_border", 0))
    prediction_float = _crop_once(prediction_float, border)
    target_float = _crop_once(target_float, border)
    prediction_rgb = np.asarray(
        prediction_float.permute(0, 2, 3, 1).contiguous().numpy(), dtype=np.float64
    )
    target_rgb = np.asarray(
        target_float.permute(0, 2, 3, 1).contiguous().numpy(), dtype=np.float64
    )
    prediction_lab = rgb2lab(
        prediction_rgb, illuminant="D65", observer="2", channel_axis=-1
    )
    target_lab = rgb2lab(target_rgb, illuminant="D65", observer="2", channel_axis=-1)
    pixel_values = delta_e00_from_lab(prediction_lab, target_lab)
    expected_shape = prediction_rgb.shape[:-1]
    if pixel_values.shape != expected_shape:
        raise RuntimeError(
            f"CIEDE2000 returned shape {pixel_values.shape}, expected {expected_shape}"
        )
    per_image = pixel_values.mean(axis=(1, 2), dtype=np.float64)
    if per_image.shape != (prediction.shape[0],):
        raise RuntimeError(f"E00 aggregation returned unexpected shape {per_image.shape}")
    if not np.isfinite(per_image).all() or np.any(per_image < 0.0):
        raise FloatingPointError("Mean per-image E00 is non-finite or negative")
    return torch.from_numpy(np.ascontiguousarray(per_image))


def e00_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Serializable definition embedded in test summaries for auditability."""
    return {
        "metric": "Delta E 2000 (CIEDE2000)",
        "direction": "lower_is_better",
        "implementation": "skimage.color.rgb2lab + skimage.color.deltaE_ciede2000",
        "scikit_image_version": skimage.__version__,
        "rgb_interpretation": "sRGB float prediction clamped to [0,1] before PNG encoding",
        "lab_illuminant": "D65",
        "lab_observer": "2 degrees",
        "kL": 1.0,
        "kC": 1.0,
        "kH": 1.0,
        "crop_border": int(config.get("crop_border", 0)),
        "pixel_aggregation": "arithmetic mean over the per-pixel CIEDE2000 map",
        "dataset_aggregation": "arithmetic mean of per-image means; every image has equal weight",
        "channel_layout": "PyTorch BCHW converted explicitly to scikit-image BHWC",
        "precision": "autocast disabled; RGB float32 on CPU, color conversion and CIEDE2000 float64",
    }
