"""UIE4-compatible RGB PSNR/SSIM plus simple dataset statistics.

The PSNR and Gaussian-window SSIM definitions intentionally mirror
``src/v1/metrics.py`` without importing any experiment package at runtime.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F


def per_image_psnr(prediction: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    values = 10.0 * torch.log10((data_range**2) / mse.clamp_min(torch.finfo(mse.dtype).eps))
    return torch.where(mse == 0, torch.full_like(values, float("inf")), values)


def _gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def per_image_ssim(
    prediction: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM expects matching BCHW tensors")
    _, channels, height, width = prediction.shape
    effective = min(window_size, height, width)
    if effective % 2 == 0:
        effective -= 1
    if effective < 1:
        raise ValueError("Images must have non-empty spatial dimensions")
    kernel = _gaussian_kernel(effective, sigma, channels, prediction.device, prediction.dtype)
    padding = effective // 2
    mu_x = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = F.conv2d(prediction.square(), kernel, padding=padding, groups=channels) - mu_x2
    sigma_y = F.conv2d(target.square(), kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mu_xy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    return score.flatten(1).mean(dim=1)


def pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


@torch.inference_mode()
def paired_metrics(
    input_image: Image.Image,
    gt_image: Image.Image,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    crop_border: int = 0,
) -> dict[str, float]:
    input_tensor = pil_to_tensor(input_image)
    gt_tensor = pil_to_tensor(gt_image)
    if input_tensor.shape != gt_tensor.shape:
        raise ValueError(f"Metric images must have matching shapes: {input_tensor.shape} vs {gt_tensor.shape}")
    if crop_border < 0:
        raise ValueError("crop_border must be non-negative")
    if crop_border:
        if input_tensor.shape[-2] <= 2 * crop_border or input_tensor.shape[-1] <= 2 * crop_border:
            raise ValueError("crop_border removes the entire image")
        input_tensor = input_tensor[..., crop_border:-crop_border, crop_border:-crop_border]
        gt_tensor = gt_tensor[..., crop_border:-crop_border, crop_border:-crop_border]
    difference = input_tensor - gt_tensor
    mse = difference.square().mean()
    return {
        "psnr": float(per_image_psnr(input_tensor, gt_tensor, data_range).item()),
        "ssim": float(per_image_ssim(input_tensor, gt_tensor, data_range, window_size, sigma).item()),
        "mae": float(difference.abs().mean().item()),
        "mse": float(mse.item()),
    }


def resize_for_evaluation(image: Image.Image, size: int) -> Image.Image:
    """Match LSUIDataset evaluation: deterministic square PIL bilinear resize."""
    return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def image_statistics(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    flat = rgb.reshape(-1, 3)
    means = flat.mean(axis=0, dtype=np.float64)
    stds = flat.std(axis=0, dtype=np.float64)
    # ITU-R BT.709 RGB luminance coefficients, applied to RGB in [0, 1].
    luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "mean_luminance": float(luminance.mean(dtype=np.float64)),
        "luminance_std": float(luminance.std(dtype=np.float64)),
        "mean_saturation": float(saturation.mean(dtype=np.float64)),
    }


@torch.inference_mode()
def psnr_and_mae_128(image_a: Image.Image, image_b: Image.Image) -> tuple[float, float]:
    resized_a = resize_for_evaluation(image_a, 128)
    resized_b = resize_for_evaluation(image_b, 128)
    tensor_a, tensor_b = pil_to_tensor(resized_a), pil_to_tensor(resized_b)
    psnr = float(per_image_psnr(tensor_a, tensor_b).item())
    mae = float((tensor_a - tensor_b).abs().mean().item())
    return psnr, mae
