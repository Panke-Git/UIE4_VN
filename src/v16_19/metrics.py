"""Identical float-RGB PSNR and Gaussian-window SSIM for all versions."""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def per_image_psnr(prediction: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    values = 10.0 * torch.log10((data_range**2) / mse.clamp_min(torch.finfo(mse.dtype).eps))
    return torch.where(mse == 0, torch.full_like(values, float("inf")), values)


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
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


def crop_border(tensor: Tensor, border: int) -> Tensor:
    if border < 0:
        raise ValueError("crop_border must be non-negative")
    if border == 0:
        return tensor
    if tensor.shape[-2] <= 2 * border or tensor.shape[-1] <= 2 * border:
        raise ValueError("crop_border removes the entire image")
    return tensor[..., border:-border, border:-border]


def batch_metrics(prediction: Tensor, target: Tensor, config: dict) -> tuple[Tensor, Tensor]:
    prediction = crop_border(prediction, int(config["crop_border"]))
    target = crop_border(target, int(config["crop_border"]))
    psnr = per_image_psnr(prediction, target, float(config["data_range"]))
    ssim = per_image_ssim(
        prediction,
        target,
        float(config["data_range"]),
        int(config["ssim_window_size"]),
        float(config["ssim_sigma"]),
    )
    return psnr, ssim

