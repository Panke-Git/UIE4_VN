#!/usr/bin/env python3
"""Measure whether v16 UICF corrections align with paired restoration demand.

The three quantities in this analysis are deliberately kept distinct:

* ``R(x)`` is the learned UICF correction *coefficient* field.
* ``delta_uicf = I_c - I = R(x) * (I - b)`` is the actual RGB correction
  induced by UICF before the backbone.
* ``delta_gt = Y - I`` is the paired-reference restoration residual.

All quantitative spatial-alignment metrics use ``delta_uicf``.  The raw
coefficient field is reported only as a separately named supplementary
diagnostic.  Visualization normalization is never used in a metric.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.e00 import batch_delta_e00, delta_e00_from_lab, e00_protocol
from src.shared.uicf_inr import UICFINROutput
from src.shared.uicf_models import UICFPreBackbone
from src.v16.dataset import LSUIDataset, ManifestEntry, validate_split_protocol
from src.v16.metrics import batch_metrics
from src.v16.test import _checkpoint_path, _torch_load
from src.v16.utils import (
    atomic_json,
    load_yaml,
    project_path,
    seed_everything,
    select_device,
    sha256_file,
    tensor_to_image,
)
from tools.visualize_v16_uicf import (
    CapturedSample,
    _safe_sample_id,
    assert_uicf_consistency,
    build_and_load_v16_model,
    capture_sample,
)


SCRIPT_VERSION = "1.0"
EXPECTED_VERSION = "v16"
EPSILON = 1e-12
PER_SAMPLE_FIELDS = [
    "sample_index",
    "sample_id",
    "filename",
    "psnr",
    "ssim",
    "e00",
    "mean_gt_rgb_demand",
    "mean_gt_e00_demand",
    "mean_uicf_effect",
    "spearman_rgb",
    "pearson_rgb",
    "top20_iou_rgb",
    "top20_precision_rgb",
    "top20_recall_rgb",
    "direction_cosine_top20_gt",
    "spearman_e00",
    "pearson_e00",
    "raw_field_spearman_rgb",
    "null_valid_shift_count",
    "null_pearson_valid_shift_count",
    "null_top20_valid_shift_count",
    "null_direction_valid_shift_count",
    "null_spearman_rgb_mean",
    "null_spearman_rgb_std",
    "null_pearson_rgb_mean",
    "null_pearson_rgb_std",
    "null_top20_iou_rgb_mean",
    "null_top20_iou_rgb_std",
    "null_direction_cosine_mean",
    "null_direction_cosine_std",
    "spearman_gain_over_null",
    "top20_iou_gain_over_null",
    "direction_gain_over_null",
    "b_r",
    "b_g",
    "b_b",
    "raw_field_mean_abs",
    "raw_field_std",
]
FAILED_FIELDS = ["sample_index", "sample_id", "filename", "error_type", "error"]
NULL_FIELDS = [
    "sample_index",
    "sample_id",
    "filename",
    "spearman_rgb",
    "null_spearman_rgb_mean",
    "spearman_gain_over_null",
    "pearson_rgb",
    "null_pearson_rgb_mean",
    "top20_iou_rgb",
    "null_top20_iou_rgb_mean",
    "top20_iou_gain_over_null",
    "direction_cosine_top20_gt",
    "null_direction_cosine_mean",
    "direction_gain_over_null",
    "null_valid_shift_count",
    "null_pearson_valid_shift_count",
    "null_top20_valid_shift_count",
    "null_direction_valid_shift_count",
]


@dataclass(frozen=True)
class OptionalMetric:
    """A finite metric value, or an explicit undefined state."""

    value: float | None
    valid: bool


class SafeAlignmentDataset(Dataset[dict[str, Any]]):
    """Preserve manifest identity when an individual paired-TSV item fails to load."""

    def __init__(self, dataset: LSUIDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.dataset.entries[index]
        try:
            if (
                Path(entry.input_relative).stem != entry.sample_id
                or Path(entry.gt_relative).stem != entry.sample_id
            ):
                raise RuntimeError(
                    "Manifest pair identity mismatch: input/GT stems must equal sample_id"
                )
            item = self.dataset[index]
        except Exception as error:
            return {
                "ok": False,
                "index": index,
                "sample_id": entry.sample_id,
                "filename": Path(entry.input_relative).name,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        return {"ok": True, "index": index, **item}


def _list_collate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze v16 UICF spatial restoration-demand alignment"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config AMP for inference (AMP remains CUDA-only)",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--num-null-shifts", type=int, default=20)
    parser.add_argument("--null-seed", type=int, default=3520)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=3520)
    parser.add_argument("--viz-k", type=int, default=12)
    parser.add_argument("--robust-percentile", type=float, default=99.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if not 0.0 < args.top_fraction < 1.0:
        raise ValueError("--top-fraction must be in (0,1)")
    if args.num_null_shifts < 1:
        raise ValueError("--num-null-shifts must be at least 1")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be at least 1")
    if args.viz_k < 1:
        raise ValueError("--viz-k must be at least 1")
    if not 0.0 < args.robust_percentile <= 100.0:
        raise ValueError("--robust-percentile must be in (0,100]")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")


def _one_dimensional_finite_pair(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("Correlation expects equally sized non-empty arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError("Correlation inputs must be finite")
    return x, y


def average_rank(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return deterministic one-based average ranks, including exact ties."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Average ranks require a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise FloatingPointError("Average-rank input must be finite")
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson_correlation(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> OptionalMetric:
    """Pearson correlation with explicit undefined handling for constant maps."""

    x, y = _one_dimensional_finite_pair(first, second)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return OptionalMetric(None, False)
    centered_x = x - x.mean(dtype=np.float64)
    centered_y = y - y.mean(dtype=np.float64)
    denominator = float(np.linalg.norm(centered_x) * np.linalg.norm(centered_y))
    if denominator == 0.0:
        return OptionalMetric(None, False)
    value = float(np.dot(centered_x, centered_y) / denominator)
    if not math.isfinite(value):
        return OptionalMetric(None, False)
    return OptionalMetric(float(np.clip(value, -1.0, 1.0)), True)


def spearman_correlation(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> OptionalMetric:
    """Spearman correlation using deterministic average-tie ranks."""

    x, y = _one_dimensional_finite_pair(first, second)
    return pearson_correlation(average_rank(x), average_rank(y))


def exact_top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    """Select exactly ``ceil(N*fraction)`` values with stable tie handling."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Top-fraction selection requires at least one value")
    if not 0.0 < fraction < 1.0:
        raise ValueError("top fraction must be in (0,1)")
    if not np.isfinite(array).all():
        raise FloatingPointError("Top-fraction input must be finite")
    count = int(math.ceil(array.size * fraction))
    order = np.argsort(-array.reshape(-1), kind="mergesort")
    mask = np.zeros(array.size, dtype=bool)
    mask[order[:count]] = True
    return mask.reshape(array.shape)


def top_fraction_overlap(
    prediction: np.ndarray, target: np.ndarray, fraction: float
) -> tuple[float, float, float]:
    """Return exact top-fraction IoU, precision, and recall."""

    if np.asarray(prediction).shape != np.asarray(target).shape:
        raise ValueError("Top-fraction overlap expects maps with identical shapes")
    predicted = exact_top_fraction_mask(prediction, fraction)
    reference = exact_top_fraction_mask(target, fraction)
    intersection = int(np.logical_and(predicted, reference).sum())
    union = int(np.logical_or(predicted, reference).sum())
    selected_prediction = int(predicted.sum())
    selected_reference = int(reference.sum())
    return (
        intersection / union,
        intersection / selected_prediction,
        intersection / selected_reference,
    )


def patch_average_pool(
    values: np.ndarray, patch_size: int, *, sample_id: str
) -> np.ndarray:
    """Non-overlapping patch average for an ``[H,W]`` or ``[C,H,W]`` map."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (2, 3):
        raise ValueError(f"Patch pooling expects [H,W] or [C,H,W], got {array.shape}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"sample_id={sample_id}: patch-pooling input is non-finite")
    height, width = array.shape[-2:]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"sample_id={sample_id}: H={height} W={width} are not divisible by "
            f"patch_size={patch_size}"
        )
    grid_h, grid_w = height // patch_size, width // patch_size
    if array.ndim == 2:
        return array.reshape(grid_h, patch_size, grid_w, patch_size).mean(
            axis=(1, 3), dtype=np.float64
        )
    channels = array.shape[0]
    return array.reshape(channels, grid_h, patch_size, grid_w, patch_size).mean(
        axis=(2, 4), dtype=np.float64
    )


def direction_cosine_top_demand(
    pooled_delta_uicf: np.ndarray,
    pooled_delta_gt: np.ndarray,
    pooled_gt_magnitude: np.ndarray,
    top_fraction: float,
    *,
    eps: float = EPSILON,
) -> OptionalMetric:
    """Mean RGB-vector cosine on high-GT-demand patches with nonzero vectors."""

    uicf = np.asarray(pooled_delta_uicf, dtype=np.float64)
    target = np.asarray(pooled_delta_gt, dtype=np.float64)
    if uicf.shape != target.shape or uicf.ndim != 3 or uicf.shape[0] != 3:
        raise ValueError("Direction alignment expects matching [3,H_grid,W_grid] arrays")
    if tuple(np.asarray(pooled_gt_magnitude).shape) != tuple(uicf.shape[-2:]):
        raise ValueError("GT magnitude shape must match the vector-field patch grid")
    if not np.isfinite(uicf).all() or not np.isfinite(target).all():
        raise FloatingPointError("Direction-alignment vectors must be finite")
    uicf_norm = np.linalg.norm(uicf, axis=0)
    target_norm = np.linalg.norm(target, axis=0)
    top_mask = exact_top_fraction_mask(pooled_gt_magnitude, top_fraction)
    valid = top_mask & (uicf_norm > eps) & (target_norm > eps)
    if not bool(valid.any()):
        return OptionalMetric(None, False)
    dot = np.sum(uicf * target, axis=0)
    cosine = dot[valid] / (uicf_norm[valid] * target_norm[valid])
    value = float(np.clip(cosine, -1.0, 1.0).mean(dtype=np.float64))
    if not math.isfinite(value):
        return OptionalMetric(None, False)
    return OptionalMetric(value, True)


def per_pixel_delta_e00_map(input_tensor: Tensor, target_tensor: Tensor) -> np.ndarray:
    """Return the paired perceptual discrepancy map ``DeltaE00(I(p),Y(p))``.

    Inputs are standard sRGB ``[3,H,W]`` tensors in ``[0,1]``.  They are
    converted to last-channel float64 CIE Lab under D65/2 degrees, then passed
    through the same :func:`src.shared.e00.delta_e00_from_lab` path used by
    model evaluation.  The returned map is not a physical degradation map.
    """

    if not isinstance(input_tensor, Tensor) or not isinstance(target_tensor, Tensor):
        raise TypeError("Per-pixel E00 expects PyTorch tensors")
    if input_tensor.shape != target_tensor.shape or input_tensor.ndim != 3:
        raise ValueError(
            "Per-pixel E00 expects matching [3,H,W] tensors, got "
            f"{tuple(input_tensor.shape)} and {tuple(target_tensor.shape)}"
        )
    if input_tensor.shape[0] != 3 or input_tensor.shape[-2] < 1 or input_tensor.shape[-1] < 1:
        raise ValueError("Per-pixel E00 expects non-empty three-channel RGB images")
    first = input_tensor.detach().to(device="cpu", dtype=torch.float32)
    second = target_tensor.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise FloatingPointError("Per-pixel E00 received non-finite RGB values")
    if bool((first < 0.0).any()) or bool((first > 1.0).any()):
        raise ValueError("Per-pixel E00 input must be sRGB in [0,1]")
    if bool((second < 0.0).any()) or bool((second > 1.0).any()):
        raise ValueError("Per-pixel E00 target must be sRGB in [0,1]")
    first_rgb = np.asarray(first.permute(1, 2, 0).contiguous().numpy(), dtype=np.float64)
    second_rgb = np.asarray(second.permute(1, 2, 0).contiguous().numpy(), dtype=np.float64)
    if np.array_equal(first_rgb, second_rgb):
        return np.zeros(first_rgb.shape[:2], dtype=np.float64)
    first_lab = rgb2lab(first_rgb, illuminant="D65", observer="2", channel_axis=-1)
    second_lab = rgb2lab(second_rgb, illuminant="D65", observer="2", channel_axis=-1)
    values = delta_e00_from_lab(first_lab, second_lab)
    if values.shape != first_rgb.shape[:2]:
        raise RuntimeError(f"Per-pixel E00 returned shape {values.shape}, expected {first_rgb.shape[:2]}")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise FloatingPointError("Per-pixel E00 produced a non-finite or negative value")
    return values


def generate_null_shifts(
    grid_height: int,
    grid_width: int,
    count: int,
    seed: int,
    sample_index: int,
) -> list[tuple[int, int]]:
    """Generate deterministic, nonzero, large circular shifts for one image."""

    if grid_height < 1 or grid_width < 1 or count < 1:
        raise ValueError("Null shifts require positive grid dimensions and count")
    candidates: list[tuple[int, int]] = []
    threshold_y, threshold_x = grid_height // 4, grid_width // 4
    for raw_y in range(grid_height):
        for raw_x in range(grid_width):
            if raw_y == 0 and raw_x == 0:
                continue
            dy = raw_y if raw_y <= grid_height // 2 else raw_y - grid_height
            dx = raw_x if raw_x <= grid_width // 2 else raw_x - grid_width
            if abs(dy) >= threshold_y or abs(dx) >= threshold_x:
                candidates.append((dy, dx))
    # A 1x1 patch grid has no permissible spatial correspondence control.
    # Returning no shifts lets the caller record null metrics as explicitly
    # undefined (count=0, JSON null, CSV blank) without losing the sample.
    if not candidates:
        return []
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_index]))
    shifts: list[tuple[int, int]] = []
    while len(shifts) < count:
        order = rng.permutation(len(candidates))
        shifts.extend(candidates[int(index)] for index in order)
    return shifts[:count]


def _optional_summary(values: Sequence[float]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError("Null-control aggregation received non-finite values")
    return float(array.mean()), float(array.std(ddof=0)), int(array.size)


def analyze_spatial_maps(
    input_tensor: Tensor,
    target_tensor: Tensor,
    enhanced_tensor: Tensor,
    correction_field: Tensor,
    *,
    sample_id: str,
    sample_index: int,
    patch_size: int,
    top_fraction: float,
    num_null_shifts: int,
    null_seed: int,
) -> tuple[dict[str, float | int | None], dict[str, np.ndarray]]:
    """Compute patch-level alignment from actual UICF and GT RGB residuals."""

    tensors = (input_tensor, target_tensor, enhanced_tensor, correction_field)
    if any(tensor.ndim != 3 or tensor.shape[0] != 3 for tensor in tensors):
        raise ValueError("Alignment expects input, target, I_c, and R as [3,H,W]")
    if any(tuple(tensor.shape) != tuple(input_tensor.shape) for tensor in tensors[1:]):
        raise ValueError("Alignment input, target, I_c, and R shapes must match")
    arrays = [tensor.detach().to(device="cpu", dtype=torch.float32).numpy() for tensor in tensors]
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError(f"sample_id={sample_id}: alignment tensor is non-finite")
    inputs, targets, enhanced, raw_field = (array.astype(np.float64) for array in arrays)
    delta_gt = targets - inputs
    delta_uicf = enhanced - inputs
    gt_rgb_demand = np.linalg.norm(delta_gt, axis=0)
    uicf_effect = np.linalg.norm(delta_uicf, axis=0)
    raw_field_magnitude = np.linalg.norm(raw_field, axis=0)
    gt_e00_demand = per_pixel_delta_e00_map(input_tensor, target_tensor)

    pooled_gt_rgb = patch_average_pool(gt_rgb_demand, patch_size, sample_id=sample_id)
    pooled_uicf = patch_average_pool(uicf_effect, patch_size, sample_id=sample_id)
    pooled_gt_e00 = patch_average_pool(gt_e00_demand, patch_size, sample_id=sample_id)
    pooled_raw = patch_average_pool(raw_field_magnitude, patch_size, sample_id=sample_id)
    pooled_delta_gt = patch_average_pool(delta_gt, patch_size, sample_id=sample_id)
    pooled_delta_uicf = patch_average_pool(delta_uicf, patch_size, sample_id=sample_id)

    spearman_rgb = spearman_correlation(pooled_uicf, pooled_gt_rgb)
    pearson_rgb = pearson_correlation(pooled_uicf, pooled_gt_rgb)
    spearman_e00 = spearman_correlation(pooled_uicf, pooled_gt_e00)
    pearson_e00 = pearson_correlation(pooled_uicf, pooled_gt_e00)
    raw_spearman = spearman_correlation(pooled_raw, pooled_gt_rgb)
    iou, precision, recall = top_fraction_overlap(pooled_uicf, pooled_gt_rgb, top_fraction)
    direction = direction_cosine_top_demand(
        pooled_delta_uicf, pooled_delta_gt, pooled_gt_rgb, top_fraction
    )

    shifts = generate_null_shifts(
        pooled_uicf.shape[0], pooled_uicf.shape[1], num_null_shifts, null_seed, sample_index
    )
    null_spearman: list[float] = []
    null_pearson: list[float] = []
    null_iou: list[float] = []
    null_direction: list[float] = []
    for dy, dx in shifts:
        shifted_uicf = np.roll(pooled_uicf, shift=(dy, dx), axis=(0, 1))
        shifted_vectors = np.roll(pooled_delta_uicf, shift=(dy, dx), axis=(1, 2))
        shifted_spearman = spearman_correlation(shifted_uicf, pooled_gt_rgb)
        shifted_pearson = pearson_correlation(shifted_uicf, pooled_gt_rgb)
        shifted_iou, _, _ = top_fraction_overlap(shifted_uicf, pooled_gt_rgb, top_fraction)
        shifted_direction = direction_cosine_top_demand(
            shifted_vectors, pooled_delta_gt, pooled_gt_rgb, top_fraction
        )
        if shifted_spearman.valid:
            null_spearman.append(float(shifted_spearman.value))
        if shifted_pearson.valid:
            null_pearson.append(float(shifted_pearson.value))
        null_iou.append(float(shifted_iou))
        if shifted_direction.valid:
            null_direction.append(float(shifted_direction.value))

    null_spearman_mean, null_spearman_std, null_spearman_count = _optional_summary(null_spearman)
    null_pearson_mean, null_pearson_std, null_pearson_count = _optional_summary(null_pearson)
    null_iou_mean, null_iou_std, null_iou_count = _optional_summary(null_iou)
    null_direction_mean, null_direction_std, null_direction_count = _optional_summary(null_direction)

    def gain(real: OptionalMetric | float, null_mean: float | None) -> float | None:
        real_value = real if isinstance(real, float) else real.value
        return None if real_value is None or null_mean is None else float(real_value - null_mean)

    row: dict[str, float | int | None] = {
        "mean_gt_rgb_demand": float(gt_rgb_demand.mean(dtype=np.float64)),
        "mean_gt_e00_demand": float(gt_e00_demand.mean(dtype=np.float64)),
        "mean_uicf_effect": float(uicf_effect.mean(dtype=np.float64)),
        "spearman_rgb": spearman_rgb.value,
        "pearson_rgb": pearson_rgb.value,
        "top20_iou_rgb": float(iou),
        "top20_precision_rgb": float(precision),
        "top20_recall_rgb": float(recall),
        "direction_cosine_top20_gt": direction.value,
        "spearman_e00": spearman_e00.value,
        "pearson_e00": pearson_e00.value,
        "raw_field_spearman_rgb": raw_spearman.value,
        "null_valid_shift_count": null_spearman_count,
        "null_pearson_valid_shift_count": null_pearson_count,
        "null_top20_valid_shift_count": null_iou_count,
        "null_direction_valid_shift_count": null_direction_count,
        "null_spearman_rgb_mean": null_spearman_mean,
        "null_spearman_rgb_std": null_spearman_std,
        "null_pearson_rgb_mean": null_pearson_mean,
        "null_pearson_rgb_std": null_pearson_std,
        "null_top20_iou_rgb_mean": null_iou_mean,
        "null_top20_iou_rgb_std": null_iou_std,
        "null_direction_cosine_mean": null_direction_mean,
        "null_direction_cosine_std": null_direction_std,
        "spearman_gain_over_null": gain(spearman_rgb, null_spearman_mean),
        "top20_iou_gain_over_null": gain(float(iou), null_iou_mean),
        "direction_gain_over_null": gain(direction, null_direction_mean),
        "raw_field_mean_abs": float(np.abs(raw_field).mean(dtype=np.float64)),
        "raw_field_std": float(raw_field.std(dtype=np.float64)),
    }
    maps = {
        "gt_rgb_demand": gt_rgb_demand,
        "uicf_effect_magnitude": uicf_effect,
        "gt_e00_demand": gt_e00_demand,
        "pooled_gt_rgb": pooled_gt_rgb,
        "pooled_uicf_effect": pooled_uicf,
    }
    return row, maps


def _failure_row(entry: ManifestEntry, index: int, error: Exception | str) -> dict[str, Any]:
    return {
        "sample_index": index,
        "sample_id": entry.sample_id,
        "filename": Path(entry.input_relative).name,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _finite_or_none(value: Tensor | float) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    for row in rows:
        for value in row.values():
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                raise FloatingPointError(f"Refusing to serialize non-finite CSV value in {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _group_valid_items(items: Sequence[dict[str, Any]]) -> tuple[dict[tuple[int, ...], list[dict[str, Any]]], list[dict[str, Any]]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    invalid: list[dict[str, Any]] = []
    for item in items:
        input_tensor, target_tensor = item.get("input"), item.get("target")
        if not isinstance(input_tensor, Tensor) or not isinstance(target_tensor, Tensor):
            item["shape_error"] = "Dataset input/target is not a tensor"
            invalid.append(item)
        elif input_tensor.ndim != 3 or input_tensor.shape[0] != 3:
            item["shape_error"] = f"Input must be [3,H,W], got {tuple(input_tensor.shape)}"
            invalid.append(item)
        elif tuple(target_tensor.shape) != tuple(input_tensor.shape):
            item["shape_error"] = (
                f"GT shape {tuple(target_tensor.shape)} differs from input {tuple(input_tensor.shape)}"
            )
            invalid.append(item)
        else:
            groups.setdefault(tuple(input_tensor.shape), []).append(item)
    return groups, invalid


def _slice_details(details: UICFINROutput, position: int) -> UICFINROutput:
    selection = slice(position, position + 1)
    return UICFINROutput(
        enhanced=details.enhanced[selection],
        correction_field=details.correction_field[selection],
        chromatic_anchor=details.chromatic_anchor[selection],
        global_feature=details.global_feature[selection],
    )


def evaluate_test_set(
    model: UICFPreBackbone,
    dataset: LSUIDataset,
    entries: Sequence[ManifestEntry],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    amp_enabled: bool,
    metric_config: Mapping[str, Any],
    patch_size: int,
    top_fraction: float,
    num_null_shifts: int,
    null_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate every manifest row and return explicit success/failure records."""

    loader = DataLoader(
        SafeAlignmentDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_list_collate,
    )
    rows: list[dict[str, Any]] = []
    failures_by_index: dict[int, dict[str, Any]] = {}
    normal_forward_checked = False
    forward = getattr(model, "forward_with_uicf_details", None)
    if not callable(forward):
        raise TypeError("v16 model lacks forward_with_uicf_details")
    progress = tqdm(total=len(entries), desc="Analyzing v16 UICF alignment", unit="sample")
    for loaded_items in loader:
        progress.update(len(loaded_items))
        valid_items: list[dict[str, Any]] = []
        for item in loaded_items:
            index = int(item["index"])
            if not item["ok"]:
                failures_by_index[index] = {
                    "sample_index": index,
                    "sample_id": item["sample_id"],
                    "filename": item["filename"],
                    "error_type": item["error_type"],
                    "error": item["error"],
                }
            else:
                valid_items.append(item)
        groups, invalid_items = _group_valid_items(valid_items)
        for item in invalid_items:
            index = int(item["index"])
            failures_by_index[index] = _failure_row(
                entries[index], index, str(item["shape_error"])
            )

        for group in groups.values():
            inputs = torch.stack([item["input"] for item in group]).to(device, non_blocking=True)
            targets = torch.stack([item["target"] for item in group]).to(device, non_blocking=True)
            try:
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, enabled=amp_enabled
                ):
                    prediction, details = forward(inputs)
            except Exception as error:
                if isinstance(error, (TypeError, AttributeError)):
                    progress.close()
                    raise
                for item in group:
                    index = int(item["index"])
                    failures_by_index[index] = _failure_row(entries[index], index, error)
                continue
            if not isinstance(prediction, Tensor) or not isinstance(details, UICFINROutput):
                progress.close()
                raise TypeError("v16 diagnostics forward returned incompatible values")

            prediction_normal_batch = None
            if not normal_forward_checked:
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, enabled=amp_enabled
                ):
                    prediction_normal_batch = model(inputs)
                if not isinstance(prediction_normal_batch, Tensor):
                    progress.close()
                    raise TypeError("v16 normal forward did not return a prediction tensor")

            for position, item in enumerate(group):
                index = int(item["index"])
                entry = entries[index]
                sample_details = _slice_details(details, position)
                sample_inputs = inputs[position : position + 1]
                sample_targets = targets[position : position + 1]
                sample_prediction = prediction[position : position + 1]
                prediction_normal = (
                    None
                    if prediction_normal_batch is None
                    else prediction_normal_batch[position : position + 1]
                )
                # A formula or normal-forward mismatch invalidates the scientific
                # protocol and therefore aborts rather than becoming a sample failure.
                assert_uicf_consistency(
                    sample_inputs, sample_prediction, sample_details, prediction_normal
                )
                try:
                    input_cpu = sample_inputs[0].detach().float().cpu()
                    target_cpu = sample_targets[0].detach().float().cpu()
                    enhanced_cpu = sample_details.enhanced[0].detach().float().cpu()
                    field_cpu = sample_details.correction_field[0].detach().float().cpu()
                    spatial, _ = analyze_spatial_maps(
                        input_cpu,
                        target_cpu,
                        enhanced_cpu,
                        field_cpu,
                        sample_id=entry.sample_id,
                        sample_index=index,
                        patch_size=patch_size,
                        top_fraction=top_fraction,
                        num_null_shifts=num_null_shifts,
                        null_seed=null_seed,
                    )
                    quality_prediction = sample_prediction.float().clamp(0.0, 1.0)
                    quality_target = sample_targets.float()
                    psnr, ssim = batch_metrics(
                        quality_prediction, quality_target, dict(metric_config)
                    )
                    e00 = batch_delta_e00(
                        quality_prediction, quality_target, dict(metric_config)
                    )
                    anchor = sample_details.chromatic_anchor[0].detach().float().cpu()
                    row = {
                        "sample_index": index,
                        "sample_id": entry.sample_id,
                        "filename": Path(entry.input_relative).name,
                        "psnr": _finite_or_none(psnr[0]),
                        "ssim": _finite_or_none(ssim[0]),
                        "e00": _finite_or_none(e00[0]),
                        **spatial,
                        "b_r": _finite_or_none(anchor[0]),
                        "b_g": _finite_or_none(anchor[1]),
                        "b_b": _finite_or_none(anchor[2]),
                    }
                    rows.append(row)
                except Exception as error:
                    failures_by_index[index] = _failure_row(entry, index, error)
            if prediction_normal_batch is not None:
                normal_forward_checked = True
    progress.close()
    rows.sort(key=lambda row: int(row["sample_index"]))
    failures = [failures_by_index[index] for index in sorted(failures_by_index)]
    if len(rows) + len(failures) != len(entries):
        raise RuntimeError(
            f"Processed count mismatch: successful={len(rows)} failed={len(failures)} "
            f"manifest={len(entries)}"
        )
    if rows and not normal_forward_checked:
        raise RuntimeError("No successful sample received the normal-forward equivalence check")
    return rows, failures


def descriptive_statistics(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {
            "valid_count": 0,
            "invalid_count": len(rows),
            "mean": None,
            "std": None,
            "median": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"Non-finite dataset statistic for {field}")
    return {
        "valid_count": int(array.size),
        "invalid_count": len(rows) - int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
    }


def paired_null_statistics(
    rows: Sequence[Mapping[str, Any]], real_field: str, null_field: str
) -> dict[str, Any]:
    pairs = [
        (float(row[real_field]), float(row[null_field]))
        for row in rows
        if row.get(real_field) is not None and row.get(null_field) is not None
    ]
    if not pairs:
        return {
            "valid_pair_count": 0,
            "mean_real": None,
            "mean_null": None,
            "mean_real_minus_null": None,
            "fraction_real_greater_than_null": None,
        }
    array = np.asarray(pairs, dtype=np.float64)
    differences = array[:, 0] - array[:, 1]
    return {
        "valid_pair_count": int(array.shape[0]),
        "mean_real": float(array[:, 0].mean()),
        "mean_null": float(array[:, 1].mean()),
        "mean_real_minus_null": float(differences.mean()),
        "fraction_real_greater_than_null": float(np.mean(differences > 0.0)),
    }


def bootstrap_mean_difference(
    differences: Sequence[float], *, samples: int, seed: int
) -> dict[str, Any]:
    """Image-level paired bootstrap CI for the mean real-minus-null difference."""

    array = np.asarray(differences, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        return {"valid_count": 0, "mean_difference": None, "ci95_low": None, "ci95_high": None}
    if samples < 1 or not np.isfinite(array).all():
        raise ValueError("Bootstrap requires finite differences and at least one resample")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        stop = min(samples, start + 512)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        bootstrap_means[start:stop] = array[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, (2.5, 97.5))
    return {
        "valid_count": int(array.size),
        "mean_difference": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    split_counts: Mapping[str, int],
    data_root: Path,
    test_manifest: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metric_fields = (
        "spearman_rgb",
        "pearson_rgb",
        "top20_iou_rgb",
        "direction_cosine_top20_gt",
        "spearman_e00",
    )
    metrics = {field: descriptive_statistics(rows, field) for field in metric_fields}
    null_specs = {
        "spearman_rgb": ("spearman_rgb", "null_spearman_rgb_mean"),
        "pearson_rgb": ("pearson_rgb", "null_pearson_rgb_mean"),
        "top20_iou_rgb": ("top20_iou_rgb", "null_top20_iou_rgb_mean"),
        "direction_cosine_top20_gt": (
            "direction_cosine_top20_gt",
            "null_direction_cosine_mean",
        ),
    }
    null_controls = {
        name: paired_null_statistics(rows, real_field, null_field)
        for name, (real_field, null_field) in null_specs.items()
    }
    bootstrap_specs = {
        "spearman_gain": "spearman_gain_over_null",
        "iou_gain": "top20_iou_gain_over_null",
        "direction_gain": "direction_gain_over_null",
    }
    bootstrap: dict[str, Any] = {
        "resamples": bootstrap_samples,
        "seed": bootstrap_seed,
        "resampling_unit": "image",
    }
    for stream, (name, field) in enumerate(bootstrap_specs.items()):
        differences = [float(row[field]) for row in rows if row.get(field) is not None]
        result = bootstrap_mean_difference(
            differences, samples=bootstrap_samples, seed=bootstrap_seed + stream
        )
        bootstrap[name] = result
        bootstrap[f"{name}_mean"] = result["mean_difference"]
        bootstrap[f"{name}_ci95_low"] = result["ci95_low"]
        bootstrap[f"{name}_ci95_high"] = result["ci95_high"]
    return {
        "dataset": dataset_name,
        "train_count": int(split_counts["train"]),
        "validation_count": int(split_counts["validation"]),
        "test_count": int(split_counts["test"]),
        "data_root": str(data_root),
        "test_manifest": str(test_manifest),
        "total_test_samples": int(split_counts["test"]),
        "processed_sample_count": len(rows) + len(failures),
        "successful_sample_count": len(rows),
        "failed_sample_count": len(failures),
        "metrics": metrics,
        "valid_sample_counts": {field: values["valid_count"] for field, values in metrics.items()},
        "null_controls": null_controls,
        "bootstrap": bootstrap,
    }


def _display_range(values: np.ndarray, percentile: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("Demand heatmap must be a finite non-negative [H,W] array")
    upper = max(float(np.percentile(array, percentile)), float(np.finfo(np.float32).eps))
    return 0.0, upper


def _heatmap_image(values: np.ndarray, upper: float) -> Image.Image:
    normalized = np.clip(np.asarray(values, dtype=np.float64) / upper, 0.0, 1.0)
    # Compact perceptually ordered dark-purple -> red -> yellow display ramp.
    stops = np.array(
        [[8, 7, 30], [78, 18, 123], [174, 42, 90], [239, 101, 52], [252, 253, 191]],
        dtype=np.float64,
    )
    scaled = normalized * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper_index = np.minimum(lower + 1, len(stops) - 1)
    weight = (scaled - lower)[..., None]
    rgb = np.rint(stops[lower] * (1.0 - weight) + stops[upper_index] * weight).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _overlap_image(
    pooled_uicf: np.ndarray,
    pooled_gt: np.ndarray,
    fraction: float,
    output_size: tuple[int, int],
) -> Image.Image:
    uicf = exact_top_fraction_mask(pooled_uicf, fraction)
    gt = exact_top_fraction_mask(pooled_gt, fraction)
    rgb = np.full((*gt.shape, 3), (28, 28, 35), dtype=np.uint8)
    rgb[gt & ~uicf] = (230, 126, 34)  # GT-only: orange
    rgb[uicf & ~gt] = (52, 152, 219)  # UICF-only: blue
    rgb[gt & uicf] = (46, 204, 113)   # overlap: green
    return Image.fromarray(rgb, mode="RGB").resize(output_size, Image.Resampling.NEAREST)


def _panel(images: Sequence[Image.Image], titles: Sequence[str], path: Path) -> None:
    if len(images) != len(titles) or not images:
        raise ValueError("Panel images and titles must be non-empty and aligned")
    source_width, source_height = images[0].size
    display_height = max(128, source_height)
    display_width = max(128, int(round(source_width * display_height / source_height)))
    resized = [image.resize((display_width, display_height), Image.Resampling.BILINEAR) for image in images]
    title_height, gap = 30, 5
    canvas = Image.new(
        "RGB",
        (len(images) * display_width + (len(images) - 1) * gap, title_height + display_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, (image, title) in enumerate(zip(resized, titles, strict=True)):
        x = column * (display_width + gap)
        canvas.paste(image, (x, title_height))
        box = draw.textbbox((0, 0), title)
        draw.text((x + (display_width - (box[2] - box[0])) // 2, 8), title, fill="black")
    canvas.save(path, dpi=(200, 200))


def save_alignment_visualization(
    sample: CapturedSample,
    row: Mapping[str, Any],
    directory: Path,
    *,
    patch_size: int,
    top_fraction: float,
    robust_percentile: float,
    selection_type: str,
) -> Path:
    """Save raw maps and a seven-column paper-candidate panel for one sample."""

    directory.mkdir(parents=True, exist_ok=False)
    _, maps = analyze_spatial_maps(
        sample.input_tensor,
        sample.target_tensor,
        sample.enhanced,
        sample.correction_field,
        sample_id=sample.sample_id,
        sample_index=sample.index,
        patch_size=patch_size,
        top_fraction=top_fraction,
        num_null_shifts=1,
        null_seed=0,
    )
    input_image = tensor_to_image(sample.input_tensor)
    gt_image = tensor_to_image(sample.target_tensor)
    corrected_image = tensor_to_image(sample.enhanced)
    input_image.save(directory / "input.png")
    gt_image.save(directory / "gt.png")
    corrected_image.save(directory / "uicf_corrected.png")
    display_ranges: dict[str, list[float]] = {}
    heatmaps: dict[str, Image.Image] = {}
    for key in ("gt_rgb_demand", "uicf_effect_magnitude", "gt_e00_demand"):
        values = maps[key]
        np.save(directory / f"{key}.npy", values, allow_pickle=False)
        lower, upper = _display_range(values, robust_percentile)
        display_ranges[key] = [lower, upper]
        heatmap = _heatmap_image(values, upper)
        heatmap.save(directory / f"{key}.png")
        heatmaps[key] = heatmap
    overlap = _overlap_image(
        maps["pooled_uicf_effect"], maps["pooled_gt_rgb"], top_fraction, input_image.size
    )
    overlap.save(directory / "top20_overlap.png")
    panel_path = directory / "alignment_panel.png"
    _panel(
        [
            input_image,
            gt_image,
            corrected_image,
            heatmaps["gt_rgb_demand"],
            heatmaps["uicf_effect_magnitude"],
            overlap,
            heatmaps["gt_e00_demand"],
        ],
        ["Input I", "GT Y", "UICF I_c", "||Y-I||_2", "||I_c-I||_2", "Top overlap", "DeltaE00(I,Y)"],
        panel_path,
    )
    metadata = {
        "sample_id": sample.sample_id,
        "sample_index": sample.index,
        "filename": sample.filename,
        "selection_type": selection_type,
        "real_metrics": {key: row.get(key) for key in PER_SAMPLE_FIELDS if not key.startswith("null_")},
        "null_metrics": {key: row.get(key) for key in PER_SAMPLE_FIELDS if key.startswith("null_") or key.endswith("gain_over_null")},
        "chromatic_anchor_b": [float(value) for value in sample.chromatic_anchor.reshape(-1)],
        "raw_correction_field_statistics": {
            "mean_abs": float(sample.correction_field.abs().mean()),
            "std": float(sample.correction_field.std(unbiased=False)),
        },
        "display_normalization_ranges": display_ranges,
        "display_normalization": f"independent [0, percentile-{robust_percentile:g}] per heatmap",
        "quantitative_normalization": "none",
        "top_overlap_colors": {
            "background": "dark gray",
            "GT_only": "orange",
            "UICF_only": "blue",
            "overlap": "green",
        },
        "uicf_actual_correction": "delta_uicf = I_c - I = R(x) * (I - b)",
        "raw_field_interpretation": "R(x) is a coefficient field, not an RGB correction residual",
    }
    atomic_json(directory / "alignment_metadata.json", metadata)
    return panel_path


def _placeholder_image(path: Path, text: str) -> None:
    image = Image.new("RGB", (900, 180), "white")
    ImageDraw.Draw(image).text((24, 76), text, fill="black")
    image.save(path)


def build_contact_sheet(panel_paths: Sequence[Path], path: Path, *, label: str) -> None:
    if not panel_paths:
        _placeholder_image(path, f"No valid {label} samples were available.")
        return
    panels: list[Image.Image] = []
    for panel_path in panel_paths:
        with Image.open(panel_path) as image:
            panels.append(image.convert("RGB"))
    target_width = max(image.width for image in panels)
    resized = [
        image.resize((target_width, round(image.height * target_width / image.width)), Image.Resampling.BILINEAR)
        if image.width != target_width else image
        for image in panels
    ]
    gap = 8
    sheet = Image.new("RGB", (target_width, sum(image.height for image in resized) + gap * (len(resized) - 1)), "white")
    y = 0
    for image in resized:
        sheet.paste(image, (0, y))
        y += image.height + gap
    sheet.save(path, dpi=(200, 200))


def save_metric_plots(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    real = [float(row["spearman_rgb"]) for row in rows if row.get("spearman_rgb") is not None]
    width, height = 800, 520
    margin_left, margin_right, margin_top, margin_bottom = 72, 30, 42, 68
    plot_left, plot_right = margin_left, width - margin_right
    plot_top, plot_bottom = margin_top, height - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    if real:
        bin_count = min(30, max(5, round(math.sqrt(len(real)))))
        counts, edges = np.histogram(real, bins=bin_count, range=(-1.0, 1.0))
        maximum = max(1, int(counts.max()))
        draw.line((plot_left, plot_top, plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)
        bin_width = (plot_right - plot_left) / bin_count
        for index, count in enumerate(counts):
            left = round(plot_left + index * bin_width)
            right = round(plot_left + (index + 1) * bin_width) - 1
            top = round(plot_bottom - int(count) / maximum * (plot_bottom - plot_top))
            draw.rectangle((left, top, right, plot_bottom - 1), fill=(53, 114, 165), outline="white")
        draw.text((plot_left, plot_bottom + 20), "-1", fill="black")
        draw.text((plot_right - 10, plot_bottom + 20), "1", fill="black")
        draw.text((plot_left + 180, height - 30), "Per-image Spearman: UICF effect vs GT RGB demand", fill="black")
        draw.text((10, plot_top), f"count (max={maximum})", fill="black")
    else:
        draw.text((width // 2 - 90, height // 2), "No valid Spearman values", fill="black")
    image.save(output_dir / "metric_histogram_spearman.png", dpi=(200, 200))

    pairs = [
        (float(row["null_spearman_rgb_mean"]), float(row["spearman_rgb"]))
        for row in rows
        if row.get("spearman_rgb") is not None and row.get("null_spearman_rgb_mean") is not None
    ]
    image = Image.new("RGB", (620, 620), "white")
    draw = ImageDraw.Draw(image)
    plot_left, plot_right, plot_top, plot_bottom = 72, 590, 35, 550
    if pairs:
        array = np.asarray(pairs, dtype=np.float64)
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="black", width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_top), fill=(176, 58, 46), width=2)
        for null_value, real_value in array:
            x = round(plot_left + (null_value + 1.0) / 2.0 * (plot_right - plot_left))
            y = round(plot_bottom - (real_value + 1.0) / 2.0 * (plot_bottom - plot_top))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(46, 134, 171))
        draw.text((plot_left, plot_bottom + 15), "-1", fill="black")
        draw.text((plot_right - 10, plot_bottom + 15), "1", fill="black")
        draw.text((plot_left + 150, 590), "Mean shifted-null Spearman", fill="black")
        draw.text((8, plot_top), "Real Spearman", fill="black")
    else:
        draw.text((220, 300), "No valid real/null pairs", fill="black")
    image.save(output_dir / "real_vs_null_spearman.png", dpi=(200, 200))


def select_visualization_rows(
    rows: Sequence[Mapping[str, Any]], viz_k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [dict(row) for row in rows if row.get("spearman_rgb") is not None]
    top = sorted(valid, key=lambda row: (-float(row["spearman_rgb"]), int(row["sample_index"])))[:viz_k]
    if not valid:
        return top, []
    demand_median = float(np.median([float(row["mean_gt_rgb_demand"]) for row in rows]))
    spearman_median = float(np.median([float(row["spearman_rgb"]) for row in valid]))
    candidates = [row for row in valid if float(row["mean_gt_rgb_demand"]) >= demand_median]
    representative = sorted(
        candidates,
        key=lambda row: (abs(float(row["spearman_rgb"]) - spearman_median), int(row["sample_index"])),
    )[:viz_k]
    return top, representative


def export_visualizations(
    model: UICFPreBackbone,
    dataset: LSUIDataset,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    output_dir: Path,
    *,
    amp_enabled: bool,
    patch_size: int,
    top_fraction: float,
    robust_percentile: float,
    viz_k: int,
) -> dict[str, list[int]]:
    top_rows, representative_rows = select_visualization_rows(rows, viz_k)
    selections = (("top_alignment", top_rows), ("representative", representative_rows))
    selected_indices: dict[str, list[int]] = {}
    for selection_type, selected_rows in selections:
        directory = output_dir / selection_type
        directory.mkdir()
        panel_paths: list[Path] = []
        selected_indices[selection_type] = []
        for rank, row in enumerate(selected_rows, 1):
            index = int(row["sample_index"])
            sample = capture_sample(
                model,
                dataset,
                index,
                device,
                amp_enabled=amp_enabled,
                compare_normal_forward=False,
            )
            folder = directory / f"{rank:02d}_{_safe_sample_id(sample.sample_id)}"
            panel_paths.append(
                save_alignment_visualization(
                    sample,
                    row,
                    folder,
                    patch_size=patch_size,
                    top_fraction=top_fraction,
                    robust_percentile=robust_percentile,
                    selection_type=selection_type,
                )
            )
            selected_indices[selection_type].append(index)
        build_contact_sheet(
            panel_paths,
            output_dir / f"{selection_type}_contact_sheet.png",
            label=selection_type.replace("_", " "),
        )
    return selected_indices


def _prepare_output_directory(
    output_dir: Path, *, overwrite: bool, run_dir: Path, data_root: Path
) -> None:
    resolved = output_dir.resolve()
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
        run_dir.resolve(),
        (run_dir / "result").resolve(),
        data_root.resolve(),
    }
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_dir}")
        contains_protected = any(
            item == resolved or item.is_relative_to(resolved) for item in protected
        )
        default_output = (run_dir / "result" / "uicf_alignment").resolve()
        nonempty_unmarked_custom = (
            resolved != default_output
            and any(output_dir.iterdir())
            and not (output_dir / "protocol.json").is_file()
        )
        if contains_protected or len(resolved.parts) < 4 or nonempty_unmarked_custom:
            raise ValueError(f"Refusing to overwrite protected or unmarked directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _write_summary_text(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        f"v16 UICF Spatial Restoration-Demand Alignment — {summary['dataset']} test set",
        f"Train/validation/test counts: {summary['train_count']}/"
        f"{summary['validation_count']}/{summary['test_count']}",
        f"Total test samples: {summary['total_test_samples']}",
        f"Successful: {summary['successful_sample_count']}",
        f"Failed: {summary['failed_sample_count']}",
        "",
        "Dataset metrics (mean / std / median / valid):",
    ]
    for name, values in summary["metrics"].items():
        if values["mean"] is None:
            lines.append(f"{name}: undefined / valid=0")
        else:
            lines.append(
                f"{name}: {values['mean']:.8f} / {values['std']:.8f} / "
                f"{values['median']:.8f} / valid={values['valid_count']}"
            )
    lines.extend(("", "Paired image-level bootstrap real-null gains (mean, 95% CI):"))
    for name in ("spearman_gain", "iou_gain", "direction_gain"):
        values = summary["bootstrap"][name]
        if values["mean_difference"] is None:
            lines.append(f"{name}: undefined / valid=0")
        else:
            lines.append(
                f"{name}: {values['mean_difference']:.8f} "
                f"[{values['ci95_low']:.8f}, {values['ci95_high']:.8f}] "
                f"valid={values['valid_count']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _protocol(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    config_path: Path,
    manifest_path: Path,
    config: Mapping[str, Any],
    data_root: Path,
    selector: str,
    amp_requested: bool,
    amp_enabled: bool,
    num_workers: int,
    dataset_name: str,
    split_counts: Mapping[str, int],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    return {
        "script": "tools/analyze_uicf_alignment.py",
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_selector": selector,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_resolved_yaml": str(config_path),
        "config_resolved_yaml_sha256": sha256_file(config_path),
        "dataset": dataset_name,
        "train_count": int(split_counts["train"]),
        "validation_count": int(split_counts["validation"]),
        "test_count": int(split_counts["test"]),
        "test_manifest": str(manifest_path),
        "test_manifest_sha256": sha256_file(manifest_path),
        "test_manifest_sample_count": int(split_counts["test"]),
        "data_root": str(data_root),
        "paired_file_verification": (
            "every manifest item checks existence/readability/RGB/pair dimensions and "
            "input/GT stem identity; failures are retained in failed_samples.csv"
        ),
        "evaluation_resize": bool(evaluation.get("resize", True)),
        "evaluation_size": evaluation.get("size"),
        "patch_size": args.patch_size,
        "top_fraction": args.top_fraction,
        "num_null_shifts": args.num_null_shifts,
        "null_seed": args.null_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "amp_requested": amp_requested,
        "amp_enabled": amp_enabled,
        "rgb_demand_definition": "D_gt_rgb(p) = ||Y(p) - I(p)||_2",
        "uicf_correction_definition": "D_uicf(p) = ||I_c(p) - I(p)||_2, where I_c = I + R(x)*(I-b)",
        "raw_field_definition": "D_raw_R(p) = ||R(p)||_2; supplementary only",
        "spearman_implementation": "custom deterministic average-tie ranks followed by Pearson correlation",
        "e00_implementation": e00_protocol(dict(config["metrics"])),
        "direction_definition": "mean RGB-vector cosine on top-GT-demand patches where both vector norms exceed eps",
        "null_control": "deterministic nonzero large circular shifts of pooled UICF scalar/vector maps",
        "normalization_policy": "no visualization normalization is used for quantitative metrics",
        "visualization_normalization": f"independent nonnegative heatmap ranges at percentile {args.robust_percentile:g}",
        "representative_selection": "closest to valid dataset-median Spearman among samples with GT RGB demand >= dataset median",
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    run_dir = project_path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run lacks config_resolved.yaml: {run_dir}")
    config = load_yaml(config_path)
    if config["experiment"]["version"] != EXPECTED_VERSION:
        raise ValueError(
            f"{EXPECTED_VERSION} alignment analysis refuses run version "
            f"{config['experiment']['version']!r}"
        )
    supported_datasets = {"LSUI19", "UIEB"}
    dataset_name = str(config["data"].get("dataset", "")).strip()
    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Alignment analysis supports {sorted(supported_datasets)}, got {dataset_name!r}"
        )
    if args.data_root is not None:
        config["data"]["root"] = str(Path(args.data_root).expanduser().resolve())
    data_root = Path(config["data"]["root"]).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data.root is unavailable: {data_root}")

    selector = args.checkpoint or config["test"]["checkpoint"]
    checkpoint_path = _checkpoint_path(run_dir, selector)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    snapshot = run_dir / "split_snapshot"
    manifests = {name: snapshot / f"{name}.tsv" for name in ("train", "validation", "test")}
    split_entries = validate_split_protocol(manifests, config["data"].get("expected_counts"))
    split_counts = {name: len(entries) for name, entries in split_entries.items()}
    test_entries = split_entries["test"]
    data = config["data"]
    test_dataset = LSUIDataset(
        manifests["test"],
        data_root,
        "test",
        int(data["patch_size"]),
        data["augmentation"],
        bool(data["pad_if_smaller"]),
        str(data["pad_mode"]),
        config["evaluation"],
        # Per-item checks are performed by SafeAlignmentDataset so every bad
        # manifest row is represented in failed_samples.csv rather than dropped.
        verify_files=False,
    )
    if test_dataset.entries != test_entries:
        raise RuntimeError("Validated test manifest order differs from Dataset order")

    output_dir = (
        run_dir / "result" / "uicf_alignment"
        if args.output_dir is None
        else project_path(args.output_dir).expanduser().resolve()
    ).resolve()
    _prepare_output_directory(output_dir, overwrite=args.overwrite, run_dir=run_dir, data_root=data_root)

    device = select_device(args.gpu)
    seed_everything(int(config["experiment"]["seed"]), deterministic=True)
    checkpoint = _torch_load(checkpoint_path, device)
    model = build_and_load_v16_model(config, checkpoint, device)
    if not isinstance(model, UICFPreBackbone):
        raise TypeError(f"Expected UICFPreBackbone, got {type(model).__name__}")
    amp_requested = bool(config["training"]["amp"]) if args.amp is None else bool(args.amp)
    amp_enabled = amp_requested and device.type == "cuda"
    num_workers = int(data["num_workers"]) if args.num_workers is None else args.num_workers
    protocol = _protocol(
        args=args,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        config_path=config_path,
        manifest_path=manifests["test"],
        config=config,
        data_root=data_root,
        selector=selector,
        amp_requested=amp_requested,
        amp_enabled=amp_enabled,
        num_workers=num_workers,
        dataset_name=dataset_name,
        split_counts=split_counts,
    )
    atomic_json(output_dir / "protocol.json", protocol)

    rows, failures = evaluate_test_set(
        model,
        test_dataset,
        test_entries,
        device,
        batch_size=args.batch_size,
        num_workers=num_workers,
        amp_enabled=amp_enabled,
        metric_config=config["metrics"],
        patch_size=args.patch_size,
        top_fraction=args.top_fraction,
        num_null_shifts=args.num_null_shifts,
        null_seed=args.null_seed,
    )
    _write_csv(output_dir / "per_sample_metrics.csv", rows, PER_SAMPLE_FIELDS)
    _write_csv(output_dir / "failed_samples.csv", failures, FAILED_FIELDS)
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            row.get("spearman_rgb") is None,
            -float(row["spearman_rgb"]) if row.get("spearman_rgb") is not None else 0.0,
            int(row["sample_index"]),
        ),
    )
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    _write_csv(output_dir / "ranking_by_spearman.csv", ranked, ["rank", *PER_SAMPLE_FIELDS])
    _write_csv(output_dir / "null_control_summary.csv", rows, NULL_FIELDS)

    summary = build_summary(
        rows,
        failures,
        dataset_name=dataset_name,
        split_counts=split_counts,
        data_root=data_root,
        test_manifest=manifests["test"],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    # Preserve the complete quantitative result before optional second-pass
    # visualization.  A later image-export error must not discard the metrics.
    atomic_json(output_dir / "summary.json", summary)
    _write_summary_text(summary, output_dir / "summary.txt")
    selected = export_visualizations(
        model,
        test_dataset,
        rows,
        device,
        output_dir,
        amp_enabled=amp_enabled,
        patch_size=args.patch_size,
        top_fraction=args.top_fraction,
        robust_percentile=args.robust_percentile,
        viz_k=args.viz_k,
    )
    summary["visualization_selection"] = selected
    save_metric_plots(rows, output_dir)
    atomic_json(output_dir / "summary.json", summary)
    _write_summary_text(summary, output_dir / "summary.txt")
    print(
        f"\n{dataset_name} v16 UICF alignment analysis completed\n"
        f"Total samples: {len(test_entries)}\n"
        f"Successful samples: {len(rows)}\n"
        f"Failed samples: {len(failures)}\n"
        f"Valid RGB Spearman samples: {summary['valid_sample_counts']['spearman_rgb']}\n"
        f"Output directory: {output_dir}"
    )


if __name__ == "__main__":
    main()
