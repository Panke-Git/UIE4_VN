"""Pure coefficient interventions and image-paired statistics (never display data)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from src.shared.e00 import batch_delta_e00
from src.shared.uicf_inr import UICFINROutput
from src.shared.uicf_models import UICFPreBackbone
from src.v16.metrics import batch_metrics
from tools.analyze_uicf_alignment import bootstrap_mean_difference
from tools.visualize_v16_uicf import assert_uicf_consistency, reconstruct_uicf_corrected


MAIN_VARIANTS = ("original", "spatial_mean", "shift_xy", "channel_cycle_1")
VARIANTS = (*MAIN_VARIANTS, "shift_y", "shift_x", "channel_cycle_2", "zero_field")
PERMUTATIONS = {"channel_cycle_1": (1, 2, 0), "channel_cycle_2": (2, 0, 1)}
METRICS = ("psnr", "ssim", "e00")
INTERPRETATION = (
    "Observational intervention on a frozen model. Changes measure model dependence, "
    "not physical causal identification, true transmission, or global optimality. "
    "No retraining; zero_field is in-model deactivation, not a trained w/o-UICF ablation. "
    "Unadjusted pointwise image-paired CIs concern this fixed checkpoint, not training-seed stability."
)


def check_tensor(
    tensor: Tensor, name: str, *, shape: Sequence[int] | None = None,
    device: torch.device | None = None, dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name}: expected floating tensor")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name}: shape={tuple(tensor.shape)}, expected={tuple(shape)}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name}: device={tensor.device}, expected={device}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name}: dtype={tensor.dtype}, expected={dtype}")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"{name}: non-finite shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} nan={int(torch.isnan(tensor).sum())} "
            f"posinf={int(torch.isposinf(tensor).sum())} neginf={int(torch.isneginf(tensor).sum())}"
        )


def shift_pixels(height: int, width: int) -> tuple[int, int]:
    # H=1 or W=1 would make at least one declared shift an identity modulo size.
    if height < 2 or width < 2:
        raise ValueError("Nonzero field shifts require H and W >= 2")
    return max(1, height // 4), max(1, width // 4)


def intervene_field(field: Tensor, variant: str) -> Tensor:
    """Return a new BCHW RGB field of the same dtype/device; never mix samples."""
    check_tensor(field, "R")
    if field.ndim != 4 or field.shape[1] != 3 or min(field.shape) < 1:
        raise ValueError("R must be nonempty [B,3,H,W]")
    if variant == "original":
        result = field.clone()
    elif variant == "spatial_mean":
        result = field.mean(dim=(-2, -1), keepdim=True).expand_as(field).clone()
    elif variant in ("shift_xy", "shift_y", "shift_x"):
        dy, dx = shift_pixels(*field.shape[-2:])
        result = torch.roll(
            field, shifts=(0 if variant == "shift_x" else dy,
                           0 if variant == "shift_y" else dx), dims=(-2, -1),
        )
    elif variant in PERMUTATIONS:
        result = field[:, list(PERMUTATIONS[variant]), :, :].clone()
    elif variant == "zero_field":
        result = torch.zeros_like(field)
    else:
        raise ValueError(f"Unknown intervention {variant!r}")
    check_tensor(result, f"R_{variant}", shape=field.shape, device=field.device, dtype=field.dtype)
    return result


def assert_field_invariants(field: Tensor, changed: Tensor, variant: str) -> None:
    check_tensor(changed, f"R_{variant}", shape=field.shape, device=field.device, dtype=field.dtype)
    first, second = field.double(), changed.double()
    tolerance = 4 * torch.finfo(field.dtype).eps
    if variant == "spatial_mean":
        if not torch.allclose(first.mean((-2, -1)), second.mean((-2, -1)),
                              atol=tolerance, rtol=tolerance):
            raise RuntimeError("spatial_mean did not preserve per-sample per-channel means")
        if not torch.equal(changed, changed[..., :1, :1].expand_as(changed)):
            raise RuntimeError("spatial_mean is not spatially constant within each channel")
    elif variant.startswith("shift_"):
        # Sorting separately for each B,C verifies the complete multiset, not just moments.
        if not torch.equal(first.flatten(2).sort(-1).values, second.flatten(2).sort(-1).values):
            raise RuntimeError(f"{variant} changed per-sample per-channel value multisets")
        for statistic in (lambda x: x.mean((-2, -1)), lambda x: x.square().sum((-2, -1))):
            if not torch.allclose(statistic(first), statistic(second), atol=1e-10, rtol=1e-10):
                raise RuntimeError(f"{variant} changed per-channel mean/L2 norm")
    elif variant in PERMUTATIONS:
        if not torch.equal(changed, field[:, list(PERMUTATIONS[variant])]):
            raise RuntimeError(f"{variant} has incorrect RGB channel order")
    elif variant == "zero_field" and torch.count_nonzero(changed):
        raise RuntimeError("zero_field is not zero")
    elif variant == "original" and not torch.equal(changed, field):
        raise RuntimeError("original field changed")


def corrected_image(inputs: Tensor, details: UICFINROutput, field: Tensor) -> Tensor:
    """Reuse the canonical diagnostic reconstruction, including native AMP promotion."""
    check_tensor(inputs, "I")
    check_tensor(field, "R_variant", shape=inputs.shape, device=inputs.device,
                 dtype=details.correction_field.dtype)
    check_tensor(details.chromatic_anchor, "b", shape=(inputs.shape[0], 3), device=inputs.device)
    result = reconstruct_uicf_corrected(inputs, replace(details, correction_field=field))
    check_tensor(result, "Ic_variant", shape=inputs.shape, device=inputs.device,
                 dtype=details.enhanced.dtype)
    return result  # No clamp: Ic is allowed outside [0,1].


def assert_prediction_close(first: Tensor, second: Tensor, *, amp: bool, label: str) -> dict[str, Any]:
    check_tensor(second, label, shape=first.shape, device=first.device, dtype=first.dtype)
    atol, rtol = (1e-5, 1e-4) if amp else (1e-6, 1e-5)
    error = float((first.float() - second.float()).abs().max())
    if not torch.allclose(first, second, atol=atol, rtol=rtol):
        raise RuntimeError(f"{label}: forward equivalence FAILED max_abs_error={error} atol={atol} rtol={rtol}")
    return {"status": "PASS", "max_abs_error": error, "atol": atol, "rtol": rtol}


def torch_rng_snapshot(device: torch.device) -> tuple[Tensor, Tensor | None]:
    return torch.get_rng_state().clone(), (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )


def assert_torch_rng_unchanged(before: tuple[Tensor, Tensor | None], device: torch.device) -> None:
    after = torch_rng_snapshot(device)
    if not torch.equal(before[0], after[0]) or (
        before[1] is not None and not torch.equal(before[1], after[1])
    ):
        raise RuntimeError("Frozen forward consumed global Torch RNG state")


@torch.inference_mode()
def predict_variants(
    model: UICFPreBackbone, inputs: Tensor, *, amp: bool, check_normal: bool,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], UICFINROutput, dict[str, Any]]:
    if any(module.training for module in model.modules()):
        raise RuntimeError("All modules must be in eval mode for frozen intervention")
    if amp and inputs.device.type != "cuda":
        raise ValueError("This native V16 inference protocol uses AMP only on CUDA")
    check_tensor(inputs, "I", dtype=torch.float32)
    rng = torch_rng_snapshot(inputs.device)
    checks: dict[str, Any] = {}
    with torch.autocast(device_type=inputs.device.type, enabled=amp):
        prediction, details = model.forward_with_uicf_details(inputs)
        assert_uicf_consistency(inputs, prediction, details)
        forward_dtype = torch.get_autocast_dtype(inputs.device.type) if amp else torch.float32
        check_tensor(details.correction_field, "R", shape=inputs.shape, device=inputs.device, dtype=forward_dtype)
        check_tensor(details.enhanced, "Ic", shape=inputs.shape, device=inputs.device, dtype=inputs.dtype)
        check_tensor(details.chromatic_anchor, "b", shape=(inputs.shape[0], 3),
                     device=inputs.device, dtype=forward_dtype)
        check_tensor(prediction, "Dout_original", shape=inputs.shape,
                     device=inputs.device, dtype=forward_dtype)
        originals = (inputs.clone(), details.correction_field.clone(), details.chromatic_anchor.clone())
        predictions = {"original": prediction}
        fields = {"original": details.correction_field}
        corrected = {"original": details.enhanced}
        if check_normal:
            # Required independent full-path audit, not another R/b used by any variant.
            checks["normal_vs_details"] = assert_prediction_close(
                prediction, model(inputs), amp=amp, label="model(I) vs details forward",
            )
            checks["backbone_vs_details"] = assert_prediction_close(
                prediction, model.backbone(details.enhanced), amp=amp,
                label="backbone(original Ic) vs details forward",
            )
            checks["program_original"] = assert_prediction_close(
                prediction, predictions["original"], amp=amp, label="program original",
            )
        for variant in VARIANTS[1:]:
            try:
                field = intervene_field(details.correction_field, variant)
                assert_field_invariants(details.correction_field, field, variant)
                ic = corrected_image(inputs, details, field)
                output = model.backbone(ic)
                check_tensor(output, f"variant={variant} Dout", shape=prediction.shape,
                             device=prediction.device, dtype=prediction.dtype)
            except Exception as error:
                raise RuntimeError(f"variant={variant}: {type(error).__name__}: {error}") from error
            fields[variant], corrected[variant], predictions[variant] = field, ic, output
        for name, before, after in zip(
            ("I", "R", "b"), originals,
            (inputs, details.correction_field, details.chromatic_anchor), strict=True,
        ):
            if not torch.equal(before, after):
                raise RuntimeError(f"An intervention modified original {name} in place")
    assert_torch_rng_unchanged(rng, inputs.device)
    return predictions, fields, corrected, details, checks


def score_prediction(prediction: Tensor, target: Tensor, config: dict) -> dict[str, float]:
    if prediction.ndim != 4 or prediction.shape[0] != 1:
        raise ValueError("Native V16 scoring requires exactly one image per batch")
    check_tensor(prediction, "Dout before metric clamp")
    check_tensor(target, "GT", shape=prediction.shape, device=prediction.device, dtype=torch.float32)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        quality = prediction.float().clamp(0.0, 1.0)
        psnr, ssim = batch_metrics(quality, target.float(), config)
        e00 = batch_delta_e00(quality, target.float(), config)
    result = {"psnr": float(psnr[0]), "ssim": float(ssim[0]), "e00": float(e00[0])}
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"Non-finite metric (including undefined/infinite PSNR): {result}")
    return result


def intervention_diagnostics(
    inputs: Tensor, field: Tensor, ic: Tensor, output: Tensor,
    details: UICFINROutput, original_output: Tensor,
) -> dict[str, float]:
    mean_abs = lambda value: float(value.double().abs().mean())
    anchor = details.chromatic_anchor
    row = {
        "mean_abs_R": mean_abs(field),
        "std_R": float(field.double().std(unbiased=False)),
        "mean_abs_delta_Ic_original": mean_abs(details.correction_field * (inputs - anchor[:, :, None, None])),
        "mean_abs_delta_Ic_variant": mean_abs(field * (inputs - anchor[:, :, None, None])),
        "mean_abs_Ic_variant_minus_Ic_original": mean_abs(ic - details.enhanced),
        "mean_abs_Dout_variant_minus_Dout_original": mean_abs(output.float() - original_output.float()),
        "fraction_clipped_final": float(((output < 0) | (output > 1)).double().mean()),
    }
    for channel, color in enumerate("rgb"):
        row[f"mean_R_{color}"] = float(field[:, channel].double().mean())
        row[f"std_R_{color}"] = float(field[:, channel].double().std(unbiased=False))
        row[f"b_{color}"] = float(anchor[0, channel])
    return row


def paired_differences(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        group = by_id.setdefault(str(row["sample_id"]), {})
        if row["variant"] in group:
            raise ValueError("Duplicate sample/variant in paired statistics")
        group[str(row["variant"])] = row
    differences = []
    for sample_id, group in by_id.items():
        if set(group) != set(VARIANTS):
            raise ValueError(f"Incomplete within-image pairing: {sample_id}")
        original = group["original"]
        for variant in VARIANTS[1:]:
            differences.append({
                "sample_id": sample_id, "filename": original["filename"], "variant": variant,
                "gain_psnr": float(original["psnr"]) - float(group[variant]["psnr"]),
                "gain_ssim": float(original["ssim"]) - float(group[variant]["ssim"]),
                "gain_e00": float(group[variant]["e00"]) - float(original["e00"]),
            })
    return differences


def summarize(rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int) -> dict[str, Any]:
    differences = paired_differences(rows)
    summary: dict[str, Any] = {"variants": {}, "paired_gains": {}, "interpretation": INTERPRETATION}
    for index, variant in enumerate(VARIANTS):
        members = [row for row in rows if row["variant"] == variant]
        if not members:
            raise ValueError("Cannot summarize an empty evaluation")
        summary["variants"][variant] = {
            "sample_count": len(members),
            **{f"mean_{metric}": float(np.mean([row[metric] for row in members], dtype=np.float64))
               for metric in METRICS},
        }
        if variant == "original":
            continue
        summary["paired_gains"][variant] = {}
        for stream, metric in enumerate(METRICS):
            values = np.asarray([row[f"gain_{metric}"] for row in differences if row["variant"] == variant])
            stream_seed = seed + (index - 1) * len(METRICS) + stream
            ci = bootstrap_mean_difference(values, samples=samples, seed=stream_seed)
            summary["paired_gains"][variant][metric] = {
                **ci, "std": float(values.std(ddof=0)), "std_ddof": 0,
                "median": float(np.median(values)), "win_fraction": float(np.mean(values > 0)),
                "bootstrap_seed": stream_seed, "bootstrap_resamples": samples,
                "resampling_unit": "image", "positive_means": "original_better",
                "ci_includes_zero": bool(ci["ci95_low"] <= 0 <= ci["ci95_high"]),
            }
    return summary
