#!/usr/bin/env python3
"""Read-only numerical-state diagnostics for an existing V18 checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v18.models import build_model
from src.v18.models.nlqc_guidance import (
    format_finite_diagnostics,
    tensor_finite_diagnostics,
)
from src.v18.test import _checkpoint_path, _torch_load
from src.v18.utils import load_yaml, project_path


EXPECTED_VERSION = "v18"
TOP_K = 30
ALPHA_KEYS = tuple(
    f"backbone.guide{stage}.nlqc_alpha" for stage in (4, 3, 2, 1)
)
GUIDE_WEIGHT_KEYS = tuple(
    f"backbone.guide{stage}.{suffix}"
    for stage in (4, 3, 2, 1)
    for suffix in (
        "query_projection.weight",
        "attention.in_proj_weight",
        "output_projection.weight",
    )
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect V18 model/optimizer numerical state without modifying it"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="last")
    return parser.parse_args(argv)


def iter_floating_tensors(value: Any, path: str) -> Iterator[tuple[str, Tensor]]:
    """Yield all floating tensors in a nested checkpoint object."""
    if torch.is_tensor(value):
        if value.is_floating_point():
            yield path, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from iter_floating_tensors(item, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{path}[{index}]"
            yield from iter_floating_tensors(item, child)


def _max_abs(tensor: Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    if not bool(torch.isfinite(tensor).all()):
        return math.inf
    return float(tensor.detach().abs().max().item())


def _format_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.10g}"


def inspect_tree(label: str, value: Any) -> list[dict[str, object]]:
    """Print whether a checkpoint tree contains non-finite floating tensors."""
    reports: list[dict[str, object]] = []
    count = 0
    for path, tensor in iter_floating_tensors(value, label):
        count += 1
        if not bool(torch.isfinite(tensor).all()):
            reports.append(tensor_finite_diagnostics(path, tensor))
    status = "NON_FINITE" if reports else "finite"
    print(
        f"{label}: status={status} floating_tensors={count} "
        f"nonfinite_tensors={len(reports)}"
    )
    for report in reports:
        formatted = format_finite_diagnostics(report).replace("\n", "\n    ")
        print(f"  non-finite tensor:\n    {formatted}")
    return reports


def print_top_model_magnitudes(model: nn.Module) -> None:
    records: list[tuple[float, str, str, tuple[int, ...], str]] = []
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point():
            status = "finite" if bool(torch.isfinite(parameter).all()) else "NON_FINITE"
            records.append(
                (_max_abs(parameter), "parameter", name, tuple(parameter.shape), status)
            )
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            status = "finite" if bool(torch.isfinite(buffer).all()) else "NON_FINITE"
            records.append((_max_abs(buffer), "buffer", name, tuple(buffer.shape), status))
    records.sort(key=lambda item: item[0], reverse=True)
    print(f"model parameter/floating-buffer max_abs Top-{TOP_K}:")
    for rank, (max_abs, kind, name, shape, status) in enumerate(records[:TOP_K], start=1):
        print(
            f"  {rank:02d}. {kind} {name} shape={shape} "
            f"max_abs={_format_number(max_abs)} status={status}"
        )


def print_batch_norm_diagnostics(model: nn.Module) -> None:
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    print(f"BatchNorm modules: {len(modules)}")
    for name, module in modules:
        running_mean = module.running_mean
        running_var = module.running_var
        if running_mean is None or running_var is None:
            print(f"  {name}: running statistics unavailable")
            continue
        mean_report = tensor_finite_diagnostics(
            f"{name}.running_mean", running_mean
        )
        var_report = tensor_finite_diagnostics(f"{name}.running_var", running_var)
        mean_finite = (
            int(mean_report["num_nan"])
            + int(mean_report["num_posinf"])
            + int(mean_report["num_neginf"])
            == 0
        )
        var_finite = (
            int(var_report["num_nan"])
            + int(var_report["num_posinf"])
            + int(var_report["num_neginf"])
            == 0
        )
        print(
            f"  {name}: running_mean_finite={mean_finite} "
            f"running_mean_max_abs={_format_number(mean_report['finite_abs_max'])} "
            f"running_var_finite={var_finite} "
            f"running_var_min={_format_number(var_report['finite_min'])} "
            f"running_var_max={_format_number(var_report['finite_max'])}"
        )


def print_selected_v18_state(model: nn.Module) -> None:
    state = model.state_dict()
    print("NLQC alpha values:")
    for key in ALPHA_KEYS:
        tensor = state[key]
        value = float(tensor.detach().cpu().item())
        status = "finite" if math.isfinite(value) else "NON_FINITE"
        print(f"  {key}: value={value:.10g} abs={abs(value):.10g} status={status}")
    print("V18 guide projection magnitudes:")
    for key in GUIDE_WEIGHT_KEYS:
        tensor = state[key]
        status = "finite" if bool(torch.isfinite(tensor).all()) else "NON_FINITE"
        print(
            f"  {key}: shape={tuple(tensor.shape)} "
            f"max_abs={_format_number(_max_abs(tensor))} status={status}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = project_path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run lacks config_resolved.yaml: {run_dir}")
    config = load_yaml(config_path)
    version = config.get("experiment", {}).get("version")
    if version != EXPECTED_VERSION:
        raise ValueError(
            f"{EXPECTED_VERSION} numerical diagnostic refuses run version {version!r}"
        )
    checkpoint_path = _checkpoint_path(run_dir, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
    if checkpoint.get("version") != EXPECTED_VERSION:
        raise ValueError(
            f"Checkpoint version is {checkpoint.get('version')!r}, expected {EXPECTED_VERSION!r}"
        )
    model_state = checkpoint.get("model_state_dict")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint lacks model_state_dict")
    if not isinstance(optimizer_state, dict):
        raise ValueError("Checkpoint lacks optimizer_state_dict")

    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint epoch: {int(checkpoint['epoch'])}")
    print(f"checkpoint version: {checkpoint['version']}")
    model_reports = inspect_tree("model_state_dict", model_state)
    optimizer_reports = inspect_tree("optimizer_state_dict", optimizer_state)

    model = build_model(config["model"]).cpu()
    model.load_state_dict(model_state, strict=True)
    print("strict model_state_dict load: passed")
    print_top_model_magnitudes(model)
    print_batch_norm_diagnostics(model)
    print_selected_v18_state(model)
    overall = "NON_FINITE" if model_reports or optimizer_reports else "finite"
    print(f"overall checkpoint numerical status: {overall}")


if __name__ == "__main__":
    main()
