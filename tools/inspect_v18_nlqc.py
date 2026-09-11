#!/usr/bin/env python3
"""Print the four learned V18 NLQC calibration scalars from a checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v18.test import _checkpoint_path, _torch_load
from src.v18.utils import load_yaml, project_path


EXPECTED_VERSION = "v18"
ALPHA_KEYS = tuple(
    f"backbone.guide{stage}.nlqc_alpha" for stage in (4, 3, 2, 1)
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the learned per-stage NLQC alpha values in a V18 checkpoint"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args(argv)


def read_alpha_values(checkpoint: dict[str, Any]) -> dict[str, float]:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint lacks model_state_dict")
    values: dict[str, float] = {}
    for key in ALPHA_KEYS:
        if key not in state:
            raise KeyError(f"Checkpoint lacks required V18 parameter: {key}")
        tensor = state[key]
        if not torch.is_tensor(tensor) or tensor.numel() != 1:
            raise ValueError(f"Expected scalar tensor for {key}")
        value = float(tensor.detach().float().cpu().item())
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite NLQC alpha in checkpoint: {key}")
        values[key] = value
    return values


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = project_path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run lacks config_resolved.yaml: {run_dir}")
    config = load_yaml(config_path)
    if config.get("experiment", {}).get("version") != EXPECTED_VERSION:
        raise ValueError(
            f"{EXPECTED_VERSION} inspector refuses run version "
            f"{config.get('experiment', {}).get('version')!r}"
        )
    selector = args.checkpoint or config["test"]["checkpoint"]
    checkpoint_path = _checkpoint_path(run_dir, selector)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
    if checkpoint.get("version") != EXPECTED_VERSION:
        raise ValueError(
            f"Checkpoint version is {checkpoint.get('version')!r}, expected {EXPECTED_VERSION!r}"
        )
    values = read_alpha_values(checkpoint)

    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint epoch: {int(checkpoint['epoch'])}")
    for key, value in values.items():
        print(f"{key}: {value:.10g}  abs={abs(value):.10g}")


if __name__ == "__main__":
    main()
