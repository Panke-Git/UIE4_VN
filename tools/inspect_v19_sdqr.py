#!/usr/bin/env python3
"""Inspect the four learned V19 SDQR coefficient vectors without modifying state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v19.models import SDQRPlainUNetColorQuery, build_model
from src.v19.utils import load_yaml, project_path


CHECKPOINT_SELECTORS = {"best_psnr", "best_ssim", "best_loss", "last"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best_psnr")
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def _checkpoint_path(run_dir: Path, selector: str) -> Path:
    if selector in CHECKPOINT_SELECTORS:
        return (
            run_dir / "checkpoint" / "last.pt"
            if selector == "last"
            else run_dir / "best" / f"{selector}.pt"
        ).resolve()
    path = Path(selector).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = ((run_dir / path).resolve(), project_path(path).resolve())
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError("Checkpoint must contain a mapping")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = project_path(args.run_dir).resolve()
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run lacks config_resolved.yaml: {run_dir}")
    config = load_yaml(config_path)
    if config.get("experiment", {}).get("version") != "v19":
        raise ValueError("SDQR inspection requires a V19 run")

    checkpoint_path = _checkpoint_path(run_dir, str(args.checkpoint))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = _torch_load(checkpoint_path)
    if checkpoint.get("version") != "v19":
        raise ValueError(f"Checkpoint version is not v19: {checkpoint.get('version')!r}")
    if checkpoint.get("resolved_config", {}).get("model") != config.get("model"):
        raise ValueError("Checkpoint model config differs from config_resolved.yaml")

    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not isinstance(model.backbone, SDQRPlainUNetColorQuery):
        raise TypeError("V19 model does not contain the expected SDQR backbone")

    report: dict[str, Any] = {
        "version": "v19",
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "all_beta_finite": True,
        "beta_count": 0,
        "stages": {},
    }
    print(f"checkpoint path: {checkpoint_path}")
    print(f"checkpoint epoch: {report['epoch']}")
    for stage in (4, 3, 2, 1):
        beta_tensor = getattr(model.backbone, f"sdqr_beta{stage}").detach().float().cpu()
        eta_tensor = torch.tanh(beta_tensor)
        multiplier_tensor = 1.0 + eta_tensor
        finite = bool(
            torch.isfinite(beta_tensor).all()
            and torch.isfinite(eta_tensor).all()
            and torch.isfinite(multiplier_tensor).all()
        )
        report["all_beta_finite"] = bool(report["all_beta_finite"] and finite)
        beta = beta_tensor.reshape(-1).tolist()
        eta = eta_tensor.reshape(-1).tolist()
        multiplier = multiplier_tensor.reshape(-1).tolist()
        stage_report = {
            "beta": beta,
            "eta": eta,
            "multiplier": multiplier,
            "finite": finite,
            "eta_min": min(eta),
            "eta_max": max(eta),
            "eta_mean": sum(eta) / len(eta),
            "eta_abs_mean": sum(abs(value) for value in eta) / len(eta),
        }
        report["stages"][str(stage)] = stage_report
        report["beta_count"] += len(beta)
        print(f"stage{stage}:")
        for query, (beta_value, eta_value, multiplier_value) in enumerate(
            zip(beta, eta, multiplier, strict=True)
        ):
            print(
                f"  query{query} beta={beta_value:.9g} eta={eta_value:.9g} "
                f"multiplier={multiplier_value:.9g}"
            )
        print(
            f"  eta_min={stage_report['eta_min']:.9g} "
            f"eta_max={stage_report['eta_max']:.9g} "
            f"eta_mean={stage_report['eta_mean']:.9g} "
            f"eta_abs_mean={stage_report['eta_abs_mean']:.9g}"
        )
    print(f"all {report['beta_count']} beta finite: {report['all_beta_finite']}")
    if not report["all_beta_finite"]:
        raise FloatingPointError("V19 checkpoint contains non-finite SDQR state")
    if args.output_json is not None:
        output_path = project_path(args.output_json).resolve()
        _atomic_json(output_path, report)
        print(f"output json: {output_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    inspect(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
