#!/usr/bin/env python3
"""Run selected UIE4 versions on official U45 and evaluate UIQM/UCIQE.

U45 is input-only and never participates in training, validation, checkpoint
selection or tuning.  All predictions and metrics are committed transactionally
only after all 45 images complete successfully.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.paper_evaluation import (
    SUPPORTED_PAPER_VERSIONS,
    resolve_dataset_name,
    sha256_file,
)
from src.shared.transactional_output import transactional_output_directory
from src.shared.u45_evaluation import (
    U45Entry,
    decode_canonical_input,
    decode_prediction_png8,
    inspect_official_u45,
    u45_metric_protocol,
    uciqe,
    uiqm,
)
from src.v16.utils import load_yaml


CHECKPOINT_SELECTORS = {"best_psnr", "best_ssim", "best_loss", "last"}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        path = Path(*path.parts[1:])
    return (PROJECT_ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-selection-criterion")
    parser.add_argument("--data-root", required=True, help="Clean official U45/U45 directory")
    parser.add_argument("--output-dir")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
    candidates = ((run_dir / path).resolve(), project_path(path))
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _selection_criterion(selector: str, explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    known = {
        "best_psnr": "source-domain validation PSNR",
        "best_ssim": "source-domain validation SSIM",
        "best_loss": "source-domain validation loss",
        "last": "last training epoch; never selected using U45",
    }
    if selector not in known:
        raise ValueError(
            "An explicit checkpoint path requires --checkpoint-selection-criterion "
            "to record how it was selected before U45 evaluation"
        )
    return known[selector]


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, dict):
        raise ValueError("Checkpoint must contain a mapping")
    return value


def _select_device(gpu: int | None) -> torch.device:
    if not torch.cuda.is_available():
        if gpu is not None:
            raise ValueError("--gpu was supplied but CUDA is unavailable")
        return torch.device("cpu")
    index = 0 if gpu is None else gpu
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"GPU index {index} is unavailable; count={torch.cuda.device_count()}")
    device = torch.device("cuda", index)
    torch.cuda.set_device(device)
    return device


def _input_tensor(rgb: np.ndarray, evaluation: dict[str, Any]) -> torch.Tensor:
    image = Image.fromarray(rgb, mode="RGB")
    if bool(evaluation.get("resize", True)):
        size = int(evaluation["size"])
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


def _prediction_image(prediction: torch.Tensor, *, native_size: tuple[int, int]) -> Image.Image:
    if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[1] != 3:
        raise ValueError(f"Expected model output [1,3,H,W], got {tuple(prediction.shape)}")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Model produced NaN/Inf on U45")
    value = prediction.detach().float()
    native_width, native_height = native_size
    if value.shape[-2:] != (native_height, native_width):
        value = F.interpolate(
            value,
            size=(native_height, native_width),
            mode="bilinear",
            align_corners=False,
        )
    array = (
        value[0]
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _write_manifest_snapshot(path: Path, entries: list[U45Entry]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=(
                "sample_id",
                "filename",
                "sha256",
                "git_blob_sha1",
                "width",
                "height",
                "original_mode",
            ),
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "sample_id": entry.sample_id,
                    "filename": entry.filename,
                    "sha256": entry.sha256,
                    "git_blob_sha1": entry.git_blob_sha1,
                    "width": entry.width,
                    "height": entry.height,
                    "original_mode": entry.original_mode,
                }
            )


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "numpy", "Pillow", "scikit-image"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    run_dir = project_path(args.run_dir)
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run lacks config_resolved.yaml: {run_dir}")
    config = load_yaml(config_path)
    version = str(config.get("experiment", {}).get("version", ""))
    if version not in SUPPORTED_PAPER_VERSIONS:
        raise ValueError(
            f"Supported versions are {sorted(SUPPORTED_PAPER_VERSIONS)}, got {version!r}"
        )
    source_dataset = resolve_dataset_name(config["data"])
    source_branch = "lsui" if source_dataset == "LSUI19" else "uieb"
    selector = str(args.checkpoint or config.get("test", {}).get("checkpoint", "best_psnr"))
    criterion = _selection_criterion(selector, args.checkpoint_selection_criterion)
    checkpoint_path = _checkpoint_path(run_dir, selector)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint_hash = sha256_file(checkpoint_path)

    # Verify all 45 original files before loading the model or creating output.
    data_root = project_path(args.data_root)
    entries = inspect_official_u45(data_root)
    if len(entries) != 45:
        raise RuntimeError(f"Official U45 inspection returned {len(entries)} entries")

    device = _select_device(args.gpu)
    model_module = importlib.import_module(f"src.{version}.models")
    model = model_module.build_model(config["model"]).to(device)
    checkpoint = _torch_load(checkpoint_path, device)
    if checkpoint.get("version") != version:
        raise ValueError(
            f"Checkpoint version {checkpoint.get('version')!r} does not match {version!r}"
        )
    checkpoint_model = checkpoint.get("resolved_config", {}).get("model")
    if checkpoint_model != config["model"]:
        raise ValueError("Checkpoint model config differs from run config_resolved.yaml")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    evaluation = config.get("evaluation", {})
    if bool(evaluation.get("resize", True)) and int(evaluation.get("size", 0)) < 1:
        raise ValueError("evaluation.size must be positive")
    amp_enabled = bool(config.get("training", {}).get("amp", False)) and device.type == "cuda"
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir is not None
        else (run_dir / "result" / "u45" / f"from_{source_branch}").resolve()
    )

    rows: list[dict[str, Any]] = []
    total_inference_seconds = 0.0
    with transactional_output_directory(output_dir, overwrite=args.overwrite) as staging:
        predictions_dir = staging / "predictions"
        predictions_dir.mkdir()
        _write_manifest_snapshot(staging / "u45_manifest_snapshot.tsv", entries)
        with torch.inference_mode():
            for index, entry in enumerate(entries, start=1):
                rgb, _ = decode_canonical_input(entry.path)
                tensor = _input_tensor(rgb, evaluation).to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    prediction = model(tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                total_inference_seconds += time.perf_counter() - started
                image = _prediction_image(
                    prediction, native_size=(entry.width, entry.height)
                )
                prediction_path = predictions_dir / entry.filename
                image.save(prediction_path, format="PNG")
                decoded = decode_prediction_png8(
                    prediction_path, expected_size=(entry.width, entry.height)
                )
                uiqm_value = uiqm(decoded)
                uciqe_value = uciqe(decoded)
                if not math.isfinite(uiqm_value) or not math.isfinite(uciqe_value):
                    raise FloatingPointError(
                        f"Non-finite U45 metric for sample_id={entry.sample_id}"
                    )
                rows.append(
                    {
                        "sample_id": entry.sample_id,
                        "input_filename": entry.filename,
                        "input_sha256": entry.sha256,
                        "prediction": entry.filename,
                        "prediction_sha256": sha256_file(prediction_path),
                        "width": entry.width,
                        "height": entry.height,
                        "uiqm": uiqm_value,
                        "uciqe": uciqe_value,
                    }
                )
                print(f"U45 inference/evaluation: {index}/45 sample_id={entry.sample_id}")

        # Revalidate every source, output and checkpoint before committing.
        revalidated = inspect_official_u45(data_root)
        if revalidated != entries:
            raise ValueError("U45 inputs changed during inference")
        if sha256_file(checkpoint_path) != checkpoint_hash:
            raise ValueError("Checkpoint changed during U45 inference")
        actual_prediction_names = {path.name for path in predictions_dir.iterdir()}
        expected_prediction_names = {entry.filename for entry in entries}
        if actual_prediction_names != expected_prediction_names:
            raise ValueError("U45 prediction directory is incomplete or contains extras")
        for entry in entries:
            decode_prediction_png8(
                predictions_dir / entry.filename,
                expected_size=(entry.width, entry.height),
            )

        with (staging / "per_image_metrics.csv").open(
            "x", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        provenance = {
            "source_training_dataset": source_dataset,
            "run_dir": str(run_dir),
            "version": version,
            "checkpoint_selector_or_path": selector,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "selection_criterion_fixed_before_u45": criterion,
            "u45_used_for_checkpoint_selection": False,
        }
        with (staging / "source_checkpoint.json").open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        summary = {
            "protocol": u45_metric_protocol(),
            "run_dir": str(run_dir),
            "version": version,
            "dataset": "U45",
            "dataset_role": "test_only_no_reference",
            "source_training_dataset": source_dataset,
            "checkpoint": provenance,
            "sample_count": len(rows),
            "prediction_count": len(rows),
            "prediction_representation": "RGB PNG8 at original U45 width and height",
            "model_input_transform": (
                f"Pillow RGB bilinear resize to {int(evaluation['size'])}x{int(evaluation['size'])}; "
                "float model output restored to original size with PyTorch bilinear "
                "align_corners=False before PNG8 encoding"
                if bool(evaluation.get("resize", True))
                else "native-size RGB input; model output must be native-size"
            ),
            "mean_uiqm": sum(float(row["uiqm"]) for row in rows) / len(rows),
            "mean_uciqe": sum(float(row["uciqe"]) for row in rows) / len(rows),
            "aggregation": "arithmetic mean of all 45 per-image scalars",
            "comparison_group": (
                f"{u45_metric_protocol()['protocol_version']}:{source_branch}:portable_png8"
            ),
            "total_inference_seconds": total_inference_seconds,
            "average_inference_seconds": total_inference_seconds / len(rows),
            "metric_time_included_in_inference": False,
            "device": str(device),
            "amp": amp_enabled,
            "environment": _dependency_versions(),
            "config_resolved_sha256": sha256_file(config_path),
            "transactional_output": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (staging / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    return output_dir, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_dir, summary = run(args)
    except (ImportError, ValueError, OSError, RuntimeError, FloatingPointError) as error:
        print(f"U45_NOT_SAVED: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_dir),
                "mean_uiqm": summary["mean_uiqm"],
                "mean_uciqe": summary["mean_uciqe"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
