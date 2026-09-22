#!/usr/bin/env python3
"""Re-evaluate selected-version PNG8 test outputs with paper metrics.

This tool never loads a model and never changes native test artifacts.  It
computes PSNR, SSIM, DeltaE00 and fixed AlexNet LPIPS from the saved RGB PNG8
predictions so every method in a paper table uses the same representation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.e00 import batch_delta_e00
from src.shared.paper_evaluation import (
    CANONICAL_TEST_MANIFESTS,
    DependencyBlocked,
    LPIPSMetric,
    METRIC_CONFIG,
    PROTOCOL_VERSION,
    SUPPORTED_PAPER_VERSIONS,
    decode_rgb_png8,
    json_score,
    official_enhanced_names,
    paper_metric_protocol,
    resolve_dataset_name,
    rgb_array_to_tensor,
    select_device,
    sha256_file,
    validate_exact_prediction_directory,
)
from src.shared.transactional_output import transactional_output_directory
from src.v16.dataset import LSUIDataset, read_manifest
from src.v16.metrics import batch_metrics
from src.v16.utils import load_yaml


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
    parser.add_argument("--data-root")
    parser.add_argument("--prediction-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--allow-lpips-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "torchvision", "lpips", "numpy", "Pillow", "scikit-image"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_result(staging: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    with (staging / "per_image_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (staging / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def evaluate(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("--batch-size and --threads must be positive")
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
    dataset_name = resolve_dataset_name(config["data"])
    expected_count, expected_manifest_hash = CANONICAL_TEST_MANIFESTS[dataset_name]
    evaluation = config.get("evaluation", {})
    if not bool(evaluation.get("resize", True)) or int(evaluation.get("size", 0)) != 256:
        raise ValueError("Paper PNG8 protocol requires evaluation.resize=true and size=256")

    manifest = (run_dir / "split_snapshot" / "test.tsv").resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Frozen test manifest is missing: {manifest}")
    manifest_hash = sha256_file(manifest)
    if manifest_hash != expected_manifest_hash:
        raise ValueError(
            f"Frozen {dataset_name} test manifest SHA256 differs from the paper protocol: "
            f"{manifest_hash}"
        )
    entries = read_manifest(manifest)
    if len(entries) != expected_count:
        raise ValueError(f"Frozen test count is {len(entries)}, expected {expected_count}")

    data_root = (
        project_path(args.data_root)
        if args.data_root is not None
        else project_path(config["data"]["root"])
    )
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root is unavailable: {data_root}; provide --data-root"
        )
    dataset = LSUIDataset(
        manifest,
        data_root,
        "test",
        int(config["data"]["patch_size"]),
        config["data"]["augmentation"],
        bool(config["data"]["pad_if_smaller"]),
        str(config["data"]["pad_mode"]),
        evaluation,
        verify_files=True,
    )
    names = official_enhanced_names(entries)
    prediction_dir = (
        project_path(args.prediction_dir)
        if args.prediction_dir is not None
        else (run_dir / "result" / "test_all_enhanced").resolve()
    )
    paths_by_name = validate_exact_prediction_directory(
        prediction_dir, [names[entry.sample_id] for entry in entries]
    )
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir is not None
        else (run_dir / "result" / "paper_metrics_png8").resolve()
    )
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite only when replacing this complete result."
        )

    # Validate every immutable input before initializing LPIPS or computing metrics.
    prediction_hashes: dict[str, str] = {}
    gt_hashes: dict[str, str] = {}
    for entry in entries:
        prediction_path = paths_by_name[names[entry.sample_id]]
        decode_rgb_png8(prediction_path, expected_size=(256, 256))
        prediction_hashes[entry.sample_id] = sha256_file(prediction_path)
        gt_path = data_root / entry.gt_relative
        gt_hashes[entry.sample_id] = sha256_file(gt_path)

    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    lpips_metric = LPIPSMetric(device, allow_download=args.allow_lpips_download)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(entries), args.batch_size):
            batch_entries = entries[start : start + args.batch_size]
            predictions = []
            targets = []
            for offset, entry in enumerate(batch_entries):
                index = start + offset
                item = dataset[index]
                if item["id"] != entry.sample_id:
                    raise RuntimeError("Dataset order differs from frozen manifest")
                prediction_path = paths_by_name[names[entry.sample_id]]
                if sha256_file(prediction_path) != prediction_hashes[entry.sample_id]:
                    raise ValueError(f"Prediction changed during evaluation: {prediction_path}")
                gt_path = data_root / entry.gt_relative
                if sha256_file(gt_path) != gt_hashes[entry.sample_id]:
                    raise ValueError(f"GT changed during evaluation: {gt_path}")
                predictions.append(
                    rgb_array_to_tensor(
                        decode_rgb_png8(prediction_path, expected_size=(256, 256))
                    )
                )
                targets.append(item["target"])
            prediction_batch = torch.stack(predictions).to(device=device, dtype=torch.float32)
            target_batch = torch.stack(targets).to(device=device, dtype=torch.float32)
            psnr, ssim = batch_metrics(prediction_batch, target_batch, METRIC_CONFIG)
            delta_e00 = batch_delta_e00(prediction_batch, target_batch, METRIC_CONFIG)
            lpips_values = lpips_metric(prediction_batch, target_batch)
            for offset, entry in enumerate(batch_entries):
                values = {
                    "psnr": float(psnr[offset]),
                    "ssim": float(ssim[offset]),
                    "delta_e00": float(delta_e00[offset]),
                    "lpips": float(lpips_values[offset]),
                }
                for key, value in values.items():
                    if not math.isfinite(value) and not (key == "psnr" and value == math.inf):
                        raise FloatingPointError(
                            f"Invalid {key} for sample_id={entry.sample_id}: {value}"
                        )
                rows.append(
                    {
                        "sample_id": entry.sample_id,
                        "input_relative": entry.input_relative,
                        "gt_relative": entry.gt_relative,
                        "prediction": names[entry.sample_id],
                        "prediction_sha256": prediction_hashes[entry.sample_id],
                        "gt_sha256": gt_hashes[entry.sample_id],
                        **values,
                    }
                )
            print(f"paper metrics: {min(start + args.batch_size, len(entries))}/{len(entries)}")

    if sha256_file(manifest) != manifest_hash:
        raise ValueError("Frozen test manifest changed during evaluation")
    means = {
        f"mean_{key}": json_score(sum(float(row[key]) for row in rows) / len(rows))
        for key in ("psnr", "ssim", "delta_e00", "lpips")
    }
    native_summary = _load_json(run_dir / "result" / "test_summary.json")
    summary = {
        "protocol": paper_metric_protocol(),
        "protocol_version": PROTOCOL_VERSION,
        "comparison_group": (
            f"{PROTOCOL_VERSION}:portable_png8:{dataset_name}:test:{manifest_hash}"
        ),
        "final_paper_table_eligible": True,
        "run_dir": str(run_dir),
        "version": version,
        "dataset": dataset_name,
        "split": "test",
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "sample_count": len(rows),
        "prediction_count": len(rows),
        "prediction_dir": str(prediction_dir),
        "data_root": str(data_root),
        "source_checkpoint": native_summary.get("checkpoint"),
        "source_checkpoint_epoch": native_summary.get("checkpoint_epoch"),
        **means,
        "lpips_runtime": {"status": "PASS", "weights": lpips_metric.weights},
        "environment": _dependency_versions(),
        "device": str(device),
        "amp": False,
        "tf32": False if device.type == "cuda" else "not_applicable",
        "transactional_output": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with transactional_output_directory(output_dir, overwrite=args.overwrite) as staging:
        _write_result(staging, rows, summary)
    return output_dir, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_dir, summary = evaluate(args)
    except (DependencyBlocked, ImportError) as error:
        print(f"BLOCKED_DEPENDENCY: {error}", file=sys.stderr)
        return 3
    except (ValueError, OSError, RuntimeError, FloatingPointError) as error:
        print(f"PAPER_EVALUATION_NOT_SAVED: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_dir),
                "mean_lpips": summary["mean_lpips"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
