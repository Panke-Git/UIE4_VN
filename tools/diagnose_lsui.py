#!/usr/bin/env python3
"""Run model-free LSUI split, difficulty, and duplicate diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import platform
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# ``python tools/diagnose_lsui.py`` puts tools/, not the repository root, on
# sys.path. Add the root so script execution and test imports use one package.
SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from tools.dataset_diagnostics.common import (
    EXPECTED_SPLIT_COUNTS,
    PROJECT_ROOT,
    SPLIT_ORDER,
    analyze_split_integrity,
    copy_manifests,
    create_run_directory,
    enforce_split_counts,
    load_config,
    read_manifest,
    resolve_project_path,
    sha256_file,
    write_csv,
    write_json,
)
from tools.dataset_diagnostics.difficulty import (
    analyze_pair,
    difficulty_extremes,
    source_distribution,
    summarize_difficulty,
    summarize_image_statistics,
    summarize_resize_effect,
    summarize_resolution,
    write_difficulty_outputs,
)
from tools.dataset_diagnostics.duplicates import build_duplicate_results, write_duplicate_outputs
from tools.dataset_diagnostics.report import write_report


ERROR_FIELDS = ("split", "sample_id", "path", "error")


def _resolve_manifest(config: Mapping[str, Any], split: str) -> Path:
    key = "validation_manifest" if split == "validation" else f"{split}_manifest"
    return resolve_project_path(config["data"][key])


def _base_run_info(
    *,
    timestamp: datetime,
    config_path: Path,
    data_root: Path,
    dhash_threshold: int,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "command": shlex.join(sys.argv),
        "data_root": str(data_root),
        "config_path": str(config_path),
        "python_version": platform.python_version(),
        "pillow_version": Image.__version__,
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "train_tsv_sha256": None,
        "validation_tsv_sha256": None,
        "test_tsv_sha256": None,
        "metric_protocol": (
            "Copied from current UIE4 evaluation: RGB float [0,1], data_range=1, "
            "Gaussian-window SSIM window_size=11 sigma=1.5 (values read from config)."
        ),
        "resize_protocol": "PIL RGB deterministic 256x256 Image.Resampling.BILINEAR",
        "dhash_threshold": dhash_threshold,
        "status": "running",
    }


def run_diagnostics(
    *,
    config_path: Path,
    data_root_override: Path | None = None,
    output_root: Path | None = None,
    dhash_threshold: int = 4,
    expected_counts: Mapping[str, int] = EXPECTED_SPLIT_COUNTS,
    generate_plots: bool = True,
    show_progress: bool = True,
) -> Path:
    started = time.monotonic()
    timestamp = datetime.now()
    config_path = config_path.expanduser()
    config_path = config_path.resolve() if config_path.is_absolute() else (PROJECT_ROOT / config_path).resolve()
    output_root = (output_root or PROJECT_ROOT / "diagnostics").expanduser().resolve()
    run_dir = create_run_directory(output_root, timestamp)
    error_path = run_dir / "diagnostic_errors.csv"
    errors: list[dict[str, Any]] = []
    write_csv(error_path, errors, ERROR_FIELDS)

    data_root = Path("<unresolved>")
    run_info = _base_run_info(
        timestamp=timestamp,
        config_path=config_path,
        data_root=data_root,
        dhash_threshold=dhash_threshold,
    )
    run_info_path = run_dir / "metadata" / "run_info.json"
    try:
        if not 0 <= dhash_threshold <= 64:
            raise ValueError("--dhash-threshold must be between 0 and 64")
        if generate_plots and importlib.util.find_spec("matplotlib") is None:
            raise RuntimeError(
                "matplotlib is required for diagnostic histograms; install requirements.txt first"
            )
        config = load_config(config_path)
        configured_root = Path(str(config["data"]["root"])).expanduser()
        data_root = (data_root_override or configured_root).resolve()
        run_info["data_root"] = str(data_root)
        if not data_root.is_dir():
            raise FileNotFoundError(f"LSUI data root is unavailable: {data_root}")

        evaluation = config.get("evaluation", {})
        evaluation_size = int(evaluation.get("size", 256))
        if not bool(evaluation.get("resize", True)) or evaluation_size != 256:
            raise ValueError(
                "This diagnostic defines Current-256 from evaluation.resize=true and evaluation.size=256"
            )
        metric_config = config.get("metrics", {})
        run_info["metric_protocol"] = (
            "Copied from current UIE4 evaluation: RGB float [0,1], "
            f"data_range={float(metric_config.get('data_range', 1.0))}, "
            f"crop_border={int(metric_config.get('crop_border', 0))}, Gaussian-window SSIM "
            f"window_size={int(metric_config.get('ssim_window_size', 11))}, "
            f"sigma={float(metric_config.get('ssim_sigma', 1.5))}."
        )
        run_info["resize_protocol"] = (
            f"PIL RGB deterministic {evaluation_size}x{evaluation_size} Image.Resampling.BILINEAR"
        )
        manifests = {split: _resolve_manifest(config, split) for split in SPLIT_ORDER}
        for split, path in manifests.items():
            if not path.is_file():
                errors.append(
                    {
                        "split": split,
                        "sample_id": "",
                        "path": str(path),
                        "error": "FileNotFoundError: manifest is unavailable",
                    }
                )
        if errors:
            raise FileNotFoundError(f"One or more split manifests are unavailable; see {error_path}")
        copy_manifests(manifests, run_dir / "metadata")
        entries = {}
        for split, path in manifests.items():
            try:
                entries[split] = read_manifest(path)
            except Exception as error:
                errors.append(
                    {
                        "split": split,
                        "sample_id": "",
                        "path": str(path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        if errors:
            raise ValueError(f"One or more split manifests are invalid; see {error_path}")
        integrity = analyze_split_integrity(entries, expected_counts)
        enforce_split_counts(integrity)
        for split in SPLIT_ORDER:
            run_info[f"{split}_tsv_sha256"] = sha256_file(manifests[split])
        write_json(run_info_path, run_info)

        metrics_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
        image_rows: list[dict[str, Any]] = []
        resolution_rows: list[dict[str, Any]] = []
        fingerprints = []
        for split in SPLIT_ORDER:
            iterator = tqdm(entries[split], desc=f"difficulty:{split}", unit="pair", disable=not show_progress)
            for entry in iterator:
                try:
                    diagnostic = analyze_pair(
                        split=split,
                        entry=entry,
                        data_root=data_root,
                        evaluation_size=evaluation_size,
                        metric_config=metric_config,
                    )
                except Exception as error:
                    errors.append(
                        {
                            "split": split,
                            "sample_id": entry.sample_id,
                            "path": f"{entry.input_relative} | {entry.gt_relative}",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                metrics_by_split[split].append(diagnostic.metrics)
                image_rows.append(diagnostic.image_statistics)
                resolution_rows.append(diagnostic.resolution)
                fingerprints.extend(diagnostic.fingerprints)
        if errors:
            write_csv(error_path, errors, ERROR_FIELDS)
            raise RuntimeError(f"Dataset diagnostics found {len(errors)} image error(s); see {error_path}")

        difficulty_summary = summarize_difficulty(metrics_by_split)
        resize_summary = summarize_resize_effect(metrics_by_split)
        resolution_summary = summarize_resolution(resolution_rows)
        image_summary = summarize_image_statistics(image_rows)
        source_rows = source_distribution(entries)
        write_difficulty_outputs(
            run_dir / "difficulty",
            metrics_by_split=metrics_by_split,
            image_rows=image_rows,
            resolution_rows=resolution_rows,
            split_summary=difficulty_summary,
            resize_summary=resize_summary,
            source_rows=source_rows,
            generate_plots=generate_plots,
        )

        duplicate_results = build_duplicate_results(fingerprints, dhash_threshold)
        write_duplicate_outputs(run_dir / "duplicates", duplicate_results)
        duplicate_summary = duplicate_results["summary"]
        summary = {
            "split_counts": integrity,
            "difficulty": difficulty_summary,
            "resize_effect": resize_summary,
            "exact_duplicates": duplicate_summary,
            "near_duplicate_candidates": {
                "dhash_threshold": dhash_threshold,
                "near_duplicate_candidate_pairs": duplicate_summary["near_duplicate_candidate_pairs"],
                "near_duplicate_candidate_pairs_by_type": duplicate_summary[
                    "near_duplicate_candidate_pairs_by_type"
                ],
            },
            "resolution": resolution_summary,
            "image_statistics": image_summary,
            "source_folder_distribution": source_rows,
        }
        extremes = difficulty_extremes(metrics_by_split)
        write_json(run_dir / "summary.json", summary)
        run_info["status"] = "completed"
        run_info["duration_seconds"] = time.monotonic() - started
        write_json(run_info_path, run_info)
        write_report(run_dir / "report.md", run_info=run_info, summary=summary, extremes=extremes)
        return run_dir
    except Exception as error:
        if not errors:
            errors.append(
                {
                    "split": "",
                    "sample_id": "",
                    "path": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        write_csv(error_path, errors, ERROR_FIELDS)
        run_info["status"] = "failed"
        run_info["duration_seconds"] = time.monotonic() - started
        run_info["error"] = f"{type(error).__name__}: {error}"
        write_json(run_info_path, run_info)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config_v1.yaml")
    parser.add_argument("--data-root", default=None, help="Override data.root from YAML")
    parser.add_argument("--dhash-threshold", type=int, default=4)
    parser.add_argument("--output-root", default=None, help="Default: <project>/diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_diagnostics(
        config_path=Path(args.config),
        data_root_override=Path(args.data_root).expanduser() if args.data_root else None,
        output_root=Path(args.output_root).expanduser() if args.output_root else None,
        dhash_threshold=args.dhash_threshold,
    )
    print(f"LSUI diagnostics completed: {run_dir}")


if __name__ == "__main__":
    main()
