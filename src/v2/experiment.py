"""Auditable experiment-directory lifecycle and provenance capture."""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .utils import PROJECT_ROOT, atomic_json, parameter_counts, project_path, save_yaml, sha256_file


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_info() -> dict[str, Any]:
    def command(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = command("rev-parse", "HEAD")
    branch = command("branch", "--show-current") if commit else None
    dirty_output = command("status", "--porcelain") if commit else None
    return {"git_branch": branch, "git_commit": commit, "git_dirty": bool(dirty_output) if commit else None}


def split_paths(config: dict) -> dict[str, Path]:
    return {
        "train": project_path(config["data"]["train_manifest"]).resolve(),
        "validation": project_path(config["data"]["validation_manifest"]).resolve(),
        "test": project_path(config["data"]["test_manifest"]).resolve(),
    }


def initialize_run(
    config: dict,
    source_config_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    sample_counts: dict[str, int],
    command_line: list[str],
) -> tuple[Path, dict]:
    timestamp = datetime.now().astimezone()
    version = config["experiment"]["version"]
    name = config["experiment"]["name"]
    seed = int(config["experiment"]["seed"])
    run_name = f"{version}_{name}_seed{seed}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    output_root = project_path(config["experiment"]["output_root"])
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Experiment directory already exists: {run_dir}")
    for child in ("best", "checkpoint", "log", "result", "split_snapshot"):
        (run_dir / child).mkdir(parents=True, exist_ok=False)

    shutil.copyfile(source_config_path, run_dir / "config_source.yaml")
    save_yaml(run_dir / "config_resolved.yaml", config)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifests = split_paths(config)
    hashes: dict[str, str] = {}
    for split, path in manifests.items():
        if not path.is_file():
            raise FileNotFoundError(f"Project-local {split} manifest is unavailable: {path}")
        target = run_dir / "split_snapshot" / path.name
        shutil.copyfile(path, target)
        source_hash, target_hash = sha256_file(path), sha256_file(target)
        if source_hash != target_hash:
            raise RuntimeError(f"Split snapshot hash mismatch for {split}")
        hashes[split] = source_hash

    total_params, trainable_params = parameter_counts(model)
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU"
    )
    start_time = timestamp.isoformat(timespec="seconds")
    run_info: dict[str, Any] = {
        "experiment_name": name,
        "version": version,
        "timestamp": timestamp.strftime("%Y%m%d_%H%M%S"),
        "seed": seed,
        "command_line": command_line,
        "source_config": str(source_config_path),
        "resolved_config": config,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu_name": device_name,
        "gpu_count": torch.cuda.device_count(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "model_total_params": total_params,
        "model_trainable_params": trainable_params,
        "train_sample_count": sample_counts["train"],
        "validation_sample_count": sample_counts["validation"],
        "test_sample_count": sample_counts["test"],
        "train_tsv_sha256": hashes["train"],
        "validation_tsv_sha256": hashes["validation"],
        "test_tsv_sha256": hashes["test"],
        "start_time": start_time,
        **_git_info(),
    }
    atomic_json(run_dir / "run_info.json", run_info)
    status = {
        "status": "running",
        "start_time": start_time,
        "end_time": None,
        "last_epoch": 0,
        "best_psnr": None,
        "best_ssim": None,
        "best_val_loss": None,
    }
    atomic_json(run_dir / "status.json", status)
    return run_dir, status


def load_status(run_dir: Path) -> dict:
    path = run_dir / "status.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "running",
        "start_time": now_iso(),
        "end_time": None,
        "last_epoch": 0,
        "best_psnr": None,
        "best_ssim": None,
        "best_val_loss": None,
    }


def update_status(run_dir: Path, state: dict, **updates: Any) -> dict:
    """Update and atomically persist run state without colliding with ``status=``."""
    state.update(updates)
    atomic_json(run_dir / "status.json", state)
    return state
