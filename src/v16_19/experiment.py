"""Independent run directories and frozen data/code identity for V16-19."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from .utils import atomic_json, parameter_counts, project_path, runtime_code_hashes, save_yaml


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def initialize_run(config: dict, source: Path, model, device, protocol_dir: Path, protocol: dict, command_line: list[str]):
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    name = str(config["experiment"]["name"])
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Experiment name must be a single safe path component")
    run = project_path(config["experiment"]["output_root"]) / f"v16_19_{name}_seed{config['experiment']['seed']}_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    for child in ("best", "checkpoint", "log", "result/validation_latest", "split_snapshot"):
        (run / child).mkdir(parents=True)
    shutil.copyfile(source, run / "config_source.yaml")
    save_yaml(run / "config_resolved.yaml", config)
    atomic_json(run / "config.json", config)
    for filename in ("train.tsv", "validation.tsv", "file_index.json", "data_quality.json", "protocol.json"):
        shutil.copyfile(protocol_dir / filename, run / "split_snapshot" / filename)
    package = Path(__file__).resolve().parent
    for relative in config["runtime_code_sha256"]:
        target = run / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / relative, target)
    shutil.copyfile(package / "snapshot_provenance.json", run / "code_snapshot/snapshot_provenance.json")
    total, trainable = parameter_counts(model)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source.parent, capture_output=True, text=True)
    quality = json.loads((protocol_dir / "data_quality.json").read_text(encoding="utf-8"))
    info = {"display_name": "V16-19", "version": "v16_19", "dataset": config["data"]["dataset"],
            "protocol": protocol["protocol"], "dataset_fingerprint": protocol["fingerprint"],
            "sample_counts": protocol["counts"], "has_test_split": False,
            "source_config": str(source), "resolved_config": config, "command_line": command_line,
            "runtime_code_sha256": runtime_code_hashes(), "git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "python_version": platform.python_version(), "python_executable": sys.executable,
            "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda, "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "model_total_params": total, "model_trainable_params": trainable,
            "cross_split_duplicate_input_groups": quality["cross_split_duplicate_groups"], "start_time": now_iso()}
    atomic_json(run / "run_info.json", info)
    status = {"status": "running", "start_time": info["start_time"], "end_time": None,
              "last_epoch": 0, "best_psnr": None, "best_ssim": None, "best_val_loss": None,
              "final_validation": None}
    atomic_json(run / "status.json", status)
    return run, status


def load_status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


def update_status(run_dir: Path, state: dict, **updates) -> dict:
    state.update(updates)
    atomic_json(run_dir / "status.json", state)
    return state
