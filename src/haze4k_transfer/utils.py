"""Audit, serialization and reproducibility helpers isolated from old runs."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch
import yaml

from src.v16.utils import seed_everything, seed_worker, rng_state, restore_rng_state, tensor_to_image
from .prepare import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
METRICS = {"data_range": 1.0, "crop_border": 0, "ssim_window_size": 11, "ssim_sigma": 1.5}


def project_path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def save_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False))


def device_for(gpu):
    if gpu == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --gpu cpu explicitly for software smoke tests")
    i = int(gpu)
    if i < 0 or i >= torch.cuda.device_count():
        raise ValueError(f"Unavailable GPU index: {i}")
    torch.cuda.set_device(i)
    return torch.device("cuda", i)


def finite(value, context):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"NaN/Inf in {context}; refusing to skip samples/steps")


def torch_load(path):
    # Only load trusted locally-created checkpoints (contains Python/NumPy RNG state).
    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def metric_signature():
    return {"version": "src.v16.metrics/float32-rgb-per-image-v1", "config": METRICS,
            "source_sha256": sha256(ROOT / "src/v16/metrics.py")}


def source_hashes():
    files = ["configs/config_v16.yaml", "src/v16/models/network.py", "src/v15/models/network.py",
             "src/v16/losses.py", "src/v16/metrics.py"]
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "src/shared").glob("*.py"))]
    files += ["src/v4/models/unet.py"]
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "src/haze4k_transfer").glob("*.py"))]
    return {p: sha256(ROOT / p) for p in files}


def audit_environment(device):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    try:
        revision, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, "git unavailable"
    return {"git_commit": revision, "git_dirty": bool(dirty), "git_status": dirty,
            "python": sys.version, "platform": platform.platform(), "command": sys.argv,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "device": str(device), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "source_hashes": source_hashes()}


def write_csv(path, rows, columns=None):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def protocol(resolution):
    return {"resolution": str(resolution), "resize": "PIL BILINEAR 256x256" if str(resolution) == "256"
            else "native; original factory internal pad/crop; no external pad/tile/resize",
            "inference": "float32; eval(); batch_size=1; clamp prediction [0,1] for metrics",
            "rgb": "PIL convert('RGB'); no EXIF transpose or ICC transform", "metrics": metric_signature()}


def fairness(config):
    # Root and variant differ; every training and evaluation choice must match.
    return {k: config[k] for k in ("seed", "data", "training", "optimizer", "loss", "metrics", "evaluation", "model", "smoke_steps")}


def run_path_guard(path):
    base = (ROOT / "experiments/haze4k_transfer").resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError(f"Run output/resume must be inside isolated {base}")
    return path


def plot_history(run, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, keys in zip(axes, [("train_loss", "val_loss"), ("val_psnr",), ("val_ssim",)]):
        for key in keys:
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=150)
    plt.close(fig)
