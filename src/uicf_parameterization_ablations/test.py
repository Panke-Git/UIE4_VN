"""Explicit held-out test, always using this run's best validation PSNR model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from src.shared.e00 import batch_delta_e00, e00_protocol
from src.v16.dataset import LSUIDataset, validate_split_protocol
from src.v16.metrics import batch_metrics
from src.v16.test import _save_grid
from src.v16.train import _torch_load
from src.v16.utils import atomic_json, load_yaml, project_path, select_device, tensor_to_image
from . import FORMULAS, VARIANTS
from .models import build_model
from .protocol import validate_config, verify_checkpoint, verify_saved_run


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--variant", choices=VARIANTS, help="Optional additional identity assertion")
    p.add_argument("--checkpoint", choices=("best_psnr",), default="best_psnr")
    p.add_argument("--gpu", type=int)
    p.add_argument("--data-root")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    run = project_path(args.run_dir).resolve()
    config = load_yaml(run/"config_resolved.yaml")
    smoke = bool(config.get("ablation", {}).get("smoke_test", False))
    variant = validate_config(config, args.variant, allow_smoke=smoke)
    paths = verify_saved_run(run, config)
    entries = validate_split_protocol(paths, config["data"]["expected_counts"])
    checkpoint_path = run/"best/best_psnr.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    status = json.loads((run/"status.json").read_text())
    if status.get("status") != "completed" or status.get("last_epoch") != config["training"]["epochs"]:
        raise ValueError("Complete the configured training before held-out test")
    if args.dry_run:
        print(json.dumps(dict(variant=variant, test_count=len(entries["test"]), checkpoint=str(checkpoint_path), dry_run=True)))
        return None
    if args.data_root:
        config["data"]["root"] = str(Path(args.data_root).expanduser().resolve())
    device = torch.device("cpu") if smoke else select_device(args.gpu)
    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
    verify_checkpoint(checkpoint, config)
    best = json.loads((run/"best/best_psnr.json").read_text())
    if checkpoint["epoch"] != best["epoch"]:
        raise ValueError("Best PSNR checkpoint metadata epoch mismatch")
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    data = config["data"]
    dataset = LSUIDataset(paths["test"], Path(data["root"]), "test", int(data["patch_size"]), data["augmentation"],
                          bool(data["pad_if_smaller"]), data["pad_mode"], config["evaluation"], verify_files=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(data["num_workers"]),
                        pin_memory=bool(data["pin_memory"]) and device.type == "cuda")
    result = run/"result"
    enhanced_dir, samples_dir = result/"test_all_enhanced", result/"test_samples"
    enhanced_dir.mkdir(exist_ok=True)
    samples_dir.mkdir(exist_ok=True)
    viz = config["test"]["visualization"]
    selected = random.Random(int(viz["random_seed"])).sample(range(len(dataset)), min(len(dataset), int(viz["num_samples"])))
    images, rows, used_names = {}, [], set()
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    inference_seconds, e00_seconds = 0.0, 0.0
    for index, batch in enumerate(loader):
        inputs, targets = (batch[k].to(device, non_blocking=True) for k in ("input", "target"))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp):
            prediction = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter()-started
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Non-finite test output: {batch['id']}")
        prediction = prediction.float().clamp(0, 1)
        psnr, ssim = batch_metrics(prediction, targets.float(), config["metrics"])
        started = time.perf_counter()
        e00 = batch_delta_e00(prediction, targets.float(), config["metrics"])
        e00_seconds += time.perf_counter()-started
        rows.append(dict(filename=batch["filename"][0], sample_id=batch["id"][0], psnr=float(psnr[0]), ssim=float(ssim[0]), e00=float(e00[0])))
        name = Path(batch["filename"][0]).stem+"_enhanced.png"
        if name in used_names:
            name = Path(batch["filename"][0]).stem+"_"+batch["id"][0]+"_enhanced.png"
        used_names.add(name)
        enhanced = tensor_to_image(prediction[0])
        if config["test"]["save_all_enhanced_images"]:
            enhanced.save(enhanced_dir/name)
        if index in selected:
            enhanced.save(samples_dir/name)
            images[index] = dict(input=tensor_to_image(inputs[0]), target=tensor_to_image(targets[0]), enhanced=enhanced)
    with (result/"test_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("filename", "sample_id", "psnr", "ssim", "e00"))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(variant=variant, version=config["experiment"]["version"], formula=FORMULAS[variant],
                   dataset="LSUI19", split="test", smoke_test=smoke, checkpoint=str(checkpoint_path),
                   checkpoint_selector="best_psnr", checkpoint_selection_split="validation", checkpoint_epoch=checkpoint["epoch"],
                   sample_count=len(rows), metric_input="float_rgb_before_png_save",
                   evaluation=config["evaluation"], metrics_config=config["metrics"], e00_protocol=e00_protocol(config["metrics"]),
                   split_sha256=config["provenance"]["split_sha256"],
                   average_inference_time_seconds=inference_seconds/len(rows), average_e00_time_seconds=e00_seconds/len(rows),
                   **{"mean_"+metric: sum(x[metric] for x in rows)/len(rows) for metric in ("psnr", "ssim", "e00")})
    if len(rows) != config["data"]["expected_counts"]["test"] or any(not torch.isfinite(torch.tensor(x[m])) for x in rows for m in ("psnr", "ssim", "e00")):
        raise ValueError("Incomplete or non-finite held-out test results")
    atomic_json(result/"test_summary.json", summary)
    if viz["enabled"]:
        _save_grid([images[i] for i in selected], result/f"test_grid_{len(selected)}x3.png", viz)
    atomic_json(result/"test_visualization_samples.json", {"random_seed": viz["random_seed"], "sample_ids": [rows[i]["sample_id"] for i in selected]})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
