"""Reload one V16-19 checkpoint and export complete Val results atomically."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from .dataset import LSUIDataset, validate_split_protocol
from .e00 import batch_delta_e00, e00_protocol
from .losses import build_loss
from .metrics import batch_metrics
from .models import build_model
from .protocol import verify_dataset
from .utils import (VERSION, atomic_json, load_yaml, project_path, runtime_code_hashes,
                    select_device, setup_logger, sha256_file, tensor_to_image, validate_config, validate_checkpoint_config)

SELECTORS = {"best_psnr", "best_ssim", "best_loss", "last"}


def checkpoint_path(run: Path, selector: str) -> Path:
    if selector in SELECTORS:
        return run / "checkpoint/last.pt" if selector == "last" else run / "best" / f"{selector}.pt"
    path = Path(selector)
    if not path.is_absolute():
        path = run / path
    return path.resolve()


def _save_grid(samples: list[dict], path: Path, visual: dict) -> None:
    width, height = int(visual["cell_width"]), int(visual["cell_height"])
    label_height = 22 if visual.get("add_labels", True) else 0
    grid = Image.new("RGB", (width * 3, (height + label_height) * len(samples)), "white")
    for row, sample in enumerate(samples):
        for col, key in enumerate(("input", "enhanced", "target")):
            image = sample[key].copy()
            if visual.get("preserve_aspect_ratio", False):
                image.thumbnail((width, height), Image.Resampling.BILINEAR)
                cell = Image.new("RGB", (width, height), "black")
                cell.paste(image, ((width-image.width)//2, (height-image.height)//2))
            else:
                cell = image.resize((width, height), Image.Resampling.BILINEAR)
            x, y = col * width, row * (height + label_height)
            grid.paste(cell, (x, y + label_height))
            if label_height:
                label = ("input", "enhanced", "GT")[col]
                ImageDraw.Draw(grid).text((x+4, y+3), f"{sample['sample_id']} {label}", fill="black")
    grid.save(path)


def _save_curves(run: Path, destination: Path) -> None:
    """Small dependency-free overview; exact values remain in CSV/JSON history."""
    history_path = run / "log/metrics_history.json"
    if not history_path.is_file():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))["epochs"]
    import math
    canvas = Image.new("RGB", (960, 720), "white")
    draw = ImageDraw.Draw(canvas)
    for panel, (key, label) in enumerate([("train_loss", "Train loss"), ("val_loss", "Val loss"), ("val_psnr", "Val PSNR"), ("val_ssim", "Val SSIM")]):
        points = [(r["epoch"], float(r[key])) for r in history if math.isfinite(float(r[key]))]
        if not points:
            continue
        left, top = 55 + (panel % 2)*480, 35 + (panel // 2)*360
        right, bottom = left+380, top+265
        draw.text((left, top-20), label, fill="black")
        draw.rectangle((left, top, right, bottom), outline="#777777")
        xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
        ymin, ymax = min(y for _, y in points), max(y for _, y in points)
        span = ymax-ymin or max(abs(ymin)*0.1, 0.01)
        ymin -= span*0.05; ymax += span*0.05
        coords = [(left+(x-xmin)/max(xmax-xmin, 1)*380, bottom-(y-ymin)/(ymax-ymin)*265) for x, y in points]
        if len(coords)>1:
            draw.line(coords, fill="#2563eb", width=2)
        for x,y in coords:
            draw.ellipse((x-2,y-2,x+2,y+2), fill="#2563eb")
        draw.text((left, bottom+8), f"epoch {xmin} to {xmax}; range {ymin:.5g} to {ymax:.5g}", fill="black")
    canvas.save(destination)


@torch.no_grad()
def export_validation(run_dir: Path, selector: str | None = None, gpu: int | None = None,
                      data_root_override: str | None = None) -> dict:
    run = project_path(run_dir).resolve()
    config = load_yaml(run / "config_resolved.yaml")
    validate_config(config)
    if not run.is_relative_to(project_path(config["experiment"]["output_root"]).resolve()):
        raise ValueError("Run must be inside its isolated configured output root")
    if config.get("runtime_code_sha256") != runtime_code_hashes():
        raise ValueError("V16-19 evaluation source differs from the recorded run")
    root = project_path(data_root_override or config["data"]["root"]).resolve()
    protocol = verify_dataset(root, run / "split_snapshot")
    if protocol["fingerprint"] != config["data"]["fingerprint"] or protocol["dataset"] != config["data"]["dataset_key"]:
        raise ValueError("Run and dataset snapshot disagree")
    if protocol["counts"] != config["data"]["expected_counts"]:
        raise ValueError("Run and snapshot counts disagree")
    manifests = {k: run / "split_snapshot" / f"{k}.tsv" for k in ("train", "validation")}
    entries = validate_split_protocol(manifests, protocol["counts"])
    selected = selector or config["validation"]["checkpoint"]
    path = checkpoint_path(run, selected)
    checkpoint_sha = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != VERSION or checkpoint.get("dataset_fingerprint") != protocol["fingerprint"]:
        raise ValueError("Checkpoint version or dataset is incompatible")
    if checkpoint.get("runtime_code_sha256") != config["runtime_code_sha256"] or checkpoint.get("resolved_config", {}).get("model") != config["model"]:
        raise ValueError("Checkpoint architecture/code identity is incompatible")
    validate_checkpoint_config(checkpoint, config)
    epoch = int(checkpoint["epoch"])
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    gc.collect()
    device = select_device(gpu)
    model = model.to(device).eval()
    criterion = build_loss(config["loss"]).to(device)
    data = config["data"]
    dataset = LSUIDataset(manifests["validation"], root, "validation", int(data["patch_size"]),
                         data["augmentation"], bool(data["pad_if_smaller"]), data["pad_mode"], config["evaluation"])
    batch_size = int(data["batch_size"]) if config["evaluation"].get("resize", True) else 1
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(data["num_workers"]),
                        pin_memory=bool(data["pin_memory"]) and device.type == "cuda")
    settings = config["validation"]
    visual = settings["visualization"]
    nvisual = min(int(visual["num_samples"]), len(dataset)) if visual["enabled"] else 0
    indices = random.Random(int(visual["random_seed"])).sample(range(len(dataset)), nvisual)
    samples = {}; rows = []; loss_total = 0.0; inference_seconds = 0.0
    label = selected if selected in SELECTORS else f"custom_{checkpoint_sha[:12]}"
    parent = run / "result/val"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / label
    temporary = Path(tempfile.mkdtemp(prefix=f".{label}-", dir=parent))
    backup = parent / f".{label}.previous"
    try:
        if settings["save_all_enhanced_images"]:
            (temporary / "enhanced").mkdir()
        if nvisual:
            (temporary / "samples").mkdir()
        start = time.perf_counter()
        for batch in loader:
            inp = batch["input"].to(device, non_blocking=True)
            gt = batch["target"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            before = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and config["training"]["amp"]):
                raw = model(inp)
                loss = criterion(raw, gt)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter()-before
            if not torch.isfinite(raw).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite final Val output: {batch['id']}")
            prediction = raw.float().clamp(0, 1)
            psnr, ssim = batch_metrics(prediction, gt.float(), config["metrics"])
            e00 = batch_delta_e00(prediction, gt.float(), config["metrics"])
            loss_total += float(loss) * len(inp)
            for i in range(len(inp)):
                index = len(rows)
                sid, filename = batch["id"][i], batch["filename"][i]
                enhanced = tensor_to_image(prediction[i])
                output_name = f"{sid}_enhanced.png"
                if settings["save_all_enhanced_images"]:
                    enhanced.save(temporary / "enhanced" / output_name)
                rows.append({"sample_id": sid, "filename": filename, "psnr": float(psnr[i]),
                             "ssim": float(ssim[i]), "e00": float(e00[i])})
                if index in indices:
                    samples[index] = {"index": index, "sample_id": sid, "filename": filename,
                                      "input": tensor_to_image(inp[i]), "enhanced": enhanced, "target": tensor_to_image(gt[i])}
                    for key, suffix in [("input", "input"), ("enhanced", "enhanced"), ("target", "gt")]:
                        samples[index][key].save(temporary / "samples" / f"{sid}_{suffix}.png")
        if len(rows) != protocol["counts"]["validation"] or len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Incomplete or duplicated final Val coverage")
        if sha256_file(path) != checkpoint_sha:
            raise ValueError("Checkpoint changed during final validation")
        verify_dataset(root, run / "split_snapshot")
        with (temporary / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "filename", "psnr", "ssim", "e00"])
            writer.writeheader(); writer.writerows(rows)
        summary = {"version": VERSION, "display_name": "V16-19", "dataset": data["dataset"],
                   "protocol": protocol["protocol"], "split": "validation", "has_test_split": False,
                   "dataset_fingerprint": protocol["fingerprint"], "checkpoint": str(path),
                   "checkpoint_selector": selected, "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": epoch,
                   "checkpoint_selection_split": "validation", "sample_count": len(rows), "mean_loss": loss_total/len(rows),
                   **{f"mean_{metric}": sum(r[metric] for r in rows)/len(rows) for metric in ("psnr", "ssim", "e00")},
                   "evaluation": config["evaluation"], "metrics_config": config["metrics"], "e00_protocol": e00_protocol(config["metrics"]),
                   "metric_representation": "clamped float RGB before PNG8 encoding; equal image weight",
                   "total_validation_time_seconds": time.perf_counter()-start,
                   "average_inference_time_seconds": inference_seconds/len(rows), "result_directory": str(destination)}
        atomic_json(temporary / "summary.json", summary)
        ordered = [samples[i] for i in indices]
        atomic_json(temporary / "visualization_samples.json", {"random_seed": int(visual["random_seed"]),
                    "samples": [{k: s[k] for k in ("index", "sample_id", "filename")} for s in ordered]})
        if ordered:
            _save_grid(ordered, temporary / f"val_grid_{len(ordered)}x3.png", visual)
        _save_curves(run, temporary / "training_curves.png")
        if backup.exists():
            raise FileExistsError(f"Interrupted export backup requires inspection: {backup}")
        if destination.exists():
            destination.rename(backup)
        try:
            temporary.rename(destination)
        except Exception:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    logger = setup_logger(f"v16_19.val.{run.name}", run / "log/final_val.log", append=True, console=config["logging"]["console"])
    logger.info("Val export completed dataset=%s epoch=%d count=%d PSNR=%.5f SSIM=%.5f E00=%.5f result=%s",
                data["dataset"], epoch, len(rows), summary["mean_psnr"], summary["mean_ssim"], summary["mean_e00"], destination)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export complete V16-19 Val results from one checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)
    summary = export_validation(Path(args.run_dir), args.checkpoint, args.gpu, args.data_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
