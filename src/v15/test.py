"""Explicit held-out test entry point. Training never calls this module automatically."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from .dataset import LSUIDataset, validate_split_protocol
from .metrics import batch_metrics
from .models import build_model
from .utils import atomic_json, load_yaml, project_path, select_device, setup_logger, tensor_to_image


EXPECTED_VERSION = "v15"
CHECKPOINT_SELECTORS = {"best_psnr", "best_ssim", "best_loss", "last"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Test {EXPECTED_VERSION}")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    return parser.parse_args(argv)


def _checkpoint_path(run_dir: Path, selector: str) -> Path:
    if selector in CHECKPOINT_SELECTORS:
        return (
            run_dir / "checkpoint" / "last.pt"
            if selector == "last"
            else run_dir / "best" / f"{selector}.pt"
        )
    path = Path(selector)
    if not path.is_absolute():
        candidates = (run_dir / path, project_path(path))
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return path.resolve()


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _fit_cell(image: Image.Image, width: int, height: int, preserve: bool) -> Image.Image:
    if not preserve:
        return image.resize((width, height), Image.Resampling.BILINEAR)
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return canvas


def _save_grid(samples: list[dict[str, Any]], path: Path, visualization: dict) -> None:
    cell_width = int(visualization["cell_width"])
    cell_height = int(visualization["cell_height"])
    add_labels = bool(visualization.get("add_labels", False))
    label_height = 20 if add_labels else 0
    grid = Image.new("RGB", (cell_width * 3, (cell_height + label_height) * len(samples)), "white")
    labels = ("input", "enhanced", "GT")
    for row, sample in enumerate(samples):
        for column, key in enumerate(("input", "enhanced", "target")):
            image = _fit_cell(
                sample[key], cell_width, cell_height,
                bool(visualization.get("preserve_aspect_ratio", False)),
            )
            x, y = column * cell_width, row * (cell_height + label_height)
            grid.paste(image, (x, y + label_height))
            if add_labels:
                ImageDraw.Draw(grid).text((x + 4, y + 3), labels[column], fill="black")
    grid.save(path)


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = project_path(args.run_dir).resolve()
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run lacks config_resolved.yaml: {run_dir}")
    config = load_yaml(config_path)
    if config["experiment"]["version"] != EXPECTED_VERSION:
        raise ValueError(f"{EXPECTED_VERSION} test refuses run version {config['experiment']['version']!r}")
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    device = select_device(args.gpu)
    selector = args.checkpoint or config["test"]["checkpoint"]
    checkpoint_path = _checkpoint_path(run_dir, selector)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    snapshot = run_dir / "split_snapshot"
    manifests = {name: snapshot / f"{name}.tsv" for name in ("train", "validation", "test")}
    entries = validate_split_protocol(manifests)
    data_root = Path(config["data"]["root"])
    if not data_root.is_dir():
        raise FileNotFoundError(f"LSUI data.root is unavailable: {data_root}")
    data = config["data"]
    test_dataset = LSUIDataset(
        manifests["test"], data_root, "test", int(data["patch_size"]), data["augmentation"],
        bool(data["pad_if_smaller"]), str(data["pad_mode"]), config["evaluation"],
        verify_files=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=int(data["num_workers"]),
        pin_memory=bool(data["pin_memory"]) and device.type == "cuda",
    )

    model = build_model(config["model"]).to(device)
    checkpoint = _torch_load(checkpoint_path, device)
    if checkpoint.get("version") != EXPECTED_VERSION:
        raise ValueError("Checkpoint version is incompatible")
    if checkpoint.get("resolved_config", {}).get("model") != config["model"]:
        raise ValueError("Checkpoint architecture differs from run config_resolved.yaml")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    result_dir = run_dir / "result"
    enhanced_dir = result_dir / "test_all_enhanced"
    samples_dir = result_dir / "test_samples"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"{EXPECTED_VERSION}.test.{run_dir.name}", result_dir.parent / "log" / "test.log",
        append=True, console=bool(config["logging"]["console"]),
    )
    visualization = config["test"]["visualization"]
    sample_count = min(int(visualization["num_samples"]), len(entries["test"]))
    selected_indices = random.Random(int(visualization["random_seed"])).sample(
        range(len(entries["test"])), sample_count
    )
    selected_set = set(selected_indices)
    selected_images: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    inference_seconds = 0.0
    total_start = time.perf_counter()
    used_names: set[str] = set()
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    for index, batch in enumerate(test_loader):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - start
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Non-finite test output sample_id={batch['id'][0]}")
        prediction = prediction.float().clamp(0.0, 1.0)
        psnr, ssim = batch_metrics(prediction, targets.float(), config["metrics"])
        filename = batch["filename"][0]
        stem = Path(filename).stem
        output_name = f"{stem}_enhanced.png"
        if output_name in used_names:
            output_name = f"{stem}_{batch['id'][0]}_enhanced.png"
        used_names.add(output_name)
        enhanced_image = tensor_to_image(prediction[0])
        if bool(config["test"]["save_all_enhanced_images"]):
            enhanced_image.save(enhanced_dir / output_name)
        rows.append({"filename": filename, "sample_id": batch["id"][0], "psnr": float(psnr[0]), "ssim": float(ssim[0])})
        if index in selected_set:
            enhanced_image.save(samples_dir / output_name)
            selected_images[index] = {
                "index": index,
                "sample_id": batch["id"][0],
                "filename": filename,
                "input": tensor_to_image(inputs[0]),
                "enhanced": enhanced_image,
                "target": tensor_to_image(targets[0]),
            }

    total_seconds = time.perf_counter() - total_start
    with (result_dir / "test_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "sample_id", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    mean_psnr = sum(row["psnr"] for row in rows) / len(rows)
    mean_ssim = sum(row["ssim"] for row in rows) / len(rows)
    atomic_json(
        result_dir / "test_summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "sample_count": len(rows),
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "total_test_time_seconds": total_seconds,
            "average_inference_time_seconds": inference_seconds / len(rows),
        },
    )
    selected_ordered = [selected_images[index] for index in selected_indices]
    atomic_json(
        result_dir / "test_visualization_samples.json",
        {
            "random_seed": int(visualization["random_seed"]),
            "samples": [
                {key: sample[key] for key in ("index", "sample_id", "filename")}
                for sample in selected_ordered
            ],
        },
    )
    if bool(visualization["enabled"]):
        _save_grid(
            selected_ordered,
            result_dir / f"test_grid_{len(selected_ordered)}x3.png",
            visualization,
        )
    logger.info(
        "test completed checkpoint_epoch=%d samples=%d mean_psnr=%.4f mean_ssim=%.4f",
        int(checkpoint["epoch"]), len(rows), mean_psnr, mean_ssim,
    )


if __name__ == "__main__":
    main()
