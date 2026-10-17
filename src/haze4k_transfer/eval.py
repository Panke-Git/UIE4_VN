"""Fixed official-test evaluation, float32 per-image canonical V16 metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import HazeDataset
from .models import build_model
from .prepare import COLUMNS, sha256, validate_bundle, verify_content, write_json
from .utils import (METRICS, device_for, fairness, finite, load_yaml, project_path, protocol,
                    source_hashes, tensor_to_image, torch_load, write_csv)
from src.v16.metrics import batch_metrics


def image_key(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()[:20] + ".png"


@torch.no_grad()
def evaluate_loader(model, loader, device, criterion=None, image_dir=None, selected=()):
    model.eval()
    rows, losses = [], []
    for batch in loader:
        x, y = batch["input"].to(device), batch["target"].to(device)
        pred = model(x)  # Deliberately float32, no autocast in validation or test.
        if pred.shape != y.shape:
            raise ValueError(f"Output shape mismatch: {pred.shape} vs {y.shape}")
        finite(pred, f"evaluation outputs IDs={batch['sample_id']}")
        finite(y, "evaluation targets")
        if criterion:
            loss = criterion(pred, y)
            finite(loss, "validation loss")
            losses.extend([loss.item()] * len(x))
        pred = pred.float().clamp(0, 1)
        psnr, ssim = batch_metrics(pred, y.float(), METRICS)
        # Even infinite perfect-image PSNR is explicitly rejected, not silently averaged/serialized.
        finite(psnr, f"PSNR IDs={batch['sample_id']}")
        finite(ssim, f"SSIM IDs={batch['sample_id']}")
        for i in range(len(x)):
            r = {k: batch[k][i] for k in COLUMNS}
            rows.append({**r, "psnr": psnr[i].item(), "ssim": ssim[i].item()})
            if image_dir is not None and r["sample_id"] in selected:
                name = image_key(r["sample_id"])
                for folder, value in (("hazy", x[i]), ("enhanced", pred[i]), ("gt", y[i])):
                    tensor_to_image(value).save(image_dir / folder / name)
    if not rows:
        raise ValueError("Evaluation dataset empty")
    summary = {"count": len(rows), "psnr": float(np.mean([r["psnr"] for r in rows])),
               "ssim": float(np.mean([r["ssim"] for r in rows]))}
    if losses:
        summary["loss"] = float(np.mean(losses))
    return rows, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", choices=("best_psnr", "last"), default="best_psnr")
    p.add_argument("--gpu", default="0")
    p.add_argument("--resolution", choices=("256", "native"), default="256")
    p.add_argument("--data-root")
    p.add_argument("--split", choices=("test", "validation"), default="test")
    p.add_argument("--save-all", action="store_true")
    a = p.parse_args(argv)
    run = project_path(a.run_dir)
    c = load_yaml(run / "config_resolved.yaml")
    meta, splits, inventory = validate_bundle(run / "split")
    root = Path(a.data_root or c["data_root"]).resolve()
    rows = splits[a.split]
    if a.split == "test":
        status = json.loads((run / "status.json").read_text())
        if status["status"] != "completed" or status["epoch"] != c["training"]["epochs"]:
            raise ValueError("Final test requires a completed target training run; use validation for interrupted runs")
    verify_content(root, rows, inventory)
    cp_path = run / f"checkpoint/{a.checkpoint}.pt"
    cp = torch_load(cp_path)
    if cp["config"] != c or cp["split_hashes"] != meta["hashes"]:
        raise ValueError("Checkpoint/run config/split mismatch")
    if cp["source_hashes"] != source_hashes():
        raise ValueError("Sources changed since training; refusing to silently evaluate changed architecture/metrics/code")
    if a.checkpoint == "best_psnr":
        history = json.loads((run / "history.json").read_text())
        best_record = max(history, key=lambda r: r["val_psnr"])  # First maximum, same strict > tie rule as training.
        if cp["epoch"] != best_record["epoch"] or cp["best_psnr"] != best_record["val_psnr"]:
            raise ValueError("best_psnr checkpoint does not match the maximum across completed validation history")
    device = device_for(a.gpu)
    model = build_model(c["variant"], c["model"]).to(device)
    model.load_state_dict(cp["model"], strict=True)
    del cp["optimizer"]  # Do not retain AdamW tensors throughout inference.
    directory = run / f"{a.split}_{a.resolution}"
    if a.checkpoint != "best_psnr":
        directory = run / f"{a.split}_{a.resolution}_{a.checkpoint}"
    directory.mkdir(exist_ok=False)  # Refuse accidental replacement of existing evaluation.
    ids = sorted(r["sample_id"] for r in rows)
    selected = ids if a.save_all else ids[:min(8, len(ids))]
    for name in ("hazy", "enhanced", "gt"):
        (directory / "images" / name).mkdir(parents=True)
    write_json(directory / "visualization_selection.json", {"rule": "lexicographic first eight fixed test sample IDs; independent of predictions" if not a.save_all else "all fixed manifest IDs",
                                                             "sample_ids": selected, "file_names": {s: image_key(s) for s in selected}})
    loader = DataLoader(HazeDataset(root, rows, a.split, resolution=a.resolution), batch_size=1, num_workers=0,
                        generator=torch.Generator().manual_seed(int(c["seed"])))
    try:
        per_image, summary = evaluate_loader(model, loader, device, image_dir=directory / "images", selected=set(selected))
        write_csv(directory / "per_image.csv", per_image)
        summary.update({"variant": c["variant"], "model_type": "UICFPreBackbone(V15 PlainUNetColorQuery)" if c["variant"] == "full" else "V15 PlainUNetColorQuery",
                        "checkpoint": a.checkpoint, "checkpoint_sha256": sha256(cp_path), "epoch": cp["epoch"],
                        "split": a.split, "split_hashes": meta["hashes"], "protocol": protocol(a.resolution),
                        "synthetic": bool(c["synthetic"]), "smoke": bool(c["smoke"]), "seed": c["seed"],
                        "training_protocol": fairness(c), "source_hashes": cp["source_hashes"],
                        "per_image_sha256": sha256(directory / "per_image.csv"),
                        "selected_sample_ids": selected, "device": str(device),
                        "training_runtime": json.loads((run / "runtime.json").read_text()),
                        "aggregation": "arithmetic mean of per-image float RGB scores"})
        write_json(directory / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("count", "psnr", "ssim", "synthetic", "smoke", "protocol")}, indent=2))
        return directory
    except BaseException as e:
        write_json(directory / "FAILED.json", {"error": repr(e)})
        raise


if __name__ == "__main__":
    main()
