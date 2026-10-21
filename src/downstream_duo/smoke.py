"""One-epoch synthetic integration run. NEVER produces publishable DUO evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .utils import fresh_dir, load_config, output_path, write_json, write_yaml


def synthetic_coco(root, counts=None):
    counts = counts or {"train": 8, "test": 3}
    root = Path(root)
    (root / "annotations").mkdir(parents=True)
    categories = [{"id": c, "name": name} for c, name in zip((7, 19, 30, 42), ("holothurian", "echinus", "scallop", "starfish"))]
    for split, count in counts.items():
        (root / "images" / split).mkdir(parents=True)
        doc = {"images": [], "annotations": [], "categories": categories}
        for i in range(count):
            iid, w, h = i + 101, 80, 64
            rng = np.random.default_rng(i + (100 if split == "test" else 0))
            image = rng.integers(0, 80, (h, w, 3), dtype=np.uint8)
            for j, category in enumerate(categories):
                x, y, bw, bh = 3+j*18, 12+j*3, 12, 15
                image[y:y+bh, x:x+bw] = [70+j*40, 200-j*30, 60+j*30]
                if i != count-1:  # Include genuinely empty-label images.
                    doc["annotations"].append({"id": i*4+j+1, "image_id": iid,
                        "category_id": category["id"], "bbox": [x, y, bw, bh], "area": bw*bh, "iscrowd": 0})
            filename = f"frame_{i:03d}.png"
            Image.fromarray(image).save(root / "images" / split / filename)
            doc["images"].append({"id": iid, "file_name": filename, "width": w, "height": h})
        write_json(root / "annotations" / f"instances_{split}.json", doc)
    return counts


def tiny_v16_config():
    return {"experiment": {"version": "v16"}, "data": {"dataset": "LSUI_SYNTHETIC_SOFTWARE_TEST"},
        "model": {"type": "plain_unet_color_query_uicf_pre_backbone", "in_channels": 3,
        "out_channels": 3, "base_channels": 2, "use_batch_norm": False, "output_activation": "sigmoid",
        "color_query": {"num_color_queries": 2, "token_dim": 4, "num_heads": 1, "ffn_expansion": 2., "dropout": 0.},
        "uicf": {"feat_dim": 2, "num_frequencies": 1, "mlp_hidden_dim": 4, "mlp_hidden_layers": 1,
                 "anchor_hidden_dim": 2, "query_chunk_size": 1024}}}


def synthetic_v16_run(run):
    from src.v16.models.network import build_model
    run = Path(run)
    (run / "best").mkdir(parents=True)
    torch.manual_seed(3520)
    config = tiny_v16_config()
    write_yaml(run / "config_resolved.yaml", config)
    model = build_model(config["model"])
    torch.save({"version": "v16", "resolved_config": config, "epoch": 0,
                "model_state_dict": model.state_dict()}, run / "best/best_psnr.pt")


def run_smoke(out):
    from .prepare import prepare
    from .train_detector import train
    from .model_inference_v16 import load_v16, enhance_test
    from .evaluate import evaluate
    from .compare import compare
    from .visualize import make_figures
    out = fresh_dir(out)
    torch.set_num_threads(2)
    counts = synthetic_coco(out / "source")
    prepare(out / "source", out / "prepared", expected=counts, synthetic=True, fraction=0.25)
    config = load_config()
    config["train"].update(model="yolo11s.yaml", imgsz=64, epochs=1, batch=2, device="cpu", workers=0,
                           pretrained=False, amp=False, plots=True, mosaic=0.0, close_mosaic=0)
    config["evaluation"].update(imgsz=64, device="cpu", workers=0, plots=True)
    registry = train(out / "prepared", out / "yolo_raw", config, synthetic=True)
    synthetic_v16_run(out / "synthetic_v16")
    model, provenance = load_v16(out / "synthetic_v16")
    provenance["synthetic_untrained"] = True
    enhance_test(out / "prepared", out / "full", model, provenance)
    raw = out / "eval_raw/metrics_raw.json"
    full = out / "eval_full/metrics_full.json"
    evaluate(out / "prepared", registry["weights"], out / "eval_raw", "raw")
    evaluate(out / "prepared", registry["weights"], out / "eval_full", "full", out / "full")
    compare(raw, full, out / "comparison")
    make_figures(raw, full, out / "figures", rows=2, width=400, gt=True)
    write_json(out / "SMOKE_ONLY.json", {"synthetic": True, "epochs": 1, "detector": "YOLO11s random initialization",
               "enhancer": "tiny random V16, not trained on real LSUI", "complete": True,
               "purpose": "software integration only; NOT DUO/LSUI research metrics"})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    run_smoke(output_path(a.output))


if __name__ == "__main__":
    main()
