"""Matched detection figures, including improving, unchanged and worsening cases."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .compare import validate_pair
from .utils import draw_boxes, fresh_dir, output_path, read_json, write_json


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    return intersection / max(1e-12, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def diagnostic_f1(predictions, gt, conf):
    """Greedy class-aware IoU .5 matching for figure selection ONLY, not mAP."""
    matched, tp, fp = set(), 0, 0
    for pred in sorted(predictions, key=lambda p: -p["confidence"]):
        if pred["confidence"] < conf:
            continue
        choices = [(iou(pred["bbox_xyxy"], g[1:]), i) for i, g in enumerate(gt)
                   if i not in matched and g[0] == pred["class_id"]]
        best = max(choices, default=(0, -1))
        if best[0] >= 0.5:
            tp += 1
            matched.add(best[1])
        else:
            fp += 1
    fn = len(gt)-tp
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2*tp/max(1, 2*tp+fp+fn)}


def load_gt(prepared, row):
    path = Path(prepared) / "labels/test_raw" / Path(row["file"]).with_suffix(".txt")
    boxes = []
    for line in path.read_text().splitlines():
        c, cx, cy, w, h = map(float, line.split())
        boxes.append((int(c), (cx-w/2)*row["width"], (cy-h/2)*row["height"],
                      (cx+w/2)*row["width"], (cy+h/2)*row["height"]))
    return boxes


def make_figures(raw_path, full_path, out, conf=0.25, rows=3, width=800, ids=None, gt=False):
    if not 0 <= conf <= 1 or min(rows, width) < 1:
        raise ValueError("Invalid figure parameters")
    raw, full = read_json(raw_path), read_json(full_path)
    validate_pair(raw, full)
    if conf < raw["parameters"]["conf"]:
        raise ValueError("Figure threshold cannot be below exported metric confidence")
    out = fresh_dir(out)
    a = {r["image_id"]: r for r in read_json(raw["predictions"])}
    b = {r["image_id"]: r for r in read_json(full["predictions"])}
    names = raw["identity"]["mapping"]["names"]
    ranking, gts = [], {}
    for row in raw["identity"]["test"]:
        iid = row["image_id"]
        gts[iid] = load_gt(raw["prepared"], row)
        sa, sb = diagnostic_f1(a[iid]["detections"], gts[iid], conf), diagnostic_f1(b[iid]["detections"], gts[iid], conf)
        delta = sb["f1"]-sa["f1"]
        ranking.append({"image_id": iid, "raw": sa, "full": sb, "delta_f1": delta,
                        "bucket": "improved" if delta > 1e-9 else "worsened" if delta < -1e-9 else "unchanged"})
    selection = {}
    for bucket in ("improved", "unchanged", "worsened"):
        candidates = [r for r in ranking if r["bucket"] == bucket]
        candidates.sort(key=lambda r: ((-r["delta_f1"] if bucket == "improved" else r["delta_f1"]), r["image_id"]))
        selection[bucket] = [r["image_id"] for r in candidates[:rows]]
    if ids:
        if any(i not in a for i in ids):
            raise ValueError("Manual figure IDs must come from official test")
        selection["manual"] = ids
    for bucket, chosen in selection.items():
        if not chosen:
            continue
        cells = []
        for iid in chosen:
            pair = []
            for result, pred, label in ((raw, a[iid], "(a) Detection on Raw"), (full, b[iid], "(b) Detection on ICAR-UIE")):
                with Image.open(Path(result["image_root"]) / pred["file"]) as image:
                    detections = pred["detections"]
                    boxed = draw_boxes(image, [(d["class_id"], *d["bbox_xyxy"]) for d in detections], names,
                                       [d["confidence"] for d in detections], conf)
                boxed.save(out / f"{bucket}_{iid}_{result['condition']}_native.png")
                h = max(1, round(boxed.height * width / boxed.width))
                boxed = boxed.resize((width, h), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (width, h+32), "white")
                canvas.paste(boxed, (0, 32))
                ImageDraw.Draw(canvas).text((8, 7), f"{label}    image_id={iid}", fill="black")
                pair.append(canvas)
            if gt:
                with Image.open(Path(raw["image_root"]) / a[iid]["file"]) as image:
                    draw_boxes(image, gts[iid], names).save(out / f"gt_{iid}.png")
            cells.append(pair)
        height = sum(max(cell.height for cell in pair) for pair in cells)
        grid = Image.new("RGB", (width*2, height), "white")
        y = 0
        for pair in cells:
            for column, cell in enumerate(pair):
                grid.paste(cell, (column*width, y))
            y += max(cell.height for cell in pair)
        grid.save(out / f"detection_{bucket}.png", dpi=(300, 300))
        grid.save(out / f"detection_{bucket}.pdf", resolution=300)
    write_json(out / "figure_selection.json", {"selection": selection, "ranking": ranking,
        "confidence": conf, "confidence_decimals": 2, "native_line_width": 2,
        "cell_width": width, "metric_nms_parameters": raw["parameters"], "synthetic": raw["synthetic"],
        "note": "Selection uses class-aware greedy IoU .5 F1 at fixed display confidence, solely to expose contrasting examples. Not per-image AP, not evidence for quantitative benefit. Empty buckets mean no qualifying examples."})
    return selection


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--full", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cell-width", type=int, default=800)
    p.add_argument("--image-ids", type=int, nargs="+")
    p.add_argument("--gt", action="store_true")
    a = p.parse_args(argv)
    make_figures(a.raw, a.full, output_path(a.output), a.conf, a.rows, a.cell_width, a.image_ids, a.gt)


if __name__ == "__main__":
    main()
