from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw

from ..utils import PROJECT, OUTPUT, digest_json, read_json, sha256, write_json, write_yaml

DIAGNOSTICS = OUTPUT / "diagnostics"
METRICS = ("precision", "recall", "map50", "map50_95")
GIVEN_RESULTS = {
    "raw": dict(zip(METRICS, (0.8908660221526422, 0.7688926458996367, 0.8678082066719932, 0.698034607749171))),
    "full": dict(zip(METRICS, (0.750440312027763, 0.5005973712677136, 0.5821621094901419, 0.4153791668479506))),
}


def fresh_output(value, inputs=()):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(DIAGNOSTICS.resolve()) or path == DIAGNOSTICS.resolve():
        raise ValueError(f"Output must be a NEW run directory below {DIAGNOSTICS}")
    for original in inputs:
        original = Path(original).resolve()
        if path == original or path.is_relative_to(original) or original.is_relative_to(path):
            raise ValueError(f"Output overlaps protected input: {original}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def runtime_dirs(out):
    for key, name in (("YOLO_CONFIG_DIR", "ultralytics"), ("MPLCONFIGDIR", "matplotlib")):
        folder = Path(out).resolve() / "runtime" / name
        (folder / "Ultralytics").mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ["YOLO_OFFLINE"] = "true"


def rgb(path):
    with Image.open(path) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f"Non-default EXIF orientation: {path}")
        return np.array(image.convert("RGB"), dtype=np.uint8)


def relative_image_file(value):
    path = Path(str(value).replace("\\","/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in path.parts[0] or path.suffix.lower() != ".png":
        raise ValueError(f"Unsafe/non-PNG prepared relative filename: {value}")
    return path


def validate_image_binding(row, official):
    relative = relative_image_file(row["file"])
    if row["image_id"] != official["id"] or row["source_file"] != official["file_name"] or row.get("official_split") != "test":
        raise ValueError("Image ID/source filename/official split mismatch")
    if relative.stem != Path(official["file_name"].replace("\\","/")).stem:
        raise ValueError("Prepared stem does not match official source filename")
    if (row["width"],row["height"]) != (official["width"],official["height"]):
        raise ValueError("Prepared dimensions do not match official COCO dimensions")
    return relative


def rgb_tensor(array):
    import torch
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected HWC uint8 RGB")
    return torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def save_tensor(tensor, path):
    """Exact same round/clamp/permute/PNG encoding convention as the old wrapper."""
    import torch
    if tensor.ndim != 4 or tensor.shape[0:2] != (1, 3) or not torch.isfinite(tensor).all():
        raise ValueError("Invalid output tensor")
    if tensor.min() < -1e-6 or tensor.max() > 1+1e-6:
        raise ValueError("Output outside RGB [0,1]")
    array = tensor[0].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return array


def pixel_difference(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Geometry mismatch: {a.shape} vs {b.shape}")
    error = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(error*error))
    return {"mae_uint8": float(np.abs(error).mean()), "max_error_uint8": float(np.abs(error).max()),
            "psnr_between_predictions_db": 10*np.log10(255**2/mse) if mse else None,
            "identical": mse == 0,
            "channel_mean_difference_uint8": error.mean(axis=(0, 1)).tolist()}


def image_statistics(array):
    f = array.astype(np.float64)
    gray = f.mean(axis=2)
    texture = [np.abs(np.diff(gray, axis=axis)).mean() for axis in (0, 1) if gray.shape[axis] > 1]
    return {"width": array.shape[1], "height": array.shape[0], "min": int(array.min()), "max": int(array.max()),
            "mean_rgb": f.mean(axis=(0, 1)).tolist(), "std_rgb": f.std(axis=(0, 1)).tolist(),
            "black_pixel_fraction": float(np.all(array == 0, axis=2).mean()),
            "white_pixel_fraction": float(np.all(array == 255, axis=2).mean()),
            "black_channel_fraction": (array == 0).mean(axis=(0, 1)).tolist(),
            "white_channel_fraction": (array == 255).mean(axis=(0, 1)).tolist(),
            "texture_mean_absolute_gradient_uint8": float(np.mean(texture)) if texture else 0.0,
            "rgb_pixel_sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def labels_xyxy(path, width, height):
    boxes = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) != 5 or not fields[0].isdigit():
            raise ValueError(f"Invalid label: {path}: {line}")
        cls, (cx, cy, w, h) = int(fields[0]), list(map(float, fields[1:]))
        if not all(np.isfinite([cx, cy, w, h])) or min(w, h) <= 0:
            raise ValueError(f"Invalid bbox: {path}")
        box = (cls, (cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height)
        if min(box[1:3]) < -1e-5 or box[3] > width+1e-5 or box[4] > height+1e-5:
            raise ValueError(f"Out of bounds: {path}")
        boxes.append(box)
    if len(set(boxes)) != len(boxes):
        raise ValueError(f"Duplicate labels: {path}")
    return sorted(boxes)


def expected_coco(doc, meta):
    """Independently derive expected boxes, including the RECORDED clipping policy."""
    images, annotations, ids = {}, {}, set()
    categories = sorted(doc["categories"], key=lambda c: c["id"])
    mapping = {c["id"]: i for i, c in enumerate(categories)}
    if len(mapping) != len(categories) or [c["name"] for c in categories] != meta["mapping"]["names"]:
        raise ValueError("COCO category mapping mismatch")
    if {str(c): i for c, i in mapping.items()} != meta["mapping"]["coco_to_yolo"]:
        raise ValueError("Original COCO IDs differ from prepared mapping")
    for image in doc["images"]:
        if image["id"] in images:
            raise ValueError("Duplicate official image ID")
        images[image["id"]] = image
        annotations[image["id"]] = []
    changes, seen_boxes = [], set()
    for ann in doc["annotations"]:
        if ann["id"] in ids or ann["image_id"] not in images or ann["category_id"] not in mapping:
            raise ValueError("Duplicate/unknown COCO annotation/image/category ID")
        ids.add(ann["id"])
        im = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0 or ann.get("ignore", 0) or ann.get("iscrowd", 0):
            raise ValueError("Unsupported/invalid original annotation")
        x2, y2 = x+w, y+h
        if min(x, y) < 0 or x2 > im["width"] or y2 > im["height"]:
            if meta.get("bbox_policy", "strict") != "clip":
                raise ValueError("Unrecorded bbox correction would be required")
            x, y, x2, y2 = max(0, x), max(0, y), min(im["width"], x2), min(im["height"], y2)
            changes.append(ann["id"])
        if x2 <= x or y2 <= y:
            if meta.get("empty_after_clip") != "drop":
                raise ValueError("Unrecorded empty-bbox drop")
            continue
        entry = (mapping[ann["category_id"]], x, y, x2, y2)
        key = (ann["image_id"],*entry)
        if key in seen_boxes:
            raise ValueError("Duplicate derived COCO bbox record")
        seen_boxes.add(key)
        annotations[ann["image_id"]].append(entry)
    return images, {iid: sorted(boxes) for iid, boxes in annotations.items()}, changes


def same_boxes(actual, expected, tolerance=1e-5):
    return len(actual) == len(expected) and (not actual or bool(np.allclose(actual, expected, rtol=0, atol=tolerance)))


def preview_grid(images, titles, path, width=640, crop=None):
    cells = []
    for array, title in zip(images, titles):
        im = Image.fromarray(array)
        if crop:
            im = im.crop(crop)
        height = max(1, round(im.height*width/im.width))
        im = im.resize((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height+28), "white")
        canvas.paste(im, (0, 28))
        ImageDraw.Draw(canvas).text((5, 6), title, fill="black")
        cells.append(canvas)
    grid = Image.new("RGB", (width*len(cells), max(c.height for c in cells)), "white")
    for i, cell in enumerate(cells):
        grid.paste(cell, (i*width, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def snapshot(paths):
    files, missing = {}, []
    for value in paths:
        path = Path(value).resolve()
        if not path.exists():
            missing.append(str(path))
        elif path.is_file():
            files[str(path)] = sha256(path)
        else:
            for item in sorted(path.rglob("*")):
                if item.is_file() and "__pycache__" not in item.parts:
                    files[str(item)] = sha256(item)
    return {"files": files, "missing": sorted(set(missing)), "roots":[str(Path(p).resolve()) for p in paths]}


def check_snapshot(before):
    changed = [p for p, digest in before["files"].items() if not Path(p).is_file() or sha256(p) != digest]
    current = set()
    for name in before.get("roots", []):
        path = Path(name)
        if path.is_file():
            current.add(str(path.resolve()))
        elif path.is_dir():
            current.update(str(p.resolve()) for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    added = sorted(current-set(before["files"])) if before.get("roots") else []
    return {"status": "pass" if not changed and not added else "fail", "changed_or_removed": changed, "new_files_in_protected_inputs":added,
            "checked_files": len(before["files"])}


def source_inventory(out):
    """Save actual source text and hashes, including missing cloud evidence paths."""
    paths = []
    for folder in ("src/downstream_duo", "scripts/downstream_duo", "configs/downstream_duo"):
        paths += [p for p in (PROJECT/folder).rglob("*") if p.is_file() and "diagnostics" not in p.parts and "__pycache__" not in p.parts]
    paths += [PROJECT/p for p in ("src/v16/models/network.py", "src/shared/uicf_models.py", "src/shared/uicf_inr.py",
              "src/shared/color_query_unet.py", "src/v4/models/unet.py", "src/v16/test.py", "src/v16/utils.py")]
    rows = []
    for path in sorted(set(paths)):
        dest = Path(out)/"source_snapshot"/path.relative_to(PROJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        rows.append({"path": str(path), "sha256": sha256(path), "saved_content": str(dest)})
    write_json(Path(out)/"source_inventory.json", rows)
    return rows


def dump_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--enhanced", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-root")
    parser.add_argument("--data-root")
    parser.add_argument("--uie-run-dir",default=os.getenv("UIE_RUN_DIR"))
    parser.add_argument("--gpu", default="0")
    return parser
