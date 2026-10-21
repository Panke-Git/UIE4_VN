"""Small shared helpers; no imports of the paired LSUI evaluator."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "experiments/downstream_duo"
PIN = "8.3.253"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def write_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def output_path(path):
    path = Path(path).expanduser().absolute()
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError(f"Generated output must be inside {OUTPUT}: {path}")
    return path


def fresh_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "numpy", "Pillow", "PyYAML", "ultralytics",
                 "opencv-python", "scipy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def require_yolo():
    if versions()["ultralytics"] != PIN:
        raise RuntimeError(f"Use isolated detector environment with ultralytics=={PIN}; see README")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(OUTPUT / "ultralytics_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib_config"))
    from ultralytics import YOLO
    return YOLO


def load_config(path=None):
    path = Path(path or PROJECT / "configs/downstream_duo/detection.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def tsv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_prepared(path, verify=True):
    path = Path(path).resolve()
    if sha256(path / "split_metadata.json") != (path / "split_metadata.sha256").read_text().strip():
        raise ValueError("Split metadata checksum mismatch")
    meta = read_json(path / "split_metadata.json")
    for relative, expected in meta["snapshot_sha256"].items():
        if sha256(path / relative) != expected:
            raise ValueError(f"Prepared snapshot was changed: {relative}")
    if meta["dataset_digest"] != digest_json(read_json(path / "dataset_snapshot.json")):
        raise ValueError("Dataset digest mismatch")
    if verify:
        for split, rows in meta["splits"].items():
            for folder, extension in (("images", ".png"), ("labels", ".txt")):
                base = path / folder / f"{split}_raw"
                suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"} if folder == "images" else {extension}
                actual = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p.suffix.lower() in suffixes}
                expected = {Path(row["file"]).with_suffix(extension).as_posix() for row in rows}
                if actual != expected:
                    raise ValueError(f"Unexpected/missing prepared {split} {folder}")
            for row in rows:
                for folder, key, ext in (("images", "image_sha256", ".png"),
                                          ("labels", "label_sha256", ".txt")):
                    item = path / folder / f"{split}_raw" / Path(row["file"]).with_suffix(ext)
                    if sha256(item) != row[key]:
                        raise ValueError(f"Prepared {folder} changed: {item}")
    return meta


def pair_identity(meta):
    return {"dataset_digest": meta["dataset_digest"], "seed": meta["seed"],
            "test": [{k: row[k] for k in ("image_id", "file", "width", "height", "label_sha256")}
                     for row in meta["splits"]["test"]], "mapping": meta["mapping"]}


def draw_boxes(image, boxes, names, confidence=None, threshold=0.25, width=2):
    from PIL import ImageDraw
    colors = ("#ff7f0e", "#22bb77", "#42a5ff", "#ed55ba", "#d1bd00", "#aa66ff")
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for i, (cls, x1, y1, x2, y2) in enumerate(boxes):
        if confidence is not None and confidence[i] < threshold:
            continue
        color = colors[int(cls) % len(colors)]
        label = names[int(cls)]
        if confidence is not None:
            label += f" {confidence[i]:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        text_y = max(0, y1 - 13)
        draw.text((x1 + 2, text_y), label, fill=color, stroke_width=1, stroke_fill="black")
    return result
