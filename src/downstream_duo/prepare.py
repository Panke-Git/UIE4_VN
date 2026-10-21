"""Audit and convert original DUO on the execution host, splitting official train only."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import random

from PIL import Image

from .inspect_duo import audit_dataset
from .utils import (OUTPUT, digest_json, fresh_dir, output_path, sha256, tsv,
                    write_json, write_yaml)


def coco_to_yolo(box, width, height, cls):
    x, y, w, h = box
    values = [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
    import math
    if not isinstance(cls, int) or cls < 0 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Invalid normalized YOLO label")
    if min(w, h) <= 0 or min(x, y) < 0 or x + w > width or y + h > height:
        raise ValueError("Invalid COCO bbox; no clipping allowed")
    return f"{cls} " + " ".join(f"{v:.12g}" for v in values)


def split_train(images, seed=3520, fraction=0.1, groups=None):
    if not 0 < fraction < 1 or len(images) < 2:
        raise ValueError("Need at least two training images and 0 < val fraction < 1")
    parent = {i: i for i in images}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    seen = {}
    for iid in sorted(images):
        keys = [("rgb", images[iid]["pixel_sha256"])]
        if groups and iid in groups:
            keys.append(("scene", groups[iid]))
        for key in keys:
            if key in seen:
                union(iid, seen[key])
            seen[key] = iid
    buckets = {}
    for iid in sorted(images):
        buckets.setdefault(find(iid), []).append(iid)
    units = sorted(buckets.values(), key=lambda b: b[0])
    if len(units) < 2:
        raise ValueError("All official training images belong to one group")
    random.Random(seed).shuffle(units)
    target = max(1, round(len(images) * fraction))
    val = []
    for group in units[:-1]:
        if not val or abs(len(val) + len(group) - target) <= abs(len(val) - target):
            val.extend(group)
    validation = set(val)
    return {"train": sorted(set(images) - validation), "val": sorted(validation)}


def prepare(root, out, seed=3520, fraction=0.1, groups_tsv=None, expected=None, synthetic=False, bbox_policy="strict", empty_after_clip="error"):
    out = fresh_dir(out)
    data, mapping, audit = audit_dataset(root, expected=expected, previews_dir=out / "gt_previews", bbox_policy=bbox_policy, empty_after_clip=empty_after_clip)
    write_json(out / "dataset_audit.json", audit)
    write_json(out / "category_mapping.json", mapping)
    write_json(out / "bbox_corrections.json", audit["bbox_corrections"])
    if not audit["conversion_allowed"]:
        raise ValueError(f"DUO audit has {audit['unresolved_error_count']} unresolved errors; see {out/'dataset_audit.json'}")
    if audit["cross_official_split_identical_rgb"]:
        raise ValueError("Exact RGB content overlaps official train/test; resolve data provenance before training")
    groups = {}
    if groups_tsv:
        with Path(groups_tsv).open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                iid = int(row["image_id"])
                if iid not in data["train"]["images"] or iid in groups or not row["group"]:
                    raise ValueError("Groups TSV requires unique official-train image_id and nonempty group")
                groups[iid] = row["group"]
    splits = split_train(data["train"]["images"], seed, fraction, groups)
    splits["test"] = sorted(data["test"]["images"])
    snapshot = {"mapping": mapping, "official": {}}
    for source_split in ("train", "test"):
        snapshot["official"][source_split] = {
            "images": [{k: image[k] for k in ("id", "file_name", "width", "height", "source_sha256", "pixel_sha256")}
                       for image in sorted(data[source_split]["images"].values(), key=lambda i: i["id"])],
            "annotations": sorted(data[source_split]["original_annotations"], key=lambda a: a["id"])}
    snapshot["bbox_policy"] = bbox_policy
    snapshot["empty_after_clip"] = empty_after_clip
    snapshot["bbox_corrections"] = audit["bbox_corrections"]
    write_json(out / "dataset_snapshot.json", snapshot)
    (out / "manifests").mkdir()
    rows_by_split = {}
    for split, ids in splits.items():
        original = "test" if split == "test" else "train"
        rows = []
        for iid in ids:
            image = data[original]["images"][iid]
            relative = Path(str(iid)) / (Path(image["file_name"]).stem + ".png")
            destination = out / "images" / f"{split}_raw" / relative
            label = out / "labels" / f"{split}_raw" / relative.with_suffix(".txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            # Decode once with Pillow, preserve exact RGB pixels, verify PNG round trip.
            with Image.open(image["source"]) as source:
                rgb = source.convert("RGB")
                rgb.save(destination)
                with Image.open(destination) as check:
                    if check.convert("RGB").tobytes() != rgb.tobytes() or check.size != rgb.size:
                        raise ValueError(f"PNG round-trip failed: {destination}")
            if sha256(image["source"]) != image["source_sha256"]:
                raise ValueError("Source changed during preparation")
            lines = [coco_to_yolo(a["bbox"], image["width"], image["height"], mapping["coco_to_yolo"][str(a["category_id"])])
                     for a in data[original]["annotations"][iid]]
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            rows.append({"image_id": iid, "official_split": original, "source_file": image["file_name"],
                         "file": relative.as_posix(), "width": image["width"], "height": image["height"],
                         "source_sha256": image["source_sha256"], "pixel_sha256": image["pixel_sha256"],
                         "image_sha256": sha256(destination), "label_sha256": sha256(label)})
        rows_by_split[split] = rows
        tsv(out / "manifests" / f"{split}.tsv", rows)
        write_json(out / "manifests" / f"{split}.json", rows)
    write_yaml(out / "duo_raw.yaml", {"path": str(Path(out).resolve()), "train": "images/train_raw",
               "val": "images/val_raw", "test": "images/test_raw", "names": mapping["names"]})
    metadata = {"seed": seed, "val_fraction_requested": fraction, "synthetic": synthetic,
        "bbox_policy": bbox_policy, "empty_after_clip": empty_after_clip,
        "counts": {s: len(r) for s, r in rows_by_split.items()}, "splits": rows_by_split,
        "mapping": mapping, "dataset_digest": digest_json(snapshot),
        "source_root": str(Path(root).resolve()), "official_counts_expected": expected or {"train": 6671, "test": 1111},
        "groups": groups, "groups_input_sha256": sha256(groups_tsv) if groups_tsv else None,
        "split_method": "seeded shuffle of train-only connected groups: exact decoded RGB + optional scene TSV; closest greedy 10%; no test selection",
        "raw_encoding": "Pillow RGB decode -> lossless 8-bit PNG; verified identical decoded pixels; original stem retained under image_id subdirectory",
        "snapshot_sha256": {p.relative_to(out).as_posix(): sha256(p) for p in sorted(Path(out).rglob("*"))
                            if p.is_file() and (p.parent.name == "manifests" or p.name in ("dataset_snapshot.json", "category_mapping.json", "duo_raw.yaml", "bbox_corrections.json"))}}
    write_json(out / "split_metadata.json", metadata)
    (out / "split_metadata.sha256").write_text(sha256(out / "split_metadata.json") + "\n", encoding="ascii")
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default=str(OUTPUT / "prepared_duo"))
    p.add_argument("--seed", type=int, default=3520)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--groups-tsv", help="Optional predeclared train-only scene groups: image_id<TAB>group")
    p.add_argument("--bbox-policy", choices=["strict", "clip"], default="strict", help="clip requires explicit choice; logs every change, never edits source")
    p.add_argument("--empty-after-clip", choices=["error", "drop"], default="error", help="Explicitly drop/log boxes with zero image intersection; keep images")
    args = p.parse_args(argv)
    meta = prepare(args.data_root, output_path(args.output), args.seed, args.val_fraction, args.groups_tsv,
                   bbox_policy=args.bbox_policy, empty_after_clip=args.empty_after_clip)
    print(meta["counts"])


if __name__ == "__main__":
    main()
