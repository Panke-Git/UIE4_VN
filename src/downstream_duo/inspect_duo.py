"""Strict COCO detection audit. Never edits source JSON or source pixels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path, PurePosixPath

from PIL import Image

from .utils import OUTPUT, digest_json, draw_boxes, fresh_dir, output_path, read_json, sha256, write_json

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED = {"train": 6671, "test": 1111}


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_image(root, split, name, files, by_basename):
    """Use explicit verifiable suffixes; basename fallback must be unique."""
    rel = PurePosixPath(str(name).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe file_name: {name}")
    suffixes = [rel]
    for prefix in (("images", split), (split,)):
        if rel.parts[:len(prefix)] == prefix:
            suffixes.append(PurePosixPath(*rel.parts[len(prefix):]))
    candidates = {root / "images" / split / Path(s) for s in suffixes} & files
    # Duplicate basenames are rejected even if one direct path exists: no implicit pairing.
    if len(by_basename[rel.name]) > 1:
        raise ValueError(f"Duplicate basename / ambiguous file_name: {name}")
    if not candidates:
        candidates = set(by_basename[rel.name])
    if len(candidates) != 1:
        raise ValueError(f"Missing or ambiguous file_name: {name}, candidates={sorted(map(str,candidates))}")
    return candidates.pop()


def audit_dataset(root, expected=None, previews_dir=None, preview_count=3, bbox_policy="strict", empty_after_clip="error"):
    root = Path(root).expanduser().resolve()
    expected = EXPECTED if expected is None else expected
    if bbox_policy not in ("strict", "clip"):
        raise ValueError("bbox_policy must be strict or clip")
    if empty_after_clip not in ("error", "drop") or (empty_after_clip == "drop" and bbox_policy != "clip"):
        raise ValueError("empty_after_clip=drop requires explicit bbox_policy=clip")
    report = {"data_root": str(root), "task": "bbox_detection", "errors": [], "splits": {},
              "bbox_policy": bbox_policy, "empty_after_clip": empty_after_clip, "bbox_corrections": [],
              "protocol": "COCO xywh pixels; source files read-only; any derived-label correction requires explicit policy and per-annotation log"}
    data, all_images, mapping = {}, {}, None
    for split in ("train", "test"):
        source = root / "annotations" / f"instances_{split}.json"
        doc = read_json(source)
        for key in ("images", "annotations", "categories"):
            if not isinstance(doc.get(key), list):
                raise ValueError(f"{source}: {key} must be a list")
        cats = doc["categories"]
        ids = [c["id"] for c in cats]
        if any(not integer(i) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{split}: invalid/duplicate category IDs")
        if len(set(c["name"] for c in cats)) != len(cats):
            raise ValueError(f"{split}: duplicate category names")
        current = {"categories": sorted(cats, key=lambda c: c["id"])}
        current["coco_to_yolo"] = {str(c["id"]): i for i, c in enumerate(current["categories"])}
        current["yolo_to_coco"] = {str(i): c["id"] for i, c in enumerate(current["categories"])}
        current["names"] = [c["name"] for c in current["categories"]]
        if mapping is not None and mapping != current:
            raise ValueError("Official train/test category definitions differ")
        mapping = current
        if len(cats) != 4:
            report["errors"].append(f"{split}: expected four detection classes, got {len(cats)}")
        directory = root / "images" / split
        files = {p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS}
        by_name = defaultdict(list)
        for path in files:
            by_name[path.name].append(path)
        images, used, annotations, annotation_ids, records, original_records = {}, set(), defaultdict(list), set(), set(), set()
        for item in doc["images"]:
            iid = item["id"]
            if not integer(iid) or iid in images:
                raise ValueError(f"{split}: invalid/duplicate image ID {iid}")
            images[iid] = dict(item)
            try:
                path = resolve_image(root, split, item["file_name"], files, by_name)
                if path in used:
                    raise ValueError(f"Multiple image records reference {path}")
                used.add(path)
                w, h = item["width"], item["height"]
                if not integer(w) or not integer(h) or min(w, h) <= 0:
                    raise ValueError("Image dimensions must be positive integers")
                with Image.open(path) as image:
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError("EXIF orientation would change detector coordinates")
                    rgb = image.convert("RGB")
                    rgb.load()
                    if rgb.size != (w, h):
                        raise ValueError(f"JSON size {(w,h)} != decoded size {rgb.size}")
                    pixel_hash = hashlib.sha256(f"{w}x{h}:RGB:".encode() + rgb.tobytes()).hexdigest()
                images[iid].update(source=str(path), source_sha256=sha256(path), pixel_sha256=pixel_hash)
            except Exception as error:
                report["errors"].append(f"{split} image_id={iid}: {error}")
        for ann in doc["annotations"]:
            aid, iid, cid = ann.get("id"), ann.get("image_id"), ann.get("category_id")
            try:
                if not integer(aid) or aid in annotation_ids:
                    raise ValueError(f"Invalid/duplicate annotation ID {aid}")
                annotation_ids.add(aid)
                if not integer(iid) or iid not in images or not integer(cid) or cid not in ids:
                    raise ValueError("Unknown image_id/category_id")
                box = ann.get("bbox")
                if not isinstance(box, list) or len(box) != 4 or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box
                ):
                    raise ValueError(f"Invalid COCO xywh bbox {box}")
                x, y, w, h = box
                original_key = (iid, cid, *box)
                if original_key in original_records:
                    raise ValueError("Duplicate original bbox record")
                original_records.add(original_key)
                image = images[iid]
                if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                    raise ValueError("iscrowd/ignore unsupported in YOLO conversion; explicit protocol decision required")
                if min(w, h) <= 0:
                    raise ValueError(f"Nonpositive bbox {box}")
                if min(x, y) < 0 or x + w > image["width"] or y + h > image["height"]:
                    message = f"Out-of-bounds bbox {box} in {image['width']}x{image['height']}"
                    if bbox_policy == "strict":
                        raise ValueError(message)
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(image["width"], x+w), min(image["height"], y+h)
                    if x2 <= x1 or y2 <= y1:
                        if empty_after_clip == "error":
                            raise ValueError(f"No positive intersection: {message}")
                        report["errors"].append(f"{split} annotation_id={aid}: {message}")
                        report["bbox_corrections"].append({"split": split, "annotation_id": aid,
                            "image_id": iid, "category_id": cid, "original_bbox": box,
                            "derived_bbox": None, "action": "drop_empty_after_clip",
                            "width": image["width"], "height": image["height"]})
                        continue  # Keep image; explicitly omit this zero-intersection target only.
                    corrected = [x1, y1, x2-x1, y2-y1]
                    report["errors"].append(f"{split} annotation_id={aid}: {message}")
                    report["bbox_corrections"].append({"split": split, "annotation_id": aid,
                        "image_id": iid, "category_id": cid, "original_bbox": box,
                        "derived_bbox": corrected, "action": "clip", "width": image["width"], "height": image["height"]})
                    ann = {**ann, "bbox": corrected}
                    box = corrected
                key = (iid, cid, *box)
                if key in records:
                    raise ValueError("Duplicate bbox record")
                records.add(key)
                annotations[iid].append(ann)
            except Exception as error:
                report["errors"].append(f"{split} annotation_id={aid}: {error}")
        counts = Counter(a["category_id"] for rows in annotations.values() for a in rows)
        missing = sorted(str(p.relative_to(directory)) for p in files - used)
        if len(images) != expected[split] or len(files) != expected[split] or missing:
            report["errors"].append(f"{split}: expected {expected[split]}, JSON={len(images)}, files={len(files)}, unreferenced={len(missing)}")
        report["splits"][split] = {"json_images": len(images), "actual_image_files": len(files),
            "annotations": len(doc["annotations"]), "valid_annotations": sum(counts.values()),
            "category_count": len(cats), "instances_per_category": {str(c): counts[c] for c in ids},
            "empty_images": sum(not annotations[i] for i in images), "unreferenced_files": missing,
            "annotation_json_sha256": sha256(source)}
        report["splits"][split]["original_instances_per_category"] = dict(Counter(str(a.get("category_id")) for a in doc["annotations"]))
        data[split] = {"images": images, "annotations": annotations, "categories": cats,
                       "original_annotations": doc["annotations"]}
        all_images[split] = list(images.values())
        if previews_dir is not None:
            Path(previews_dir).mkdir(parents=True, exist_ok=True)
            for image in sorted(images.values(), key=lambda i: i["id"])[:preview_count]:
                if "source" not in image:
                    continue
                boxes = [(mapping["coco_to_yolo"][str(a["category_id"])], a["bbox"][0], a["bbox"][1],
                          a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3])
                         for a in annotations[image["id"]]]
                with Image.open(image["source"]) as im:
                    preview = draw_boxes(im, boxes, mapping["names"])
                    preview.thumbnail((1600, 1200))
                    preview.save(Path(previews_dir) / f"gt_{split}_{image['id']}.png")
    hashes = defaultdict(list)
    for split, items in all_images.items():
        for item in items:
            if "pixel_sha256" in item:
                hashes[item["pixel_sha256"]].append({"split": split, "image_id": item["id"], "source": item["source"]})
    report["identical_rgb_groups"] = [rows for rows in hashes.values() if len(rows) > 1]
    report["cross_official_split_identical_rgb"] = [rows for rows in report["identical_rgb_groups"]
                                                      if len({r["split"] for r in rows}) > 1]
    report["overlapping_numeric_ids"] = sorted(set(data["train"]["images"]) & set(data["test"]["images"]))
    report["overlapping_basenames_count"] = len({Path(i["file_name"]).name for i in all_images["train"]}
                                               & {Path(i["file_name"]).name for i in all_images["test"]})
    report["leakage_note"] = "ID/basename overlap alone is not leakage; exact decoded RGB hashes are checked. Near-duplicate scene identity is not inferred from filenames; optional train-only group TSV is supported."
    report["image_path_mapping"] = {split: [{k: item[k] for k in ("id", "file_name", "width", "height", "source", "source_sha256", "pixel_sha256") if k in item}
                                                for item in sorted(items, key=lambda r: r["id"])]
                                    for split, items in all_images.items()}
    report["valid"] = not report["errors"]
    report["conversion_allowed"] = len(report["errors"]) == len(report["bbox_corrections"])
    report["unresolved_error_count"] = len(report["errors"]) - len(report["bbox_corrections"])
    report["correction_actions"] = dict(Counter(c["action"] for c in report["bbox_corrections"]))
    report["mapping_sha256"] = digest_json(mapping)
    return data, mapping, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default=str(OUTPUT / "audit"))
    p.add_argument("--preview-count", type=int, default=3)
    args = p.parse_args(argv)
    out = fresh_dir(output_path(args.output))
    _, mapping, report = audit_dataset(args.data_root, previews_dir=out / "gt_previews", preview_count=args.preview_count)
    write_json(out / "dataset_audit.json", report)
    write_json(out / "categories_mapping.json", mapping)
    print(f"Audit valid={report['valid']} errors={len(report['errors'])}: {out / 'dataset_audit.json'}")
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
