"""Read-only Haze4K decoding, pairing, content audit and deterministic grouped split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

VERSION = "haze4k-split-v1"
COLUMNS = ["sample_id", "input_relative", "gt_relative", "clean_id"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "validation", "test")
DEFAULT_SPLIT = Path(__file__).resolve().parents[2] / "split/haze4k"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(json_bytes(value))
    temp.replace(path)


def select_folder(root, official, names):
    found = [root / official / name for name in names if (root / official / name).is_dir()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {names} directory in {root / official}; found {found}")
    return found[0]


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative or str(p) != relative:
        raise ValueError(f"Unsafe/noncanonical dataset-relative path: {relative!r}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {path}")
    return path


def check_rows(rows, split):
    if not rows:
        raise ValueError(f"Empty manifest: {split}")
    for field in ("sample_id", "input_relative"):
        values = [r[field] for r in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {field} in {split}")
    for r in rows:
        official = "test" if split == "test" else "train"
        for field, folders in (("input_relative", {"haze", "hazy"}), ("gt_relative", {"gt", "clear"})):
            p = PurePosixPath(r[field])
            safe_path(Path("."), r[field])
            if len(p.parts) != 3 or p.parts[0] != official or p.parts[1] not in folders:
                raise ValueError(f"Wrong official split/role: {r}")
        haze = PurePosixPath(r["input_relative"])
        gt = PurePosixPath(r["gt_relative"])
        if (haze.stem.split("_", 1)[0] != r["clean_id"] or gt.stem != r["clean_id"]
                or r["sample_id"] != f"{official}/{haze.stem}"):
            raise ValueError(f"Incorrect pairing/identity: {r}")


def manifest_bytes(rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def read_manifest(path, split):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Expected header {COLUMNS}: {path}")
        rows = list(reader)
    if any(set(r) != set(COLUMNS) or any(not v for v in r.values()) for r in rows):
        raise ValueError(f"Malformed manifest: {path}")
    check_rows(rows, split)
    return rows


def decode_info(path):
    try:
        with Image.open(path) as image:
            image.load()  # Full decode; never enable LOAD_TRUNCATED_IMAGES.
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-frame image is unsupported")
            rgb = image.convert("RGB")  # PIL deterministic conversion, no EXIF transpose/ICC conversion.
            pixels = np.asarray(rgb)
            return {"size": list(image.size), "mode": image.mode, "sha256": sha256(path),
                    "rgb_sha256": hashlib.sha256(str(image.size).encode() + pixels.tobytes()).hexdigest()}
    except Exception as e:
        raise ValueError(f"Cannot fully decode {path}: {e}") from e


def scan(root):
    inventory, official_rows, report = {}, {}, {}
    problems = []
    for official in ("train", "test"):
        folders = {"input": select_folder(root, official, ("haze", "hazy")),
                   "gt": select_folder(root, official, ("gt", "clear"))}
        images, miscellaneous = {}, []
        for role, folder in folders.items():
            images[role] = []
            for path in sorted(folder.iterdir(), key=lambda p: p.name):
                if not path.is_file():
                    problems.append(f"Unexpected non-file in image directory: {path}")
                    continue
                relative = path.relative_to(root).as_posix()
                safe_path(root, relative)
                metadata_file = path.name in {".DS_Store", "Thumbs.db", "desktop.ini"} or path.name.startswith("._")
                if metadata_file or path.suffix.lower() not in EXTENSIONS:
                    miscellaneous.append({"path": relative, "ignored_metadata": metadata_file})
                    continue
                images[role].append(path)
                try:
                    inventory[relative] = {**decode_info(path), "role": role, "official_split": official}
                except ValueError as e:
                    problems.append(str(e))
        gt_by_id = defaultdict(list)
        for gt in images["gt"]:
            gt_by_id[gt.stem].append(gt)
        for clean, paths in gt_by_id.items():
            if len(paths) != 1:
                problems.append(f"Ambiguous GT stem {official}/{clean}: {paths}")
        rows, used = [], set()
        for haze in images["input"]:
            clean = haze.stem.split("_", 1)[0]
            matches = gt_by_id.get(clean, [])
            if len(matches) != 1:
                problems.append(f"Missing/ambiguous GT for {haze}: {matches}")
                continue
            gt = matches[0]
            ir, gr = haze.relative_to(root).as_posix(), gt.relative_to(root).as_posix()
            used.add(gt)
            if ir in inventory and gr in inventory and inventory[ir]["size"] != inventory[gr]["size"]:
                problems.append(f"Pair size mismatch: {haze} / {gt}")
            rows.append(dict(zip(COLUMNS, (f"{official}/{haze.stem}", ir, gr, clean))))
        unused = set(images["gt"]) - used
        if unused:
            problems.append(f"Unused GTs: {sorted(map(str, unused))}")
        if rows:
            check_rows(rows, "test" if official == "test" else "train")
        else:
            problems.append(f"No paired images: {official}")
        official_rows[official] = rows
        multiplicities = Counter(Counter(r["clean_id"] for r in rows).values())
        report[official] = {"haze_count": len(images["input"]), "gt_count": len(images["gt"]),
                            "haze_per_clean_distribution": dict(sorted(multiplicities.items())),
                            "resolution_counts": dict(sorted(Counter(
                                f"{v['size'][0]}x{v['size'][1]}" for v in inventory.values()
                                if v["official_split"] == official and v["role"] == "input").items())),
                            "original_mode_counts": dict(Counter(v["mode"] for v in inventory.values()
                                if v["official_split"] == official)),
                            "non_images": miscellaneous, "pair_examples": rows[:5]}
    if problems:
        raise ValueError("Preflight FAILED:\n" + "\n".join(problems))
    overlaps = {}
    # Check all roles, including input-vs-GT across official splits. IDs play no role.
    for field in ("sha256", "rgb_sha256"):
        by_hash = defaultdict(lambda: {"train": [], "test": []})
        for path, info in inventory.items():
            by_hash[info[field]][info["official_split"]].append(path)
        overlaps[field] = [dict(hash=h, **parts) for h, parts in sorted(by_hash.items())
                           if parts["train"] and parts["test"]]
    report["cross_official_content_overlap"] = overlaps
    report["rgb_conversion"] = "PIL convert('RGB'); no EXIF transpose or ICC transform; single frame only"
    report["trans"] = "Not scanned or read"
    return official_rows, inventory, report


def grouped_split(rows, seed=3520):
    ids = sorted({r["clean_id"] for r in rows})  # Unicode lexicographic, not numeric.
    if len(ids) < 2:
        raise ValueError("At least two training clean IDs required")
    random.Random(seed).shuffle(ids)
    nval = min(len(ids) - 1, max(1, int(len(ids) * 0.1 + 0.5)))
    validation = set(ids[:nval])
    return {"train": [r for r in rows if r["clean_id"] not in validation],
            "validation": [r for r in rows if r["clean_id"] in validation]}


def prepare(root, split_dir=DEFAULT_SPLIT, seed=3520, accept_content_overlap=False, synthetic=False):
    root, split_dir = Path(root).resolve(), Path(split_dir).resolve()
    print(f"Absolute data root: {root}", flush=True)
    if not root.is_dir():
        raise FileNotFoundError(f"Haze4K data root unavailable: {root}")
    if split_dir == root or split_dir.is_relative_to(root):
        raise ValueError("Split output must be outside the read-only dataset root")
    split_dir.mkdir(parents=True, exist_ok=True)
    official, inventory, report = scan(root)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    overlap = any(report["cross_official_content_overlap"].values())
    if overlap and not accept_content_overlap:
        # Only write a report, never change the official split or produce trainable manifests.
        rejected = split_dir / "overlap_review_required.json"
        if not rejected.exists():
            write_json(rejected, report)
        raise ValueError(f"Cross-official content overlap requires user review: {rejected}. "
                         "After review only, use --accept-content-overlap to preserve official splits.")
    rows = {**grouped_split(official["train"], seed), "test": official["test"]}
    payloads = {f"{s}.tsv": manifest_bytes(rows[s]) for s in SPLITS}
    payloads["inventory.json"] = json_bytes(inventory)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    metadata = {"version": VERSION, "seed": seed, "data_root_at_prepare": str(root),
                "synthetic": synthetic, "accepted_content_overlap": bool(overlap and accept_content_overlap),
                "algorithm": "sort Unicode clean IDs; Python random.Random(seed).shuffle; first round_half_up(10%) val; input paths lexicographic",
                "counts": {s: len(rows[s]) for s in SPLITS}, "hashes": hashes, "report": report}
    meta_path = split_dir / "split_metadata.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for key in ("version", "seed", "synthetic", "accepted_content_overlap", "counts", "hashes"):
            if old[key] != metadata[key]:
                raise ValueError(f"Existing split differs ({key}); refusing rewrite. Choose a NEW --split-dir explicitly.")
        for name, digest in hashes.items():
            if not (split_dir / name).is_file() or sha256(split_dir / name) != digest:
                raise ValueError(f"Existing split artifact modified/missing: {split_dir / name}; refusing rewrite")
        print("Existing manifests/content inventory verified byte-for-byte; nothing rewritten.")
    else:
        existing = [p for p in split_dir.iterdir() if p.name != "overlap_review_required.json"]
        if existing:
            raise ValueError(f"Nonempty split directory without metadata; refusing rewrite: {existing}")
        for name, data in payloads.items():
            (split_dir / name).write_bytes(data)
        write_json(meta_path, metadata)  # Write approval marker last.
    print(json.dumps({"counts": metadata["counts"], "hashes": hashes}, indent=2))
    return metadata


def validate_bundle(split_dir):
    split_dir = Path(split_dir)
    meta = json.loads((split_dir / "split_metadata.json").read_text())
    if meta["version"] != VERSION:
        raise ValueError("Unknown split version")
    for name in ["train.tsv", "validation.tsv", "test.tsv", "inventory.json"]:
        if sha256(split_dir / name) != meta["hashes"][name]:
            raise ValueError(f"Split hash mismatch: {split_dir / name}")
    rows = {s: read_manifest(split_dir / f"{s}.tsv", s) for s in SPLITS}
    if any(len(rows[s]) != meta["counts"][s] for s in SPLITS):
        raise ValueError("Manifest count mismatch")
    if {r["clean_id"] for r in rows["train"]} & {r["clean_id"] for r in rows["validation"]}:
        raise ValueError("Clean ID leakage train/validation")
    all_rows = sum(rows.values(), [])
    if len({r["input_relative"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate input across splits")
    expected = grouped_split(sorted(rows["train"] + rows["validation"], key=lambda r: r["input_relative"]), meta["seed"])
    if any(rows[s] != expected[s] for s in ("train", "validation")):
        raise ValueError("Manifests violate deterministic grouped split")
    inventory = json.loads((split_dir / "inventory.json").read_text())
    referenced = {r[f] for r in all_rows for f in ("input_relative", "gt_relative")}
    if referenced != set(inventory):
        raise ValueError("Manifest coverage differs from preflight inventory")
    return meta, rows, inventory


def verify_content(root, rows, inventory):
    # Detect added/removed images and newly ambiguous aliases without decoding test data during training.
    observed = set()
    for official in ("train", "test"):
        for names in (("haze", "hazy"), ("gt", "clear")):
            folder = select_folder(Path(root), official, names)
            for path in folder.iterdir():
                metadata = path.name in {".DS_Store", "Thumbs.db", "desktop.ini"} or path.name.startswith("._")
                if path.is_file() and path.suffix.lower() in EXTENSIONS and not metadata:
                    observed.add(path.relative_to(root).as_posix())
    if observed != set(inventory):
        raise ValueError(f"Dataset image inventory changed since preflight; added={sorted(observed-set(inventory))}; missing={sorted(set(inventory)-observed)}")
    for relative in sorted({r[f] for r in rows for f in ("input_relative", "gt_relative")}):
        path = safe_path(Path(root), relative)
        if sha256(path) != inventory[relative]["sha256"]:
            raise ValueError(f"Data changed since full-decode preflight: {path}; run prepare in a new split directory")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split-dir", default=str(DEFAULT_SPLIT))
    p.add_argument("--seed", type=int, default=3520)
    p.add_argument("--accept-content-overlap", action="store_true", help="Explicit user acceptance after reviewing overlap report")
    p.add_argument("--synthetic", action="store_true", help="Mark generated software-test data; never paper evidence")
    a = p.parse_args(argv)
    try:
        prepare(a.data_root, a.split_dir, a.seed, a.accept_content_overlap, a.synthetic)
    except OSError as e:
        raise SystemExit(f"Filesystem operation failed (path/permissions/disk): {e}") from e


if __name__ == "__main__":
    main()
