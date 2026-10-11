"""Freeze and verify physical Train/Val data without importing legacy code or torch."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROTOCOL = "V16_19_physical_train_val_v1"
COUNTS = {"lsui19": {"train": 3851, "validation": 428}, "uieb19": {"train": 801, "validation": 89}}
LABELS = {"lsui19": "LSUI19", "uieb19": "UIEB19"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing paired directory: {directory}")
    result = {}
    for path in sorted(directory.iterdir()):
        if path.name == ".DS_Store" or path.name.startswith("._"):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Unexpected entry in paired directory: {path}")
        if any(c in path.name for c in "\t\r\n") or path.stem in {".", ".."}:
            raise ValueError(f"Invalid sample filename: {path.name!r}")
        result[path.name] = path
    return result


def _image(path: Path, cache: dict) -> dict:
    sha = file_sha256(path)
    if sha not in cache:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Expected single-frame RGB image: {path}")
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            cache[sha] = {"sha256": sha, "width": image.width, "height": image.height,
                          "format": image.format, "pixel_sha256": digest}
    return dict(cache[sha])


def collect_dataset(root: Path, dataset: str, expected_counts: dict | None = None) -> tuple[dict, dict]:
    if dataset not in COUNTS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    counts = COUNTS[dataset] if expected_counts is None else expected_counts
    if set(counts) != {"train", "validation"}:
        raise ValueError("Only train and validation counts are supported")
    entries = {}; cache = {}; identities = set(); duplicate_inputs = defaultdict(list)
    format_mismatches = []
    for split, folder in [("train", "Train"), ("validation", "Val")]:
        inputs, targets = _files(root / folder / "input"), _files(root / folder / "GT")
        if inputs.keys() != targets.keys():
            raise ValueError(f"Unpaired {folder} files: input-only={sorted(inputs.keys()-targets.keys())[:5]}, GT-only={sorted(targets.keys()-inputs.keys())[:5]}")
        if len(inputs) != int(counts[split]) or not inputs:
            raise ValueError(f"{dataset}/{split} has {len(inputs)} pairs; expected {counts[split]}")
        entries[split] = []
        for name in sorted(inputs):
            sid = Path(name).stem
            if sid in identities:
                raise ValueError(f"Duplicate sample id within/across folders: {sid}")
            identities.add(sid)
            inp, gt = _image(inputs[name], cache), _image(targets[name], cache)
            if (inp["width"], inp["height"]) != (gt["width"], gt["height"]):
                raise ValueError(f"Input/GT dimensions differ: {folder}/{name}")
            entry = {"sample_id": sid, "split": split, "input_relative": f"{folder}/input/{name}",
                     "gt_relative": f"{folder}/GT/{name}", "input": inp, "gt": gt}
            entries[split].append(entry)
            duplicate_inputs[inp["pixel_sha256"]].append(entry)
            for role, meta in [("input", inp), ("GT", gt)]:
                if Path(name).suffix.lower() == ".png" and meta["format"] != "PNG":
                    format_mismatches.append({"path": f"{folder}/{role}/{name}", "actual_format": meta["format"]})
    groups = []
    for group in duplicate_inputs.values():
        if len(group) < 2:
            continue
        groups.append({"samples": [{"sample_id": e["sample_id"], "split": e["split"], "path": e["input_relative"]} for e in group],
                       "cross_split": len({e["split"] for e in group}) > 1,
                       "identical_gt_pixels": len({e["gt"]["pixel_sha256"] for e in group}) == 1})
    quality = {"duplicate_definition": "identical RGB dimensions and every pixel",
               "duplicate_input_groups": groups, "cross_split_duplicate_groups": sum(g["cross_split"] for g in groups),
               "format_extension_mismatches": format_mismatches,
               "policy": "Preserve physical membership and originals; duplicates are reported, not removed."}
    return entries, quality


def freeze_dataset(root: Path, dataset: str, output: Path, expected_counts: dict | None = None) -> dict:
    """Create once. Existing protocols are verified, never silently regenerated."""
    root, output = root.resolve(), output.resolve()
    if output.exists():
        protocol = verify_dataset(root, output)
        if protocol["dataset"] != dataset:
            raise ValueError("Existing protocol is for a different dataset")
        return protocol
    entries, quality = collect_dataset(root, dataset, expected_counts)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for split, values in entries.items():
            with (temporary / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerows((e["sample_id"], e["input_relative"], e["gt_relative"]) for e in values)
        write_json(temporary / "file_index.json", {"entries": [e for values in entries.values() for e in values]})
        write_json(temporary / "data_quality.json", quality)
        hashes = {n: file_sha256(temporary / n) for n in ["train.tsv", "validation.tsv", "file_index.json", "data_quality.json"]}
        fingerprint = hashlib.sha256(json.dumps({"protocol": PROTOCOL, "dataset": dataset, "hashes": hashes}, sort_keys=True).encode()).hexdigest()
        protocol = {"protocol": PROTOCOL, "dataset": dataset, "label": LABELS[dataset],
                    "counts": {k: len(v) for k, v in entries.items()}, "files_sha256": hashes,
                    "fingerprint": fingerprint, "membership": "existing physical Train/Val folders", "has_test_split": False}
        write_json(temporary / "protocol.json", protocol)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return protocol


def read_protocol(directory: Path) -> dict:
    directory = directory.resolve()
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol["protocol"] != PROTOCOL or protocol.get("has_test_split") is not False:
        raise ValueError("Not a V16-19 two-split protocol")
    if protocol["dataset"] not in COUNTS or set(protocol["counts"]) != {"train", "validation"}:
        raise ValueError("Invalid dataset or split keys")
    expected_files = {"train.tsv", "validation.tsv", "file_index.json", "data_quality.json"}
    if set(protocol["files_sha256"]) != expected_files or (directory / "test.tsv").exists():
        raise ValueError("Invalid frozen protocol files")
    for filename, expected in protocol["files_sha256"].items():
        if file_sha256(directory / filename) != expected:
            raise ValueError(f"Frozen protocol file changed: {filename}")
    fingerprint = hashlib.sha256(json.dumps({"protocol": PROTOCOL, "dataset": protocol["dataset"], "hashes": protocol["files_sha256"]}, sort_keys=True).encode()).hexdigest()
    if fingerprint != protocol["fingerprint"]:
        raise ValueError("Protocol fingerprint mismatch")
    return protocol


def verify_dataset(root: Path, directory: Path) -> dict:
    """Verify all input/GT bytes and exact membership against the frozen index."""
    protocol = read_protocol(directory)
    index = json.loads((directory / "file_index.json").read_text(encoding="utf-8"))["entries"]
    if len(index) != sum(protocol["counts"].values()):
        raise ValueError("Frozen index count mismatch")
    for split, folder in [("train", "Train"), ("validation", "Val")]:
        entries = [e for e in index if e["split"] == split]
        if len(entries) != protocol["counts"][split]:
            raise ValueError(f"Frozen {split} index count mismatch")
        expected_rows = [[e["sample_id"], e["input_relative"], e["gt_relative"]] for e in entries]
        with (directory / f"{split}.tsv").open(encoding="utf-8", newline="") as handle:
            if list(csv.reader(handle, delimiter="\t")) != expected_rows:
                raise ValueError(f"Manifest/index disagreement for {split}")
        for role, key in [("input", "input_relative"), ("GT", "gt_relative")]:
            actual = _files(root / folder / role)
            expected_names = {Path(e[key]).name for e in entries}
            if set(actual) != expected_names:
                raise ValueError(f"Dataset membership changed: {folder}/{role}")
            for e in entries:
                path = Path(e[key])
                if path.parts != (folder, role, path.name) or path.stem != e["sample_id"]:
                    raise ValueError(f"Unsafe or mispaired index path: {path}")
                metadata = e["input" if role == "input" else "gt"]
                if file_sha256(root / path) != metadata["sha256"]:
                    raise ValueError(f"Dataset file bytes changed: {root / path}")
    return protocol
