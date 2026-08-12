#!/usr/bin/env python3
"""Validate fixed LSUI TSV protocol independently of model code."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"train": 3466, "validation": 385, "test": 428}


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def read(path: Path) -> list[tuple[str, str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if rows and tuple(cell.lower() for cell in rows[0]) in {
        ("sample_id", "input_path", "gt_path"), ("id", "input", "gt")
    }:
        rows = rows[1:]
    for line, row in enumerate(rows, 1):
        if len(row) != 3 or any(not value.strip() for value in row):
            raise ValueError(f"{path}:{line}: expected 3 non-empty TSV columns")
        if Path(row[1]).is_absolute() or Path(row[2]).is_absolute():
            raise ValueError(f"{path}:{line}: paths must be relative")
    return [(row[0].strip(), row[1].strip(), row[2].strip()) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_v1.yaml")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = PROJECT_ROOT / "split" / "lsui19"
    paths = {split: base / f"{split}.tsv" for split in EXPECTED}
    parsed = {split: read(path) for split, path in paths.items()}
    hashes = {split: sha256(path) for split, path in paths.items()}

    print("TSV schema: no header; 3 columns = sample_id, relative input path, relative GT path")
    for split in EXPECTED:
        print(f"{split} count: {len(parsed[split])}")
        print(f"{split} SHA256: {hashes[split]}")
        if len(parsed[split]) != EXPECTED[split]:
            raise ValueError(f"{split} count mismatch")
    print(f"total count: {sum(map(len, parsed.values()))}")
    if len(parsed["train"]) + len(parsed["validation"]) != 3851:
        raise ValueError("train + validation must equal 3851")

    for split, rows in parsed.items():
        duplicate_count = 0
        for column in range(3):
            values = [row[column] for row in rows]
            duplicate_count += len(values) - len(set(values))
        print(f"{split} within-split duplicate count: {duplicate_count}")
        if duplicate_count:
            raise ValueError(f"Duplicate identity in {split}")
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap_count = sum(
            len({row[column] for row in parsed[first]} & {row[column] for row in parsed[second]})
            for column in range(3)
        )
        print(f"{first} vs {second} cross-split overlap count: {overlap_count}")
        if overlap_count:
            raise ValueError(f"Cross-split leakage between {first} and {second}")

    data_root = Path(args.data_root or config["data"]["root"])
    if not data_root.is_dir():
        print(f"image check: data root unavailable ({data_root})")
        print("missing input count: N/A")
        print("missing GT count: N/A")
        return
    missing_input = missing_gt = unreadable = non_rgb = mismatched = 0
    first_problems: list[str] = []
    for split, rows in parsed.items():
        for sample_id, input_relative, gt_relative in rows:
            input_path, gt_path = data_root / input_relative, data_root / gt_relative
            if not input_path.is_file():
                missing_input += 1
                first_problems.append(f"split={split} id={sample_id} missing input={input_path} gt={gt_path}")
            if not gt_path.is_file():
                missing_gt += 1
                first_problems.append(f"split={split} id={sample_id} input={input_path} missing gt={gt_path}")
            if not input_path.is_file() or not gt_path.is_file():
                continue
            try:
                with Image.open(input_path) as input_image, Image.open(gt_path) as gt_image:
                    input_image.load()
                    gt_image.load()
                    non_rgb += int(input_image.mode != "RGB") + int(gt_image.mode != "RGB")
                    mismatched += int(input_image.size != gt_image.size)
            except Exception as error:
                unreadable += 1
                first_problems.append(f"split={split} id={sample_id} unreadable={error} input={input_path} gt={gt_path}")
            if input_path.stem != gt_path.stem or input_path.stem != sample_id:
                mismatched += 1
    print(f"missing input count: {missing_input}")
    print(f"missing GT count: {missing_gt}")
    print(f"unreadable pair count: {unreadable}")
    print(f"non-RGB image count: {non_rgb}")
    print(f"mismatched pair count: {mismatched}")
    if missing_input or missing_gt or unreadable or non_rgb or mismatched:
        raise RuntimeError("Image validation failed:\n" + "\n".join(first_problems[:20]))


if __name__ == "__main__":
    main()

