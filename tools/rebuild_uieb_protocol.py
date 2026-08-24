#!/usr/bin/env python3
"""
Rebuild a reproducible UIEB protocol without moving/copying any images.

Expected current dataset layout:
UIEB19/
├── Train/
│   ├── input/
│   └── GT/
└── Val/
    ├── input/
    └── GT/

The script:
1. Collects all 890 paired images from the existing Train/Val folders.
2. Locks the public UIEB T90 test split (90 images).
3. Splits the remaining 800 images into:
       train      = 720
       validation = 80
   using a fixed seed (default: 3520).
4. Writes project-compatible TSV manifests:
       split/uieb/train.tsv
       split/uieb/validation.tsv
       split/uieb/test.tsv
5. Writes protocol metadata and hashes for reproducibility.

No image is moved, copied, renamed, or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# Public UIEB T90 list from:
# https://github.com/ddz16/UIE_Benckmark/blob/main/data/UIEB/test.txt
#
# The list is intentionally embedded here so that the experimental protocol
# does not depend on network access or future repository changes.
PUBLIC_T90_FILENAMES: tuple[str, ...] = (
    "10909.png",
    "10945.png",
    "11052.png",
    "11064.png",
    "11374.png",
    "11398.png",
    "1225.png",
    "12290.png",
    "12336.png",
    "12348.png",
    "12422.png",
    "12433.png",
    "1407.png",
    "1491.png",
    "15001.png",
    "15045.png",
    "15094.png",
    "15113.png",
    "15136.png",
    "1539.png",
    "15418.png",
    "15426.png",
    "15704.png",
    "1573.png",
    "1660.png",
    "1742.png",
    "1957.png",
    "208_img_.png",
    "209_img_.png",
    "210_img_.png",
    "211_img_.png",
    "212_img_.png",
    "213_img_.png",
    "2546.png",
    "25_img_.png",
    "2629.png",
    "2677.png",
    "26_img_.png",
    "2701.png",
    "2774.png",
    "2787.png",
    "2882.png",
    "2977.png",
    "3001.png",
    "3015.png",
    "3196.png",
    "3202.png",
    "3330.png",
    "3650.png",
    "3728.png",
    "379_img_.png",
    "380_img_.png",
    "381_img_.png",
    "382_img_.png",
    "383_img_.png",
    "3925.png",
    "3947.png",
    "4070.png",
    "540_img_.png",
    "541_img_.png",
    "542_img_.png",
    "543_img_.png",
    "544_img_.png",
    "5554.png",
    "5818.png",
    "5830.png",
    "601_img_.png",
    "602_img_.png",
    "603_img_.png",
    "604_img_.png",
    "605_img_.png",
    "6062.png",
    "6082.png",
    "6788.png",
    "6820.png",
    "8010.png",
    "8046.png",
    "82_img_.png",
    "86_img_.png",
    "87_img_.png",
    "89_img_.png",
    "90_img_.png",
    "928_img_.png",
    "929_img_.png",
    "9547.png",
    "9554.png",
    "9557.png",
    "9567.png",
    "9896.png",
    "9900.png",
)

T90_SOURCE_URL = (
    "https://github.com/ddz16/UIE_Benckmark/blob/main/data/UIEB/test.txt"
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


@dataclass(frozen=True)
class PairEntry:
    sample_id: str
    filename: str
    input_relative: str
    gt_relative: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {folder}")

    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.as_posix().lower())


def add_unique_by_filename(
    destination: dict[str, Path],
    files: Iterable[Path],
    label: str,
) -> None:
    for path in files:
        filename = path.name
        if filename in destination:
            raise ValueError(
                f"Duplicate filename across source folders for {label}: "
                f"{filename}\n"
                f"  first : {destination[filename]}\n"
                f"  second: {path}"
            )
        destination[filename] = path


def collect_all_pairs(root: Path) -> dict[str, PairEntry]:
    """
    Collect all pairs from the *existing physical* Train/Val folders.

    Important:
    The physical folder name does NOT determine the new experimental split.
    After this function, the new split is determined only by the generated TSVs.
    """
    input_by_filename: dict[str, Path] = {}
    gt_by_filename: dict[str, Path] = {}

    for physical_split in ("Train", "Val"):
        input_dir = root / physical_split / "input"
        gt_dir = root / physical_split / "GT"

        add_unique_by_filename(
            input_by_filename,
            image_files(input_dir),
            label="input",
        )
        add_unique_by_filename(
            gt_by_filename,
            image_files(gt_dir),
            label="GT",
        )

    input_names = set(input_by_filename)
    gt_names = set(gt_by_filename)

    only_input = sorted(input_names - gt_names)
    only_gt = sorted(gt_names - input_names)

    if only_input or only_gt:
        message = ["Input/GT pairing mismatch detected."]
        if only_input:
            message.append(
                "Files present only in input (first 20): "
                + ", ".join(only_input[:20])
            )
        if only_gt:
            message.append(
                "Files present only in GT (first 20): "
                + ", ".join(only_gt[:20])
            )
        raise ValueError("\n".join(message))

    if len(input_names) != 890:
        raise ValueError(
            f"Expected exactly 890 paired UIEB images, found {len(input_names)}."
        )

    # Downstream dataset.py requires:
    # input_path.stem == gt_path.stem == sample_id
    stem_to_filename: dict[str, str] = {}
    for filename in sorted(input_names):
        stem = Path(filename).stem
        if stem in stem_to_filename:
            raise ValueError(
                "Duplicate sample_id (filename stem) detected. "
                "This would violate the current dataset.py contract:\n"
                f"  sample_id: {stem}\n"
                f"  file 1   : {stem_to_filename[stem]}\n"
                f"  file 2   : {filename}"
            )
        stem_to_filename[stem] = filename

    pairs: dict[str, PairEntry] = {}

    for filename in sorted(input_names):
        input_path = input_by_filename[filename]
        gt_path = gt_by_filename[filename]

        if input_path.stem != gt_path.stem:
            raise ValueError(
                f"Stem mismatch for pair {filename}: "
                f"{input_path.name} vs {gt_path.name}"
            )

        pairs[filename] = PairEntry(
            sample_id=input_path.stem,
            filename=filename,
            input_relative=input_path.relative_to(root).as_posix(),
            gt_relative=gt_path.relative_to(root).as_posix(),
        )

    return pairs


def validate_public_t90(pairs: dict[str, PairEntry]) -> list[str]:
    if len(PUBLIC_T90_FILENAMES) != 90:
        raise RuntimeError(
            f"Internal error: T90 list has {len(PUBLIC_T90_FILENAMES)} entries, "
            "expected 90."
        )

    if len(set(PUBLIC_T90_FILENAMES)) != 90:
        raise RuntimeError("Internal error: duplicate filename in T90 list.")

    missing = sorted(set(PUBLIC_T90_FILENAMES) - set(pairs))

    if missing:
        raise ValueError(
            "The local UIEB19 does not contain all public T90 files.\n"
            f"Missing {len(missing)} file(s):\n  "
            + "\n  ".join(missing)
        )

    return sorted(PUBLIC_T90_FILENAMES)


def build_protocol(
    pairs: dict[str, PairEntry],
    seed: int,
    validation_count: int,
) -> tuple[list[PairEntry], list[PairEntry], list[PairEntry]]:
    test_names = validate_public_t90(pairs)
    test_set = set(test_names)

    development_names = sorted(set(pairs) - test_set)

    if len(development_names) != 800:
        raise ValueError(
            f"Expected 800 non-T90 development images, "
            f"found {len(development_names)}."
        )

    if not (1 <= validation_count < 800):
        raise ValueError(
            f"validation_count must be in [1, 799], got {validation_count}."
        )

    # Deterministic split of the public 800-image training pool.
    shuffled = development_names.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    validation_names = sorted(shuffled[:validation_count])
    train_names = sorted(shuffled[validation_count:])

    train_entries = [pairs[name] for name in train_names]
    validation_entries = [pairs[name] for name in validation_names]
    test_entries = [pairs[name] for name in test_names]

    return train_entries, validation_entries, test_entries


def validate_final_protocol(
    train_entries: list[PairEntry],
    validation_entries: list[PairEntry],
    test_entries: list[PairEntry],
    expected_validation_count: int,
) -> None:
    expected_train_count = 800 - expected_validation_count

    expected = {
        "train": expected_train_count,
        "validation": expected_validation_count,
        "test": 90,
    }

    split_entries = {
        "train": train_entries,
        "validation": validation_entries,
        "test": test_entries,
    }

    for split_name, entries in split_entries.items():
        if len(entries) != expected[split_name]:
            raise ValueError(
                f"{split_name} count mismatch: "
                f"{len(entries)} != {expected[split_name]}"
            )

        for field in (
            "sample_id",
            "filename",
            "input_relative",
            "gt_relative",
        ):
            values = [getattr(entry, field) for entry in entries]
            if len(values) != len(set(values)):
                raise ValueError(
                    f"Duplicate {field} within {split_name}."
                )

    split_pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )

    for first, second in split_pairs:
        for field in (
            "sample_id",
            "filename",
            "input_relative",
            "gt_relative",
        ):
            left = {
                getattr(entry, field)
                for entry in split_entries[first]
            }
            right = {
                getattr(entry, field)
                for entry in split_entries[second]
            }

            overlap = left & right
            if overlap:
                raise ValueError(
                    f"Cross-split leakage between {first} and {second} "
                    f"by {field}: {sorted(overlap)[:10]}"
                )

    all_entries = (
        train_entries + validation_entries + test_entries
    )

    if len(all_entries) != 890:
        raise ValueError(
            f"Final protocol contains {len(all_entries)} entries, expected 890."
        )

    all_filenames = {entry.filename for entry in all_entries}
    if len(all_filenames) != 890:
        raise ValueError(
            "Final protocol does not contain 890 unique filenames."
        )

    actual_test = {entry.filename for entry in test_entries}
    expected_test = set(PUBLIC_T90_FILENAMES)

    if actual_test != expected_test:
        missing = sorted(expected_test - actual_test)
        extra = sorted(actual_test - expected_test)
        raise ValueError(
            "Final test set does not exactly match public T90.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )


def manifest_text(entries: list[PairEntry]) -> str:
    """
    Produce exactly the schema accepted by the current project:
        sample_id<TAB>input_relative<TAB>gt_relative

    No header is written, matching the existing LSUI manifest convention.
    """
    lines = [
        f"{entry.sample_id}\t"
        f"{entry.input_relative}\t"
        f"{entry.gt_relative}"
        for entry in entries
    ]
    return "\n".join(lines) + "\n"


def safe_write_text(
    path: Path,
    text: str,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}\n"
            "Re-run with --overwrite only if you intentionally want "
            "to regenerate the protocol."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_protocol_files(
    output_dir: Path,
    root: Path,
    seed: int,
    validation_count: int,
    train_entries: list[PairEntry],
    validation_entries: list[PairEntry],
    test_entries: list[PairEntry],
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "train.tsv": manifest_text(train_entries),
        "validation.tsv": manifest_text(validation_entries),
        "test.tsv": manifest_text(test_entries),
    }

    for filename, text in manifests.items():
        safe_write_text(
            output_dir / filename,
            text,
            overwrite=overwrite,
        )

    manifest_hashes = {
        filename: sha256_bytes(text.encode("utf-8"))
        for filename, text in manifests.items()
    }

    metadata = {
        "protocol_name": "UIEB_public_T90_720_80_90",
        "dataset": "UIEB",
        "dataset_root_at_generation": str(root),
        "total_pairs": 890,
        "public_training_pool": 800,
        "train_count": len(train_entries),
        "validation_count": len(validation_entries),
        "test_count": len(test_entries),
        "validation_split_seed": seed,
        "test_protocol": "Public UIEB T90",
        "t90_source": T90_SOURCE_URL,
        "t90_filenames": list(PUBLIC_T90_FILENAMES),
        "manifest_sha256": manifest_hashes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Images were not moved or copied. TSV paths point to the "
            "existing physical Train/Val folders. Experimental membership "
            "is determined solely by these manifests."
        ),
    }

    metadata_text = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"

    safe_write_text(
        output_dir / "protocol.json",
        metadata_text,
        overwrite=overwrite,
    )

    t90_text = "\n".join(PUBLIC_T90_FILENAMES) + "\n"
    safe_write_text(
        output_dir / "public_t90.txt",
        t90_text,
        overwrite=overwrite,
    )

    summary_lines = [
        "UIEB PROTOCOL SUMMARY",
        "=" * 80,
        f"Dataset root              : {root}",
        f"Output directory          : {output_dir}",
        "",
        "Protocol",
        "-" * 80,
        "Total paired images       : 890",
        "Public T90 test images    : 90",
        "Public training pool      : 800",
        f"Train                     : {len(train_entries)}",
        f"Validation                : {len(validation_entries)}",
        f"Test                      : {len(test_entries)}",
        f"Validation split seed     : {seed}",
        "",
        "Manifest SHA256",
        "-" * 80,
        f"train.tsv                 : {manifest_hashes['train.tsv']}",
        f"validation.tsv            : {manifest_hashes['validation.tsv']}",
        f"test.tsv                  : {manifest_hashes['test.tsv']}",
        "",
        "T90 source",
        "-" * 80,
        T90_SOURCE_URL,
        "",
        "[OK] train / validation / test are mutually disjoint.",
        "[OK] test.tsv exactly matches the embedded public T90 list.",
        "[OK] all 890 local paired samples are used exactly once.",
        "[OK] no image was moved, copied, renamed, or modified.",
        "",
    ]

    safe_write_text(
        output_dir / "protocol_summary.txt",
        "\n".join(summary_lines),
        overwrite=overwrite,
    )


def print_preview(
    name: str,
    entries: list[PairEntry],
    n: int = 5,
) -> None:
    print(f"\n{name} preview ({len(entries)} samples):")
    for entry in entries[:n]:
        print(
            f"  {entry.sample_id}\t"
            f"{entry.input_relative}\t"
            f"{entry.gt_relative}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild UIEB into public T90 + deterministic train/validation "
            "manifests without modifying image files."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "UIEB19 dataset root containing "
            "Train/input, Train/GT, Val/input, Val/GT."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("split/uieb"),
        help=(
            "Directory for generated manifests. "
            "Default: split/uieb"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3520,
        help="Seed used only to split the public 800-image pool. Default: 3520",
    )

    parser.add_argument(
        "--validation-count",
        type=int,
        default=80,
        help=(
            "Number of validation images taken from the public 800-image "
            "training pool. Default: 80"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing generated protocol.",
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"UIEB root does not exist: {root}"
        )

    print("=" * 80)
    print("UIEB protocol reconstruction")
    print("=" * 80)
    print(f"Dataset root     : {root}")
    print(f"Output directory : {output_dir}")
    print(f"Seed             : {args.seed}")
    print(f"Validation count : {args.validation_count}")

    print("\n[1/5] Collecting existing paired files...")
    pairs = collect_all_pairs(root)
    print(f"[OK] Found {len(pairs)} unique paired images.")

    print("\n[2/5] Validating embedded public T90...")
    test_names = validate_public_t90(pairs)
    print(f"[OK] All {len(test_names)} public T90 files exist locally.")

    print("\n[3/5] Building deterministic 720/80/90 protocol...")
    train_entries, validation_entries, test_entries = build_protocol(
        pairs=pairs,
        seed=args.seed,
        validation_count=args.validation_count,
    )

    print("\n[4/5] Checking counts and cross-split leakage...")
    validate_final_protocol(
        train_entries=train_entries,
        validation_entries=validation_entries,
        test_entries=test_entries,
        expected_validation_count=args.validation_count,
    )
    print("[OK] Final protocol passed all integrity checks.")

    print("\n[5/5] Writing manifests and protocol metadata...")
    write_protocol_files(
        output_dir=output_dir,
        root=root,
        seed=args.seed,
        validation_count=args.validation_count,
        train_entries=train_entries,
        validation_entries=validation_entries,
        test_entries=test_entries,
        overwrite=args.overwrite,
    )

    print_preview("TRAIN", train_entries)
    print_preview("VALIDATION", validation_entries)
    print_preview("TEST/T90", test_entries)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"train.tsv       : {output_dir / 'train.tsv'}")
    print(f"validation.tsv  : {output_dir / 'validation.tsv'}")
    print(f"test.tsv        : {output_dir / 'test.tsv'}")
    print(f"protocol.json   : {output_dir / 'protocol.json'}")
    print(f"summary         : {output_dir / 'protocol_summary.txt'}")
    print("")
    print("Expected final counts:")
    print(f"  train      = {len(train_entries)}")
    print(f"  validation = {len(validation_entries)}")
    print(f"  test       = {len(test_entries)}")
    print(f"  total      = {len(train_entries) + len(validation_entries) + len(test_entries)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
