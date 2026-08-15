"""Raw-file, decoded-pixel, and cross-split perceptual-hash duplicate audits."""

from __future__ import annotations

import hashlib
import itertools
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from .common import SPLIT_ORDER, SPLIT_PAIRS, sha256_file, write_csv, write_json
from .metrics import psnr_and_mae_128


@dataclass(frozen=True)
class ImageFingerprint:
    split: str
    sample_id: str
    image_type: str
    relative_path: str
    absolute_path: Path
    file_sha256: str
    pixel_sha256: str
    dhash: int


def decoded_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    width, height = rgb.size
    digest = hashlib.sha256()
    digest.update(struct.pack(">II", width, height))
    digest.update(np.asarray(rgb, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def dhash64(image: Image.Image) -> int:
    """Return the standard 64-bit horizontal difference hash."""
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(grayscale, dtype=np.uint8)
    comparisons = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in comparisons.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return (int(hash_a) ^ int(hash_b)).bit_count()


def fingerprint_image(
    *,
    split: str,
    sample_id: str,
    image_type: str,
    relative_path: str,
    absolute_path: Path,
    image: Image.Image,
) -> ImageFingerprint:
    return ImageFingerprint(
        split=split,
        sample_id=sample_id,
        image_type=image_type,
        relative_path=relative_path,
        absolute_path=absolute_path,
        file_sha256=sha256_file(absolute_path),
        pixel_sha256=decoded_pixel_sha256(image),
        dhash=dhash64(image),
    )


def _ordered_pair(a: ImageFingerprint, b: ImageFingerprint) -> tuple[ImageFingerprint, ImageFingerprint]:
    order = {split: index for index, split in enumerate(SPLIT_ORDER)}
    key_a = (order[a.split], a.sample_id, a.relative_path)
    key_b = (order[b.split], b.sample_id, b.relative_path)
    return (a, b) if key_a <= key_b else (b, a)


def exact_duplicate_pairs(
    fingerprints: Sequence[ImageFingerprint],
    hash_attribute: str,
    *,
    cross_split_only: bool = False,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[ImageFingerprint]] = defaultdict(list)
    for item in fingerprints:
        groups[(item.image_type, str(getattr(item, hash_attribute)))].append(item)

    rows: list[dict[str, Any]] = []
    for (image_type, digest), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        for first, second in itertools.combinations(group, 2):
            if cross_split_only and first.split == second.split:
                continue
            first, second = _ordered_pair(first, second)
            rows.append(
                {
                    "split_a": first.split,
                    "sample_a": first.sample_id,
                    "path_a": first.relative_path,
                    "split_b": second.split,
                    "sample_b": second.sample_id,
                    "path_b": second.relative_path,
                    "type": image_type,
                    "hash": digest,
                }
            )
    return rows


def near_duplicate_candidates(
    fingerprints: Sequence[ImageFingerprint],
    threshold: int,
) -> list[dict[str, Any]]:
    if threshold < 0 or threshold > 64:
        raise ValueError("dHash threshold must be between 0 and 64")
    buckets: dict[tuple[str, str], list[ImageFingerprint]] = defaultdict(list)
    for item in fingerprints:
        buckets[(item.image_type, item.split)].append(item)

    candidates: list[dict[str, Any]] = []
    for image_type in ("input", "gt"):
        for split_a, split_b in SPLIT_PAIRS:
            for first in buckets[(image_type, split_a)]:
                for second in buckets[(image_type, split_b)]:
                    distance = hamming_distance(first.dhash, second.dhash)
                    if distance > threshold:
                        continue
                    with Image.open(first.absolute_path) as image_a, Image.open(second.absolute_path) as image_b:
                        image_a.load()
                        image_b.load()
                        candidate_psnr, candidate_mae = psnr_and_mae_128(image_a, image_b)
                    candidates.append(
                        {
                            "split_a": first.split,
                            "sample_a": first.sample_id,
                            "input_path_a": first.relative_path,
                            "split_b": second.split,
                            "sample_b": second.sample_id,
                            "input_path_b": second.relative_path,
                            "image_type": image_type,
                            "dhash_a": f"{first.dhash:016x}",
                            "dhash_b": f"{second.dhash:016x}",
                            "hamming_distance": distance,
                            "candidate_psnr_128": candidate_psnr,
                            "candidate_mae_128": candidate_mae,
                        }
                    )
    candidates.sort(
        key=lambda row: (
            row["hamming_distance"],
            row["image_type"],
            row["split_a"],
            row["sample_a"],
            row["split_b"],
            row["sample_b"],
        )
    )
    return candidates


def _group_count(fingerprints: Sequence[ImageFingerprint], attribute: str) -> int:
    groups: dict[tuple[str, str], int] = defaultdict(int)
    for item in fingerprints:
        groups[(item.image_type, str(getattr(item, attribute)))] += 1
    return sum(count > 1 for count in groups.values())


def build_duplicate_results(
    fingerprints: Sequence[ImageFingerprint],
    dhash_threshold: int,
) -> dict[str, Any]:
    file_rows = exact_duplicate_pairs(fingerprints, "file_sha256")
    pixel_rows = exact_duplicate_pairs(fingerprints, "pixel_sha256")
    cross_rows = exact_duplicate_pairs(fingerprints, "pixel_sha256", cross_split_only=True)
    near_rows = near_duplicate_candidates(fingerprints, dhash_threshold)
    summary = {
        "file_duplicate_groups": _group_count(fingerprints, "file_sha256"),
        "file_duplicate_pairs": len(file_rows),
        "pixel_duplicate_groups": _group_count(fingerprints, "pixel_sha256"),
        "pixel_duplicate_pairs": len(pixel_rows),
        "cross_split_exact_pixel_duplicate_pairs": len(cross_rows),
        "cross_split_exact_pixel_duplicate_pairs_by_type": {
            image_type: sum(row["type"] == image_type for row in cross_rows)
            for image_type in ("input", "gt")
        },
        "near_duplicate_candidate_pairs": len(near_rows),
        "near_duplicate_candidate_pairs_by_type": {
            image_type: sum(row["image_type"] == image_type for row in near_rows)
            for image_type in ("input", "gt")
        },
        "dhash_threshold": dhash_threshold,
    }
    return {
        "exact_file": file_rows,
        "exact_pixel": pixel_rows,
        "cross_exact": cross_rows,
        "near": near_rows,
        "summary": summary,
    }


EXACT_FIELDS = (
    "split_a", "sample_a", "path_a", "split_b", "sample_b", "path_b", "type", "hash"
)
NEAR_FIELDS = (
    "split_a", "sample_a", "input_path_a", "split_b", "sample_b", "input_path_b",
    "image_type", "dhash_a", "dhash_b", "hamming_distance", "candidate_psnr_128",
    "candidate_mae_128",
)


def write_duplicate_outputs(directory: Path, results: dict[str, Any]) -> None:
    write_csv(directory / "exact_file_duplicates.csv", results["exact_file"], EXACT_FIELDS)
    write_csv(directory / "exact_pixel_duplicates.csv", results["exact_pixel"], EXACT_FIELDS)
    write_csv(directory / "cross_split_exact_duplicates.csv", results["cross_exact"], EXACT_FIELDS)
    write_csv(directory / "near_duplicate_candidates.csv", results["near"], NEAR_FIELDS)
    write_json(directory / "duplicate_summary.json", results["summary"])
