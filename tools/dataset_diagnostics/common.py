"""Shared manifest, serialization, and descriptive-statistics helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ORDER = ("train", "validation", "test")
SPLIT_PAIRS = (("train", "validation"), ("train", "test"), ("validation", "test"))
EXPECTED_SPLIT_COUNTS = {"train": 3466, "validation": 385, "test": 428}


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    input_relative: str
    gt_relative: str


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read LSUI's optional-header, three-column, tab-separated manifest."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    header = tuple(cell.strip().lower() for cell in rows[0])
    has_header = header in {
        ("sample_id", "input_path", "gt_path"),
        ("id", "input", "gt"),
    }
    entries: list[ManifestEntry] = []
    for index, row in enumerate(rows[1:] if has_header else rows, start=2 if has_header else 1):
        if len(row) != 3 or any(not cell.strip() for cell in row):
            raise ValueError(f"{path}:{index}: expected three non-empty tab-separated columns")
        sample_id, input_relative, gt_relative = (cell.strip() for cell in row)
        if Path(input_relative).is_absolute() or Path(gt_relative).is_absolute():
            raise ValueError(f"{path}:{index}: image paths must be relative to data root")
        entries.append(ManifestEntry(sample_id, input_relative, gt_relative))
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_values(values: Iterable[str]) -> dict[str, Any]:
    counts = Counter(values)
    duplicates = {value: count for value, count in counts.items() if count > 1}
    return {
        "duplicate_occurrences": sum(count - 1 for count in duplicates.values()),
        "duplicate_value_count": len(duplicates),
        "values": duplicates,
    }


def analyze_split_integrity(
    entries: Mapping[str, Sequence[ManifestEntry]],
    expected_counts: Mapping[str, int] = EXPECTED_SPLIT_COUNTS,
) -> dict[str, Any]:
    counts = {split: len(entries[split]) for split in SPLIT_ORDER}
    within: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        rows = entries[split]
        within[split] = {
            "sample_id": _duplicate_values(row.sample_id for row in rows),
            "input_path": _duplicate_values(row.input_relative for row in rows),
            "gt_path": _duplicate_values(row.gt_relative for row in rows),
        }

    overlaps: dict[str, Any] = {}
    for split_a, split_b in SPLIT_PAIRS:
        pair_key = f"{split_a}_vs_{split_b}"
        overlaps[pair_key] = {}
        for field in ("sample_id", "input_relative", "gt_relative"):
            values_a = {getattr(row, field) for row in entries[split_a]}
            values_b = {getattr(row, field) for row in entries[split_b]}
            values = sorted(values_a & values_b)
            overlaps[pair_key][field.replace("_relative", "_path")] = {
                "count": len(values),
                "values": values,
            }

    expected = dict(expected_counts)
    return {
        "counts": counts,
        "expected_counts": expected,
        "counts_match_expected": counts == expected,
        "train_plus_validation": counts["train"] + counts["validation"],
        "total": sum(counts.values()),
        "within_split_duplicates": within,
        "cross_split_path_overlaps": overlaps,
    }


def enforce_split_counts(integrity: Mapping[str, Any]) -> None:
    if not integrity["counts_match_expected"]:
        raise ValueError(
            f"Split counts {integrity['counts']} do not match expected "
            f"{integrity['expected_counts']}"
        )
    expected = integrity["expected_counts"]
    if expected == EXPECTED_SPLIT_COUNTS:
        if integrity["train_plus_validation"] != 3851 or integrity["total"] != 4279:
            raise ValueError("Fixed LSUI split must have train+validation=3851 and total=4279")


def create_run_directory(output_root: Path, timestamp: datetime | None = None) -> Path:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = output_root / f"lsui19_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base.name}_{suffix:02d}"
        suffix += 1
    for relative in ("difficulty", "duplicates", "metadata"):
        (candidate / relative).mkdir(parents=True, exist_ok=False)
    return candidate


def copy_manifests(manifests: Mapping[str, Path], metadata_dir: Path) -> None:
    for split in SPLIT_ORDER:
        shutil.copy2(manifests[split], metadata_dir / f"{split}.tsv")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def descriptive_statistics(values: Iterable[float | int | None]) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and not math.isnan(float(value))],
        dtype=np.float64,
    )
    if array.size == 0:
        return {
            key: None
            for key in (
                "count", "mean", "median", "std", "min", "max",
                "p05", "p10", "p25", "p50", "p75", "p90", "p95",
            )
        } | {"count": 0, "finite_count": 0, "infinite_count": 0}
    finite = array[np.isfinite(array)]
    if finite.size:
        if finite.size == array.size:
            percentiles = np.percentile(array, [5, 10, 25, 50, 75, 90, 95])
            mean = float(np.mean(array))
            median = float(np.median(array))
            std = float(np.std(array))
        else:
            # Nearest-rank percentiles include infinite observations without
            # undefined interpolation such as ``inf - inf``.
            percentiles = np.percentile(
                array, [5, 10, 25, 50, 75, 90, 95], method="nearest"
            )
            has_positive = bool(np.isposinf(array).any())
            has_negative = bool(np.isneginf(array).any())
            mean = float("nan") if has_positive and has_negative else (
                float("inf") if has_positive else float("-inf")
            )
            median = float(percentiles[3])
            std = float("inf")
        result: dict[str, Any] = {
            "count": int(array.size),
            "finite_count": int(finite.size),
            "infinite_count": int(np.isinf(array).sum()),
            "mean": mean,
            "median": median,
            "std": std,
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
        result.update(
            dict(zip(("p05", "p10", "p25", "p50", "p75", "p90", "p95"), map(float, percentiles)))
        )
        return result
    infinity = float(array[0])
    return {
        "count": int(array.size),
        "finite_count": 0,
        "infinite_count": int(array.size),
        **{key: infinity for key in ("mean", "median", "min", "max")},
        "std": 0.0 if np.all(array == infinity) else float("nan"),
        **{key: infinity for key in ("p05", "p10", "p25", "p50", "p75", "p90", "p95")},
    }


def source_folder(relative_path: str) -> str:
    parent = Path(relative_path).parent.as_posix()
    return parent if parent != "." else "<root>"
