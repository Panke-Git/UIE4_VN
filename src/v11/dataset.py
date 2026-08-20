"""LSUI paired dataset driven exclusively by the project-local fixed TSV manifests."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestEntry:
    sample_id: str
    input_relative: str
    gt_relative: str


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read actual LSUI schema: no header, sample_id, input path, GT path."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    header = tuple(cell.strip().lower() for cell in rows[0])
    if header in {
        ("sample_id", "input_path", "gt_path"),
        ("id", "input", "gt"),
    }:
        rows = rows[1:]
    entries: list[ManifestEntry] = []
    for line_number, row in enumerate(rows, start=1):
        if len(row) != 3 or any(not cell.strip() for cell in row):
            raise ValueError(f"{path}:{line_number}: expected three non-empty tab-separated columns")
        sample_id, input_relative, gt_relative = (cell.strip() for cell in row)
        if Path(input_relative).is_absolute() or Path(gt_relative).is_absolute():
            raise ValueError(f"{path}:{line_number}: image paths must be relative to data.root")
        entries.append(ManifestEntry(sample_id, input_relative, gt_relative))
    return entries


def duplicate_keys(entries: Sequence[ManifestEntry]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in entries:
        key = f"{entry.sample_id}\t{entry.input_relative}\t{entry.gt_relative}"
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def entry_identity(entry: ManifestEntry) -> tuple[str, str, str]:
    return entry.sample_id, entry.input_relative, entry.gt_relative


def validate_split_protocol(manifests: dict[str, Path]) -> dict[str, list[ManifestEntry]]:
    """Enforce fixed counts, uniqueness by every identity field, and zero leakage."""
    parsed = {name: read_manifest(path) for name, path in manifests.items()}
    expected = {"train": 3466, "validation": 385, "test": 428}
    for split, count in expected.items():
        if len(parsed[split]) != count:
            raise ValueError(f"{split} count is {len(parsed[split])}, expected {count}")
    if len(parsed["train"]) + len(parsed["validation"]) != 3851:
        raise ValueError("train + validation must equal 3851")

    fields = ("sample_id", "input_relative", "gt_relative")
    for split, entries in parsed.items():
        for field in fields:
            values = [getattr(entry, field) for entry in entries]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {field} within {split}")
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for first, second in pairs:
        for field in fields:
            overlap = {getattr(entry, field) for entry in parsed[first]} & {
                getattr(entry, field) for entry in parsed[second]
            }
            if overlap:
                examples = sorted(overlap)[:5]
                raise ValueError(f"Cross-split leakage {first}/{second} by {field}: {examples}")
    return parsed


def verify_image_entries(entries: Sequence[ManifestEntry], data_root: Path, split: str) -> None:
    """Fail loudly on missing, unreadable, non-RGB, mismatched, or mispaired files."""
    problems: list[str] = []
    for entry in entries:
        input_path = data_root / entry.input_relative
        gt_path = data_root / entry.gt_relative
        context = f"split={split} sample_id={entry.sample_id} input={input_path} gt={gt_path}"
        if not input_path.is_file() or not gt_path.is_file():
            missing = []
            if not input_path.is_file():
                missing.append("input")
            if not gt_path.is_file():
                missing.append("GT")
            problems.append(f"missing {','.join(missing)}: {context}")
            continue
        try:
            with Image.open(input_path) as input_image, Image.open(gt_path) as gt_image:
                input_image.load()
                gt_image.load()
                if input_image.mode != "RGB" or gt_image.mode != "RGB":
                    problems.append(
                        f"non-RGB mode ({input_image.mode}, {gt_image.mode}): {context}"
                    )
                if input_image.size != gt_image.size:
                    problems.append(
                        f"pair size mismatch ({input_image.size}, {gt_image.size}): {context}"
                    )
        except Exception as error:  # PIL emits several format-specific exception types.
            problems.append(f"unreadable ({type(error).__name__}: {error}): {context}")
        if input_path.stem != gt_path.stem or input_path.stem != entry.sample_id:
            problems.append(f"pair identity mismatch: {context}")
        if len(problems) >= 20:
            break
    if problems:
        raise RuntimeError("Image validation failed:\n" + "\n".join(problems))


def _to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def _pad_pair(input_array: np.ndarray, gt_array: np.ndarray, patch_size: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    height, width = input_array.shape[:2]
    pad_h, pad_w = max(0, patch_size - height), max(0, patch_size - width)
    if not (pad_h or pad_w):
        return input_array, gt_array
    np_mode = mode
    if mode == "reflect" and (height < 2 or width < 2):
        np_mode = "edge"
    padding = ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(input_array, padding, mode=np_mode), np.pad(gt_array, padding, mode=np_mode)


class LSUIDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        split: str,
        patch_size: int,
        augmentation: dict,
        pad_if_smaller: bool,
        pad_mode: str,
        evaluation: dict,
        verify_files: bool = True,
    ) -> None:
        self.entries = read_manifest(manifest_path)
        self.data_root = data_root
        self.split = split
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.pad_if_smaller = pad_if_smaller
        self.pad_mode = pad_mode
        self.evaluation = evaluation
        if duplicate_keys(self.entries):
            raise ValueError(f"Duplicate manifest entries in {manifest_path}")
        if verify_files:
            verify_image_entries(self.entries, data_root, split)

    def __len__(self) -> int:
        return len(self.entries)

    def _load_pair(self, entry: ManifestEntry) -> tuple[Image.Image, Image.Image]:
        input_path = self.data_root / entry.input_relative
        gt_path = self.data_root / entry.gt_relative
        with Image.open(input_path) as image:
            if image.mode != "RGB":
                raise RuntimeError(f"Expected RGB input: split={self.split} sample_id={entry.sample_id} path={input_path}")
            input_image = image.copy()
        with Image.open(gt_path) as image:
            if image.mode != "RGB":
                raise RuntimeError(f"Expected RGB GT: split={self.split} sample_id={entry.sample_id} path={gt_path}")
            gt_image = image.copy()
        if input_image.size != gt_image.size:
            raise RuntimeError(f"Pair dimensions differ: split={self.split} sample_id={entry.sample_id}")
        return input_image, gt_image

    def _training_transform(self, input_image: Image.Image, gt_image: Image.Image) -> tuple[Tensor, Tensor]:
        input_array = np.array(input_image, dtype=np.uint8, copy=True)
        gt_array = np.array(gt_image, dtype=np.uint8, copy=True)
        height, width = input_array.shape[:2]
        if (height < self.patch_size or width < self.patch_size) and not self.pad_if_smaller:
            raise ValueError(f"Image {width}x{height} is smaller than patch_size={self.patch_size}")
        input_array, gt_array = _pad_pair(
            input_array, gt_array, self.patch_size, self.pad_mode
        )
        height, width = input_array.shape[:2]
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        slices = (slice(top, top + self.patch_size), slice(left, left + self.patch_size))
        input_array, gt_array = input_array[slices], gt_array[slices]

        if self.augmentation.get("hflip", False) and random.random() < 0.5:
            input_array, gt_array = np.flip(input_array, 1), np.flip(gt_array, 1)
        if self.augmentation.get("vflip", False) and random.random() < 0.5:
            input_array, gt_array = np.flip(input_array, 0), np.flip(gt_array, 0)
        if self.augmentation.get("rot90", False):
            rotations = random.randrange(4)
            input_array, gt_array = np.rot90(input_array, rotations), np.rot90(gt_array, rotations)
        input_array = np.ascontiguousarray(input_array).copy()
        gt_array = np.ascontiguousarray(gt_array).copy()
        input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).float() / 255.0
        gt_tensor = torch.from_numpy(gt_array).permute(2, 0, 1).float() / 255.0
        return input_tensor, gt_tensor

    def _evaluation_transform(self, input_image: Image.Image, gt_image: Image.Image) -> tuple[Tensor, Tensor]:
        if self.evaluation.get("resize", True):
            size = int(self.evaluation["size"])
            target_size = (size, size)
            input_image = input_image.resize(target_size, Image.Resampling.BILINEAR)
            gt_image = gt_image.resize(target_size, Image.Resampling.BILINEAR)
        return _to_tensor(input_image), _to_tensor(gt_image)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        entry = self.entries[index]
        input_image, gt_image = self._load_pair(entry)
        if self.split == "train":
            input_tensor, gt_tensor = self._training_transform(input_image, gt_image)
        else:
            input_tensor, gt_tensor = self._evaluation_transform(input_image, gt_image)
        return {
            "id": entry.sample_id,
            "filename": Path(entry.input_relative).name,
            "input": input_tensor,
            "target": gt_tensor,
        }
