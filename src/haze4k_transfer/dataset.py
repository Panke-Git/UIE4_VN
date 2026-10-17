"""Haze4K-specific manifests, paired V16-equivalent geometry and PIL evaluation."""
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .prepare import check_rows, safe_path


def load_rgb(path):
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


def tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255.0


class HazeDataset(Dataset):
    def __init__(self, root, rows, split, patch_size=256, resolution="256"):
        check_rows(rows, split)
        self.root, self.rows, self.split = Path(root), rows, split
        self.patch_size, self.resolution = patch_size, str(resolution)
        if self.resolution not in {"256", "native"}:
            raise ValueError("Evaluation supports only 256 or native")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, y = [load_rgb(safe_path(self.root, row[f])) for f in ("input_relative", "gt_relative")]
        if x.size != y.size:
            raise ValueError(f"Pair size mismatch: {row}")
        if self.split == "train":
            x, y = np.array(x), np.array(y)
            h, w = x.shape[:2]
            padding = ((0, max(0, self.patch_size - h)), (0, max(0, self.patch_size - w)), (0, 0))
            # Exact source V16 reflection padding, edge fallback for singleton axes.
            if any(p[1] for p in padding):
                mode = "edge" if min(h, w) < 2 else "reflect"
                x, y = np.pad(x, padding, mode=mode), np.pad(y, padding, mode=mode)
            h, w = x.shape[:2]
            top, left = random.randint(0, h - self.patch_size), random.randint(0, w - self.patch_size)
            sl = (slice(top, top + self.patch_size), slice(left, left + self.patch_size))
            x, y = x[sl], y[sl]
            if random.random() < 0.5:
                x, y = np.flip(x, 1), np.flip(y, 1)
            if random.random() < 0.5:
                x, y = np.flip(x, 0), np.flip(y, 0)
            k = random.randrange(4)
            x, y = np.rot90(x, k), np.rot90(y, k)
            x, y = tensor(x), tensor(y)
        else:
            if self.resolution == "256":
                x, y = x.resize((256, 256), Image.Resampling.BILINEAR), y.resize((256, 256), Image.Resampling.BILINEAR)
            x, y = tensor(x), tensor(y)
        return {**row, "input": x, "target": y}
