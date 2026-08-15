from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
import torch
import yaml

from src.v1.metrics import per_image_psnr as uie4_psnr
from src.v1.metrics import per_image_ssim as uie4_ssim
from tools.dataset_diagnostics.common import ManifestEntry, read_manifest, sha256_file
from tools.dataset_diagnostics.difficulty import analyze_pair
from tools.dataset_diagnostics.duplicates import (
    build_duplicate_results,
    decoded_pixel_sha256,
    dhash64,
    fingerprint_image,
    hamming_distance,
)
from tools.dataset_diagnostics.metrics import (
    paired_metrics,
    per_image_psnr as diagnostic_psnr,
    per_image_ssim as diagnostic_ssim,
    resize_for_evaluation,
)
from tools.diagnose_lsui import run_diagnostics


def _gradient(width: int = 18, height: int = 14, offset: int = 0) -> Image.Image:
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    array = np.stack(
        (
            np.broadcast_to((x * 11 + offset) % 256, (height, width)),
            np.broadcast_to((y * 17 + offset) % 256, (height, width)),
            (x * 7 + y * 13 + offset) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_diagnostic_psnr_ssim_match_current_uie4_implementation() -> None:
    generator = torch.Generator().manual_seed(3520)
    prediction = torch.rand(2, 3, 19, 23, generator=generator)
    target = torch.rand(2, 3, 19, 23, generator=generator)
    assert torch.equal(diagnostic_psnr(prediction, target), uie4_psnr(prediction, target))
    assert torch.equal(diagnostic_ssim(prediction, target), uie4_ssim(prediction, target))


def test_psnr_native_and_current_256_metrics(tmp_path: Path) -> None:
    image = _gradient()
    identical = paired_metrics(image, image)
    assert identical["psnr"] == float("inf")
    assert identical["ssim"] == 1.0
    assert identical["mae"] == 0.0

    different = _gradient(offset=25)
    assert torch.isfinite(torch.tensor(paired_metrics(image, different)["psnr"]))
    assert paired_metrics(image, different)["mae"] > 0
    resized = paired_metrics(resize_for_evaluation(image, 256), resize_for_evaluation(different, 256))
    assert torch.isfinite(torch.tensor(resized["psnr"]))

    data_root = tmp_path / "data"
    (data_root / "input").mkdir(parents=True)
    (data_root / "gt").mkdir()
    image.save(data_root / "input/a.png")
    different.save(data_root / "gt/a.png")
    row = analyze_pair(
        split="train",
        entry=ManifestEntry("a", "input/a.png", "gt/a.png"),
        data_root=data_root,
        evaluation_size=256,
        metric_config={"data_range": 1.0, "ssim_window_size": 11, "ssim_sigma": 1.5},
    ).metrics
    assert row["native_shape_match"] is True
    assert row["psnr_native"] is not None
    assert row["psnr_256"] is not None

    image.save(data_root / "input/b.png")
    _gradient(width=20, height=16, offset=25).save(data_root / "gt/b.png")
    mismatch = analyze_pair(
        split="test",
        entry=ManifestEntry("b", "input/b.png", "gt/b.png"),
        data_root=data_root,
        evaluation_size=256,
        metric_config={"data_range": 1.0, "ssim_window_size": 11, "ssim_sigma": 1.5},
    ).metrics
    assert mismatch["native_shape_match"] is False
    assert mismatch["psnr_native"] is None
    assert mismatch["psnr_256"] is not None


def test_raw_pixel_hash_and_dhash_duplicate_detection(tmp_path: Path) -> None:
    image = _gradient()
    png_a = tmp_path / "a.png"
    png_copy = tmp_path / "a-copy.png"
    png_metadata = tmp_path / "a-metadata.png"
    image.save(png_a)
    shutil.copy2(png_a, png_copy)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("diagnostic", "different file bytes, identical decoded pixels")
    image.save(png_metadata, pnginfo=metadata)

    assert sha256_file(png_a) == sha256_file(png_copy)
    assert sha256_file(png_a) != sha256_file(png_metadata)
    with Image.open(png_a) as first, Image.open(png_copy) as copied, Image.open(png_metadata) as second:
        assert decoded_pixel_sha256(first) == decoded_pixel_sha256(second)
        assert hamming_distance(dhash64(first), dhash64(second)) == 0

        fingerprints = [
            fingerprint_image(
                split="train", sample_id="a", image_type="input", relative_path="a.png",
                absolute_path=png_a, image=first,
            ),
            fingerprint_image(
                split="validation", sample_id="c", image_type="input", relative_path="a-copy.png",
                absolute_path=png_copy, image=copied,
            ),
            fingerprint_image(
                split="test", sample_id="b", image_type="input", relative_path="a-metadata.png",
                absolute_path=png_metadata, image=second,
            ),
        ]
    results = build_duplicate_results(fingerprints, dhash_threshold=0)
    assert results["summary"]["file_duplicate_pairs"] == 1
    assert results["summary"]["pixel_duplicate_pairs"] == 3
    assert results["summary"]["cross_split_exact_pixel_duplicate_pairs"] == 3
    assert results["summary"]["near_duplicate_candidate_pairs"] == 3


def _write_manifest(path: Path, sample_id: str, input_relative: str, gt_relative: str) -> None:
    path.write_text(f"{sample_id}\t{input_relative}\t{gt_relative}\n", encoding="utf-8")


def test_synthetic_manifests_and_summary_generation(tmp_path: Path) -> None:
    data_root = tmp_path / "lsui"
    manifest_root = tmp_path / "split"
    manifest_root.mkdir()
    definitions = {
        "train": ("t", "Train/input/t.png", "Train/GT/t.png", 0, 20),
        "validation": ("v", "Val/input/v.png", "Val/GT/v.png", 0, 35),
        "test": ("e", "Test/input/e.png", "Test/GT/e.png", 60, 80),
    }
    for split, (sample_id, input_relative, gt_relative, input_offset, gt_offset) in definitions.items():
        input_path, gt_path = data_root / input_relative, data_root / gt_relative
        input_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        _gradient(offset=input_offset).save(input_path)
        _gradient(offset=gt_offset).save(gt_path)
        manifest = manifest_root / f"{split}.tsv"
        _write_manifest(manifest, sample_id, input_relative, gt_relative)
        parsed = read_manifest(manifest)
        assert parsed == [ManifestEntry(sample_id, input_relative, gt_relative)]

    config = {
        "data": {
            "root": str(data_root),
            "train_manifest": str(manifest_root / "train.tsv"),
            "validation_manifest": str(manifest_root / "validation.tsv"),
            "test_manifest": str(manifest_root / "test.tsv"),
        },
        "evaluation": {"resize": True, "size": 256},
        "metrics": {
            "data_range": 1.0,
            "crop_border": 0,
            "ssim_window_size": 11,
            "ssim_sigma": 1.5,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = run_diagnostics(
        config_path=config_path,
        output_root=tmp_path / "diagnostics",
        expected_counts={"train": 1, "validation": 1, "test": 1},
        dhash_threshold=0,
        generate_plots=False,
        show_progress=False,
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["split_counts"]["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert set(summary) >= {
        "split_counts", "difficulty", "resize_effect", "exact_duplicates",
        "near_duplicate_candidates", "resolution", "image_statistics",
    }
    assert summary["difficulty"]["test"]["psnr_256"]["count"] == 1
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "duplicates/cross_split_exact_duplicates.csv").is_file()
    assert (run_dir / "difficulty/difficulty_extremes.csv").is_file()
    assert (run_dir / "metadata/train.tsv").read_bytes() == (manifest_root / "train.tsv").read_bytes()
