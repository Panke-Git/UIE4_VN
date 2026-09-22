from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.shared.paper_evaluation import (
    LPIPSMetric,
    SUPPORTED_PAPER_VERSIONS,
    decode_rgb_png8,
    official_enhanced_names,
    resolve_dataset_name,
    sha256_file,
    validate_exact_prediction_directory,
)
from src.shared.transactional_output import transactional_output_directory
from src.v16.dataset import ManifestEntry
from tools import evaluate_paper_metrics
from tools.compare_runs import main as compare_runs_main


def test_supported_paper_versions_are_the_requested_seven() -> None:
    assert SUPPORTED_PAPER_VERSIONS == {
        "v4",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
    }


def test_legacy_and_modern_dataset_resolution() -> None:
    assert resolve_dataset_name({"dataset": "UIEB"}) == "UIEB"
    assert resolve_dataset_name(
        {"root": "/data/LSUI19_dup_train", "test_manifest": "split/lsui19/test.tsv"}
    ) == "LSUI19"
    with pytest.raises(ValueError, match="exactly one"):
        resolve_dataset_name({"root": "/unknown"})


def test_official_enhanced_name_collision_is_ordered() -> None:
    entries = [
        ManifestEntry("first", "a/shared.png", "gt/first.png"),
        ManifestEntry("second", "b/shared.png", "gt/second.png"),
    ]
    assert official_enhanced_names(entries) == {
        "first": "shared_enhanced.png",
        "second": "shared_second_enhanced.png",
    }


def test_rgb_png8_and_exact_directory_validation(tmp_path: Path) -> None:
    directory = tmp_path / "predictions"
    directory.mkdir()
    path = directory / "one.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    array = decode_rgb_png8(path, expected_size=(8, 8))
    assert array.shape == (8, 8, 3)
    assert array.dtype == np.float32
    assert validate_exact_prediction_directory(directory, ["one.png"]) == {
        "one.png": path
    }
    (directory / "unexpected.txt").write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        validate_exact_prediction_directory(directory, ["one.png"])


def test_lpips_call_uses_official_normalize_once_contract() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.normalize: bool | None = None

        def forward(
            self, prediction: torch.Tensor, target: torch.Tensor, *, normalize: bool
        ) -> torch.Tensor:
            self.normalize = normalize
            return (prediction - target).square().mean(dim=(1, 2, 3), keepdim=True)

    metric = object.__new__(LPIPSMetric)
    metric.device = torch.device("cpu")
    metric.model = FakeModel()
    first = torch.zeros(2, 3, 8, 8)
    second = torch.ones(2, 3, 8, 8)
    values = metric(first, second)
    assert metric.model.normalize is True
    torch.testing.assert_close(values, torch.ones(2))


def test_transactional_output_removes_failures_and_preserves_old_result(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    with pytest.raises(RuntimeError, match="injected"):
        with transactional_output_directory(output, overwrite=False) as staging:
            (staging / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("injected failure")
    assert not output.exists()
    assert not list(tmp_path.glob(".result.staging-*"))

    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="injected"):
        with transactional_output_directory(output, overwrite=True) as staging:
            (staging / "new.txt").write_text("new", encoding="utf-8")
            raise RuntimeError("injected failure")
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (output / "new.txt").exists()

    with transactional_output_directory(output, overwrite=True) as staging:
        (staging / "new.txt").write_text("new", encoding="utf-8")
    assert not (output / "old.txt").exists()
    assert (output / "new.txt").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".result.backup-*"))


def _write_synthetic_paired_run(tmp_path: Path) -> tuple[Path, str]:
    run_dir = tmp_path / "run"
    data_root = tmp_path / "data"
    (run_dir / "split_snapshot").mkdir(parents=True)
    prediction_dir = run_dir / "result" / "test_all_enhanced"
    prediction_dir.mkdir(parents=True)
    (data_root / "input").mkdir(parents=True)
    (data_root / "gt").mkdir(parents=True)
    rows = []
    for index in range(2):
        sample_id = f"sample_{index}"
        image = np.full((13, 17, 3), 40 + 30 * index, dtype=np.uint8)
        target = np.clip(image.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        Image.fromarray(image, mode="RGB").save(data_root / "input" / f"{sample_id}.png")
        Image.fromarray(target, mode="RGB").save(data_root / "gt" / f"{sample_id}.png")
        rows.append((sample_id, f"input/{sample_id}.png", f"gt/{sample_id}.png"))
        enhanced = Image.fromarray(target, mode="RGB").resize(
            (256, 256), Image.Resampling.BILINEAR
        )
        enhanced.save(prediction_dir / f"{sample_id}_enhanced.png")
    manifest = run_dir / "split_snapshot" / "test.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)
    config = {
        "experiment": {"version": "v4"},
        "data": {
            "dataset": "LSUI19",
            "root": str(data_root),
            "patch_size": 8,
            "augmentation": {"hflip": False, "vflip": False, "rot90": False},
            "pad_if_smaller": True,
            "pad_mode": "reflect",
        },
        "evaluation": {"resize": True, "size": 256},
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return run_dir, sha256_file(manifest)


def test_paired_evaluator_commits_only_complete_png8_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest_hash = _write_synthetic_paired_run(tmp_path)
    monkeypatch.setitem(
        evaluate_paper_metrics.CANONICAL_TEST_MANIFESTS,
        "LSUI19",
        (2, manifest_hash),
    )

    class FakeLPIPS:
        def __init__(self, device: torch.device, *, allow_download: bool) -> None:
            del device, allow_download
            self.weights = {"synthetic": {"sha256": "test"}}

        def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (prediction - target).abs().mean(dim=(1, 2, 3)).cpu()

    monkeypatch.setattr(evaluate_paper_metrics, "LPIPSMetric", FakeLPIPS)
    args = evaluate_paper_metrics.parse_args(
        ["--run-dir", str(run_dir), "--batch-size", "2"]
    )
    output, summary = evaluate_paper_metrics.evaluate(args)
    assert output == run_dir / "result" / "paper_metrics_png8"
    assert summary["sample_count"] == 2
    assert summary["final_paper_table_eligible"] is True
    assert (output / "summary.json").is_file()
    with (output / "per_image_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        result_rows = list(csv.DictReader(handle))
    assert len(result_rows) == 2
    assert set(result_rows[0]) >= {"psnr", "ssim", "delta_e00", "lpips"}
    parsed = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert parsed["comparison_group"].startswith("uie4-paper-png8-1.0.0")

    with pytest.raises(FileExistsError, match="--overwrite"):
        evaluate_paper_metrics.evaluate(args)

    original_summary = (output / "summary.json").read_bytes()

    class FailingLPIPS(FakeLPIPS):
        def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            del prediction, target
            raise RuntimeError("injected LPIPS failure")

    monkeypatch.setattr(evaluate_paper_metrics, "LPIPSMetric", FailingLPIPS)
    overwrite_args = evaluate_paper_metrics.parse_args(
        ["--run-dir", str(run_dir), "--batch-size", "2", "--overwrite"]
    )
    with pytest.raises(RuntimeError, match="injected LPIPS failure"):
        evaluate_paper_metrics.evaluate(overwrite_args)
    assert (output / "summary.json").read_bytes() == original_summary
    assert not list((run_dir / "result").glob(".paper_metrics_png8.staging-*"))


def test_compare_runs_reads_paper_and_u45_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    paper_dir = run_dir / "result" / "paper_metrics_png8"
    u45_dir = run_dir / "result" / "u45" / "from_lsui"
    paper_dir.mkdir(parents=True)
    u45_dir.mkdir(parents=True)
    (paper_dir / "summary.json").write_text(
        json.dumps(
            {
                "mean_psnr": 25.0,
                "mean_ssim": 0.9,
                "mean_delta_e00": 4.0,
                "mean_lpips": 0.1,
            }
        ),
        encoding="utf-8",
    )
    (u45_dir / "summary.json").write_text(
        json.dumps({"mean_uiqm": 3.2, "mean_uciqe": 28.1}), encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["compare_runs.py", str(run_dir)])
    compare_runs_main()
    lines = capsys.readouterr().out.strip().splitlines()
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    values = dict(zip(header, row, strict=True))
    assert values["paper_png8_lpips"] == "0.1"
    assert values["paper_png8_e00"] == "4.0"
    assert values["u45_uiqm"] == "3.2"
    assert values["u45_uciqe"] == "28.1"
