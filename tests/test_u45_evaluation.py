from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.shared.u45_evaluation import (
    OFFICIAL_U45_INDEX,
    U45Entry,
    UCIQE_VARIANT,
    UIQM_VARIANT,
    decode_prediction_png8,
    inspect_official_u45,
    uciqe_components,
    uiqm_components,
)
from tools import test_u45


def test_official_u45_index_is_complete_and_pinned() -> None:
    assert len(OFFICIAL_U45_INDEX) == 45
    assert set(OFFICIAL_U45_INDEX) == {f"{index}.png" for index in range(1, 46)}
    assert OFFICIAL_U45_INDEX["1.png"] == (
        "7ee9ba053a44b97ee9dd6e3fea8832bbffb05fa0",
        79577,
    )
    assert UIQM_VARIANT == "FX_UIQM_v2"
    assert UCIQE_VARIANT == "FX_UCIQE_CIELAB_v1"


def test_uiqm_v2_and_uciqe_regression_values() -> None:
    rng = np.random.default_rng(3520)
    rgb = rng.integers(0, 256, size=(16, 17, 3), dtype=np.uint8)
    uiqm = uiqm_components(rgb)
    uciqe = uciqe_components(rgb)
    assert uiqm["uiqm"] == pytest.approx(1.884752457698262, abs=1e-12)
    assert uiqm["uicm"] == pytest.approx(10.015550853517357, abs=1e-12)
    assert uciqe["uciqe"] == pytest.approx(34.51060374467883, abs=1e-12)
    assert uciqe["tail_count"] == 3

    black = np.zeros((16, 16, 3), dtype=np.uint8)
    assert uiqm_components(black)["uiqm"] == 0.0
    assert uciqe_components(black)["uciqe"] == 0.0


def test_u45_official_inspection_rejects_count_only_impostors(tmp_path: Path) -> None:
    for index in range(1, 46):
        Image.new("RGB", (8, 8), (index, index, index)).save(tmp_path / f"{index}.png")
    with pytest.raises(ValueError, match="source identity mismatch"):
        inspect_official_u45(tmp_path)


def test_prediction_must_be_rgb_png8_at_native_size(tmp_path: Path) -> None:
    path = tmp_path / "prediction.png"
    Image.new("RGB", (11, 9), (10, 20, 30)).save(path)
    assert decode_prediction_png8(path, expected_size=(11, 9)).shape == (9, 11, 3)
    with pytest.raises(ValueError, match="original input"):
        decode_prediction_png8(path, expected_size=(10, 9))


def _synthetic_entries(tmp_path: Path) -> list[U45Entry]:
    root = tmp_path / "u45"
    root.mkdir()
    entries = []
    for index in range(45):
        filename = f"{index + 1}.png"
        array = np.full((10, 12, 3), 20 + index, dtype=np.uint8)
        path = root / filename
        Image.fromarray(array, mode="RGB").save(path)
        raw = path.read_bytes()
        entries.append(
            U45Entry(
                sample_id=str(index + 1),
                filename=filename,
                path=path,
                sha256=hashlib.sha256(raw).hexdigest(),
                git_blob_sha1="synthetic",
                width=12,
                height=10,
                original_mode="RGB",
            )
        )
    return entries


def _synthetic_run(tmp_path: Path) -> tuple[Path, list[U45Entry]]:
    run_dir = tmp_path / "run"
    (run_dir / "best").mkdir(parents=True)
    model_config = {"type": "synthetic"}
    config = {
        "experiment": {"version": "v4"},
        "data": {"dataset": "UIEB", "root": "/unused"},
        "model": model_config,
        "training": {"amp": False},
        "evaluation": {"resize": True, "size": 8},
        "test": {"checkpoint": "best_psnr"},
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    torch.save(
        {
            "version": "v4",
            "epoch": 3,
            "resolved_config": {"model": model_config},
            "model_state_dict": {},
        },
        run_dir / "best" / "best_psnr.pt",
    )
    return run_dir, _synthetic_entries(tmp_path)


def test_u45_run_commits_complete_result_and_failure_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, entries = _synthetic_run(tmp_path)
    monkeypatch.setattr(test_u45, "inspect_official_u45", lambda _: entries)

    class IdentityModel(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    original_import_module = importlib.import_module
    monkeypatch.setattr(
        test_u45.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(build_model=lambda _: IdentityModel())
            if name == "src.v4.models"
            else original_import_module(name)
        ),
    )
    args = test_u45.parse_args(
        ["--run-dir", str(run_dir), "--data-root", str(tmp_path / "u45")]
    )
    output, summary = test_u45.run(args)
    assert output == run_dir / "result" / "u45" / "from_uieb"
    assert summary["sample_count"] == 45
    assert summary["source_training_dataset"] == "UIEB"
    assert len(list((output / "predictions").glob("*.png"))) == 45
    assert (output / "per_image_metrics.csv").is_file()
    assert (output / "source_checkpoint.json").is_file()
    parsed = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert parsed["metric_time_included_in_inference"] is False

    class FailingModel(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("injected U45 failure")

    monkeypatch.setattr(
        test_u45.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(build_model=lambda _: FailingModel())
            if name == "src.v4.models"
            else original_import_module(name)
        ),
    )
    failure_output = tmp_path / "failed-output"
    failure_args = test_u45.parse_args(
        [
            "--run-dir",
            str(run_dir),
            "--data-root",
            str(tmp_path / "u45"),
            "--output-dir",
            str(failure_output),
        ]
    )
    with pytest.raises(RuntimeError, match="injected U45 failure"):
        test_u45.run(failure_args)
    assert not failure_output.exists()
    assert not list(tmp_path.glob(".failed-output.staging-*"))
