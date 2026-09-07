from __future__ import annotations

import csv
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.shared.e00 import batch_delta_e00, delta_e00_from_lab, e00_protocol
from src.v1.metrics import batch_metrics
from src.v4.engine import _write_history
from tools.clean_test_analysis.io import ModelMetric, TestSample, read_model_metrics
from tools.compare_runs import main as compare_runs_main


ROOT = Path(__file__).resolve().parents[1]
PRIORITY_VERSIONS = ("v4", "v13", "v14", "v15", "v16", "v17")
ALL_VERSIONS = tuple(f"v{index}" for index in range(1, 18))
METRIC_CONFIG = {
    "data_range": 1.0,
    "crop_border": 0,
    "ssim_window_size": 11,
    "ssim_sigma": 1.5,
}


# Published CIEDE2000 supplementary test pairs by Sharma, Wu, and Dalal.
SHARMA_LAB1 = np.array(
    [
        [50.0000, 2.6772, -79.7751], [50.0000, 3.1571, -77.2803],
        [50.0000, 2.8361, -74.0200], [50.0000, -1.3802, -84.2814],
        [50.0000, -1.1848, -84.8006], [50.0000, -0.9009, -85.5211],
        [50.0000, 0.0000, 0.0000], [50.0000, -1.0000, 2.0000],
        [50.0000, 2.4900, -0.0010], [50.0000, 2.4900, -0.0010],
        [50.0000, 2.4900, -0.0010], [50.0000, 2.4900, -0.0010],
        [50.0000, -0.0010, 2.4900], [50.0000, -0.0010, 2.4900],
        [50.0000, -0.0010, 2.4900], [50.0000, 2.5000, 0.0000],
        [50.0000, 2.5000, 0.0000], [50.0000, 2.5000, 0.0000],
        [50.0000, 2.5000, 0.0000], [50.0000, 2.5000, 0.0000],
        [50.0000, 2.5000, 0.0000], [50.0000, 2.5000, 0.0000],
        [50.0000, 2.5000, 0.0000], [50.0000, 2.5000, 0.0000],
        [60.2574, -34.0099, 36.2677], [63.0109, -31.0961, -5.8663],
        [61.2901, 3.7196, -5.3901], [35.0831, -44.1164, 3.7933],
        [22.7233, 20.0904, -46.6940], [36.4612, 47.8580, 18.3852],
        [90.8027, -2.0831, 1.4410], [90.9257, -0.5406, -0.9208],
        [6.7747, -0.2908, -2.4247], [2.0776, 0.0795, -1.1350],
    ],
    dtype=np.float64,
)
SHARMA_LAB2 = np.array(
    [
        [50.0000, 0.0000, -82.7485], [50.0000, 0.0000, -82.7485],
        [50.0000, 0.0000, -82.7485], [50.0000, 0.0000, -82.7485],
        [50.0000, 0.0000, -82.7485], [50.0000, 0.0000, -82.7485],
        [50.0000, -1.0000, 2.0000], [50.0000, 0.0000, 0.0000],
        [50.0000, -2.4900, 0.0009], [50.0000, -2.4900, 0.0010],
        [50.0000, -2.4900, 0.0011], [50.0000, -2.4900, 0.0012],
        [50.0000, 0.0009, -2.4900], [50.0000, 0.0010, -2.4900],
        [50.0000, 0.0011, -2.4900], [50.0000, 0.0000, -2.5000],
        [73.0000, 25.0000, -18.0000], [61.0000, -5.0000, 29.0000],
        [56.0000, -27.0000, -3.0000], [58.0000, 24.0000, 15.0000],
        [50.0000, 3.1736, 0.5854], [50.0000, 3.2972, 0.0000],
        [50.0000, 1.8634, 0.5757], [50.0000, 3.2592, 0.3350],
        [60.4626, -34.1751, 39.4387], [62.8187, -29.7946, -4.0864],
        [61.4292, 2.2480, -4.9620], [35.0232, -40.0716, 1.5901],
        [23.0331, 14.9730, -42.5619], [36.2715, 50.5065, 21.2231],
        [91.1528, -1.6435, 0.0447], [88.6381, -0.8985, -0.7239],
        [5.8714, -0.0985, -2.2286], [0.9033, -0.0636, -0.5514],
    ],
    dtype=np.float64,
)
SHARMA_EXPECTED = np.array(
    [
        2.0425, 2.8615, 3.4412, 1.0000, 1.0000, 1.0000, 2.3669, 2.3669,
        7.1792, 7.1792, 7.2195, 7.2195, 4.8045, 4.8045, 4.7461, 4.3065,
        27.1492, 22.8977, 31.9030, 19.4535, 1.0000, 1.0000, 1.0000, 1.0000,
        1.2644, 1.2630, 1.8731, 1.8645, 2.0373, 1.4146, 1.4441, 1.5381,
        0.6377, 0.9082,
    ],
    dtype=np.float64,
)


def test_sharma_standard_lab_pairs_match_published_delta_e00() -> None:
    actual = delta_e00_from_lab(SHARMA_LAB1, SHARMA_LAB2)
    np.testing.assert_allclose(actual, SHARMA_EXPECTED, rtol=0.0, atol=1e-4)


def test_identical_and_swapped_rgb_images_are_zero_and_symmetric() -> None:
    generator = torch.Generator().manual_seed(3520)
    first = torch.rand(3, 3, 12, 10, generator=generator)
    second = torch.rand(3, 3, 12, 10, generator=generator)
    identical = batch_delta_e00(first, first, METRIC_CONFIG)
    assert identical.shape == (3,)
    torch.testing.assert_close(identical, torch.zeros_like(identical), rtol=0.0, atol=1e-12)
    forward = batch_delta_e00(first, second, METRIC_CONFIG)
    reverse = batch_delta_e00(second, first, METRIC_CONFIG)
    torch.testing.assert_close(forward, reverse, rtol=1e-12, atol=1e-12)


def test_black_white_gray_and_color_inputs_are_finite_and_nonnegative() -> None:
    colors = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.1, 0.3]]
    )
    prediction = colors[:, :, None, None].expand(-1, -1, 4, 5).clone()
    target = prediction.roll(1, dims=0)
    values = batch_delta_e00(prediction, target, METRIC_CONFIG)
    assert values.shape == (4,)
    assert torch.isfinite(values).all()
    assert (values >= 0.0).all()


def test_batch_matches_per_image_and_dataset_mean_is_sample_weighted() -> None:
    generator = torch.Generator().manual_seed(3407)
    prediction = torch.rand(5, 3, 9, 11, generator=generator)
    target = torch.rand(5, 3, 9, 11, generator=generator)
    batched = batch_delta_e00(prediction, target, METRIC_CONFIG)
    individual = torch.cat(
        [
            batch_delta_e00(prediction[index : index + 1], target[index : index + 1], METRIC_CONFIG)
            for index in range(5)
        ]
    )
    torch.testing.assert_close(batched, individual, rtol=0.0, atol=1e-12)
    unequal_batches = (batched[:2], batched[2:])
    sample_weighted = sum(float(value) for batch in unequal_batches for value in batch) / 5
    assert sample_weighted == pytest.approx(float(batched.mean()), abs=1e-12)


def test_crop_border_is_applied_exactly_once() -> None:
    target = torch.zeros(1, 3, 6, 6)
    prediction = target.clone()
    prediction[0, :, 1, 1] = torch.tensor([1.0, 0.5, 0.25])
    cropped_config = {**METRIC_CONFIG, "crop_border": 1}
    once = batch_delta_e00(prediction, target, cropped_config)
    explicit = batch_delta_e00(
        prediction[..., 1:-1, 1:-1], target[..., 1:-1, 1:-1], METRIC_CONFIG
    )
    torch.testing.assert_close(once, explicit, rtol=0.0, atol=1e-12)
    assert float(once[0]) > 0.0  # A second crop would incorrectly remove this corner pixel.


def test_old_metric_config_needs_no_e00_fields_and_psnr_ssim_are_unchanged() -> None:
    prediction = torch.tensor(
        [[[[0.0, 0.25], [0.5, 0.75]], [[1.0, 0.75], [0.5, 0.25]], [[0.1, 0.2], [0.3, 0.4]]]]
    )
    target = torch.tensor(
        [[[[0.1, 0.2], [0.4, 0.8]], [[0.9, 0.8], [0.6, 0.2]], [[0.2, 0.1], [0.35, 0.45]]]]
    )
    assert "e00" not in METRIC_CONFIG
    assert torch.isfinite(batch_delta_e00(prediction, target, METRIC_CONFIG)).all()
    psnr, ssim = batch_metrics(prediction, target, METRIC_CONFIG)
    assert float(psnr[0]) == pytest.approx(22.04119873046875, abs=1e-7)
    assert float(ssim[0]) == pytest.approx(0.8744046092033386, abs=1e-7)


def test_old_history_rows_remain_missing_instead_of_becoming_zero(tmp_path: Path) -> None:
    (tmp_path / "log").mkdir()
    old_row = {
        "epoch": 1, "lr": 2e-4, "train_loss": 0.1, "val_loss": 0.2,
        "val_psnr": 20.0, "val_ssim": 0.8, "epoch_time_seconds": 1.0,
    }
    new_row = {**old_row, "epoch": 2, "val_e00": math.nan}
    _write_history(tmp_path, [old_row, new_row])
    persisted = json.loads((tmp_path / "log/metrics_history.json").read_text())
    assert "val_e00" not in persisted["epochs"][0]
    assert math.isnan(persisted["epochs"][1]["val_e00"])
    with (tmp_path / "log/metrics_history.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["val_e00"] == ""
    assert rows[1]["val_e00"].lower() == "nan"


def _small_model_config(config: dict) -> dict:
    model = deepcopy(config["model"])
    if "base_channels" in model:
        model["base_channels"] = 4
    if "color_query" in model:
        model["color_query"] = {
            **model["color_query"],
            "token_dim": 16,
            "num_heads": 4,
        }
    if "uicf" in model:
        model["uicf"] = {
            "feat_dim": 4,
            "num_frequencies": 2,
            "mlp_hidden_dim": 8,
            "mlp_hidden_layers": 1,
            "anchor_hidden_dim": 4,
            "query_chunk_size": 64,
        }
    return model


def _write_rgb_pair(data_root: Path, sample_id: str, value: int) -> tuple[str, str]:
    input_relative = f"input/{sample_id}.png"
    gt_relative = f"gt/{sample_id}.png"
    array = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(data_root / input_relative)
    Image.fromarray(np.clip(array.astype(np.int16) + 7, 0, 255).astype(np.uint8), mode="RGB").save(
        data_root / gt_relative
    )
    return input_relative, gt_relative


@pytest.mark.parametrize("version", PRIORITY_VERSIONS)
def test_priority_version_test_entry_writes_e00_with_old_style_config(
    tmp_path: Path, version: str
) -> None:
    run_dir = tmp_path / f"{version}_run"
    data_root = tmp_path / "data"
    for path in (
        run_dir / "best", run_dir / "split_snapshot", run_dir / "log",
        data_root / "input", data_root / "gt",
    ):
        path.mkdir(parents=True, exist_ok=True)
    split_ids = {"train": "train_id", "validation": "validation_id", "test": "test_id"}
    for offset, (split, sample_id) in enumerate(split_ids.items()):
        input_relative, gt_relative = _write_rgb_pair(data_root, sample_id, 32 + 48 * offset)
        (run_dir / "split_snapshot" / f"{split}.tsv").write_text(
            f"{sample_id}\t{input_relative}\t{gt_relative}\n", encoding="utf-8"
        )

    config = yaml.safe_load((ROOT / f"configs/config_{version}.yaml").read_text(encoding="utf-8"))
    config["model"] = _small_model_config(config)
    config["data"] = {
        **config["data"],
        "root": str(data_root),
        "expected_counts": {"train": 1, "validation": 1, "test": 1},
        "num_workers": 0,
        "pin_memory": False,
    }
    config["evaluation"] = {"resize": True, "size": 16}
    config["training"]["amp"] = False
    config["test"] = {
        **config["test"],
        "save_all_enhanced_images": False,
        "visualization": {**config["test"]["visualization"], "enabled": False, "num_samples": 1},
    }
    config["logging"]["console"] = False
    assert "e00" not in config and "e00" not in config["metrics"]
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    module = __import__(f"src.{version}.test", fromlist=["main"])
    model_module = __import__(f"src.{version}.models", fromlist=["build_model"])
    model = model_module.build_model(config["model"])
    torch.save(
        {
            "version": version,
            "epoch": 9,
            "resolved_config": {"model": config["model"]},
            "model_state_dict": model.state_dict(),
        },
        run_dir / "best/best_psnr.pt",
    )
    module.main(["--run-dir", str(run_dir), "--checkpoint", "best_psnr"])

    with (run_dir / "result/test_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((run_dir / "result/test_summary.json").read_text())
    assert list(rows[0]) == ["filename", "sample_id", "psnr", "ssim", "e00"]
    assert float(rows[0]["e00"]) >= 0.0
    assert summary["mean_e00"] == pytest.approx(float(rows[0]["e00"]))
    assert summary["e00_protocol"] == e00_protocol(config["metrics"])
    assert summary["total_e00_time_seconds"] >= 0.0
    assert summary["average_inference_time_seconds"] >= 0.0


def test_all_versions_use_the_same_shared_e00_wiring() -> None:
    for version in ALL_VERSIONS:
        engine = (ROOT / f"src/{version}/engine.py").read_text(encoding="utf-8")
        test = (ROOT / f"src/{version}/test.py").read_text(encoding="utf-8")
        assert "from src.shared.e00 import batch_delta_e00" in engine
        assert "batch_delta_e00(clipped, targets.float(), config[\"metrics\"])" in engine
        assert '"val_e00": val_metrics["e00"]' in engine
        assert "from src.shared.e00 import batch_delta_e00, e00_protocol" in test
        assert "batch_delta_e00(prediction, targets.float(), config[\"metrics\"])" in test
        assert '"mean_e00": mean_e00' in test


def test_compare_runs_adds_e00_and_old_results_show_na(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    old_run, new_run = tmp_path / "old", tmp_path / "new"
    for run, summary in ((old_run, {"mean_psnr": 20, "mean_ssim": 0.8}), (new_run, {"mean_psnr": 21, "mean_ssim": 0.81, "mean_e00": 4.2})):
        (run / "result").mkdir(parents=True)
        (run / "run_info.json").write_text(json.dumps({"version": "v4"}), encoding="utf-8")
        (run / "status.json").write_text("{}", encoding="utf-8")
        (run / "result/test_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["compare_runs.py", str(old_run), str(new_run)])
    compare_runs_main()
    lines = capsys.readouterr().out.strip().splitlines()
    assert "test_e00" in lines[0].split("\t")
    assert lines[1].split("\t")[9] == "N/A"
    assert lines[2].split("\t")[9] == "4.2"


def test_clean_test_reader_ignores_the_new_extra_e00_column(tmp_path: Path) -> None:
    samples = {"a": TestSample("a", "input/a.png", "gt/a.png", 0)}
    path = tmp_path / "test_metrics.csv"
    path.write_text("sample_id,psnr,ssim,e00\na,20.0,0.8,3.5\n", encoding="utf-8")
    metrics, fields = read_model_metrics(path, samples, model_label="v4")
    assert metrics == {"a": ModelMetric("a", 20.0, 0.8)}
    assert fields == ["sample_id", "psnr", "ssim", "e00"]
