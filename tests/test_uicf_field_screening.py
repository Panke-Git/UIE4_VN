from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.v16.models import build_model
from tools.visualize_uicf_field import (
    DEFAULT_VIZ_WEIGHTS,
    _write_csv,
    build_contact_sheet,
    compute_field_metrics,
    export_ranking_html,
    main as screening_main,
    percentile_ranks,
    rank_samples,
)


ROOT = Path(__file__).resolve().parents[1]


def _metrics(field: np.ndarray) -> dict[str, float]:
    inputs = np.linspace(0.0, 1.0, field.size, dtype=np.float32).reshape(field.shape)
    anchor = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    corrected = inputs + field * (inputs - anchor[:, None, None])
    return compute_field_metrics(field, inputs, corrected, anchor)


def _small_model_config() -> dict:
    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text(encoding="utf-8"))
    model = deepcopy(config["model"])
    model["base_channels"] = 4
    model["color_query"] = {
        **model["color_query"],
        "token_dim": 16,
        "num_heads": 4,
    }
    model["uicf"] = {
        "feat_dim": 4,
        "num_frequencies": 2,
        "mlp_hidden_dim": 8,
        "mlp_hidden_layers": 1,
        "anchor_hidden_dim": 4,
        "query_chunk_size": 31,
    }
    return model


def test_spatial_metrics_are_zero_for_constant_and_higher_for_varying_field() -> None:
    constant = np.full((3, 12, 14), 0.5, dtype=np.float32)
    gradient = np.linspace(-1.0, 1.0, 12 * 14, dtype=np.float32).reshape(12, 14)
    varying = np.stack((gradient, gradient * 0.75, gradient * 1.25))
    constant_metrics, varying_metrics = _metrics(constant), _metrics(varying)
    assert constant_metrics["spatial_std"] == pytest.approx(0.0, abs=1e-12)
    assert constant_metrics["spatial_robust_range"] == pytest.approx(0.0, abs=1e-12)
    assert constant_metrics["spatial_nonuniformity"] == pytest.approx(0.0, abs=1e-12)
    assert varying_metrics["spatial_std"] > constant_metrics["spatial_std"]
    assert varying_metrics["spatial_nonuniformity"] > constant_metrics["spatial_nonuniformity"]
    assert 0.0 <= varying_metrics["spatial_coherence"] <= 1.0


def test_channel_specificity_distinguishes_identical_and_different_maps() -> None:
    base = np.linspace(-0.8, 1.0, 10 * 12, dtype=np.float32).reshape(10, 12)
    identical = np.stack((base, base, base))
    different = np.stack((base, -base, np.zeros_like(base)))
    identical_metrics, different_metrics = _metrics(identical), _metrics(different)
    assert identical_metrics["D_rg"] == pytest.approx(0.0)
    assert identical_metrics["channel_specificity"] == pytest.approx(0.0)
    assert different_metrics["D_rg"] > 0.0
    assert different_metrics["D_rb"] > 0.0
    assert different_metrics["channel_specificity"] > 0.0


def test_percentile_ranks_use_average_ties_and_cover_zero_to_one() -> None:
    assert percentile_ranks([10.0, 20.0, 20.0, 40.0]) == pytest.approx(
        [0.0, 0.5, 0.5, 1.0]
    )
    assert percentile_ranks([7.0]) == [1.0]


def test_viz_score_uses_exact_weights_and_not_psnr() -> None:
    rows = [
        {
            "sample_index": 0,
            "sample_id": "spatial",
            "spatial_nonuniformity": 10.0,
            "channel_specificity": 0.0,
            "spatial_coherence": 0.0,
            "image_correction_strength": 0.0,
            "psnr": 99.0,
        },
        {
            "sample_index": 1,
            "sample_id": "other_three",
            "spatial_nonuniformity": 0.0,
            "channel_specificity": 10.0,
            "spatial_coherence": 10.0,
            "image_correction_strength": 10.0,
            "psnr": 1.0,
        },
    ]
    ranked = rank_samples(rows)
    assert DEFAULT_VIZ_WEIGHTS == {
        "spatial": 0.45,
        "channel": 0.35,
        "coherence": 0.15,
        "delta": 0.05,
    }
    assert ranked[0]["sample_id"] == "other_three"
    assert ranked[0]["viz_score"] == pytest.approx(0.55)
    assert ranked[1]["viz_score"] == pytest.approx(0.45)


def test_csv_serialization_contains_no_nan_and_rejects_nonfinite(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write_csv(path, [{"sample_id": "ok", "score": 0.5}], ["sample_id", "score"])
    assert "nan" not in path.read_text(encoding="utf-8").lower()
    with pytest.raises(FloatingPointError, match="non-finite CSV"):
        _write_csv(path, [{"sample_id": "bad", "score": float("nan")}], ["sample_id", "score"])


def test_static_ranking_html_uses_relative_thumbnail_paths(tmp_path: Path) -> None:
    rows = [
        {
            "rank": 1,
            "sample_id": "sample&one",
            "thumbnail_path": "thumbnails/0001_sample.png",
            "viz_score": 0.9,
            "P_spatial": 0.95,
            "P_channel": 0.85,
            "P_coherence": 0.8,
            "P_delta": 0.4,
        }
    ]
    path = tmp_path / "ranking.html"
    export_ranking_html(rows, path)
    document = path.read_text(encoding="utf-8")
    assert "<!doctype html>" in document.lower()
    assert 'src="thumbnails/0001_sample.png"' in document
    assert "sample&amp;one" in document
    assert "http://" not in document and "https://" not in document


def test_contact_sheet_is_generated_as_a_readable_candidate_grid(tmp_path: Path) -> None:
    thumbnails = tmp_path / "thumbnails"
    thumbnails.mkdir()
    rows = []
    for rank in range(1, 6):
        relative = Path("thumbnails") / f"{rank}.png"
        Image.new("RGB", (240, 70), (rank * 30, 80, 120)).save(tmp_path / relative)
        rows.append({"rank": rank, "sample_id": f"id{rank}", "thumbnail_path": relative.as_posix()})
    output = tmp_path / "contact.png"
    build_contact_sheet(rows, tmp_path, output, columns=4)
    with Image.open(output) as sheet:
        assert sheet.mode == "RGB"
        assert sheet.width > 4 * 200
        assert sheet.height > 2 * 70


def test_complete_synthetic_screening_run_exports_ranking_and_top_k(tmp_path: Path) -> None:
    run_dir = tmp_path / "v16_lsui_run"
    data_root = tmp_path / "LSUI19"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "split_snapshot").mkdir()
    for directory in (data_root / "input", data_root / "gt"):
        directory.mkdir(parents=True)
    split_ids = {
        "train": ["train_id"],
        "validation": ["validation_id"],
        "test": ["test_0", "test_1", "test_2", "test_3"],
    }
    for offset, sample_id in enumerate(sum(split_ids.values(), [])):
        if sample_id == "test_3":
            continue  # Exercise per-sample failure recording without aborting screening.
        value = 20 + offset * 20
        input_array = np.full((16, 16, 3), value, dtype=np.uint8)
        gt_array = np.full((16, 16, 3), min(value + 5, 255), dtype=np.uint8)
        Image.fromarray(input_array, mode="RGB").save(data_root / "input" / f"{sample_id}.png")
        Image.fromarray(gt_array, mode="RGB").save(data_root / "gt" / f"{sample_id}.png")
    for split, sample_ids in split_ids.items():
        lines = [f"{sample_id}\tinput/{sample_id}.png\tgt/{sample_id}.png" for sample_id in sample_ids]
        (run_dir / "split_snapshot" / f"{split}.tsv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text(encoding="utf-8"))
    config["model"] = _small_model_config()
    config["data"] = {
        **config["data"],
        "root": str(data_root),
        "expected_counts": {split: len(ids) for split, ids in split_ids.items()},
        "num_workers": 0,
    }
    config["evaluation"] = {"resize": True, "size": 16}
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    model = build_model(config["model"])
    torch.save(
        {
            "version": "v16",
            "epoch": 9,
            "resolved_config": {"model": config["model"]},
            "model_state_dict": model.state_dict(),
        },
        run_dir / "best" / "best_psnr.pt",
    )

    screening_main(
        [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best_psnr",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--top-k",
            "2",
            "--thumbnail-size",
            "64",
        ]
    )
    output = run_dir / "result" / "uicf_screening"
    required = {
        "all_samples_metrics.csv",
        "ranking.csv",
        "ranking.html",
        "top2.csv",
        "top2_summary.csv",
        "top2_summary.txt",
        "failed_samples.csv",
        "top2_contact_sheet.png",
        "screening_config.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert len(list((output / "thumbnails").glob("*.png"))) == 3
    assert len(list((output / "top2").glob("rank*"))) == 2
    with (output / "ranking.csv").open(newline="", encoding="utf-8") as handle:
        ranking = list(csv.DictReader(handle))
    assert len(ranking) == 3
    assert [int(row["rank"]) for row in ranking] == [1, 2, 3]
    screening_config = json.loads((output / "screening_config.json").read_text())
    assert screening_config["total_test_samples"] == 4
    assert screening_config["processed_sample_count"] == 4
    assert screening_config["successful_sample_count"] == 3
    assert screening_config["failed_sample_count"] == 1
    assert screening_config["score_uses_psnr_or_ssim"] is False
    with (output / "failed_samples.csv").open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))
    assert len(failures) == 1 and failures[0]["sample_id"] == "test_3"
