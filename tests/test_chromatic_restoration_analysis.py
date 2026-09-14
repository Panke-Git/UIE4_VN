from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from src.v16.dataset import LSUIDataset, ManifestEntry, read_manifest
from tools.plot_chromatic_restoration_analysis import (
    PROJECT_ROOT,
    ab_bin_edges,
    ab_probability_histogram,
    analyze_sample,
    delta_e00_map_from_rgb,
    enhanced_filename_mapping,
    main,
    resolve_enhanced_paths,
    resolve_dataset_label,
    resolve_project_path,
    rgb_to_project_lab,
    shared_delta_e_vmax,
)


def _rgb_pattern(index: int, *, height: int = 19, width: int = 23) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    red = (17 * xx + 5 * yy + 31 * index) % 256
    green = (3 * xx + 13 * yy + 19 * index + 20) % 256
    blue = (11 * xx + 7 * yy + 23 * index + 45) % 256
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def _write_synthetic_run(tmp_path: Path, *, sample_count: int = 3) -> tuple[Path, Path]:
    run_dir = tmp_path / "experiments" / "synthetic_v16_uieb"
    data_root = tmp_path / "synthetic_UIEB_data"
    input_dir = data_root / "input"
    gt_dir = data_root / "gt"
    enhanced_dir = run_dir / "result" / "test_all_enhanced"
    manifest_dir = run_dir / "split_snapshot"
    for directory in (input_dir, gt_dir, enhanced_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    entries: list[ManifestEntry] = []
    metric_rows: list[dict[str, object]] = []
    for index in range(sample_count):
        sample_id = f"sample_{index:02d}"
        input_array = _rgb_pattern(index)
        reference_array = np.clip(
            input_array.astype(np.int16) + np.array([25, -10, 18]), 0, 255
        ).astype(np.uint8)
        input_path = input_dir / f"{sample_id}.png"
        gt_path = gt_dir / f"{sample_id}.png"
        Image.fromarray(input_array, mode="RGB").save(input_path)
        Image.fromarray(reference_array, mode="RGB").save(gt_path)
        entries.append(
            ManifestEntry(
                sample_id,
                f"input/{sample_id}.png",
                f"gt/{sample_id}.png",
            )
        )

        resized_input = np.asarray(
            Image.fromarray(input_array, mode="RGB").resize(
                (16, 16), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
        resized_reference = np.asarray(
            Image.fromarray(reference_array, mode="RGB").resize(
                (16, 16), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
        blend = np.rint(0.2 * resized_input + 0.8 * resized_reference).astype(np.uint8)
        enhanced_name = f"{sample_id}_enhanced.png"
        Image.fromarray(blend, mode="RGB").save(enhanced_dir / enhanced_name)
        official_e00 = float(
            delta_e00_map_from_rgb(blend / 255.0, resized_reference / 255.0).mean()
        )
        metric_rows.append(
            {
                "filename": f"{sample_id}.png",
                "sample_id": sample_id,
                "psnr": 25.0 + index,
                "ssim": 0.90 + index * 0.01,
                "e00": official_e00,
            }
        )

    manifest_path = manifest_dir / "test.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for entry in entries:
            writer.writerow((entry.sample_id, entry.input_relative, entry.gt_relative))

    config = {
        "experiment": {"version": "v16", "name": "synthetic"},
        "data": {
            "dataset": "UIEB",
            "root": str(data_root),
            "patch_size": 8,
            "augmentation": {"hflip": False, "vflip": False, "rot90": False},
            "pad_if_smaller": True,
            "pad_mode": "reflect",
            "expected_counts": {"train": 1, "validation": 1, "test": sample_count},
        },
        "evaluation": {"resize": True, "size": 16},
        "metrics": {"crop_border": 0},
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    metrics_path = run_dir / "result" / "test_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("filename", "sample_id", "psnr", "ssim", "e00")
        )
        writer.writeheader()
        writer.writerows(metric_rows)
    return run_dir, data_root


def test_project_root_and_path_resolution_are_cwd_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert PROJECT_ROOT.name == "UIE4_VN"
    expected = (PROJECT_ROOT / "experiments" / "example").resolve()
    monkeypatch.chdir(PROJECT_ROOT)
    from_repo = resolve_project_path("experiments/example")
    monkeypatch.chdir(PROJECT_ROOT.parent)
    from_workspace = resolve_project_path("UIE4_VN/experiments/example")
    assert from_repo == expected
    assert from_workspace == expected


def test_manifest_mapping_uses_frozen_three_column_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "test.tsv"
    manifest.write_text(
        "one\tinput/one.png\tgt/one.png\n"
        "two\tinput/two.png\tgt/two.png\n",
        encoding="utf-8",
    )
    assert read_manifest(manifest) == [
        ManifestEntry("one", "input/one.png", "gt/one.png"),
        ManifestEntry("two", "input/two.png", "gt/two.png"),
    ]


def test_enhanced_filename_resolution_reproduces_ordered_collision_fallback(
    tmp_path: Path,
) -> None:
    entries = [
        ManifestEntry("first", "a/shared.png", "gt/first.png"),
        ManifestEntry("second", "b/shared.png", "gt/second.png"),
    ]
    expected = {
        "first": "shared_enhanced.png",
        "second": "shared_second_enhanced.png",
    }
    assert enhanced_filename_mapping(entries) == expected
    enhanced_dir = tmp_path / "test_all_enhanced"
    enhanced_dir.mkdir()
    for name in expected.values():
        Image.new("RGB", (4, 4), "black").save(enhanced_dir / name)
    resolved = resolve_enhanced_paths(entries, enhanced_dir, tmp_path)
    assert {key: path.name for key, path in resolved.items()} == expected


def test_evaluation_resize_and_enhanced_shape_match_official_dataset(
    tmp_path: Path,
) -> None:
    run_dir, data_root = _write_synthetic_run(tmp_path, sample_count=1)
    manifest = run_dir / "split_snapshot" / "test.tsv"
    dataset = LSUIDataset(
        manifest,
        data_root,
        "test",
        8,
        {"hflip": False, "vflip": False, "rot90": False},
        True,
        "reflect",
        {"resize": True, "size": 16},
        verify_files=True,
    )
    item = dataset[0]
    with Image.open(data_root / "input" / "sample_00.png") as source:
        expected = np.asarray(
            source.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32
        ) / 255.0
    actual = item["input"].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(actual, expected)
    enhanced = run_dir / "result" / "test_all_enhanced" / "sample_00_enhanced.png"
    official = {"filename": "sample_00.png", "psnr": 25.0, "ssim": 0.9, "e00": 1.0}
    analyzed = analyze_sample(dataset, 0, enhanced, official, crop_border=0)
    assert analyzed.input_rgb.shape == (16, 16, 3)
    assert analyzed.output_rgb.shape == (16, 16, 3)
    assert analyzed.reference_rgb.shape == (16, 16, 3)


def test_identical_image_delta_e00_map_is_zero() -> None:
    rgb = _rgb_pattern(0, height=7, width=9).astype(np.float64) / 255.0
    result = delta_e00_map_from_rgb(rgb, rgb.copy())
    assert result.shape == (7, 9)
    np.testing.assert_allclose(result, 0.0, rtol=0.0, atol=1e-12)


def test_ab_histogram_is_probability_mass_and_reuses_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edges = ab_bin_edges(32, -128.0, 127.0)
    labs = [
        rgb_to_project_lab(_rgb_pattern(index, height=8, width=9) / 255.0)
        for index in range(3)
    ]
    original_histogram2d = np.histogram2d
    observed_bins: list[tuple[np.ndarray, np.ndarray]] = []

    def recording_histogram2d(*args: object, **kwargs: object) -> object:
        bins = kwargs["bins"]
        assert isinstance(bins, tuple)
        observed_bins.append(bins)
        return original_histogram2d(*args, **kwargs)

    monkeypatch.setattr(np, "histogram2d", recording_histogram2d)
    histograms = [ab_probability_histogram(lab, edges, edges) for lab in labs]
    for histogram in histograms:
        assert histogram.shape == (32, 32)
        assert float(histogram.sum()) == pytest.approx(1.0, abs=1e-12)
    assert all(histogram.shape == histograms[0].shape for histogram in histograms)
    assert len(observed_bins) == 3
    assert all(a_edges is edges and b_edges is edges for a_edges, b_edges in observed_bins)


def test_legacy_dataset_label_is_inferred_without_restricting_modern_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert resolve_dataset_label({"dataset": "UIEB"}) == "UIEB"
    assert resolve_dataset_label({"dataset": "future-paired-dataset"}) == "future-paired-dataset"
    assert resolve_dataset_label(
        {"root": "/data/LSUI19_dup_train", "test_manifest": "split/lsui19/test.tsv"}
    ) == "LSUI19"
    assert "inferred legacy dataset label" in capsys.readouterr().err


def test_shared_delta_e_vmax_uses_pooled_percentile() -> None:
    maps = [np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([[4.0, 5.0]])]
    expected = float(np.percentile(np.arange(6, dtype=np.float64), 80.0))
    assert shared_delta_e_vmax(maps, percentile=80.0, explicit_vmax=None) == expected
    assert shared_delta_e_vmax(maps, percentile=80.0, explicit_vmax=7.5) == 7.5


def test_missing_enhanced_output_fails_with_official_test_instruction(
    tmp_path: Path,
) -> None:
    entries = [ManifestEntry("one", "input/one.png", "gt/one.png")]
    enhanced_dir = tmp_path / "result" / "test_all_enhanced"
    enhanced_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Run the official v16 test first") as error:
        resolve_enhanced_paths(entries, enhanced_dir, tmp_path)
    assert "python -m src.v16.test" in str(error.value)


def test_screening_and_final_figure_outputs_are_generated(tmp_path: Path) -> None:
    run_dir, _ = _write_synthetic_run(tmp_path, sample_count=3)
    screening_output = tmp_path / "screening_output"
    main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(screening_output),
            "--num-candidates",
            "2",
            "--ab-bins",
            "16",
            "--dpi",
            "72",
        ]
    )
    expected_outputs = {
        "chromatic_restoration_analysis.png",
        "chromatic_restoration_analysis.pdf",
        "chromatic_restoration_analysis.svg",
        "chromatic_per_sample.csv",
        "selected_samples.csv",
        "analysis_protocol.json",
        "candidate_contact_sheet.png",
    }
    assert {path.name for path in screening_output.iterdir()} == expected_outputs
    protocol = json.loads(
        (screening_output / "analysis_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["dataset"] == "UIEB"
    assert protocol["analysis_mode"] == "screening"
    assert protocol["analyzed_sample_count"] == 3
    assert protocol["sample_count"] == 2
    assert len(protocol["candidate_sample_ids"]) == 2
    assert protocol["model_or_checkpoint_loaded"] is False
    assert protocol["inference_performed"] is False
    with (screening_output / "chromatic_per_sample.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        per_sample_rows = list(csv.DictReader(handle))
    assert len(per_sample_rows) == 3
    assert {
        "official_output_e00",
        "output_png_e00",
        "png_quantization_delta",
    }.issubset(per_sample_rows[0])
    with (screening_output / "selected_samples.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        selected_rows = list(csv.DictReader(handle))
    assert all(row["candidate_selection_rule"] for row in selected_rows)
    for filename in expected_outputs:
        assert (screening_output / filename).stat().st_size > 0
    with Image.open(screening_output / "chromatic_restoration_analysis.png") as image:
        assert image.mode == "RGBA"
        assert float(np.asarray(image.convert("RGB"), dtype=np.float64).std()) > 0.0
    assert (screening_output / "chromatic_restoration_analysis.pdf").read_bytes().startswith(
        b"%PDF"
    )

    selected = protocol["selected_sample_ids"]
    final_output = tmp_path / "final_output"
    main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(final_output),
            "--sample-ids",
            *selected,
            "--ab-bins",
            "16",
            "--dpi",
            "72",
        ]
    )
    final_protocol = json.loads(
        (final_output / "analysis_protocol.json").read_text(encoding="utf-8")
    )
    assert final_protocol["analysis_mode"] == "final_figure"
    assert final_protocol["selected_sample_ids"] == selected
    assert (final_output / "chromatic_restoration_analysis.png").stat().st_size > 0
    assert (final_output / "chromatic_restoration_analysis.pdf").stat().st_size > 0


def test_existing_outputs_require_overwrite(tmp_path: Path) -> None:
    run_dir, _ = _write_synthetic_run(tmp_path, sample_count=1)
    output = tmp_path / "existing"
    main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output),
            "--ab-bins",
            "8",
            "--dpi",
            "36",
        ]
    )
    with pytest.raises(FileExistsError, match="--overwrite"):
        main(
            [
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output),
                "--ab-bins",
                "8",
                "--dpi",
                "36",
            ]
        )


def test_unknown_final_sample_id_fails_loudly(tmp_path: Path) -> None:
    run_dir, _ = _write_synthetic_run(tmp_path, sample_count=1)
    with pytest.raises(ValueError, match="not in frozen test split"):
        main(
            [
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(tmp_path / "unused"),
                "--sample-ids",
                "not-a-real-id",
                "--dpi",
                "36",
            ]
        )
