"""Known-transform software checks; synthetic scores are not V16 research results."""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from tools.evaluate_sift import main, parse_args
from tools.sift_eval.matching import Settings, match_images, summarize, verify_geometry


ROOT = Path(__file__).resolve().parents[1]


def texture(seed=4, size=192):
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    array = cv2.GaussianBlur(array, (3, 3), 0)
    for _ in range(70):
        x, y = rng.integers(8, size - 8, 2)
        cv2.circle(array, (int(x), int(y)), int(rng.integers(2, 7)),
                   tuple(map(int, rng.integers(0, 256, 3))), -1)
    return Image.fromarray(array)


def test_identity_rejects_displacement_instead_of_fitting_it_away():
    a = np.array([[1, 2], [5, 8], [9, 4], [4, 6]], dtype=np.float32)
    b = a + [[0, 0], [2, 0], [4, 0], [0, 4]]
    mask, matrix, status = verify_geometry(a, b, "identity", Settings(threshold_px=3))
    assert mask.tolist() == [True, True, False, False]
    np.testing.assert_array_equal(matrix, np.eye(3))
    assert status == "ok"


def test_known_homography_and_degenerate_cases():
    rng = np.random.default_rng(7)
    a = rng.uniform(0, 150, (40, 2)).astype(np.float32)
    expected = np.array([[1.01, .03, 12], [-.02, .99, 8], [.0001, .0003, 1.]])
    b = cv2.perspectiveTransform(a[None], expected)[0]
    b[-5:] = rng.uniform(0, 150, (5, 2))
    mask, matrix, status = verify_geometry(a, b, "homography", Settings(threshold_px=1))
    assert status == "ok" and mask[:35].all() and not mask[-5:].any()
    np.testing.assert_allclose(matrix / matrix[2, 2], expected, atol=1e-4)
    mask, matrix, status = verify_geometry(a[:3], b[:3], "homography", Settings())
    assert status == "insufficient_matches" and matrix is None and not mask.any()
    collinear = np.array([[i, 2 * i] for i in range(10)], np.float32)
    mask, matrix, status = verify_geometry(collinear, collinear + 10, "homography", Settings())
    assert status == "estimation_failed" and matrix is None and not mask.any()


def test_fundamental_model_on_known_3d_projections():
    rng = np.random.default_rng(17)
    xyz = rng.uniform([-1, -1, 3], [1, 1, 7], (40, 3))
    a = (xyz[:, :2] / xyz[:, 2:] * 300 + 150).astype(np.float32)
    xyz[:, 0] += .3
    b = (xyz[:, :2] / xyz[:, 2:] * 300 + 150).astype(np.float32)
    mask, matrix, status = verify_geometry(a, b, "fundamental", Settings(threshold_px=.5))
    assert status == "ok" and mask.sum() >= 38 and matrix.shape == (3, 3)


def test_sift_recovers_known_translation_and_runs_deterministically():
    a = texture()
    transform = np.array([[1., 0, 8], [0, 1., 6]])
    b = Image.fromarray(cv2.warpAffine(np.asarray(a), transform, a.size))
    settings = Settings(threshold_px=1)
    first = match_images(a, b, "homography", settings)
    second = match_images(a, b, "homography", settings)
    assert first.metrics["verified_matches"] > 20
    assert first.metrics["inlier_ratio"] > .9
    assert first.metrics == second.metrics
    np.testing.assert_array_equal(first.inliers, second.inliers)
    displacement = first.points_b[first.inliers] - first.points_a[first.inliers]
    np.testing.assert_allclose(np.median(displacement, axis=0), [8, 6], atol=.3)


def test_zero_features_are_counted_and_zero_baseline_gain_is_null():
    image = Image.new("RGB", (64, 64))
    blank = match_images(image, image, "identity", Settings())
    assert blank.metrics["verified_matches"] == blank.metrics["tentative_matches"] == 0
    assert blank.metrics["inlier_ratio"] == 0 and blank.metrics["geometry_status"] == "no_keypoints"
    rows = [{"pair_id": "blank", "condition": c, **blank.metrics} for c in ("raw", "v16")]
    result = summarize(rows)
    assert result["raw"]["pair_count"] == result["raw"]["zero_tentative_pairs"] == 1
    assert result["raw"]["macro_inlier_ratio"] == 0
    assert result["paired_change"]["total_verified_relative_gain"] is None
    json.dumps(result, allow_nan=False)


@pytest.fixture
def saved_run(tmp_path):
    run, data, predictions = (tmp_path / key for key in ("run", "data", "predictions"))
    (run / "split_snapshot").mkdir(parents=True)
    data.mkdir()
    predictions.mkdir()
    config = {"experiment": {"version": "v16"}, "evaluation": {"resize": True, "size": 192},
              "data": {"root": "/old/server/path", "expected_counts": {"test": 2}}}
    (run / "config_resolved.yaml").write_text(yaml.safe_dump(config))
    (run / "split_snapshot/test.tsv").write_text("one\ta.jpg\tref_a.jpg\ntwo\tb.jpg\tref_b.jpg\n")
    a, b = texture(), Image.new("RGB", (192, 192))
    for image, name in ((a, "a"), (b, "b")):
        image.save(data / f"{name}.jpg")
        # Use decoded JPEG pixels to give exact known identity predictions/references.
        with Image.open(data / f"{name}.jpg") as decoded:
            decoded.save(data / f"ref_{name}.jpg", quality=100)
            decoded.save(predictions / f"{name}_enhanced.png")
    return run, data, predictions


def base_args(saved_run):
    run, data, predictions = saved_run
    return ["gt", "--run-dir", str(run), "--data-root", str(data),
            "--prediction-dir", str(predictions), "--device", "cpu"]


def test_saved_predictions_cli_artifacts_and_no_checkpoint_required(saved_run):
    run, _, _ = saved_run
    command = [sys.executable, str(ROOT / "tools/evaluate_sift.py"), *base_args(saved_run)]
    process = subprocess.run(command, capture_output=True, text=True, cwd=run.parent, timeout=60)
    assert process.returncode == 0, process.stderr
    output = run / "result/sift_gt"
    summary = json.loads((output / "sift_evaluation.json").read_text())
    assert summary["complete"] and not summary["scope"]["partial"]
    assert summary["metrics"]["raw"]["zero_tentative_pairs"] == 1
    assert summary["metrics"]["v16"]["pair_count"] == 2
    assert summary["protocol"]["geometry"] == "identity"
    assert not summary["inference"]["checkpoint_loaded"]
    assert (output / "figures/overview.png").is_file()
    with np.load(output / "matches/00000_v16.npz") as points:
        assert points["points_a"].shape[1] == 2
        assert points["inliers"].sum() > 10
    assert len(list((output / "matches").iterdir())) == 4
    with pytest.raises(FileExistsError):
        main(base_args(saved_run))


def test_partial_runs_and_failed_replacement_keep_previous_output(saved_run):
    run, _, predictions = saved_run
    result = main(base_args(saved_run) + ["--limit", "1", "--num-visualizations", "0"])
    assert result["scope"]["partial"] and result["scope"]["evaluated_pairs"] == 1
    summary_path = run / "result/sift_gt/sift_evaluation.json"
    before = summary_path.read_bytes()
    Image.new("RGB", (64, 64)).save(predictions / "a_enhanced.png")
    with pytest.raises(ValueError, match="Prediction size"):
        main(base_args(saved_run) + ["--overwrite"])
    assert summary_path.read_bytes() == before
    assert not list(summary_path.parent.parent.glob(".sift_gt.staging-*"))


def test_pairs_prediction_mapping_and_source_protection(saved_run):
    run, data, predictions = saved_run
    pairs = run.parent / "pairs.csv"
    pairs.write_text("pair_id,image_a,image_b,enhanced_a,enhanced_b\n"
                     "scene,a.jpg,b.jpg,a_enhanced.png,b_enhanced.png\n")
    args = ["pairs", "--run-dir", str(run), "--data-root", str(data), "--pairs", str(pairs),
            "--prediction-dir", str(predictions), "--resolution", "native"]
    result = main(args)
    assert result["protocol"]["geometry"] == "homography"
    assert result["metrics"]["raw"]["geometry_failure_pairs"] == 1
    with pytest.raises(ValueError, match="source data"):
        main(args + ["--output-dir", str(run.parent), "--overwrite"])
    pairs.write_text(pairs.read_text() + "duplicate,b.jpg,a.jpg,b_enhanced.png,a_enhanced.png\n")
    with pytest.raises(ValueError, match="duplicate/reverse"):
        main(args)


def test_pairs_known_translation_exports_equal_raw_and_v16_scores(saved_run):
    run, data, predictions = saved_run
    a = texture()
    b = Image.fromarray(cv2.warpAffine(np.asarray(a), np.float32([[1, 0, 8], [0, 1, 6]]), a.size))
    a.save(data / "left.png")
    b.save(data / "right.png")
    a.save(predictions / "left.png")
    b.save(predictions / "right.png")
    pairs = run.parent / "known_pairs.csv"
    pairs.write_text("pair_id,image_a,image_b,enhanced_a,enhanced_b\n"
                     "known,left.png,right.png,left.png,right.png\n")
    result = main(["pairs", "--run-dir", str(run), "--data-root", str(data), "--pairs", str(pairs),
                   "--prediction-dir", str(predictions), "--resolution", "native", "--max-draw", "1"])
    raw, enhanced = result["metrics"]["raw"], result["metrics"]["v16"]
    assert raw["total_verified_matches"] > 20
    assert raw == enhanced  # Displaying only one line must not cap the score.
    assert result["visualization"]["max_draw"] == 1


def test_original_gt_size_mismatch_is_rejected_even_when_resizing(saved_run):
    _, data, _ = saved_run
    Image.new("RGB", (300, 300)).save(data / "ref_a.jpg")
    with pytest.raises(ValueError, match="Aligned GT requires equal dimensions"):
        main(base_args(saved_run))


def test_real_v16_factory_checkpoint_inference_on_cpu(saved_run):
    from src.v16.models import build_model

    run, data, _ = saved_run
    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text())
    config["model"].update({"base_channels": 2, "use_batch_norm": False})
    config["model"]["color_query"].update({"num_color_queries": 2, "token_dim": 4, "num_heads": 1})
    config["model"]["uicf"].update({"feat_dim": 2, "num_frequencies": 1, "mlp_hidden_dim": 4,
                                  "mlp_hidden_layers": 1, "anchor_hidden_dim": 2, "query_chunk_size": 1024})
    config["data"] = {"root": str(data), "expected_counts": {"test": 2}}
    config["evaluation"] = {"resize": True, "size": 32}
    (run / "config_resolved.yaml").write_text(yaml.safe_dump(config))
    (run / "best").mkdir()
    model = build_model(config["model"])
    torch.save({"version": "v16", "resolved_config": config, "epoch": 7,
                "model_state_dict": model.state_dict()}, run / "best/best_psnr.pt")
    torch.set_num_threads(1)
    result = main(["gt", "--run-dir", str(run), "--data-root", str(data),
                   "--device", "cpu", "--limit", "1"])
    assert result["inference"]["checkpoint_loaded"] and result["inference"]["epoch"] == 7
    output = run / "result/sift_gt"
    generated = list((output / "enhanced").glob("*.png"))
    assert len(generated) == 1
    with Image.open(generated[0]) as image:
        assert image.size == (32, 32) and image.mode == "RGB"
    # Strict state/config validation refuses a migrated checkpoint from another architecture.
    checkpoint_path = run / "best/best_psnr.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["resolved_config"]["model"]["base_channels"] = 999
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="architecture"):
        main(["gt", "--run-dir", str(run), "--data-root", str(data), "--device", "cpu", "--overwrite"])


def test_resolution_and_settings_validation():
    with pytest.raises(SystemExit):
        parse_args(["gt", "--run-dir", ".", "--max-side", "512"])
    with pytest.raises(ValueError):
        Settings(ratio=1)
    with pytest.raises(ValueError):
        Settings(threshold_px=float("nan"))
