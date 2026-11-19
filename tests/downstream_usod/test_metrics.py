import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.downstream_usod.evaluation import (METRICS, N_THRESHOLDS, MetricRows, measure,
                                            reduce_metrics)
from src.downstream_usod.statistics import paired_bootstrap


def test_perfect_maps_and_native_alignment():
    gt = np.zeros((8, 8), dtype=np.uint8)
    gt[2:6, 2:6] = 255
    metrics = dict(zip(METRICS, reduce_metrics(measure(gt, gt)), strict=True))
    assert metrics["MAE"] == 0
    assert metrics["MaxF"] == pytest.approx(1.0)
    assert metrics["S-measure"] == pytest.approx(1.0)
    assert metrics["MaxEm"] == pytest.approx(gt.size / (gt.size - 1))
    assert metrics["MeanF"] < metrics["MaxF"]
    with pytest.raises(ValueError, match="aligned grayscale"):
        measure(gt[:, :-1], gt)


def test_constant_map_mae_uses_original_png_intensity():
    gt = np.zeros((7, 9), dtype=np.uint8)
    value = np.full_like(gt, 128)
    scores = dict(zip(METRICS, reduce_metrics(measure(value, gt)), strict=True))
    assert scores["MAE"] == pytest.approx(128 / 255)
    assert scores["MeanF"] == 0
    assert scores["MaxEm"] == pytest.approx(0.25 * gt.size / (gt.size - 1))
    assert scores["S-measure"] == 1
    gt[0, 0] = 255
    assert np.isfinite(measure(value, gt)).all()


def test_bootstrap_reaggregates_curves_each_draw():
    ids = ("a", "b")
    raw = MetricRows(ids, np.zeros((2, 2 * N_THRESHOLDS + 2)))
    scores = raw.values.copy()
    scores[0, [0, N_THRESHOLDS]] = 1
    scores[1, [1, N_THRESHOLDS + 1]] = 1
    scores[:, -1] = (0.2, 0.4)
    candidate = MetricRows(ids, scores)
    assert candidate.overall()["MaxF"] == 0.5
    assert candidate.overall()["MaxEm"] == 0.5
    assert candidate.per_image()[:, 0].mean() == 1
    rows = {r["metric"]: r for r in paired_bootstrap(raw, candidate, 200, 3520)}
    for key in ("MaxF", "MaxEm"):
        assert rows[key]["enhanced_minus_raw"] == 0.5
        assert rows[key]["ci95_low"] == 0.5
        assert rows[key]["ci95_high"] == 1
        assert rows[key]["improved_fraction"] == 1
    assert rows["MAE"]["worse"] == 2
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(raw, MetricRows(ids[::-1], scores), 3, 0)


@pytest.mark.skipif(not os.getenv("USOD_OFFICIAL_REPO"), reason="requires the pinned author checkout")
def test_author_metric_parity_on_rgb8_predictions_and_soft_gt(tmp_path):
    path = Path(os.environ["USOD_OFFICIAL_REPO"]) / "TC_USOD/Evaluation/evaluator.py"
    module = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location(
        "author_evaluator", path
    ))
    spec.loader.exec_module(module)
    rng = np.random.default_rng(71)
    pair1 = rng.integers(0, 256, (2, 17, 13), dtype=np.uint8)
    pair2 = rng.integers(60, 176, (2, 17, 13), dtype=np.uint8)
    pair2[1] = (pair2[1] > 120).astype(np.uint8) * 255
    pairs = [pair1, pair2]
    images = [(Image.fromarray(prediction), Image.fromarray(mask)) for prediction, mask in pairs]
    official = module.Eval_thread(images, "fixture", "fixture", str(tmp_path), cuda=False)
    f, _, _ = official.Eval_fmeasure()
    e = official.Eval_Emeasure()
    expected = [f.max().item(), f.mean().item(), official.Eval_Smeasure(),
                e.max().item(), e.mean().item(), official.Eval_mae()]
    actual = np.stack([measure(pred, gt) for pred, gt in pairs]).mean(axis=0)
    np.testing.assert_allclose(actual[:N_THRESHOLDS], f.numpy(), atol=2e-6, rtol=0)
    np.testing.assert_allclose(actual[N_THRESHOLDS:2 * N_THRESHOLDS], e.numpy(), atol=2e-6, rtol=0)
    np.testing.assert_allclose(reduce_metrics(actual), expected, atol=2e-6, rtol=0)
