"""The author's 255-threshold protocol and auditable per-image scores."""

from dataclasses import dataclass

import numpy as np
import torch

from .dataset import read_input
from .storage import atomic_file, seal_json, write_csv

THRESHOLDS = torch.linspace(0, 1 - 1e-10, 255, dtype=torch.float32).numpy()
LEVELS = torch.arange(256, dtype=torch.float32).div(255).numpy()
N_THRESHOLDS = len(THRESHOLDS)
METRICS = ("MaxF", "MeanF", "S-measure", "MaxEm", "MeanEm", "MAE")
METRIC_PROTOCOL = "tc-usod-evaluator-255-v1"
EPS = 1e-20


def _normalize_levels(lo, hi):
    return (LEVELS - LEVELS[lo]) / (LEVELS[hi] - LEVELS[lo] + np.float32(EPS))


def _threshold_curves(pred, gt, normalized_levels):
    # Count each grayscale pair once. Suffix sums give the positive set of each threshold.
    joint = np.bincount(pred.astype(np.int32).ravel() * 256 + gt.ravel(),
                        minlength=256 * 256).reshape(256, 256)
    active = joint.any(axis=0)
    joint = joint[:, active]
    levels = LEVELS[active].astype(np.float64)
    totals = joint.sum(axis=0)
    suffix = np.vstack((joint[::-1].cumsum(axis=0)[::-1], np.zeros(len(levels), dtype=np.int64)))
    cuts = np.searchsorted(normalized_levels, THRESHOLDS, side="left")
    positives = suffix[cuts]
    foreground = positives.sum(axis=1)
    truth_mass = np.sum(totals * levels)
    overlap = np.sum(positives * levels, axis=1)
    precision = overlap / (foreground + EPS)
    recall = overlap / (truth_mass + EPS)
    f_curve = 1.3 * precision * recall / (0.3 * precision + recall + EPS)

    pred_mean = foreground[:, None] / pred.size
    gt_residual = levels[None, :] - truth_mass / gt.size

    def enhancement(residual):
        alignment = 2 * residual * gt_residual / (residual ** 2 + gt_residual ** 2 + EPS)
        return ((1 + alignment) / 2) ** 2

    weighted = positives * enhancement(1 - pred_mean) + (totals - positives) * enhancement(-pred_mean)
    return f_curve, weighted.sum(axis=1) / (pred.size - 1 + EPS)


def _s_object(values):
    mean = values.mean()
    spread = values.std(ddof=1) if values.size > 1 else 0.0
    return 2 * mean / (1 + mean ** 2 + spread + EPS)


def _s_region(pred, mask):
    rows, columns = mask.shape
    area = mask.sum()
    x = int(np.rint(mask.sum(axis=0) @ np.arange(columns) / area))
    y = int(np.rint(mask.sum(axis=1) @ np.arange(rows) / area))
    quadrants = ((slice(0, y), slice(0, x)), (slice(0, y), slice(x, columns)),
                 (slice(y, rows), slice(0, x)), (slice(y, rows), slice(x, columns)))
    score = 0.0
    for section in quadrants:
        a, b = pred[section], mask[section].astype(np.float64)
        if not a.size:
            continue  # Zero area contributes zero weight.
        ma, mb = a.mean(), b.mean()
        da, db = a - ma, b - mb
        denom = a.size - 1 + EPS
        coefficient = 4 * ma * mb * (da * db).sum() / denom
        scale = (ma ** 2 + mb ** 2) * ((da ** 2 + db ** 2).sum() / denom)
        local = (1.0 if scale == 0 else 0.0) if coefficient == 0 else coefficient / (scale + EPS)
        score += local * a.size / mask.size
    return score


def _s_measure(pred, gt):
    mask = gt >= 128
    if not mask.any():
        return float(1 - pred.mean())
    if mask.all():
        return float(pred.mean())
    fraction = mask.mean()
    obj = fraction * _s_object(pred[mask]) + (1 - fraction) * _s_object(1 - pred[~mask])
    return max(0.0, float(0.5 * (obj + _s_region(pred, mask))))


def measure(prediction, ground_truth):
    """One image's F/E curves and S/MAE scalars in the author's PNG8 protocol."""
    pred, gt = np.asarray(prediction), np.asarray(ground_truth)
    if (pred.ndim != 2 or gt.shape != pred.shape or pred.size < 2
            or pred.dtype != np.uint8 or gt.dtype != np.uint8):
        raise ValueError("prediction and GT must be aligned grayscale uint8 arrays")
    levels = _normalize_levels(int(pred.min()), int(pred.max()))
    f, e = _threshold_curves(pred, gt, levels)
    s = _s_measure(levels[pred].astype(np.float64), gt)
    mae = np.abs(LEVELS[pred] - LEVELS[gt]).mean(dtype=np.float64)
    return np.concatenate((f, e, [s, mae]))


def reduce_metrics(mean_values):
    """Summarize a mean curve (single image, dataset or resampled dataset)."""
    f = mean_values[..., :N_THRESHOLDS]
    e = mean_values[..., N_THRESHOLDS:2 * N_THRESHOLDS]
    return np.stack((f.max(axis=-1), f.mean(axis=-1), mean_values[..., -2],
                     e.max(axis=-1), e.mean(axis=-1), mean_values[..., -1]), axis=-1)


@dataclass(frozen=True)
class MetricRows:
    ids: tuple[str, ...]
    values: np.ndarray

    def overall(self):
        return dict(zip(METRICS, reduce_metrics(self.values.mean(axis=0)).tolist(), strict=True))

    def per_image(self):
        return reduce_metrics(self.values)

    def save(self, destination, identity):
        destination.mkdir(parents=True, exist_ok=True)
        seal_json(destination / "identity.json", {"protocol": METRIC_PROTOCOL, **identity})
        with atomic_file(destination / "per_image_curves.npz") as stream:
            np.savez_compressed(stream, ids=np.asarray(self.ids), scores=self.values, thresholds=THRESHOLDS)
        rows = [{"sample_id": identifier, **dict(zip(METRICS, values.tolist(), strict=True))}
                for identifier, values in zip(self.ids, self.per_image(), strict=True)]
        write_csv(destination / "per_image.csv", rows)
        means = self.values.mean(axis=0)
        write_csv(destination / "threshold_curves.csv", [
            {"threshold": float(threshold), "F": f, "E": e}
            for threshold, f, e in zip(THRESHOLDS, means[:N_THRESHOLDS],
                                       means[N_THRESHOLDS:2 * N_THRESHOLDS], strict=True)
        ])
        seal_json(destination / "overall.json", {
            "protocol": METRIC_PROTOCOL, "count": len(self.ids), "metrics": self.overall(),
        })


def evaluate_method(samples, test_root, load_prediction):
    values = []
    for sample in samples:
        try:
            values.append(measure(load_prediction(sample), read_input(test_root, sample, "gt")))
        except (ValueError, OSError) as error:
            raise ValueError(f"evaluation failed for {sample.sample_id}: {error}") from error
    return MetricRows(tuple(sample.sample_id for sample in samples), np.stack(values))
