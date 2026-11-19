"""Image-paired comparisons with dataset-curve bootstrap estimates."""

import numpy as np

from .evaluation import METRICS, reduce_metrics
from .storage import write_csv, write_json

LOWER_IS_BETTER = np.asarray([False, False, False, False, False, True])
TIE_ATOL = 1e-12


def paired_bootstrap(raw, enhanced, replicates, seed):
    if raw.ids != enhanced.ids or len(set(raw.ids)) != len(raw.ids):
        raise ValueError("paired bootstrap requires identical ordered, unique sample IDs")
    if replicates < 2:
        raise ValueError("paired bootstrap requires at least two resamples")
    count = len(raw.ids)
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(METRICS)), dtype=np.float64)
    probability = np.full(count, 1 / count)
    for start in range(0, replicates, 64):
        # Multinomial weights are equivalent to drawing N common image indices with replacement.
        size = min(replicates - start, 64)
        weights = rng.multinomial(count, probability, size=size).astype(np.float64) / count
        enhanced_mean = np.einsum("bi,ij->bj", weights, enhanced.values)
        raw_mean = np.einsum("bi,ij->bj", weights, raw.values)
        draws[start:start + size] = reduce_metrics(enhanced_mean) - reduce_metrics(raw_mean)
    delta = reduce_metrics(enhanced.values.mean(axis=0)) - reduce_metrics(raw.values.mean(axis=0))
    interval = np.percentile(draws, [2.5, 97.5], axis=0)
    per_image = enhanced.per_image() - raw.per_image()
    gains = per_image * np.where(LOWER_IS_BETTER, -1, 1)
    improved = (gains > TIE_ATOL).sum(axis=0)
    worse = (gains < -TIE_ATOL).sum(axis=0)
    records = []
    for i, name in enumerate(METRICS):
        records.append({
            "metric": name, "enhanced_minus_raw": float(delta[i]),
            "ci95_low": float(interval[0, i]), "ci95_high": float(interval[1, i]),
            "better": int(improved[i]), "tied": int(count - improved[i] - worse[i]),
            "worse": int(worse[i]), "improved_fraction": float(improved[i] / count),
        })
    return records


def compare_methods(methods, output, replicates, seed):
    raw = methods["raw"]
    table, pairs = [], []
    for method, rows in methods.items():
        if method == "raw":
            continue
        table.extend({"method": method, **record}
                     for record in paired_bootstrap(raw, rows, replicates, seed))
        delta = rows.per_image() - raw.per_image()
        for identifier, values in zip(rows.ids, delta, strict=True):
            pairs.append({"method": method, "sample_id": identifier,
                          **dict(zip(METRICS, values.tolist(), strict=True))})
    write_csv(output / "paired_bootstrap.csv", table)
    write_csv(output / "paired_per_image.csv", pairs)
    write_json(output / "paired_bootstrap.json", {
        "n": len(raw.ids), "replicates": replicates, "seed": seed,
        "interval": "paired image bootstrap, 2.5/97.5 percentile",
        "delta": "condition minus Raw; negative MAE delta is improvement",
        "per_image_improvement_tie_atol": TIE_ATOL, "rows": table,
    })
    return table
