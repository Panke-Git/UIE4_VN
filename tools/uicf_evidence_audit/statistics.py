"""Independent aggregation of saved numbers; no image or model imports."""
from __future__ import annotations

from collections import Counter
import math
import statistics

from . import MEAN_ATOL


def number(value):
    if value is None or value == "":
        return None
    try:
        if isinstance(value, bool):
            raise ValueError("Boolean is not a numeric observation")
        result = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid numeric value: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite numeric value: {value!r}")
    return result


def describe(rows, field):
    values, missing, invalid = [], 0, 0
    for row in rows:
        try:
            value = number(row.get(field))
            if value is None:
                missing += 1
            else:
                values.append(value)
        except ValueError:
            invalid += 1
    return dict(mean=math.fsum(values)/len(values) if values else None,
                std=statistics.pstdev(values) if values else None,
                median=statistics.median(values) if values else None,
                valid_count=len(values), invalid_count=missing+invalid,
                missing_count=missing, nonfinite_or_malformed_count=invalid)


def equal_number(first, second, atol=MEAN_ATOL):
    try:
        a, b = number(first), number(second)
        return a is None and b is None or (
            a is not None and b is not None and abs(a-b) <= atol)
    except ValueError:
        return False


def compare_saved(audit, section, name, actual, saved, source, *, atol=MEAN_ATOL):
    if saved is None or actual is None:
        return audit.check(section, name, "UNVERIFIED", f"缺少可比较值：重算={actual!r}; 保存={saved!r}", [source])
    ok = equal_number(actual, saved, atol)
    return audit.check(section, name, "PASS" if ok else "FAIL",
        f"重算={actual!r}; 保存={saved!r}; absolute_tolerance={atol}", [source], "P0" if not ok else "P1")


def check_rows(audit, section, rows, expected, source, *, allow_variants=False):
    if rows is None:
        return "UNVERIFIED"
    keys = [(r.get("sample_id"), r.get("variant")) if allow_variants else r.get("sample_id") for r in rows]
    duplicates = [str(k) for k, count in Counter(keys).items() if count > 1]
    audit.check(section, "unique_sample_variant" if allow_variants else "unique_sample_id",
        "FAIL" if duplicates or any(not r.get("sample_id") for r in rows) else "PASS",
        f"行数={len(rows)}, 重复键={duplicates[:8]}", [source], "P0")
    if expected is None:
        audit.check(section, "sample_membership", "UNVERIFIED", "缺少冻结 test manifest", [source])
    else:
        expected_by_id = {r[0]: r[1].replace("\\", "/").rsplit("/", 1)[-1] for r in expected}
        actual = {r.get("sample_id") for r in rows}
        missing, extra = set(expected_by_id)-actual, actual-set(expected_by_id)
        status = "FAIL" if extra else ("UNVERIFIED" if missing else "PASS")
        audit.check(section, "sample_membership", status,
            f"缺少={len(missing)}, 额外={len(extra)}; 示例={sorted(str(x) for x in missing|extra)[:8]}", [source], "P0")
        mismatch = [r.get("sample_id") for r in rows if r.get("sample_id") in expected_by_id
                    and r.get("filename") != expected_by_id[r["sample_id"]]]
        audit.check(section, "sample_filename", "FAIL" if mismatch else "PASS",
                    f"文件名冲突={mismatch[:8]}", [source], "P0")
        index = {r[0]: i for i, r in enumerate(expected)}
        wrong = [r["sample_id"] for r in rows if r.get("sample_id") in index
                 and r.get("sample_index") not in (None, "", str(index[r["sample_id"]]), index[r["sample_id"]])]
        audit.check(section, "sample_index", "FAIL" if wrong else "PASS",
                    f"冻结清单索引冲突={wrong[:8]}", [source], "P0")
    return audit.status(section)


def paired_gains(original, changed, metric):
    """Positive always means original is better; align IDs, never CSV row positions."""
    first = {r["sample_id"]: r for r in original}
    second = {r["sample_id"]: r for r in changed}
    values, ids = [], []
    for row in original:
        sample_id = row["sample_id"]
        if sample_id not in second:
            continue
        try:
            a, b = number(first[sample_id].get(metric)), number(second[sample_id].get(metric))
        except ValueError:
            continue
        if a is None or b is None:
            continue
        values.append(b-a if metric == "e00" else a-b)
        ids.append(sample_id)
    return values, ids


def bootstrap(values, samples, seed):
    """Match the saved analysis's NumPy PCG64 percentile image-bootstrap protocol."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires nonempty finite paired differences")
    if not isinstance(samples, int) or isinstance(samples, bool) or not 1 <= samples <= 100000:
        raise ValueError("Bootstrap resamples must be an integer in [1,100000]")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Invalid bootstrap seed")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        stop = min(samples, start+512)
        indices = rng.integers(0, values.size, size=(stop-start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return dict(valid_count=int(values.size), mean_difference=float(values.mean()),
                ci95_low=float(low), ci95_high=float(high))
