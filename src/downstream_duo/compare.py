"""Validate paired protocol and write dataset-level Raw/Full deltas."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .utils import fresh_dir, output_path, read_json, sha256, write_json

METRICS = ("precision", "recall", "map50", "map50_95")


def validate_pair(raw, full):
    if raw["condition"] != "raw" or full["condition"] != "full":
        raise ValueError("Expected A=Raw, B=Full")
    for key in ("identity", "weights", "weights_sha256", "versions", "parameters", "test_count", "synthetic"):
        if raw[key] != full[key]:
            raise ValueError(f"Unfair or mismatched pair: {key}")
    if sha256(raw["weights"]) != raw["weights_sha256"]:
        raise ValueError("Registered detector weights changed")
    if [c["category_id"] for c in raw["per_class"]] != [c["category_id"] for c in full["per_class"]]:
        raise ValueError("Different per-class mappings")
    expected = [(r["image_id"], r["file"], r["width"], r["height"]) for r in raw["identity"]["test"]]
    for result in (raw, full):
        if sha256(result["predictions"]) != result["predictions_sha256"]:
            raise ValueError("Prediction export was changed")
        actual = [(r["image_id"], r["file"], r["width"], r["height"]) for r in read_json(result["predictions"])]
        if actual != expected:
            raise ValueError("Prediction image IDs or geometry differ from official test manifest")


def compare(raw_path, full_path, out):
    raw, full = read_json(raw_path), read_json(full_path)
    validate_pair(raw, full)
    out = fresh_dir(out)
    rows = []
    for label, a, b in [("all", raw, full)] + [
        (a["name"], a, b) for a, b in zip(raw["per_class"], full["per_class"])
    ]:
        for key in METRICS:
            av, bv = a[key], b[key]
            rows.append({"class": label, "metric": key, "raw": av, "full": bv,
                         "full_minus_raw": bv-av if av is not None and bv is not None else None})
    result = {"synthetic": raw["synthetic"], "test_count": raw["test_count"],
              "weights": raw["weights"], "weights_sha256": raw["weights_sha256"],
              "identity": raw["identity"], "parameters": raw["parameters"], "rows": rows,
              "interpretation": "Frozen Raw-trained detector downstream utility; enhancement need not improve mAP. No cross-degradation generalization claim.",
              "raw_metrics": str(Path(raw_path).resolve()), "full_metrics": str(Path(full_path).resolve())}
    write_json(out / "comparison.json", result)
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["SYNTHETIC SOFTWARE SMOKE — NOT PAPER RESULTS" if raw["synthetic"] else "DUO entire official test / frozen detector",
             "", f"Images: {raw['test_count']}; all values 0–1; delta is absolute B−A.", "",
             "| Class | Metric | A Raw | B ICAR-UIE | B−A |", "|---|---|---:|---:|---:|"]
    def fmt(v):
        return "N/A (no GT)" if v is None else f"{v:.6f}"
    for row in rows:
        lines.append(f"| {row['class']} | {row['metric']} | {fmt(row['raw'])} | {fmt(row['full'])} | {fmt(row['full_minus_raw'])} |")
    lines.extend(["", result["interpretation"], "", raw["metric_definition"]])
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--full", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    compare(a.raw, a.full, output_path(a.output))


if __name__ == "__main__":
    main()
