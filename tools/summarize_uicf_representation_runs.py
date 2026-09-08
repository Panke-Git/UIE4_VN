#!/usr/bin/env python3
"""Summarize existing UICF representation-analysis outputs without recomputation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v16.utils import atomic_json, project_path


FIELDS = [
    "run_name", "run_dir", "dataset", "field_variant",
    "use_spatial_conditioning", "use_global_field_conditioning",
    "use_learned_anchor", "psnr", "ssim", "e00",
    "raw_field_spearman_rgb", "raw_field_shifted_null_spearman_rgb",
    "raw_field_spearman_gain_over_null", "raw_field_top20_iou_rgb",
    "raw_field_shifted_null_top20_iou_rgb", "raw_field_spearman_e00",
    "raw_field_shifted_null_spearman_e00",
    "raw_field_spearman_gain_ci95_low", "raw_field_spearman_gain_ci95_high",
    "raw_minus_anchor_spearman_ci95_low", "raw_minus_anchor_spearman_ci95_high",
    "raw_minus_gradient_spearman_ci95_low", "raw_minus_gradient_spearman_ci95_high",
]


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _validate_finite_json(value: Any, context: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_json(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_json(child, f"{context}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"Non-finite JSON value at {context}")


def summarize_run(run_dir: Path) -> dict[str, Any]:
    analysis_dir = run_dir / "result" / "uicf_representation_alignment"
    summary_path, protocol_path = analysis_dir / "summary.json", analysis_dir / "protocol.json"
    if not summary_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(
            f"Run lacks representation summary/protocol: {run_dir}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_finite_json(summary, str(summary_path))
    _validate_finite_json(protocol, str(protocol_path))
    raw_metrics = _nested(summary, "raw_field_representation", "metrics") or {}
    raw_null = _nested(summary, "raw_field_representation", "null_controls") or {}
    bootstrap = summary.get("bootstrap", {})
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "dataset": summary.get("evaluation_dataset", summary.get("dataset")),
        "field_variant": protocol.get("field_variant", "implicit"),
        "use_spatial_conditioning": protocol.get("use_spatial_conditioning", True),
        "use_global_field_conditioning": protocol.get("use_global_field_conditioning", True),
        "use_learned_anchor": protocol.get("use_learned_anchor", True),
        "psnr": _nested(summary, "quality_metrics", "psnr", "mean"),
        "ssim": _nested(summary, "quality_metrics", "ssim", "mean"),
        "e00": _nested(summary, "quality_metrics", "e00", "mean"),
        "raw_field_spearman_rgb": _nested(raw_metrics, "raw_field_spearman_rgb", "mean"),
        "raw_field_shifted_null_spearman_rgb": _nested(raw_null, "spearman_rgb", "mean_null"),
        "raw_field_spearman_gain_over_null": _nested(raw_null, "spearman_rgb", "mean_real_minus_null"),
        "raw_field_top20_iou_rgb": _nested(raw_metrics, "raw_field_top20_iou_rgb", "mean"),
        "raw_field_shifted_null_top20_iou_rgb": _nested(raw_null, "top20_iou_rgb", "mean_null"),
        "raw_field_spearman_e00": _nested(raw_metrics, "raw_field_spearman_e00", "mean"),
        "raw_field_shifted_null_spearman_e00": _nested(raw_null, "spearman_e00", "mean_null"),
        "raw_field_spearman_gain_ci95_low": _nested(bootstrap, "raw_field_spearman_gain", "ci95_low"),
        "raw_field_spearman_gain_ci95_high": _nested(bootstrap, "raw_field_spearman_gain", "ci95_high"),
        "raw_minus_anchor_spearman_ci95_low": _nested(bootstrap, "raw_minus_anchor_spearman_rgb", "ci95_low"),
        "raw_minus_anchor_spearman_ci95_high": _nested(bootstrap, "raw_minus_anchor_spearman_rgb", "ci95_high"),
        "raw_minus_gradient_spearman_ci95_low": _nested(bootstrap, "raw_minus_gradient_spearman_rgb", "ci95_low"),
        "raw_minus_gradient_spearman_ci95_high": _nested(bootstrap, "raw_minus_gradient_spearman_rgb", "ci95_high"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", nargs="+", required=True)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)
    run_arguments = [item for group in args.run_dir for item in group]
    rows = [
        summarize_run(project_path(item).expanduser().resolve()) for item in run_arguments
    ]
    output_dir = project_path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "representation_run_comparison.csv"
    json_path = output_dir / "representation_run_comparison.json"
    _write_csv(csv_path, rows)
    atomic_json(
        json_path,
        {
            "source": "existing summary.json/protocol.json only; no image recomputation",
            "run_count": len(rows),
            "runs": rows,
        },
    )
    print(f"Wrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
