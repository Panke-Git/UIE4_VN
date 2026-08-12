#!/usr/bin/env python3
"""Print compact comparable provenance and validation/test results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def value(data: dict, key: str) -> str:
    result = data.get(key)
    return "N/A" if result is None else str(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    columns = [
        "run", "version", "name", "seed", "best_val_loss", "best_val_psnr",
        "best_val_ssim", "test_psnr", "test_ssim", "total_params", "trainable_params",
    ]
    print("\t".join(columns))
    for raw in args.run_dirs:
        run = Path(raw).resolve()
        info = read_json(run / "run_info.json")
        status = read_json(run / "status.json")
        test = read_json(run / "result" / "test_summary.json")
        row = [
            run.name,
            value(info, "version"), value(info, "experiment_name"), value(info, "seed"),
            value(status, "best_val_loss"), value(status, "best_psnr"), value(status, "best_ssim"),
            value(test, "mean_psnr"), value(test, "mean_ssim"),
            value(info, "model_total_params"), value(info, "model_trainable_params"),
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()

