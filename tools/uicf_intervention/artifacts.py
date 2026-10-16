"""Strict serialization, conservative legacy-result audits and recoverable output."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.v16.utils import PROJECT_ROOT, atomic_json, sha256_file, tensor_to_image
from tools.analyze_uicf_alignment import _write_csv
from .core import MAIN_VARIANTS, METRICS


SCRIPT_ID = "v16_uicf_field_intervention"
SCRIPT_VERSION = "1.0"
OUTPUT_NAME = "uicf_field_intervention"


def strict_json(path: Path, value: Any) -> None:
    json.dumps(value, allow_nan=False)  # Existing atomic helper plus a strict finite gate.
    atomic_json(path, value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    _write_csv(path, rows, fields)


def safe_identity(sample_id: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")[:60] or "sample"
    return f"{label}_{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:16]}"


def model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    # Include non-persistent buffers too (e.g. positional frequency bands).
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True)
        git = {"commit": commit, "dirty": bool(status), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError) as error:
        git = {"commit": None, "dirty": None, "unavailable": str(error)}
    sources = {PROJECT_ROOT / "tools/analyze_uicf_intervention.py",
               PROJECT_ROOT / "tools/analyze_uicf_alignment.py",
               PROJECT_ROOT / "tools/visualize_v16_uicf.py"}
    for directory in ("src/shared", "src/v1/models", "src/v4", "src/v15", "src/v16",
                      "tools/uicf_intervention"):
        sources.update((PROJECT_ROOT / directory).rglob("*.py"))
    return {"git": git, "source_sha256": {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in sorted(sources)
    }}


def validate_output_path(output: Path, *, run_dir: Path, data_root: Path, overwrite: bool) -> Path:
    # Check before resolve so a symlink cannot silently redirect a dedicated path.
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError(f"Output path cannot traverse symlinks: {output}")
    output, run_dir, data_root = output.resolve(), run_dir.resolve(), data_root.resolve()
    default = run_dir / "result" / OUTPUT_NAME
    if output != default and not output.is_relative_to(default):
        if output.name != OUTPUT_NAME:
            raise ValueError(f"Custom output must have dedicated leaf name {OUTPUT_NAME!r}")
        if output.is_relative_to(run_dir) or output.is_relative_to(PROJECT_ROOT / "experiments"):
            raise ValueError("Custom output cannot target another experiment or old result directory")
    protected = (Path("/"), Path.home(), PROJECT_ROOT, run_dir, run_dir / "result", data_root)
    if any(path == output or path.is_relative_to(output) for path in protected):
        raise ValueError(f"Refusing output at/above protected directory: {output}")
    forbidden = (data_root, run_dir / "checkpoint", run_dir / "best", run_dir / "split_snapshot",
                 PROJECT_ROOT / "src", PROJECT_ROOT / "configs", PROJECT_ROOT / "split",
                 PROJECT_ROOT / "tests", PROJECT_ROOT / "tools", PROJECT_ROOT / ".git")
    if any(output.is_relative_to(path) for path in forbidden):
        raise ValueError(f"Refusing output inside protected data/source/checkpoint directory: {output}")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output is not a directory: {output}")
        if any(output.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Output is nonempty; explicitly use --overwrite: {output}")
            try:
                marker = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError(f"Refusing overwrite of unmarked directory: {output}") from error
            if marker.get("script_id") != SCRIPT_ID or marker.get("run_dir") != str(run_dir):
                raise ValueError("Overwrite marker does not belong to this tool and V16 run")
            if any(path.is_symlink() for path in output.rglob("*")):
                raise ValueError("Refusing overwrite of directory containing symlinks")
    return output


class OutputTransaction:
    """Never delete prior results: stage, retain failures, archive old success on replacement."""

    def __init__(self, output: Path, *, run_dir: Path, data_root: Path, overwrite: bool) -> None:
        self.output = output
        self.run_dir, self.data_root, self.overwrite = run_dir, data_root, overwrite
        output.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
        self.previous: Path | None = None

    def commit(self) -> None:
        # Recheck at publication time: another process may have created a result
        # during this long evaluation. Never implicitly authorize overwriting it.
        validate_output_path(self.output, run_dir=self.run_dir, data_root=self.data_root,
                             overwrite=self.overwrite)
        if self.output.exists():
            self.previous = self.output.with_name(f".{self.output.name}.previous-{uuid.uuid4().hex}")
            os.replace(self.output, self.previous)
        try:
            os.replace(self.stage, self.output)
        except BaseException:
            if self.previous is not None:
                os.replace(self.previous, self.output)
            raise

    def retain_failure(self) -> Path:
        failed = self.output.with_name(f".{self.output.name}.failed-{uuid.uuid4().hex}")
        os.replace(self.stage, failed)
        return failed


def official_result_snapshot(run_dir: Path) -> dict[str, Any]:
    paths = [run_dir / "result" / name for name in ("test_summary.json", "test_metrics.csv")]
    return {str(path): sha256_file(path) if path.is_file() else None for path in paths}


def audit_official_results(
    run_dir: Path, rows: Sequence[Mapping[str, Any]], config: dict,
    protocol: Mapping[str, Any], *, debug_partial: bool,
) -> dict[str, Any]:
    """Compare float CSV metrics only, never use PNGs as metric inputs.

    Old official test summaries omit full AMP/resize/metric provenance. A numerical
    match is reported distinctly from a fully verifiable protocol match. In the
    absence of a known mismatch, a numerical discrepancy still fails conservatively.
    """
    summary_path, csv_path = run_dir / "result/test_summary.json", run_dir / "result/test_metrics.csv"
    audit: dict[str, Any] = {"files": official_result_snapshot(run_dir)}
    if debug_partial:
        return {**audit, "status": "NOT_COMPARABLE", "reasons": ["debug_partial evaluation"]}
    if not summary_path.is_file() and not csv_path.is_file():
        return {**audit, "status": "NOT_AVAILABLE", "reasons": ["No official test results"]}
    if not summary_path.is_file() or not csv_path.is_file():
        return {**audit, "status": "NOT_COMPARABLE", "reasons": ["Official summary/CSV pair is incomplete"]}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        old_rows = list(csv.DictReader(handle))
    originals = [row for row in rows if row["variant"] == "original"]
    reasons, unknown = [], []
    if summary.get("checkpoint_epoch") != protocol["checkpoint_epoch"]:
        reasons.append("Different checkpoint epoch")
    old_checkpoint = str(summary.get("checkpoint", "")).replace("\\", "/")
    if old_checkpoint.rsplit("/", 1)[-1] != Path(protocol["checkpoint"]).name:
        reasons.append("Different/unknown checkpoint selector filename")
    for key, expected in (("checkpoint_sha256", protocol["checkpoint_sha256"]),
                          ("test_manifest_sha256", protocol["test_manifest_sha256"]),
                          ("amp_enabled", protocol["amp_enabled"]),
                          ("evaluation", config["evaluation"]), ("metrics", config["metrics"])):
        if key not in summary:
            unknown.append(f"Legacy summary has no {key}")
        elif summary[key] != expected:
            reasons.append(f"Different {key}")
    native_amp = bool(config["training"]["amp"]) and protocol["device_type"] == "cuda"
    if "amp_enabled" not in summary and native_amp != protocol["amp_enabled"]:
        reasons.append("AMP override differs from native test config")
    if summary.get("e00_protocol") != protocol["e00_protocol"]:
        reasons.append("Different/missing E00 protocol (including implementation version)")
    if summary.get("sample_count") != len(originals) or len(old_rows) != len(originals):
        reasons.append("Different sample count")
    ids = [row.get("sample_id") for row in old_rows]
    if len(set(ids)) != len(ids) or set(ids) != {row["sample_id"] for row in originals}:
        reasons.append("Different/duplicate sample IDs")
    by_id = {row.get("sample_id"): row for row in old_rows}
    if not reasons and any(by_id[row["sample_id"]].get("filename") != row["filename"] for row in originals):
        reasons.append("Different sample filenames")
    if reasons:
        return {**audit, "status": "NOT_COMPARABLE", "reasons": reasons, "unverified_metadata": unknown}
    tolerances = {"psnr": 1e-3, "ssim": 1e-4, "e00": 1e-3} if protocol["amp_enabled"] else {
        "psnr": 1e-5, "ssim": 1e-6, "e00": 1e-5,
    }
    errors: dict[str, Any] = {}
    failures = []
    for metric in METRICS:
        try:
            previous = np.asarray([float(by_id[row["sample_id"]][metric]) for row in originals])
            current = np.asarray([row[metric] for row in originals], dtype=np.float64)
            old_mean = float(summary[f"mean_{metric}"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Malformed official {metric} results: {error}") from error
        if not np.isfinite(previous).all() or not math.isfinite(old_mean):
            raise FloatingPointError(f"Official {metric} contains non-finite values")
        max_error = float(np.abs(previous - current).max())
        mean_error = abs(old_mean - float(current.mean()))
        if abs(old_mean - float(previous.mean())) > tolerances[metric]:
            failures.append(f"Official {metric} summary disagrees with its own CSV")
        if max_error > tolerances[metric] or mean_error > tolerances[metric]:
            failures.append(f"{metric} exceeds tolerance {tolerances[metric]}")
        errors[metric] = {"max_per_image_error": max_error, "dataset_mean_error": mean_error,
                          "absolute_tolerance": tolerances[metric]}
    audit.update({"status": "FAIL" if failures else (
        "NUMERIC_MATCH_LEGACY_METADATA_INCOMPLETE" if unknown else "PASS"),
        "reasons": failures, "unverified_metadata": unknown, "errors": errors})
    return audit


def save_visualization(
    directory: Path, inputs: torch.Tensor, target: torch.Tensor,
    predictions: Mapping[str, torch.Tensor], details: Any,
    scores: Mapping[str, Mapping[str, float]],
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    items = [("Input", inputs[0], None), ("GT", target[0], None)] + [
        (variant, predictions[variant][0], scores[variant]) for variant in MAIN_VARIANTS
    ]
    width = max(224, inputs.shape[-1])
    height = max(160, round(inputs.shape[-2] * width / inputs.shape[-1]))
    label_height, gap = 44, 6
    canvas = Image.new("RGB", (width * 3 + gap * 2, (height + label_height) * 2 + gap), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, tensor, score) in enumerate(items):
        x, y = (index % 3) * (width + gap), (index // 3) * (height + label_height + gap)
        image = tensor_to_image(tensor)
        image.save(directory / f"{name.lower()}.png")
        canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (x, y + label_height))
        draw.text((x + 5, y + 4), name, fill="black")
        if score is not None:
            draw.text((x + 5, y + 22),
                      f"P {score['psnr']:.3f}  S {score['ssim']:.4f}  E00 {score['e00']:.3f}", fill="black")
    canvas.save(directory / "comparison.png", dpi=(300, 300))
    # These are raw tensors for audit, not normalized heatmaps or PNG round-trips.
    np.save(directory / "R_original.npy", details.correction_field[0].float().cpu().numpy(), allow_pickle=False)
    np.save(directory / "Ic_original.npy", details.enhanced[0].float().cpu().numpy(), allow_pickle=False)
    strict_json(directory / "metadata.json", {"scores": scores, "display_only_clamp": True,
                                              "metrics_from": "float prediction before PNG encoding"})


def write_paper_tables(directory: Path, summary: Mapping[str, Any]) -> None:
    if not summary["paper_ready"]:
        raise RuntimeError("Refusing paper tables for an incomplete/debug/failed experiment")
    rows = [{"variant": name, **summary["variants"][name]} for name in MAIN_VARIANTS]
    write_csv(directory / "paper_table.csv", rows,
              ["variant", "sample_count", "mean_psnr", "mean_ssim", "mean_e00"])
    lines = ["| Variant | PSNR ↑ | SSIM ↑ | ΔE00 ↓ |", "|---|---:|---:|---:|"]
    lines += [f"| {row['variant']} | {row['mean_psnr']:.4f} | {row['mean_ssim']:.6f} | {row['mean_e00']:.4f} |"
              for row in rows]
    lines += ["", "Observational intervention on a frozen model; not a retrained ablation or physical causal proof.",
              "Paired positive gains favor original. Image-level CIs crossing zero indicate uncertainty."]
    (directory / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
