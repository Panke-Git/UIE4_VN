"""Read-only native V16/QGRD result comparison with strict frozen-protocol audit.

No model imports, checkpoint loading, evaluation, or experiment writes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("no_qgrd", "shared_query", "single_stage", "no_local_refine", "v16")
LABELS = {"no_qgrd": "w/o QGRD", "shared_query": "Shared Query",
          "single_stage": "Single-stage", "no_local_refine": "w/o Local Refine",
          "v16": "Full V16"}
TYPES = {
    "v16": "plain_unet_color_query_uicf_pre_backbone",
    "no_qgrd": "plain_unet_uicf_pre_backbone_no_qgrd",
    "shared_query": "plain_unet_color_query_uicf_pre_backbone_shared_query",
    "single_stage": "plain_unet_color_query_uicf_pre_backbone_single_stage",
    "no_local_refine": "plain_unet_color_query_uicf_pre_backbone_no_local_refine",
}
PROTOCOLS = {"lsui": {"dataset": "LSUI19", "counts": {"train": 3466, "validation": 385, "test": 428}},
             "uieb": {"dataset": "UIEB", "counts": {"train": 720, "validation": 80, "test": 90}}}


def audit_evaluation_precision(config, info, summary, role):
    # Metadata audit only; never import a model/evaluation runner here.
    explicit = config["evaluation"].get("precision")
    precision = explicit if explicit is not None else (
        "native_amp" if config["training"]["amp"] else "fp32"
    )
    if precision not in {"fp32", "native_amp"}:
        raise ValueError(f"{role}: unknown evaluation_precision {precision!r}")
    for label, record in (("run_info", info), ("test_summary", summary)):
        declared = record.get("evaluation_precision")
        if explicit is not None and declared != precision:
            raise ValueError(f"{role}: {label} evaluation_precision missing or inconsistent")
        if declared is not None and declared != precision:
            raise ValueError(f"{role}: {label} evaluation_precision differs from checkpoint-selection config")
        if precision == "fp32" and explicit is not None:
            expected = {
                "precision": "fp32", "autocast_enabled": False,
                "input_dtype": "torch.float32", "prediction_dtype": "torch.float32",
                "loss_dtype": "torch.float32", "model_weights_changed_for_evaluation": False,
            }
            if record.get("evaluation_forward_protocol") != expected:
                raise ValueError(f"{role}: {label} FP32 evaluation forward protocol inconsistent")
    return precision


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_name(data):
    explicit = str(data.get("dataset", "")).strip()
    if explicit:
        if explicit not in {"LSUI19", "UIEB"}:
            raise ValueError(f"Unsupported dataset: {explicit!r}")
        return explicit
    metadata = [str(data.get(k, "")).replace("\\", "/").lower()
                for k in ("root", "train_manifest", "validation_manifest", "test_manifest")]
    lsui = any("lsui19" in p or "/lsui/" in p for p in metadata) or "lsui" in metadata[0]
    uieb = any("uieb" in p for p in metadata)
    if lsui == uieb:
        raise ValueError("Legacy dataset metadata is unknown or conflicting")
    return "LSUI19" if lsui else "UIEB"


def _manifest(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if rows and tuple(x.lower().strip() for x in rows[0]) in {
        ("sample_id", "input_path", "gt_path"), ("id", "input", "gt")
    }:
        rows = rows[1:]
    if not rows or any(len(row) != 3 or any(not x.strip() for x in row) for row in rows):
        raise ValueError(f"Invalid frozen TSV: {path}")
    for column in range(3):
        if len({r[column] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate frozen manifest identity: {path}")
    return rows


def audit_run(path: Path, role: str, dataset: str):
    config = yaml.safe_load((path / "config_resolved.yaml").read_text(encoding="utf-8"))
    info = read_json(path / "run_info.json")
    summary = read_json(path / "result/test_summary.json")
    expected_version = "v16" if role == "v16" else "qgrd_" + role
    if config["experiment"]["version"] != expected_version or info["version"] != expected_version:
        raise ValueError(f"{role}: run version is not {expected_version}")
    if config["model"]["type"] != TYPES[role]:
        raise ValueError(f"{role}: model type mismatch")
    resolved = info.get("resolved_config")
    if resolved is None or resolved != config:
        raise ValueError(f"{role}: run_info resolved_config differs from frozen config")
    protocol = PROTOCOLS[dataset]
    if dataset_name(config["data"]) != protocol["dataset"]:
        raise ValueError(f"{role}: dataset differs from requested {dataset}")
    if "dataset" in summary and summary["dataset"] != protocol["dataset"]:
        raise ValueError(f"{role}: summary dataset mismatch")
    manifests = {s: path / "split_snapshot" / (s + ".tsv") for s in ("train", "validation", "test")}
    entries = {s: _manifest(p) for s, p in manifests.items()}
    counts = {s: len(r) for s, r in entries.items()}
    if counts != protocol["counts"] or config["data"].get("expected_counts", counts) != counts:
        raise ValueError(f"{role}: not the canonical three-set counts: {counts}")
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for column in range(3):
            if {r[column] for r in entries[first]} & {r[column] for r in entries[second]}:
                raise ValueError(f"{role}: frozen split leakage {first}/{second}")
    hashes = {s: sha256(p) for s, p in manifests.items()}
    for s, digest in hashes.items():
        if info.get(s + "_tsv_sha256") != digest:
            raise ValueError(f"{role}: {s} manifest hash differs from run_info")
        if info.get(s + "_sample_count") != counts[s]:
            raise ValueError(f"{role}: {s} count differs from run_info")
    if summary.get("split_sha256", hashes) != hashes:
        raise ValueError(f"{role}: summary manifest hashes mismatch")
    checkpoint = summary.get("checkpoint", "")
    selector = summary.get("checkpoint_selector", Path(checkpoint).stem)
    if selector != "best_psnr" or Path(checkpoint).name != "best_psnr.pt":
        raise ValueError(f"{role}: checkpoint selector must be best_psnr")
    # Legacy V16 did not serialize metric_input; its E00 provenance explicitly
    # identifies pre-PNG float sRGB. Do not accept PNG8 summaries as native results.
    e00 = summary.get("e00_protocol")
    if not isinstance(e00, dict) or e00.get("rgb_interpretation") != "sRGB float prediction clamped to [0,1] before PNG encoding":
        raise ValueError(f"{role}: missing/non-native float E00 protocol")
    if summary.get("metric_input", "float_rgb_before_png_save") != "float_rgb_before_png_save":
        raise ValueError(f"{role}: PNG8/non-native metric input cannot be compared")
    if summary.get("metric_config", config["metrics"]) != config["metrics"]:
        raise ValueError(f"{role}: summary metric protocol mismatch")
    if summary.get("evaluation", config["evaluation"]) != config["evaluation"]:
        raise ValueError(f"{role}: summary evaluation resize/size mismatch")
    precision = audit_evaluation_precision(config, info, summary, role)
    with (path / "result/test_metrics.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"filename", "sample_id", "psnr", "ssim", "e00"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"{role}: test_metrics lacks native PSNR/SSIM/E00 fields")
        raw = list(reader)
    by_id = {}
    expected_files = {row[0]: Path(row[1]).name for row in entries["test"]}
    for row in raw:
        sample = row["sample_id"]
        if sample in by_id:
            raise ValueError(f"{role}: duplicate test sample ID {sample!r}")
        if expected_files.get(sample) != row["filename"]:
            raise ValueError(f"{role}: sample ID/filename differs from frozen test TSV: {sample}")
        values = {m: float(row[m]) for m in ("psnr", "ssim", "e00")}
        if not all(math.isfinite(v) for v in values.values()) or values["e00"] < 0:
            raise ValueError(f"{role}: non-finite/negative color metrics; inspect individual results")
        by_id[sample] = values
    if len(by_id) != counts["test"] or set(by_id) != set(expected_files):
        raise ValueError(f"{role}: test sample count/IDs differ from frozen manifest")
    if summary["sample_count"] != len(by_id):
        raise ValueError(f"{role}: summary sample_count mismatch")
    means = {m: math.fsum(r[m] for r in by_id.values()) / len(by_id) for m in ("psnr", "ssim", "e00")}
    for metric, value in means.items():
        if not math.isclose(value, float(summary["mean_" + metric]), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"{role}: summary mean_{metric} differs from all per-image rows")
    fairness = {
        "dataset": protocol["dataset"], "split_sha256": hashes, "counts": counts,
        "test_sample_ids": sorted(by_id), "metrics": config["metrics"],
        "evaluation": config["evaluation"], "e00_protocol": e00,
        "evaluation_precision": precision,
        "checkpoint_selector": selector, "metric_input": "float_rgb_before_png_save",
        "seed": config["experiment"]["seed"],
        "training_protocol": {k: config[k] for k in ("loss", "optimizer", "scheduler", "training", "checkpoint")},
        "data_preprocessing": {k: config["data"][k] for k in (
            "patch_size", "batch_size", "num_workers", "pin_memory", "pad_if_smaller", "pad_mode", "augmentation")},
        "common_model_config": {k: v for k, v in config["model"].items() if k != "type"},
    }
    return {"run_dir": str(path), "means": means, "rows": by_id, "fairness": fairness,
            "checkpoint": checkpoint, "checkpoint_epoch": summary["checkpoint_epoch"]}


def compare(runs, dataset):
    audited = {}
    errors = []
    for role, path in runs.items():
        try:
            audited[role] = audit_run(path, role, dataset)
        except (ValueError, KeyError, TypeError, OSError, csv.Error, yaml.YAMLError) as error:
            errors.append(f"{role}: {error}")
    if errors:
        raise ValueError("Protocol audit refused comparison:\n" + "\n".join(errors))
    baseline = audited["v16"]["fairness"]
    for role in VARIANTS[:-1]:
        differing = [k for k in baseline if audited[role]["fairness"][k] != baseline[k]]
        if differing:
            errors.append(f"{role} vs Full V16: mismatch in {', '.join(differing)}")
    if errors:
        raise ValueError("Protocol audit refused comparison:\n" + "\n".join(errors))
    comparison = []
    deltas = []
    for role in VARIANTS:
        means = audited[role]["means"]
        comparison.append({"variant": LABELS[role], "psnr": means["psnr"], "ssim": means["ssim"], "e00": means["e00"],
                           **{"delta_" + m: means[m] - audited["v16"]["means"][m] for m in means}})
        if role != "v16":
            for sample in baseline["test_sample_ids"]:
                deltas.append({"variant": LABELS[role], "sample_id": sample,
                               **{"delta_" + m: audited[role]["rows"][sample][m] - audited["v16"]["rows"][sample][m]
                                  for m in means}})
    audit = {"status": "PASS", "dataset": baseline["dataset"], "protocol": baseline,
             "runs": {r: {k: v for k, v in a.items() if k not in ("rows", "fairness")} for r, a in audited.items()},
             "interpretation": "Paired test-image differences for these fixed trained models; not cross-seed evidence."}
    return comparison, deltas, audit


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path, comparison, deltas, audit):
    # Audit completes before any output. Failed regeneration retains a prior report.
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".qgrd-report-", dir=output.parent))
    backup = None
    try:
        write_csv(stage / "comparison.csv", comparison)
        write_csv(stage / "per_image_deltas.csv", deltas)
        (stage / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
        (stage / "protocol_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        lines = ["# QGRD native test comparison", "", "PSNR ↑ / SSIM ↑ / ΔE00 ↓. Deltas are variant minus Full V16.", "",
                 "| Variant | PSNR↑ | SSIM↑ | ΔE00↓ | ΔPSNR | ΔSSIM | ΔE00 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for row in comparison:
            lines.append("| " + row["variant"] + " | " + " | ".join(f"{row[k]:.6f}" for k in
                         ("psnr", "ssim", "e00", "delta_psnr", "delta_ssim", "delta_e00")) + " |")
        lines += ["", "All frozen Train/Validation/Test hashes, IDs, counts, native metric and training protocols passed.",
                  "These paired differences describe fixed trained models, not stability across training seeds.", ""]
        (stage / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=".qgrd-report-backup-", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except BaseException:
            if backup is not None:
                os.replace(backup, output)
                backup = None
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("lsui", "uieb"), required=True)
    parser.add_argument("--baseline-run", required=True)
    for variant in VARIANTS[:-1]:
        parser.add_argument("--" + variant.replace("_", "-") + "-run", required=True)
    args = parser.parse_args(argv)
    runs = {"v16": Path(args.baseline_run).resolve(),
            **{v: Path(getattr(args, v + "_run")).resolve() for v in VARIANTS[:-1]}}
    try:
        comparison, deltas, audit = compare(runs, args.dataset)
    except ValueError as error:
        parser.exit(2, str(error) + "\n")
    output = PROJECT_ROOT / "reports/qgrd_ablations" / args.dataset
    write_report(output, comparison, deltas, audit)
    print(json.dumps({"status": "PASS", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
