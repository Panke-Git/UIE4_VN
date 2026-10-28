"""Explicit WSL2 host migration: preserve originals, record platform change,
then replay the frozen Raw detector before permitting a new P1 plan.

Only the platform string may differ. Package/Python versions, data hashes,
weights, enhancement protocol and detector parameters are never waived.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time
import traceback

from ..utils import load_prepared, pair_identity, read_json, sha256, versions, write_json
from .common import fresh_output, runtime_dirs, snapshot, check_snapshot, METRICS


def platform_only_changes(original, current):
    differences = {k: {"original": original.get(k), "current": current.get(k)}
                   for k in sorted(set(original) | set(current)) if original.get(k) != current.get(k)}
    if set(differences) - {"platform"}:
        raise ValueError(f"Install original package/Python versions first; only platform may change: {differences}")
    return differences


def is_wsl2():
    return platform.system() == "Linux" and "microsoft" in (platform.release() + platform.version()).lower()


def checked_registry(prepared, weights, meta):
    registry = read_json(Path(prepared) / "detector_training_lock.json")
    if registry["status"] != "complete" or registry["identity"] != pair_identity(meta):
        raise ValueError("Completed original baseline and identical data/GT are required")
    if Path(registry["weights"]).resolve() != Path(weights).resolve() or sha256(weights) != registry["weights_sha256"]:
        raise ValueError("Restore the original absolute paths and exact Raw best.pt bytes")
    if Path(registry["prepared"]).resolve() != Path(prepared).resolve():
        raise ValueError("This migration preserves original Linux paths; it does not rewrite metadata")
    platform_only_changes(registry["versions"], versions())
    return registry


def create(prepared, weights, enhanced, p0_run, output, *, _synthetic=False):
    if not _synthetic and not is_wsl2():
        raise ValueError("This explicit migration route supports WSL2 Linux, not native Windows")
    prepared, weights, enhanced, p0_run = [Path(x).resolve() for x in (prepared, weights, enhanced, p0_run)]
    meta = load_prepared(prepared)
    if bool(meta["synthetic"]) != _synthetic:
        raise ValueError("Production migration refuses synthetic data")
    registry = checked_registry(prepared, weights, meta)
    from ..model_inference_v16 import validate_enhanced
    full = validate_enhanced(prepared, enhanced, meta)
    audit = read_json(p0_run / "audit/audit_protocol.json")
    identity = read_json(p0_run / "identity/identity_control.json")
    runner = read_json(p0_run / "runner_status.json")
    integrity = read_json(p0_run / "input_integrity_after.json")
    if audit["status"] != "pass" or identity["status"] != "pass" or integrity["status"] != "pass" or runner["status"] != "completed_with_explicit_native_eligibility":
        raise ValueError("Need the completed original P0 evidence, not a stopped audit")
    if any(Path(audit["paths"][key]).resolve() != value for key, value in
           (("prepared", prepared), ("weights", weights), ("enhanced", enhanced))):
        raise ValueError("Restore original P0 Linux paths without changing its JSON")
    uie = Path(full["provenance"]["run_dir"])
    initial = Path(registry["initial_model"]).resolve()
    assets = [prepared / "split_metadata.json", prepared / "detector_training_lock.json", weights,
              enhanced / "enhancement_manifest.json", uie / "best/best_psnr.pt", uie / "config_resolved.yaml",
              initial, p0_run / "audit/audit_protocol.json", p0_run / "identity/identity_control.json",
              p0_run / "runner_status.json", p0_run / "input_integrity_after.json"]
    if sha256(initial) != registry.get("initial_model_sha256"):
        raise ValueError("Restore the exact original COCO initialization too")
    if sha256(uie / "best/best_psnr.pt") != full["provenance"]["checkpoint_sha256"] or sha256(uie / "config_resolved.yaml") != full["provenance"]["config_sha256"]:
        raise ValueError("Original V16 checkpoint/config bytes changed")
    out = fresh_output(output, [prepared, weights, enhanced, p0_run, uie, initial])
    runtime_dirs(out)
    print("WSL migration: hashing restored originals; this may take several minutes", flush=True)
    write_json(out / "input_integrity_before.json", snapshot([prepared, enhanced, *assets]))
    import torch
    hardware = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                "devices": [{"name": torch.cuda.get_device_name(i), "total_bytes": torch.cuda.get_device_properties(i).total_memory}
                            for i in range(torch.cuda.device_count())]}
    value = {"schema": 1, "kind": "wsl2-platform-only", "synthetic": _synthetic,
        "prepared": str(prepared), "raw_weights": str(weights), "enhanced_test": str(enhanced), "p0_run": str(p0_run),
        "identity": pair_identity(meta), "historical_detector_versions": registry["versions"], "target_detector_versions": versions(),
        "differences": platform_only_changes(registry["versions"], versions()),
        "target_hardware": hardware,
        "asset_hashes": {str(p): sha256(p) for p in assets}, "historical_metrics": audit["existing_metrics"],
        "note": "Original locks/JSON/checkpoints remain unchanged. Target platform is recorded separately; no package or protocol waiver.",
        "created_unix": time.time()}
    write_json(out / "migration_record.json", value)
    (out / "migration_record.sha256").write_text(sha256(out / "migration_record.json") + "\n")
    return out / "migration_record.json"


def verify_record(record_path, prepared, weights, *, runtime_check=True, require_recheck=False):
    record_path = Path(record_path).resolve()
    from .common import DIAGNOSTICS
    if not record_path.is_relative_to(DIAGNOSTICS.resolve()) or sha256(record_path) != record_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Missing/changed explicit WSL migration record")
    value = read_json(record_path)
    if value["kind"] != "wsl2-platform-only" or Path(value["prepared"]).resolve() != Path(prepared).resolve() or Path(value["raw_weights"]).resolve() != Path(weights).resolve():
        raise ValueError("Migration record belongs to different paths/weights")
    platform_only_changes(value["historical_detector_versions"], value["target_detector_versions"])
    if runtime_check:
        if not value["synthetic"] and not is_wsl2():
            raise ValueError("Recorded WSL2 execution required")
        if versions() != value["target_detector_versions"]:
            raise ValueError("Target environment changed after migration record")
    for path, expected in value["asset_hashes"].items():
        if sha256(path) != expected:
            raise ValueError(f"Original migration asset changed: {path}")
    meta = load_prepared(prepared, verify=False)
    registry = read_json(Path(prepared) / "detector_training_lock.json")
    if registry["identity"] != value["identity"] or pair_identity(meta) != value["identity"] or registry["weights_sha256"] != sha256(weights):
        raise ValueError("Original baseline/data identity changed")
    if require_recheck:
        replay = read_json(record_path.parent / "baseline_recheck.json")
        if replay["status"] != "pass" or replay["migration_record_sha256"] != sha256(record_path) or replay["input_integrity"]["status"] != "pass":
            raise ValueError("Frozen baseline recheck must pass on the new host before P1")
        for condition, expected in replay["metric_file_hashes"].items():
            if sha256(record_path.parent / "baseline" / condition / f"metrics_{condition}.json") != expected:
                raise ValueError("Migration baseline metrics changed")
    return registry, value


def recheck(record_path, gpu="0"):
    path = Path(record_path).resolve()
    initial = read_json(path)
    registry, value = verify_record(path, initial["prepared"], initial["raw_weights"])
    from .evaluator import frozen_evaluate
    meta = load_prepared(value["prepared"])
    current, deltas = {}, {}
    for condition in ("raw", "full"):
        out = path.parent / "baseline" / condition
        image_root = Path(value["prepared"]) / "images/test_raw" if condition == "raw" else Path(value["enhanced_test"]) / "images/test_full"
        metric_path = out / f"metrics_{condition}.json"
        if metric_path.is_file():
            result = read_json(metric_path)
            if result["migration_record_sha256"] != sha256(path) or result["predictions_sha256"] != sha256(out / "per_image_predictions.json"):
                raise ValueError("Existing migration recheck changed")
        else:
            result = frozen_evaluate(value["prepared"], value["raw_weights"], image_root, meta["splits"]["test"],
                                     out, condition, value["historical_metrics"][condition], gpu, migration_record=path)
        if result["identity"] != value["identity"] or result["weights_sha256"] != registry["weights_sha256"] or result["parameters"] != registry["evaluation"] or result["versions"] != value["target_detector_versions"] or result["test_count"] != len(meta["splits"]["test"]):
            raise ValueError("Migration baseline protocol/coverage differs")
        current[condition] = result
        deltas[condition] = {k: result[k]-value["historical_metrics"][condition][k] for k in METRICS}
    guard = check_snapshot(read_json(path.parent / "input_integrity_before.json"))
    passed = guard["status"] == "pass" and all(abs(deltas[c][k]) <= .005 for c in deltas for k in ("map50", "map50_95"))
    summary = {"status": "pass" if passed else "stop", "migration_record_sha256": sha256(path),
        "historical_metrics": value["historical_metrics"], "target_metrics": current, "target_minus_historical": deltas,
        "metric_file_hashes": {c: sha256(path.parent / "baseline" / c / f"metrics_{c}.json") for c in current},
        "input_integrity": guard, "engineering_AP_alert_threshold": .005,
        "caveat": "Hardware/kernel migration, not bitwise reproducibility. The threshold is an engineering alert, not statistical significance or test tuning."}
    write_json(path.parent / "baseline_recheck.json", summary)
    write_json(path.parent / "input_integrity_after.json", guard)
    lines = ["# WSL2 migration / frozen detector recheck", "", f"Status: {summary['status']}", "",
             "Original files and negative results retained. Only platform string differences accepted.", "",
             "| Test | Original mAP50 / 95 | WSL mAP50 / 95 |", "|---|---|---|"]
    for c in current:
        a, b = value["historical_metrics"][c], current[c]
        lines.append(f"| {c} | {a['map50']:.6f} / {a['map50_95']:.6f} | {b['map50']:.6f} / {b['map50_95']:.6f} |")
    lines += ["", summary["caveat"], "", f"Input integrity: {guard['status']}"]
    (path.parent / "migration_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def probe_uie(record_path, device="cuda:0", *, output=None):
    """Fixed largest original-train image, same enhancer settings; no labels/test tuning."""
    record_path = Path(record_path).resolve()
    record = read_json(record_path)
    verify_record(record_path, record["prepared"], record["raw_weights"], runtime_check=False, require_recheck=True)
    from ..model_inference_v16 import load_v16, enhance_tensor
    from .common import rgb, rgb_tensor, save_tensor
    import torch
    full = read_json(Path(record["enhanced_test"]) / "enhancement_manifest.json")
    for key in ("torch", "numpy", "Pillow"):
        if versions()[key] != full["versions"][key]:
            raise ValueError(f"Restore original pax package first: {key}")
    meta = load_prepared(record["prepared"], verify=False)
    row = sorted(meta["splits"]["train"], key=lambda r: (-(r["width"]*r["height"]), r["image_id"]))[0]
    protected = [Path(record["prepared"]), Path(record["enhanced_test"]), record_path,
                 record_path.with_suffix(".sha256"), record_path.parent / "baseline",
                 *map(Path, record["asset_hashes"])]
    if record.get("p0_run"):
        protected.append(Path(record["p0_run"]))
    out = fresh_output(output if output is not None else record_path.parent / "uie_memory_probe", protected)
    cuda_index = None
    status = {"status": "not_run", "split": "official train only", "image_id": row["image_id"],
              "width": row["width"], "height": row["height"], "device": device,
              "tile_size": full["tile_size"], "overlap": full["overlap"], "dtype": "float32", "amp": False,
              "output": str(out), "migration_record_sha256": sha256(record_path),
              "probe_source_sha256": sha256(Path(__file__)), "versions": versions(),
              "inference_started": False, "inference_completed": False,
              "peak_allocated_bytes": None, "peak_reserved_bytes": None,
              "memory_scope": "CUDA allocator peaks reset after explicit CUDA initialization and before model loading; includes model and inference, excludes driver/context and other processes",
              "note": "One memory probe, no model optimization or test-based parameter selection. Does not certify YOLO training batch memory."}
    def stage(name):
        status["stage"] = name
        print(f"V16 memory probe: {name} ({out})", flush=True)
    try:
        stage("parse_device")
        target = torch.device(device)
        if target.type == "cuda":
            stage("initialize_cuda")
            if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
            status["cuda_initialized_before_probe"] = torch.cuda.is_initialized()
            # is_available() does not initialize the allocator. reset_peak_memory_stats()
            # also does not initialize it and can reject even a valid device 0.
            torch.cuda.init()
            cuda_index = target.index if target.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(cuda_index)
            target = torch.device("cuda", cuda_index)
            status.update(device=str(target), cuda_device_index=cuda_index)
            torch.cuda.synchronize(cuda_index)
            stage("reset_peak_statistics")
            torch.cuda.reset_peak_memory_stats(cuda_index)
        elif target.type != "cpu":
            raise ValueError("WSL memory probe supports explicit CUDA or CPU only")
        stage("load_v16")
        model, provenance = load_v16(full["provenance"]["run_dir"], device=str(target))
        if provenance["checkpoint_sha256"] != full["provenance"]["checkpoint_sha256"]:
            raise ValueError("Probe checkpoint differs")
        stage("verify_train_source")
        source = Path(record["prepared"]) / "images/train_raw" / row["file"]
        if sha256(source) != row["image_sha256"]: raise ValueError("Train probe source changed")
        stage("inference")
        status["inference_started"] = True
        prediction = enhance_tensor(model, rgb_tensor(rgb(source)), str(target), full["tile_size"], full["overlap"])
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
        status["inference_completed"] = True
        stage("save_output")
        save_tensor(prediction, out / "probe.png")
        status["output_sha256"] = sha256(out / "probe.png")
        status["status"] = "pass"
        stage("complete")
    except Exception as error:
        status.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "error",
                      failed_stage=status["stage"], error=repr(error),
                      error_type=type(error).__name__, traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        # Reporting must not lazily initialize CUDA after a failure or hide its traceback.
        if cuda_index is not None and torch.cuda.is_initialized():
            for key, query in (("gpu_name", torch.cuda.get_device_name),
                               ("peak_allocated_bytes", torch.cuda.max_memory_allocated),
                               ("peak_reserved_bytes", torch.cuda.max_memory_reserved)):
                try:
                    status[key] = query(cuda_index)
                except Exception as error:
                    status.setdefault("memory_reporting_errors", {})[key] = repr(error)
        write_json(out / "probe.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("create")
    for name in ("prepared", "raw-weights", "enhanced-test", "p0-run", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("recheck")
    p.add_argument("--record", required=True); p.add_argument("--gpu", default="0")
    p = sub.add_parser("probe-uie")
    p.add_argument("--record", required=True); p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", help="New directory below diagnostics; default: <record-dir>/uie_memory_probe. Existing outputs are never overwritten.")
    args = parser.parse_args(argv)
    if args.phase == "create":
        print(create(args.prepared, args.raw_weights, args.enhanced_test, args.p0_run, args.output))
    elif args.phase == "recheck":
        outcome = recheck(args.record, args.gpu)
        print(outcome["status"])
        if outcome["status"] != "pass": raise SystemExit(2)
    else:
        outcome = probe_uie(args.record, args.device, output=args.output)
        print(outcome)
        if outcome["status"] != "pass": raise SystemExit(2)


if __name__ == "__main__":
    main()
