"""Create a NEW diagnostic recovery fork when changing AMP scale policy.

The source run is read-only. This is not an equivalent resume and is always
marked smoke/non-paper: epochs before/after the fork have different policies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

import torch

from .precision import cap_scale_state, validate_scale_policy
from .prepare import sha256, validate_bundle, verify_content, write_json
from .train import validate_config
from .utils import (ROOT, audit_environment, load_yaml, plot_history, project_path, run_path_guard,
                    save_checkpoint, save_yaml, source_hashes, torch_load, write_csv)

# Audited pre-ceiling Haze4K trainer from the 2026-10-17 fix. Architecture/loss/metrics
# and all other common sources must still match exactly; no generic hash bypass.
LEGACY_TRAIN_SHA256 = "0c4ab2cf2b5f22e4dab450ddd0f29fd0fcb5e6d10ab0ac99f069fe5525390256"
LEGACY_PREPARE_SHA256 = "d48726fd27606136197ae6eda431805287b98361eb495158fffda3f62d01afc5"
ADDED_SOURCES = {"src/haze4k_transfer/precision.py", "src/haze4k_transfer/recover.py"}


def validate_recovery_sources(saved, current):
    if set(saved) - set(current) or (set(current) - set(saved)) - ADDED_SOURCES:
        raise ValueError("Recovery source inventory is incompatible")
    differences = {}
    reviewed = {"src/haze4k_transfer/train.py": LEGACY_TRAIN_SHA256,
                "src/haze4k_transfer/prepare.py": LEGACY_PREPARE_SHA256}
    for name, digest in saved.items():
        if current[name] != digest:
            if reviewed.get(name) != digest:
                raise ValueError(f"Recovery refuses an unreviewed source change: {name}")
            differences[name] = {"saved": digest, "current": current[name]}
    return differences


def check_finite_checkpoint(payload):
    for name, tensor in payload["model"].items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Saved model already nonfinite: {name}")
    for state in payload["optimizer"]["state"].values():
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f"Saved optimizer already nonfinite: {name}")
    if payload["amp_scaler"]:
        scale = float(payload["amp_scaler"]["scale"])
        validate_scale_policy(scale, None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Source run checkpoint/last.pt")
    p.add_argument("--amp-max-scale", type=float, required=True, help="New initial scale and growth ceiling")
    p.add_argument("--data-root")
    a = p.parse_args(argv)
    validate_scale_policy(a.amp_max_scale, a.amp_max_scale)
    cp_path = project_path(a.checkpoint)
    source = run_path_guard(cp_path.parent.parent)
    if cp_path != source / "checkpoint/last.pt":
        raise ValueError("Recovery must start from an epoch-boundary last.pt")
    last = torch_load(cp_path)
    old_config = load_yaml(source / "config_resolved.yaml")
    if old_config != last["config"]:
        raise ValueError("Source checkpoint/config mismatch")
    if old_config.get("recovery"):
        raise ValueError("Nested recovery is not supported; keep a single explicit policy change")
    if last["epoch"] >= old_config["training"]["epochs"]:
        raise ValueError("Source run has already reached its target epochs")
    if not last["history"] or last["history"][-1]["epoch"] != last["epoch"]:
        raise ValueError("Source history is inconsistent with checkpoint epoch")
    current = source_hashes()
    differences = validate_recovery_sources(last["source_hashes"], current)
    meta, rows, inventory = validate_bundle(source / "split")
    if meta["hashes"] != last["split_hashes"]:
        raise ValueError("Source split/checkpoint mismatch")
    root = Path(a.data_root or old_config["data_root"]).resolve()
    verify_content(root, rows["train"] + rows["validation"], inventory)
    best_path = source / "checkpoint/best_psnr.pt"
    best = torch_load(best_path)
    if best["config"] != old_config or best["source_hashes"] != last["source_hashes"] or best["split_hashes"] != meta["hashes"]:
        raise ValueError("Source best checkpoint provenance mismatch")
    best_record = max(last["history"], key=lambda r: r["val_psnr"])
    if (best["epoch"] != best_record["epoch"] or best["best_psnr"] != best_record["val_psnr"]
            or last["best_psnr"] != best_record["val_psnr"]):
        raise ValueError("Source best does not match saved validation history")
    for payload in (last, best):
        check_finite_checkpoint(payload)
        for required in ("rng", "loader_generator", "val_generator", "amp_scaler", "optimizer"):
            if required not in payload:
                raise ValueError(f"Missing resume state: {required}")
    c = deepcopy(old_config)
    c["data_root"] = str(root)
    c["training"].update(amp_init_scale=float(a.amp_max_scale), amp_max_scale=float(a.amp_max_scale))
    c["smoke"] = True  # Diagnostic mixed-policy continuation, never a uniform-protocol paper result.
    provenance = {"kind": "diagnostic_precision_policy_change", "paper_eligible": False,
                  "source_run": str(source), "source_checkpoint": str(cp_path),
                  "source_last_sha256": sha256(cp_path), "source_best_sha256": sha256(best_path),
                  "source_epoch": last["epoch"], "next_epoch": last["epoch"] + 1,
                  "source_config": old_config, "source_hashes": last["source_hashes"],
                  "reviewed_source_differences": differences, "source_scaler_state": last["amp_scaler"],
                  "new_initial_and_maximum_scale": a.amp_max_scale,
                  "state_preserved": ["model", "optimizer", "best_validation_PSNR", "epoch", "rng", "loader_generator", "val_generator"],
                  "note": "Only scaler scale/growth tracker and explicit numerical policy change. The source run is untouched."}
    c["recovery"] = provenance
    validate_config(c)
    name = f"{c['variant']}_recovery_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{c['seed']}_{uuid.uuid4().hex[:6]}"
    run = run_path_guard(ROOT / "experiments/haze4k_transfer" / name)
    run.mkdir(parents=True, exist_ok=False)
    (run / "checkpoint").mkdir()
    (run / "validation").mkdir()
    write_json(run / "recovery.json", provenance)
    try:
        shutil.copytree(source / "split", run / "split")
        shutil.copyfile(source / "config_source.yaml", run / "config_source.yaml")
        save_yaml(run / "config_resolved.yaml", c)
        if (source / "model.json").is_file():
            shutil.copyfile(source / "model.json", run / "model.json")
        # Copy only pre-checkpoint validation records. Later records from failed attempts are not inherited.
        for epoch in range(1, last["epoch"] + 1):
            path = source / "validation" / f"epoch_{epoch:04d}.csv"
            if path.is_file():
                shutil.copyfile(path, run / "validation" / path.name)
        write_json(run / "history.json", last["history"])
        write_csv(run / "history.csv", last["history"])
        plot_history(run, last["history"])
        with (run / "train.log").open("w") as log:
            for record in last["history"]:
                log.write(json.dumps(record, allow_nan=False) + "\n")
        write_json(run / "environment.json", audit_environment(torch.device("cpu")))
        for name, payload in (("last", last), ("best_psnr", best)):
            payload["config"] = c
            payload["source_hashes"] = current
            # Keep growth parameters but restart scale and growth counter under the recorded new policy.
            if payload["amp_scaler"]:
                state = dict(payload["amp_scaler"])
                state.update(scale=float(a.amp_max_scale), _growth_tracker=0)
                payload["amp_scaler"] = cap_scale_state(state, a.amp_max_scale)
            payload["recovery"] = provenance
            save_checkpoint(run / "checkpoint" / f"{name}.pt", payload)
        write_json(run / "status.json", {"status": "diagnostic_recovery_ready", "epoch": last["epoch"],
                                        "best_validation_psnr": last["best_psnr"], "paper_eligible": False})
    except BaseException as e:
        write_json(run / "status.json", {"status": "recovery_failed", "error": repr(e)})
        raise
    print(f"RECOVERY_RUN_DIR={run}")
    print(f"DIAGNOSTIC ONLY: original run unchanged; next epoch={last['epoch'] + 1}; smoke=true; not a uniform-protocol paper run.")
    print(f'python -m src.haze4k_transfer.train --resume "{run}/checkpoint/last.pt" --gpu 0 --data-root "{root}"')
    return run


if __name__ == "__main__":
    main()
