"""Explicit P1 matched-domain experiment. Never called by P0; never trains V16.

Plan first, enhance the original train/val, then explicitly authorize ONE new
detector training. Test enhancement and the original Raw baseline stay read-only.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
import time

from ..utils import (PROJECT, load_prepared, pair_identity, read_json, require_yolo,
                     sha256, versions, write_yaml)
from .common import fresh_output, runtime_dirs, rgb, rgb_tensor, save_tensor, snapshot, check_snapshot
from .evaluator import build_dataset


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_partition(meta):
    """Numeric train/test IDs may overlap; official split membership is decisive."""
    train, val, test = (meta["splits"][s] for s in ("train", "val", "test"))
    for split, rows in (("train", train), ("val", val), ("test", test)):
        expected = "test" if split == "test" else "train"
        if not rows or any(r["official_split"] != expected for r in rows):
            raise ValueError(f"Wrong official membership for {split}")
        if len({r["image_id"] for r in rows}) != len(rows) or len({r["file"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {split} IDs/files")
    if {r["image_id"] for r in train} & {r["image_id"] for r in val}:
        raise ValueError("Original train/val must be disjoint")
    if meta["seed"] != 3520:
        raise ValueError("P1 keeps the original seed 3520 split")


def validate_training_config(config, synthetic=False):
    if any(k in config for k in ("data", "project", "name", "exist_ok", "model")):
        raise ValueError("Training paths/model must be constructed by P1")
    if config.get("resume") or config.get("amp") or config.get("cache"):
        raise ValueError("No automatic resume, AMP self-check downloads, or source image cache")
    if not synthetic and (config.get("seed") != 3520 or config.get("imgsz") != 640 or config.get("epochs") != 80):
        raise ValueError("Production P1 fixes original seed/imgsz/80 epochs")


def validate_initial_weights(path, expected_sha256, synthetic=False):
    path = Path(path)
    if not path.is_file() or not expected_sha256 or sha256(path) != expected_sha256:
        raise ValueError("Need original COCO initialization file with the recorded SHA256; no new download or Raw best.pt initialization")
    if not synthetic and path.suffix != ".pt":
        raise ValueError("Production needs original COCO-pretrained .pt")


def source_hashes():
    files = list((PROJECT / "src/downstream_duo").glob("*.py"))
    files += list((PROJECT / "src/downstream_duo/diagnostics").glob("*.py"))
    files += [PROJECT / "src/v16/models/network.py", PROJECT / "src/shared/uicf_models.py",
              PROJECT / "src/shared/uicf_inr.py", PROJECT / "src/shared/color_query_unet.py",
              PROJECT / "src/v4/models/unet.py", PROJECT / "scripts/downstream_duo/diagnostics/run_p1.sh"]
    return {str(p): sha256(p) for p in files if p.is_file()}


def plan(prepared, raw_weights, enhanced_test, p0_run, output, initial_weights=None, uie_run_dir=None, *, _synthetic=False):
    prepared, raw_weights, enhanced_test, p0_run = [Path(p).resolve() for p in (prepared, raw_weights, enhanced_test, p0_run)]
    from ..evaluate import frozen_registry
    from ..model_inference_v16 import validate_enhanced
    meta = load_prepared(prepared)
    if bool(meta["synthetic"]) != _synthetic:
        raise ValueError("Real P1 refuses synthetic data; software tests use a private API")
    validate_partition(meta)
    baseline = frozen_registry(prepared, raw_weights, meta)
    for split in ("train", "val"):
        if sha256(prepared / "manifests" / f"{split}.tsv") != baseline.get(f"{split}_manifest_sha256"):
            raise ValueError("P1 must retain precisely the original detector train/val partition")
    validate_training_config(baseline["train"], _synthetic)
    audit = read_json(p0_run / "audit/audit_protocol.json")
    identity = read_json(p0_run / "identity/identity_control.json")
    status = read_json(p0_run / "runner_status.json")
    integrity = read_json(p0_run / "input_integrity_after.json")
    if audit["status"] != "pass" or identity["status"] != "pass" or integrity["status"] != "pass":
        raise ValueError("Review P0 first: protocol, Identity and input integrity must pass")
    if status["status"] != "completed_with_explicit_native_eligibility":
        raise ValueError("P0 did not complete its explicitly scoped native comparison")
    if bool(audit["synthetic"]) != _synthetic or any(Path(audit["paths"][k]).resolve() != p for k, p in
        (("prepared", prepared), ("weights", raw_weights), ("enhanced", enhanced_test))):
        raise ValueError("P0 and P1 inputs must match exactly")
    existing = validate_enhanced(prepared, enhanced_test, meta)
    if sha256(raw_weights) != audit["existing_metrics"]["raw"]["weights_sha256"]:
        raise ValueError("P0 detector differs from the baseline")
    initial = Path(initial_weights or baseline["initial_model"]).expanduser().resolve()
    validate_initial_weights(initial, baseline.get("initial_model_sha256"), _synthetic)
    uie = Path(uie_run_dir or existing["provenance"]["run_dir"]).resolve()
    checkpoint, cfg = uie / "best/best_psnr.pt", uie / "config_resolved.yaml"
    if sha256(checkpoint) != existing["provenance"]["checkpoint_sha256"] or sha256(cfg) != existing["provenance"]["config_sha256"]:
        raise ValueError("P1 enhancer must be identical to existing test enhancement")
    if sha256(checkpoint) != audit["uie_checkpoint_sha256"]:
        raise ValueError("P0 and P1 enhancer differ")
    if not isinstance(existing["tile_size"], int) or existing["tile_size"] <= 0 or not 0 < existing["overlap"] < existing["tile_size"]:
        raise ValueError("P1 freezes the existing tiled E domain; no new enhancement parameter search")
    out = fresh_output(output, [prepared, raw_weights, enhanced_test, p0_run, initial, uie])
    runtime_dirs(out)
    baseline_metrics = {c: audit["existing_metrics"][c] for c in ("raw", "full")}
    for c, metric in baseline_metrics.items():
        if metric["weights_sha256"] != baseline["weights_sha256"] or metric["identity"] != pair_identity(meta) or metric["parameters"] != baseline["evaluation"]:
            raise ValueError(f"Original {c} protocol mismatch")
    protected = [prepared, enhanced_test, Path(baseline["output"]), checkpoint, cfg, initial,
                 p0_run / "audit/audit_protocol.json", p0_run / "identity/identity_control.json",
                 p0_run / "runner_status.json", p0_run / "input_integrity_after.json"]
    print("P1 plan: hashing protected inputs; this may take several minutes", flush=True)
    atomic_json(out / "input_integrity_before.json", snapshot(protected))
    sources = source_hashes()
    for file in sources:
        dest = out / "source_snapshot" / Path(file).relative_to(PROJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
    value = {"schema": 1, "synthetic": _synthetic, "prepared": str(prepared), "raw_weights": str(raw_weights),
        "raw_weights_sha256": sha256(raw_weights), "enhanced_test": str(enhanced_test), "p0_run": str(p0_run),
        "initial_weights": str(initial), "initial_weights_sha256": sha256(initial), "uie_run_dir": str(uie),
        "uie_checkpoint_sha256": sha256(checkpoint), "uie_config_sha256": sha256(cfg),
        "enhancement_manifest_sha256": sha256(enhanced_test / "enhancement_manifest.json"),
        "prepared_metadata_sha256": sha256(prepared / "split_metadata.json"), "identity": pair_identity(meta),
        "counts": meta["counts"], "train": baseline["train"], "evaluation": baseline["evaluation"],
        "detector_versions": baseline["versions"], "uie_versions": existing["versions"],
        "tile_size": existing["tile_size"], "overlap": existing["overlap"],
        "baseline_metrics": baseline_metrics, "source_hashes": sources,
        "detector_python": sys.executable, "created_unix": time.time(),
        "protocol": "New E-trained detector from identical COCO initialization; E-val selects best.pt. Existing E-test reused read-only. Evaluate E->R and E->E on identical official test IDs.",
        "limits": "Single seed; P0 native OOM limits retained. Recovery supports domain mismatch, not proof of zero information loss. No recovery also requires checking training/optimization."}
    atomic_json(out / "plan.json", value)
    (out / "plan.sha256").write_text(sha256(out / "plan.json") + "\n")
    write_yaml(out / "training_config_snapshot.yaml", {**baseline["train"], "model": str(initial)})
    report(out)
    return out


def load_plan(run_dir, detector=False, verify_original=False):
    root = Path(run_dir).resolve()
    from .common import DIAGNOSTICS
    if not root.is_relative_to(DIAGNOSTICS.resolve()) or sha256(root / "plan.json") != (root / "plan.sha256").read_text().strip():
        raise ValueError("P1 plan missing, changed or outside diagnostics")
    p = read_json(root / "plan.json")
    for path, expected in p["source_hashes"].items():
        if sha256(path) != expected:
            raise ValueError(f"Executed source changed since plan: {path}; use a new reviewed plan")
    for path, expected in ((p["raw_weights"], p["raw_weights_sha256"]), (p["initial_weights"], p["initial_weights_sha256"]),
        (Path(p["uie_run_dir"]) / "best/best_psnr.pt", p["uie_checkpoint_sha256"]),
        (Path(p["uie_run_dir"]) / "config_resolved.yaml", p["uie_config_sha256"]),
        (Path(p["enhanced_test"]) / "enhancement_manifest.json", p["enhancement_manifest_sha256"]),
        (Path(p["prepared"]) / "split_metadata.json", p["prepared_metadata_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Protected P1 input changed: {path}")
    if detector and versions() != p["detector_versions"]:
        raise ValueError("Use exactly the original detector environment, without in-place upgrades")
    meta = load_prepared(p["prepared"], verify=verify_original)
    if pair_identity(meta) != p["identity"]:
        raise ValueError("P1 image/GT identity changed")
    runtime_dirs(root)
    return root, p, meta


def journal_records(path):
    records = {}
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            records[(row["split"], row["image_id"])] = row
    return records


def verify_enhanced_train(root, p, meta):
    manifest = read_json(root / "enhanced/enhancement_manifest.json")
    expected = {(s, r["image_id"]): r for s in ("train", "val") for r in meta["splits"][s]}
    records = manifest["records"]
    if not manifest["complete"] or len(records) != len(expected) or {(r["split"], r["image_id"]) for r in records} != set(expected):
        raise ValueError("All original train/val images must be enhanced successfully before training")
    if manifest["checkpoint_sha256"] != p["uie_checkpoint_sha256"] or manifest["tile_size"] != p["tile_size"] or manifest["overlap"] != p["overlap"]:
        raise ValueError("Enhancement protocol changed")
    for r in records:
        row = expected[(r["split"], r["image_id"])]
        image = root / "enhanced/images" / (r["split"] + "_enhanced") / row["file"]
        label = root / "enhanced/labels" / (r["split"] + "_enhanced") / Path(row["file"]).with_suffix(".txt")
        if r["status"] != "ok" or r["file"] != row["file"] or sha256(image) != r["sha256"] or sha256(label) != row["label_sha256"]:
            raise ValueError(f"P1 enhanced data changed: {image}")
        if rgb(image).shape != (row["height"], row["width"], 3):
            raise ValueError("P1 geometry changed")
    for s in ("train", "val"):
        actual = {x.relative_to(root / f"enhanced/images/{s}_enhanced").as_posix()
                  for x in (root / f"enhanced/images/{s}_enhanced").rglob("*.png")}
        if actual != {r["file"] for r in meta["splits"][s]}:
            raise ValueError("Unexpected/missing enhanced input images")
    return manifest


def enhance(run_dir, device):
    root, p, meta = load_plan(run_dir, verify_original=True)
    target = root / "enhanced"
    if (target / "enhancement_manifest.json").is_file() and read_json(target / "enhancement_manifest.json")["complete"]:
        verify_enhanced_train(root, p, meta)
        print("P1 enhancement already complete and verified; not regenerated", flush=True)
        return
    for key in ("torch", "numpy", "Pillow"):
        if versions()[key] != p["uie_versions"][key]:
            raise ValueError(f"Use original V16 environment: {key} differs")
    import torch
    from ..model_inference_v16 import load_v16, enhance_tensor
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU/resize fallback")
    if (target / "active.lock").exists():
        raise FileExistsError("Enhancement active/stale lock exists; review process before starting another job")
    model, provenance = load_v16(p["uie_run_dir"], device=device)
    if provenance["checkpoint_sha256"] != p["uie_checkpoint_sha256"] or any(x.dtype != torch.float32 for x in model.parameters()):
        raise ValueError("Frozen enhancer checkpoint/FP32 mismatch")
    target.mkdir(exist_ok=True)
    active = target / "active.lock"
    with active.open("x") as f:
        f.write(str(__import__('os').getpid()))
    progress = target / "enhancement_progress.jsonl"
    records = journal_records(progress)
    manifest = {"complete": False, "checkpoint_sha256": p["uie_checkpoint_sha256"],
        "tile_size": p["tile_size"], "overlap": p["overlap"], "dtype": "float32", "amp": False,
        "pixel_chain": "Raw PNG RGB /255 -> original enhance_tensor -> same-size round(255*x) RGB PNG; no resizing",
        "versions": versions(), "device": device, "records": []}
    try:
        total = len(meta["splits"]["train"]) + len(meta["splits"]["val"])
        done = 0
        for split in ("train", "val"):
            for row in meta["splits"][split]:
                key = (split, row["image_id"])
                image = target / "images" / (split + "_enhanced") / row["file"]
                label = target / "labels" / (split + "_enhanced") / Path(row["file"]).with_suffix(".txt")
                if key in records and records[key]["status"] == "ok":
                    if sha256(image) != records[key]["sha256"] or sha256(label) != row["label_sha256"]:
                        raise ValueError("Logged enhancement output changed; no silent overwrite")
                    done += 1
                    continue
                if image.exists() or label.exists():
                    raise FileExistsError(f"Unlogged output needs review; no overwrite: {image}")
                entry = {"split": split, "image_id": row["image_id"], "file": row["file"], "status": "failed"}
                start = time.perf_counter()
                try:
                    array = rgb(Path(p["prepared"]) / "images" / (split + "_raw") / row["file"])
                    result = enhance_tensor(model, rgb_tensor(array), device, p["tile_size"], p["overlap"])
                    save_tensor(result, image)
                    if rgb(image).shape != (row["height"], row["width"], 3):
                        raise ValueError("Enhancer changed bbox coordinate geometry")
                    label.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(Path(p["prepared"]) / "labels" / (split + "_raw") / Path(row["file"]).with_suffix(".txt"), label)
                    entry.update(status="ok", sha256=sha256(image), label_sha256=sha256(label), seconds=time.perf_counter()-start)
                    if entry["label_sha256"] != row["label_sha256"]:
                        raise ValueError("GT byte content changed")
                except Exception as error:
                    entry["error"] = repr(error)
                    raise
                finally:
                    records[key] = entry
                    with progress.open("a") as f:
                        f.write(json.dumps(entry) + "\n")
                done += 1
                print(f"P1 enhance {done}/{total}: {split} image_id={row['image_id']}", flush=True)
        manifest["complete"] = len(records) == total and all(r["status"] == "ok" for r in records.values())
    finally:
        manifest["records"] = list(records.values())
        atomic_json(target / "enhancement_manifest.json", manifest)
        active.unlink(missing_ok=True)
    verify_enhanced_train(root, p, meta)


def authorize_training(root, start_training):
    if not start_training:
        raise ValueError("P1 needs explicit --start-training; plan/enhancement alone never authorizes training")
    with (Path(root) / "detector_training_attempt.lock").open("x") as f:
        f.write("One P1 training attempt. No automatic resume/retraining.\n")


def train(run_dir, gpu, start_training=False):
    root, p, meta = load_plan(run_dir, detector=True)
    if (root / "detector/registry.json").is_file() and read_json(root / "detector/registry.json").get("status") == "complete":
        old = read_json(root / "detector/registry.json")
        if sha256(old["weights"]) != old["weights_sha256"]:
            raise ValueError("P1 trained weights changed")
        print("P1 detector already trained; not retrained", flush=True)
        return old
    if str(gpu) != str(p["train"]["device"]):
        raise ValueError("Keep the recorded baseline training device; no implicit override")
    verify_enhanced_train(root, p, meta)
    authorize_training(root, start_training)
    out = root / "detector"
    out.mkdir(exist_ok=False)
    data = out / "enhanced_train_val.yaml"
    write_yaml(data, {"path": str(root / "enhanced"), "train": "images/train_enhanced", "val": "images/val_enhanced", "names": meta["mapping"]["names"]})
    registry = {"status": "started", "train": p["train"], "identity": p["identity"], "initial_weights_sha256": p["initial_weights_sha256"],
                "versions": versions(), "data": str(data), "started_unix": time.time()}
    atomic_json(out / "registry.json", registry)
    YOLO = require_yolo()
    from ultralytics.utils import LOGGER
    handler = logging.FileHandler(out / "train.log")
    LOGGER.addHandler(handler)
    try:
        model = YOLO(p["initial_weights"], task="detect")
        if not p["synthetic"]:
            from ultralytics.nn.tasks import yaml_model_load
            expected = yaml_model_load("yolo11s.yaml")
            if model.model.yaml.get("backbone") != expected["backbone"] or model.model.yaml.get("head") != expected["head"] or model.model.yaml.get("scale") != "s" or model.model.yaml.get("nc") != 80:
                raise ValueError("Initialization must be the original COCO 80-class YOLO11s")
        model.train(data=str(data), project=str(out), name="run", exist_ok=False, **p["train"])
        best = out / "run/weights/best.pt"
        registry.update(status="complete", weights=str(best), weights_sha256=sha256(best), finished_unix=time.time())
    except Exception as error:
        registry.update(status="failed", error=repr(error))
        raise
    finally:
        atomic_json(out / "registry.json", registry)
        LOGGER.removeHandler(handler); handler.close()
    load_plan(root, detector=True)
    return registry


def evaluate(run_dir, gpu):
    root, p, meta = load_plan(run_dir, detector=True, verify_original=True)
    verify_enhanced_train(root, p, meta)
    from ..model_inference_v16 import validate_enhanced
    validate_enhanced(p["prepared"], p["enhanced_test"], meta)
    registry = read_json(root / "detector/registry.json")
    weights = Path(registry["weights"])
    if registry["status"] != "complete" or sha256(weights) != registry["weights_sha256"]:
        raise ValueError("Need completed, unchanged P1 best.pt")
    if str(gpu) != str(p["evaluation"]["device"]):
        raise ValueError("Keep original evaluation device and parameters")
    rows = meta["splits"]["test"]
    from ultralytics.models.yolo.detect import DetectionValidator
    for condition in ("raw", "full"):
        out = root / "evaluation" / condition
        metric_path = out / f"metrics_{condition}.json"
        if metric_path.is_file():
            old = read_json(metric_path)
            if old["weights_sha256"] != registry["weights_sha256"] or old["parameters"] != p["evaluation"] or old["identity"] != p["identity"] or old["test_count"] != len(rows) or old["predictions_sha256"] != sha256(out / "per_image_predictions.json"):
                raise ValueError("Existing P1 evaluation protocol changed")
            print(f"P1 {condition} evaluation already complete; retained", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=False)
        image_root = Path(p["prepared"]) / "images/test_raw" if condition == "raw" else Path(p["enhanced_test"]) / "images/test_full"
        data = build_dataset(p["prepared"], image_root, rows, out / "dataset", meta["mapping"]["names"])
        by_path = {str((out / "dataset/images/test" / r["file"]).resolve()): r for r in rows}
        predictions = {r["image_id"]: {k: r[k] for k in ("image_id", "file", "width", "height")} | {"detections": []} for r in rows}
        seen, effective = set(), {}
        class P1Validator(DetectionValidator):
            def update_metrics(self, preds, batch):
                for filename in batch["im_file"]:
                    key = str(Path(filename).resolve())
                    if key not in by_path or key in seen:
                        raise ValueError("Unexpected/duplicate P1 test image")
                    seen.add(key)
                super().update_metrics(preds, batch)

            def pred_to_json(self, predn, pbatch):
                row = by_path[str(Path(pbatch["im_file"]).resolve())]
                for box, conf, cls in zip(predn["bboxes"].tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
                    cls = int(cls); cid = meta["mapping"]["yolo_to_coco"][str(cls)]
                    predictions[row["image_id"]]["detections"].append({"bbox_xyxy": box, "confidence": conf, "class_id": cls, "category_id": cid})
                    self.jdict.append({"image_id": row["image_id"], "category_id": cid, "bbox": [box[0], box[1], box[2]-box[0], box[3]-box[1]], "score": conf})

            def finalize_metrics(self):
                super().finalize_metrics()
                effective.update({k: v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v) for k, v in vars(self.args).items()})
        model = require_yolo()(str(weights), task="detect")
        model.model.eval().requires_grad_(False)
        metrics = model.val(validator=P1Validator, data=str(data), split="test", save_json=True, save_txt=False,
                            project=str(out), name="validator", exist_ok=False, **p["evaluation"])
        if seen != set(by_path) or sha256(weights) != registry["weights_sha256"]:
            raise ValueError("P1 test coverage or fixed weights changed")
        box = metrics.box
        value = {"condition": "E->R" if condition == "raw" else "E->E", "synthetic": p["synthetic"],
            "test_count": len(rows), "identity": p["identity"], "weights": str(weights), "weights_sha256": sha256(weights),
            "parameters": p["evaluation"], "effective_validator_args": effective, "versions": versions(),
            "precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50), "map50_95": float(box.map),
            "per_class": [{"class_id": int(c), "name": meta["mapping"]["names"][int(c)], "precision": float(box.p[i]), "recall": float(box.r[i]),
                           "map50": float(box.ap50[i]), "map50_95": float(box.ap[i])} for i, c in enumerate(box.ap_class_index)]}
        atomic_json(out / "per_image_predictions.json", list(predictions.values()))
        value["predictions_sha256"] = sha256(out / "per_image_predictions.json")
        atomic_json(metric_path, value)
    load_plan(root, detector=True)


def report(run_dir):
    root, p, meta = load_plan(run_dir)
    rows = [{"condition": "R->R", **{k: p["baseline_metrics"]["raw"][k] for k in ("precision", "recall", "map50", "map50_95")}},
            {"condition": "R->E", **{k: p["baseline_metrics"]["full"][k] for k in ("precision", "recall", "map50", "map50_95")}}]
    complete = True
    for condition, title in (("raw", "E->R"), ("full", "E->E")):
        file = root / "evaluation" / condition / f"metrics_{condition}.json"
        if file.is_file():
            value = read_json(file)
            if value["identity"] != p["identity"] or value["parameters"] != p["evaluation"] or value["test_count"] != len(meta["splits"]["test"]) or value["predictions_sha256"] != sha256(file.parent / "per_image_predictions.json"):
                raise ValueError("P1 summary cannot mix test protocols")
            rows.append({"condition": title, **{k: value[k] for k in ("precision", "recall", "map50", "map50_95")}, "weights_sha256": value["weights_sha256"]})
        else:
            rows.append({"condition": title, "status": "not_evaluated"}); complete = False
    guard = check_snapshot(read_json(root / "input_integrity_before.json"))
    atomic_json(root / "input_integrity_after.json", guard)
    if complete and rows[2]["weights_sha256"] != rows[3]["weights_sha256"]:
        raise ValueError("E->R/E->E used different detector weights")
    if complete:
        registry = read_json(root / "detector/registry.json")
        if registry["status"] != "complete" or sha256(registry["weights"]) != rows[2]["weights_sha256"]:
            raise ValueError("P1 detector weights changed after evaluation")
    progress = journal_records(root / "enhanced/enhancement_progress.jsonl")
    training = read_json(root / "detector/registry.json") if (root / "detector/registry.json").is_file() else {}
    summary = {"status": "complete" if complete and guard["status"] == "pass" else "not_complete", "synthetic": p["synthetic"],
        "counts": p["counts"], "train": p["train"], "initial_weights_sha256": p["initial_weights_sha256"],
        "raw_weights_sha256": p["raw_weights_sha256"], "uie_checkpoint_sha256": p["uie_checkpoint_sha256"],
        "rows": rows, "input_integrity": guard, "interpretation_limits": p["limits"],
        "phases": {"enhancement_logged_success": sum(r["status"] == "ok" for r in progress.values()),
                   "enhancement_expected": p["counts"]["train"] + p["counts"]["val"],
                   "training_status": training.get("status", "not_started")}}
    if complete:
        summary["E_to_E_minus_R_to_E"] = {k: rows[3][k]-rows[1][k] for k in ("precision", "recall", "map50", "map50_95")}
        summary["E_to_E_minus_R_to_R"] = {k: rows[3][k]-rows[0][k] for k in ("precision", "recall", "map50", "map50_95")}
    atomic_json(root / "p1_summary.json", summary)
    lines = ["# P1 matched-domain DUO", "", f"Status: {summary['status']}",
        "SYNTHETIC SOFTWARE TEST ONLY" if p["synthetic"] else "Original negative R->E result retained; new detector is selected on enhanced val only.",
        "", "| Train->Test | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for row in rows:
        values = [f"{row[k]:.6f}" if k in row else "not evaluated" for k in ("precision", "recall", "map50", "map50_95")]
        lines.append("| " + " | ".join([row["condition"], *values]) + " |")
    lines += ["", f"Input integrity: {guard['status']}", "", p["protocol"], "", p["limits"], "",
              "No test-based epoch/threshold/enhancer selection. R-trained and E-trained detectors differ; this is not the original frozen-detector experiment."]
    (root / "p1_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    a = sub.add_parser("plan", help="Read-only input validation and locked plan; no inference/training")
    for name in ("prepared", "raw-weights", "enhanced-test", "p0-run", "output"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--initial-weights"); a.add_argument("--uie-run-dir")
    for phase in ("check", "enhance", "train", "evaluate", "report"):
        a = sub.add_parser(phase)
        a.add_argument("--run-dir", required=True)
        if phase == "enhance": a.add_argument("--device", default="cuda:0")
        if phase in ("train", "evaluate"): a.add_argument("--gpu", default="0")
        if phase == "train": a.add_argument("--start-training", action="store_true")
    args = parser.parse_args(argv)
    if args.phase == "plan":
        print(plan(args.prepared, args.raw_weights, args.enhanced_test, args.p0_run, args.output, args.initial_weights, args.uie_run_dir))
    elif args.phase == "check":
        root, p, meta = load_plan(args.run_dir, detector=True, verify_original=True)
        print({"status": "pass", "counts": p["counts"], "train": p["train"], "initial_weights": p["initial_weights"]})
    elif args.phase == "enhance": enhance(args.run_dir, args.device)
    elif args.phase == "train": train(args.run_dir, args.gpu, args.start_training)
    elif args.phase == "evaluate": evaluate(args.run_dir, args.gpu)
    else: print(report(args.run_dir)["status"])


if __name__ == "__main__":
    main()
