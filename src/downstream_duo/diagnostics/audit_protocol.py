"""P0-A: independent per-ID COCO/label/pixel/protocol audit, read-only inputs."""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
from PIL import Image

from ..utils import load_prepared, pair_identity, draw_boxes, versions
from ..inspect_duo import resolve_image, EXTENSIONS
from .common import *


def audit(prepared, weights, enhanced, out, evaluation_root=None, data_root=None, uie_run_dir=None,
          allow_synthetic=False, previews=30):
    prepared, weights, enhanced, out = map(lambda p: Path(p).resolve(), (prepared, weights, enhanced, out))
    out.mkdir(parents=True, exist_ok=False)
    runtime_dirs(out)
    checks, records = [], []
    result = {"status": "unknown", "checks": checks, "images": records, "versions": versions(),
              "paths": {"prepared": str(prepared), "weights": str(weights), "enhanced": str(enhanced)},
              "source_inventory": source_inventory(out), "synthetic": None,
              "bbox_comparison": {"method": "same-class one-to-one bipartite matching; independent of row order",
                                  "absolute_tolerance_pixels": 1e-5}}
    def check(name, state, detail, critical=True):
        checks.append({"check": name, "status": state, "detail": detail, "critical": critical})
    try:
        meta = load_prepared(prepared, verify=False)
        result["synthetic"] = bool(meta["synthetic"])
        if meta["synthetic"] and not allow_synthetic:
            raise ValueError("Real diagnostics refuse synthetic prepared data")
        rows = meta["splits"]["test"]
        ids = [r["image_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate prepared test image IDs")
        root = Path(data_root or meta["source_root"]).resolve()
        evaluation = Path(evaluation_root or prepared.parent/"evaluation").resolve()
        source_json = root/"annotations/instances_test.json"
        result["paths"].update(data_root=str(root), coco_test=str(source_json), evaluation=str(evaluation))
        doc = read_json(source_json)
        images, boxes, corrected = expected_coco(doc, meta)
        check("official_test_id_set", "pass" if set(ids) == set(images) else "fail",
              {"prepared": len(ids), "official": len(images), "missing": sorted(set(images)-set(ids)), "extra": sorted(set(ids)-set(images))})
        if not meta["synthetic"]:
            check("official_population", "pass" if len(images) == 1111 and len(doc["annotations"]) == 10517 else "fail",
                  {"images": len(images), "original_annotations": len(doc["annotations"])})
        snapshot_doc = read_json(prepared/"dataset_snapshot.json")
        check("official_annotation_snapshot", "pass" if
              sorted(doc["annotations"], key=lambda a:a["id"]) == snapshot_doc["official"]["test"]["annotations"] else "fail",
              "Original COCO annotations compared with the preparation snapshot, not aggregate counts")
        logged_corrections = read_json(prepared/"bbox_corrections.json") if (prepared/"bbox_corrections.json").exists() else []
        check("recorded_bbox_corrections", "pass" if sorted(corrected) == sorted(c["annotation_id"] for c in logged_corrections if c["split"] == "test") else "fail",
              {"expected_test_corrections": corrected, "policy": meta.get("bbox_policy", "strict")})
        full_manifest = read_json(enhanced/"enhancement_manifest.json")
        full_rows = full_manifest.get("records", [])
        full_by_id = {r["image_id"]:r for r in full_rows}
        check("enhancement_ids_and_identity", "pass" if full_manifest.get("complete") is True and
              len(full_by_id) == len(full_rows) == len(rows) and set(full_by_id) == set(ids) and
              full_manifest.get("identity") == pair_identity(meta) else "fail", "Exact ID/file/GT identity and completion flag")
        files = {p for p in (root/"images/test").rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS}
        by_name = defaultdict(list)
        for path in files:
            by_name[path.name].append(path)
        used = set()
        sample = set(random.Random(3520).sample(sorted(ids), min(previews, len(ids))))
        result["gt_preview_ids"] = sorted(sample)
        source_hashes = {str(source_json):sha256(source_json), str(weights):sha256(weights)}
        raw_base, full_base = prepared/"images/test_raw", enhanced/"images/test_full"
        for base, label, extension in ((raw_base,"raw images",".png"), (full_base,"full images",".png"),
                                      (prepared/"labels/test_raw","raw labels",".txt"), (enhanced/"labels/test_full","full labels",".txt")):
            actual = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and
                      (p.suffix.lower() in EXTENSIONS if extension == ".png" else p.suffix == extension)}
            wanted = {Path(r["file"]).with_suffix(extension).as_posix() for r in rows}
            check(label+"_file_set", "pass" if actual == wanted else "fail",
                  {"unexpected":sorted(actual-wanted), "missing":sorted(wanted-actual)})
        for index, row in enumerate(rows):
            iid = row["image_id"]
            record = {"image_id": iid, "file": row["file"], "status": "fail"}
            records.append(record)
            try:
                official = images[iid]
                validate_image_binding(row,official)
                original = resolve_image(root, "test", official["file_name"], files, by_name)
                if original in used:
                    raise ValueError("Two official records map to one source file")
                used.add(original)
                raw_path, full_path = raw_base/row["file"], full_base/row["file"]
                raw_label = prepared/"labels/test_raw"/Path(row["file"]).with_suffix(".txt")
                full_label = enhanced/"labels/test_full"/Path(row["file"]).with_suffix(".txt")
                record["paths"] = {"original":str(original), "raw":str(raw_path), "enhanced":str(full_path),
                                   "raw_label":str(raw_label), "enhanced_label":str(full_label)}
                hashes = {str(p):sha256(p) for p in (original,raw_path,full_path,raw_label,full_label)}
                source_hashes.update(hashes)
                if hashes[str(original)] != row["source_sha256"] or hashes[str(raw_path)] != row["image_sha256"] or hashes[str(raw_label)] != row["label_sha256"]:
                    raise ValueError("Prepared/source file checksum mismatch")
                fr = full_by_id[iid]
                if fr.get("status") != "ok" or fr["file"] != row["file"] or hashes[str(full_path)] != fr["sha256"] or hashes[str(full_label)] != row["label_sha256"]:
                    raise ValueError("Enhanced image/label/filename hash mismatch")
                arrays = {"original":rgb(original), "raw":rgb(raw_path), "full":rgb(full_path)}
                expected_size = (official["height"], official["width"], 3)
                if (row["width"],row["height"]) != (official["width"],official["height"]) or any(a.shape != expected_size for a in arrays.values()):
                    raise ValueError("Per-image decoded size or JSON coordinate system mismatch")
                for path in (raw_path, full_path):
                    with Image.open(path) as im:
                        if im.mode != "RGB" or im.format != "PNG":
                            raise ValueError(f"Expected RGB PNG: {path}; got {im.mode}/{im.format}")
                source_decode_diff = pixel_difference(arrays["original"],arrays["raw"])
                if source_decode_diff["max_error_uint8"] > 1:
                    raise ValueError(f"Prepared Raw/source decoded RGB differs beyond the declared 1-level decoder engineering tolerance: {source_decode_diff}")
                actual = labels_xyxy(raw_label, official["width"], official["height"])
                if not same_boxes(actual, boxes[iid]) or raw_label.read_bytes() != full_label.read_bytes():
                    raise ValueError("Per-image GT bbox/class/label-byte mismatch with original COCO")
                record.update(status="pass", sha256=hashes, gt_count=len(actual),
                              source_vs_raw_decode_difference=source_decode_diff,
                              raw=image_statistics(arrays["raw"]), full=image_statistics(arrays["full"]),
                              raw_full_pixel_difference=pixel_difference(arrays["raw"],arrays["full"]),
                              bgr_alternative_mae_uint8=pixel_difference(arrays["raw"],arrays["full"][...,::-1])["mae_uint8"])
                if iid in sample:
                    for condition in ("raw", "full"):
                        image = draw_boxes(Image.fromarray(arrays[condition]), boxes[iid], meta["mapping"]["names"])
                        image.thumbnail((1600,1200))
                        dest = out/"gt_previews"/f"{iid}_{condition}.png"
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        image.save(dest)
            except Exception as error:
                record["error"] = repr(error)
            if (index+1) % 100 == 0:
                print(f"Audit {index+1}/{len(rows)}", flush=True)
        check("all_per_image_mapping_geometry_gt_and_hashes", "pass" if all(r["status"] == "pass" for r in records) else "fail",
              {"failed_ids":[r["image_id"] for r in records if r["status"] != "pass"]})
        check("source_vs_prepared_decode", "pass" if all(r.get("source_vs_raw_decode_difference",{}).get("identical") for r in records) else "unknown",
              {"allowed_engineering_max_uint8":1,"note":"Existing Raw PNG is the detector baseline's canonical input. Small original-JPEG decode differences are recorded, not automatically attributed to libraries.",
               "nonidentical_ids":[r["image_id"] for r in records if not r.get("source_vs_raw_decode_difference",{}).get("identical")]},critical=False)
        result["test_ids"] = ids
        result["source_sha256"] = source_hashes
        result["original_gt_count"] = len(doc["annotations"])
        result["derived_gt_count"] = sum(len(v) for v in boxes.values())
        result["enhancement"] = {key:full_manifest.get(key, "unknown") for key in
                                  ("provenance", "versions", "device", "tile_size", "overlap", "padding", "output_encoding")}
        inference_file = PROJECT/"src/downstream_duo/model_inference_v16.py"
        tree = ast.parse(inference_file.read_text())
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else "unknown"
                 for node in ast.walk(tree) if isinstance(node, ast.Call)]
        result["processing_chain"] = {"observed_source_sha256":sha256(inference_file),
            "decode":"Pillow RGB (source -> prepared Raw PNG -> float32 /255 -> BCHW)",
            "inference":"complete V16 independently per padded tile; inference_mode; FP32 input/model in current source",
            "encode":"clamp[0,1] -> round(255*x) -> uint8 RGB PNG, exact original width/height",
            "autocast_call_in_current_wrapper": "autocast" in calls,
            "resize_call_in_current_wrapper": "resize" in calls,
            "original_run_amp_recorded":full_manifest.get("amp", "unknown (not explicitly recorded)"),
            "original_run_dtype_recorded":full_manifest.get("dtype", "unknown (not explicitly recorded)"),
            "original_executed_source_hash":full_manifest.get("source_sha256", "unknown (current source snapshot is not historical execution proof)"),
            "v16_paired_test_amp":"src/v16/test.py uses training.amp; this paired evaluator was not used for DUO"}
        prov = full_manifest.get("provenance", {})
        uie = Path(uie_run_dir or prov.get("run_dir", "/unavailable"))
        ckpt = uie/"best/best_psnr.pt"
        result["paths"].update(uie_run_dir=str(uie), uie_checkpoint=str(ckpt))
        if ckpt.is_file():
            result["uie_checkpoint_sha256"] = sha256(ckpt)
            source_hashes[str(ckpt.resolve())] = sha256(ckpt)
            source_hashes[str((uie/"config_resolved.yaml").resolve())] = sha256(uie/"config_resolved.yaml")
            check("uie_checkpoint_hash", "pass" if sha256(ckpt) == prov.get("checkpoint_sha256") else "fail", prov)
            check("uie_config_hash", "pass" if sha256(uie/"config_resolved.yaml") == prov.get("config_sha256") else "fail",
                  {"actual":sha256(uie/"config_resolved.yaml"),"recorded":prov.get("config_sha256")})
        else:
            check("uie_checkpoint_hash", "unknown", f"Unavailable: {ckpt}")
        baseline = {}
        for condition in ("raw", "full"):
            folder = evaluation/condition
            metrics_path = folder/f"metrics_{condition}.json"
            validator_preds = folder/"validator/predictions.json"
            exported_preds = folder/f"predictions_{condition}.json"
            for path in (metrics_path, validator_preds, exported_preds):
                if not path.is_file():
                    check(f"{condition}_existing_output", "unknown", f"Unavailable: {path}")
            if not all(p.is_file() for p in (metrics_path,validator_preds,exported_preds)):
                continue
            metric = read_json(metrics_path)
            baseline[condition] = metric
            check(f"{condition}_reported_population", "pass" if metric.get("test_count") == len(rows) and
                  metric.get("synthetic") == meta["synthetic"] else "fail", {"test_count":metric.get("test_count"),"synthetic":metric.get("synthetic")})
            if not meta["synthetic"]:
                check(f"{condition}_preserved_main_numbers", "pass" if all(
                    isinstance(metric.get(k),(int,float)) and abs(metric[k]-GIVEN_RESULTS[condition][k]) <= 1e-6 for k in METRICS) else "fail",
                    {k:metric.get(k,"unknown") for k in METRICS})
            source_hashes.update({str(p):sha256(p) for p in (metrics_path,validator_preds,exported_preds)})
            preds = read_json(exported_preds)
            pred_ids = [p["image_id"] for p in preds]
            paired = {p["image_id"]:p for p in preds}
            exports_valid = len(pred_ids) == len(set(pred_ids)) == len(ids) and set(pred_ids) == set(ids)
            exports_valid = exports_valid and all(paired[r["image_id"]]["file"] == r["file"] and
                       (paired[r["image_id"]]["width"],paired[r["image_id"]]["height"]) == (r["width"],r["height"]) for r in rows)
            check(f"{condition}_exported_prediction_id_geometry", "pass" if exports_valid else "fail", "Includes images with zero predictions")
            check(f"{condition}_prediction_export_hash", "pass" if sha256(exported_preds) == metric.get("predictions_sha256") else "fail", str(exported_preds))
            flat = []
            for item in preds:
                for detection in item["detections"]:
                    cls = detection["class_id"]
                    if meta["mapping"]["yolo_to_coco"].get(str(cls)) != detection.get("category_id"):
                        raise ValueError(f"{condition} exported YOLO/COCO class mapping differs")
                    x1,y1,x2,y2 = detection["bbox_xyxy"]
                    flat.append((item["image_id"],detection["category_id"],x1,y1,x2-x1,y2-y1,detection["confidence"]))
            coco_export = [(p["image_id"],p["category_id"],*p["bbox"],p["score"]) for p in read_json(validator_preds)]
            export_match = len(flat) == len(coco_export) and (not flat or bool(np.allclose(sorted(flat),sorted(coco_export),rtol=0,atol=.001)))
            check(f"{condition}_two_prediction_exports_match", "pass" if export_match else "fail", "ID/class/xywh/score multiset checked; <=0.001 export rounding tolerance")
            validator_valid = all(p.get("image_id") in images and p.get("category_id") in {c["id"] for c in doc["categories"]} and
                                  len(p.get("bbox", [])) == 4 and np.isfinite(p["bbox"]).all() and
                                  0 <= p.get("score", -1) <= 1 for p in read_json(validator_preds))
            check(f"{condition}_validator_prediction_ids", "pass" if validator_valid else "fail", "COCO export may omit zero-prediction images")
            check(f"{condition}_weights_hash", "pass" if metric.get("weights_sha256") == sha256(weights) and
                  Path(metric.get("weights", "")).resolve() == weights else "fail", {"actual_sha256":sha256(weights), "recorded":metric.get("weights_sha256")})
            check(f"{condition}_identity", "pass" if metric.get("identity") == pair_identity(meta) else "fail", "Full per-image identity/GT hash, not counts")
        result["existing_metrics"] = baseline
        if set(baseline) == {"raw", "full"}:
            for key in ("parameters", "versions", "identity", "weights", "weights_sha256", "test_count"):
                check("paired_"+key, "pass" if baseline["raw"].get(key) == baseline["full"].get(key) and key in baseline["raw"] else "fail", baseline["raw"].get(key, "unknown"))
            result["evaluation_parameters"] = baseline["raw"].get("parameters", {})
            for key in ("imgsz", "conf", "iou", "agnostic_nms", "max_det", "half", "rect", "batch", "device", "augment"):
                check("recorded_eval_"+key, "pass" if key in result["evaluation_parameters"] else "unknown", result["evaluation_parameters"].get(key,"unknown"))
        else:
            check("paired_evaluation_protocol", "unknown", "Existing Raw/Full metrics JSON unavailable")
        from ..evaluate import frozen_registry
        try:
            registry = frozen_registry(prepared, weights, meta)
            result["detector_registry"] = registry
            for split,key in (("train","train_manifest_sha256"),("val","val_manifest_sha256")):
                check("detector_training_"+split+"_manifest", "pass" if sha256(prepared/"manifests"/f"{split}.tsv") == registry.get(key) else "fail",
                      {"actual":sha256(prepared/"manifests"/f"{split}.tsv"),"recorded":registry.get(key,"unknown")})
            training_yaml = Path(registry["output"])/"raw_train_val.yaml"
            if training_yaml.is_file():
                import yaml
                data_cfg = yaml.safe_load(training_yaml.read_text())
                root_cfg = Path(data_cfg["path"])
                raw_training = (root_cfg/str(data_cfg["train"])).resolve() == (prepared/"images/train_raw").resolve()
                raw_training = raw_training and (root_cfg/str(data_cfg["val"])).resolve() == (prepared/"images/val_raw").resolve() and "test" not in data_cfg
                check("detector_trained_on_raw_train_val_only", "pass" if raw_training else "fail", {"path":str(training_yaml),"content":data_cfg})
            else:
                check("detector_trained_on_raw_train_val_only", "unknown", f"Training YAML missing: {training_yaml}")
            check("training_locked_evaluation_parameters", "pass" if registry.get("evaluation") == result.get("evaluation_parameters") else "fail",
                  {"training_locked":registry.get("evaluation"),"existing_evaluation":result.get("evaluation_parameters","unknown")})
            from ..utils import require_yolo
            detector = require_yolo()(str(weights), task="detect")
            result["actual_detector"] = {"class_names":detector.names,"task":detector.task,"yaml":detector.model.yaml,
                                         "sha256":sha256(weights),"checkpoint_training_args":{
                                             k:detector.ckpt.get("train_args",{}).get(k,"unknown") for k in ("data","seed","epochs","batch","imgsz")}}
            check("actual_frozen_detector", "pass" if list(detector.names.values()) == meta["mapping"]["names"] else "fail", "Loaded existing local best.pt; no substitute/download/training")
            del detector
        except Exception as error:
            check("actual_frozen_detector", "fail", repr(error))
        valid = [r for r in records if r["status"] == "pass"]
        result["pixel_statistics"] = {}
        for condition in ("raw","full"):
            if valid:
                counts = np.array([r[condition]["width"]*r[condition]["height"] for r in valid], dtype=np.float64)
                means = np.array([r[condition]["mean_rgb"] for r in valid])
                stds = np.array([r[condition]["std_rgb"] for r in valid])
                mean = (means*counts[:,None]).sum(axis=0)/counts.sum()
                variance = ((stds**2+means**2)*counts[:,None]).sum(axis=0)/counts.sum()-mean**2
                result["pixel_statistics"][condition] = {"image_count":len(valid),"pixel_weighted_mean_rgb":mean.tolist(),
                    "pixel_weighted_std_rgb":np.sqrt(np.maximum(0,variance)).tolist(),
                    "mean_of_image_channel_means":means.mean(axis=0).tolist(),
                    **{k:float(sum(r[condition][k]*n for r,n in zip(valid,counts))/counts.sum()) for k in ("black_pixel_fraction","white_pixel_fraction")}}
        extremes = sorted(valid,key=lambda r: -r["raw_full_pixel_difference"]["mae_uint8"])[:20]
        result["extreme_sample_ids"] = {"largest_rgb_change":[r["image_id"] for r in extremes],
            "lowest_full_channel_std":[r["image_id"] for r in sorted(valid,key=lambda r:sum(r["full"]["std_rgb"]))[:20]],
            "highest_black_white_saturation":[r["image_id"] for r in sorted(valid,key=lambda r:-(r["full"]["black_pixel_fraction"]+r["full"]["white_pixel_fraction"]))[:20]]}
        dump_csv(out/"extreme_samples.csv", [{"image_id":r["image_id"],"file":r["file"],"rgb_mae_uint8":r["raw_full_pixel_difference"]["mae_uint8"],
                    "full_mean_rgb":r["full"]["mean_rgb"],"full_std_rgb":r["full"]["std_rgb"],"black_fraction":r["full"]["black_pixel_fraction"],"white_fraction":r["full"]["white_pixel_fraction"]} for r in extremes])
        result["input_guard"] = {"files":source_hashes}
    except Exception as error:
        check("audit_execution", "fail", repr(error))
    result["status"] = "fail" if any(c["critical"] and c["status"] == "fail" for c in checks) else "unknown" if any(c["critical"] and c["status"] == "unknown" for c in checks) else "pass"
    result["statistics_caveat"] = "Pixel statistics, BGR-alternative MAE, duplicate content and extremes are prompts for inspection, not proof of a cause. Equal dimensions/labels do not prove semantic content was not shifted; inspect GT previews."
    if records:
        for condition in ("raw","full"):
            groups = defaultdict(list)
            for row in records:
                if row["status"] == "pass":
                    groups[row[condition]["rgb_pixel_sha256"]].append(row["image_id"])
            result.setdefault("duplicate_content_groups",{})[condition] = [v for v in groups.values() if len(v)>1]
    write_json(out/"audit_protocol.json", result)
    lines = ["# P0-A protocol audit", "", f"Status: **{result['status']}**", "", "| Check | Status | Evidence |", "|---|---|---|"]
    lines += [f"| {c['check']} | {c['status']} | {str(c['detail']).replace('|','/')} |" for c in checks]
    lines += ["", "Paths:", "```json", json.dumps(result["paths"],indent=2), "```", "", result["statistics_caveat"],
              "", "Processing chain / recorded precision:", "```json", json.dumps(result.get("processing_chain",{}),indent=2), "```",
              "", "Pixel statistics (pixel-weighted global mean/std, and image-wise means):", "```json", json.dumps(result.get("pixel_statistics",{}),indent=2), "```",
              "", "Model hashes:", "```json", json.dumps({"yolo":result.get("actual_detector",{}).get("sha256","unknown"),"v16":result.get("uie_checkpoint_sha256","unknown")},indent=2), "```",
              "", "Actual source contents: source_snapshot/ (SHA256 and exact paths in source_inventory.json).", "GT overlays are copies only, selected once with seed 3520."]
    (out/"audit_protocol.md").write_text("\n".join(lines)+"\n")
    return result


def main(argv=None):
    args = common_parser(__doc__).parse_args(argv)
    out = fresh_output(args.output, (args.prepared,args.weights,args.enhanced))
    result = audit(args.prepared,args.weights,args.enhanced,out/"audit",args.evaluation_root,args.data_root,args.uie_run_dir)
    print(result["status"],out/"audit/audit_protocol.md")
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
