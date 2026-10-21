"""Evaluate frozen Raw-trained best.pt on exactly the same entire official test."""
from __future__ import annotations

import argparse
from pathlib import Path

from .utils import (OUTPUT, digest_json, fresh_dir, load_prepared, output_path, pair_identity,
                    read_json, require_yolo, sha256, versions, write_json, write_yaml)


def frozen_registry(prepared, weights, meta):
    registry = read_json(Path(prepared) / "detector_training_lock.json")
    if registry["status"] != "complete" or registry["identity"] != pair_identity(meta):
        raise ValueError("Detector training is incomplete or belongs to a different dataset")
    if str(Path(weights).resolve()) != registry["weights"] or sha256(weights) != registry["weights_sha256"]:
        raise ValueError("A/B must use precisely the registered best.pt path AND SHA256")
    if registry["versions"] != versions():
        raise ValueError("Use the same detector software environment as training")
    return registry


def evaluate(prepared, weights, out, condition="raw", enhanced=None, device=None):
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    meta = load_prepared(prepared)
    registry = frozen_registry(prepared, weights, meta)
    if condition not in ("raw", "full"):
        raise ValueError("Only mandatory A=raw and B=full are implemented")
    YOLO = require_yolo()
    enhancement = None
    if condition == "full":
        if enhanced is None:
            raise ValueError("--enhanced is required for Full")
        from .model_inference_v16 import validate_enhanced
        enhancement = validate_enhanced(prepared, enhanced, meta)
    fresh_dir(out)
    image_root = (prepared / "images/test_raw" if condition == "raw"
                  else Path(enhanced).resolve() / "images/test_full")
    root = prepared if condition == "raw" else Path(enhanced).resolve()
    # All three paths intentionally point to test ONLY in this evaluation-only YAML.
    eval_yaml = out / "test_dataset.yaml"
    write_yaml(eval_yaml, {"path": str(root), "train": str(image_root), "val": str(image_root),
                          "test": str(image_root), "names": meta["mapping"]["names"]})
    rows = meta["splits"]["test"]
    by_path = {str((image_root / row["file"]).resolve()): row for row in rows}
    predictions = {row["image_id"]: {"image_id": row["image_id"], "file": row["file"],
                                    "width": row["width"], "height": row["height"], "detections": []}
                   for row in rows}
    seen = set()
    from ultralytics.models.yolo.detect import DetectionValidator
    class PairedValidator(DetectionValidator):
        def update_metrics(self, preds, batch):
            for filename in batch["im_file"]:
                path = str(Path(filename).resolve())
                if path not in by_path or path in seen:
                    raise ValueError(f"Unexpected/duplicate evaluation file {filename}")
                seen.add(path)
            super().update_metrics(preds, batch)

        def pred_to_json(self, predn, pbatch):
            # This hook receives the SAME post-NMS predictions as metric aggregation,
            # scaled back by the validator to the original image coordinates.
            row = by_path[str(Path(pbatch["im_file"]).resolve())]
            target = predictions[row["image_id"]]["detections"]
            for box, conf, cls in zip(predn["bboxes"].tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
                cls = int(cls)
                detection = {"class_id": cls, "category_id": meta["mapping"]["yolo_to_coco"][str(cls)],
                             "bbox_xyxy": box, "confidence": conf}
                target.append(detection)
                self.jdict.append({"image_id": row["image_id"], "category_id": detection["category_id"],
                                   "bbox": [box[0], box[1], box[2]-box[0], box[3]-box[1]], "score": conf})
    params = dict(registry["evaluation"])
    if device is not None:
        params["device"] = device
    model = YOLO(str(Path(weights).resolve()), task="detect")
    model.model.eval().requires_grad_(False)
    metrics = model.val(validator=PairedValidator, data=str(eval_yaml), split="test", save_json=True,
                        save_txt=False, project=str(out), name="validator", exist_ok=False, **params)
    if len(seen) != len(rows):
        raise ValueError(f"Validator skipped images: seen={len(seen)}, expected={len(rows)}")
    if sha256(weights) != registry["weights_sha256"]:
        raise ValueError("Detector weights changed during evaluation")
    box = metrics.box
    class_indices = {int(cls): i for i, cls in enumerate(box.ap_class_index)}
    per_class = []
    for cls, name in enumerate(meta["mapping"]["names"]):
        index = class_indices.get(cls)
        per_class.append({"class_id": cls, "category_id": meta["mapping"]["yolo_to_coco"][str(cls)], "name": name,
            "precision": float(box.p[index]) if index is not None else None,
            "recall": float(box.r[index]) if index is not None else None,
            "map50": float(box.ap50[index]) if index is not None else None,
            "map50_95": float(box.ap[index]) if index is not None else None})
    prediction_file = out / f"predictions_{condition}.json"
    write_json(prediction_file, [predictions[row["image_id"]] for row in rows])
    result = {"condition": condition, "synthetic": meta["synthetic"], "identity": pair_identity(meta),
              "weights": registry["weights"], "weights_sha256": registry["weights_sha256"],
              "versions": versions(), "parameters": params, "test_count": len(seen),
              "precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50),
              "map50_95": float(box.map), "per_class": per_class,
              "metric_definition": "Ultralytics detection val: dataset AP IoU=.50:.05:.95, class macro means; P/R at internally selected smoothed mean-F1 operating point, not at figure confidence; no per-image AP averaging",
              "predictions": str(prediction_file), "predictions_sha256": sha256(prediction_file),
              "image_root": str(image_root), "prepared": str(prepared),
              "enhancement": {k: enhancement[k] for k in ("provenance", "tile_size", "overlap", "output_encoding")} if enhancement else None}
    write_json(out / f"metrics_{condition}.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default=str(OUTPUT / "prepared_duo"))
    p.add_argument("--weights", required=True)
    p.add_argument("--condition", choices=["raw", "full"], required=True)
    p.add_argument("--enhanced")
    p.add_argument("--output", required=True)
    p.add_argument("--device")
    args = p.parse_args(argv)
    result = evaluate(args.prepared, args.weights, output_path(args.output), args.condition, args.enhanced, args.device)
    print({k: result[k] for k in ("condition", "test_count", "precision", "recall", "map50", "map50_95")})


if __name__ == "__main__":
    main()
