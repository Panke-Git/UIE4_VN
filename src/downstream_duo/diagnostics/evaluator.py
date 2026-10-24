"""Frozen generic diagnostic evaluator. All cacheable inputs are private COPIES."""
from __future__ import annotations

from pathlib import Path
import shutil

from ..utils import load_prepared, require_yolo, versions
from ..evaluate import frozen_registry
from .common import *


def build_dataset(prepared, images, rows, out, names):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for row in rows:
        filename = relative_image_file(row["file"])
        original_image = Path(images)/filename
        original_label = Path(prepared)/"labels/test_raw"/filename.with_suffix(".txt")
        dest_image = out/"images/test"/filename
        dest_label = out/"labels/test"/filename.with_suffix(".txt")
        for source, dest in ((original_image,dest_image),(original_label,dest_label)):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source,dest)  # no hardlink/symlink which might allow source mutation
            if sha256(source) != sha256(dest):
                raise ValueError("Diagnostic copy checksum mismatch")
        if sha256(original_label) != row["label_sha256"]:
            raise ValueError("Diagnostic GT differs from audited test GT")
        if rgb(dest_image).shape != (row["height"],row["width"],3):
            raise ValueError("Diagnostic image geometry changed")
    yaml_path = out/"dataset.yaml"
    write_yaml(yaml_path,{"path":str(out.resolve()),"train":"images/test","val":"images/test","test":"images/test","names":names})
    return yaml_path


def frozen_evaluate(prepared, weights, images, rows, out, condition, baseline, gpu="0"):
    prepared, weights, out = Path(prepared).resolve(), Path(weights).resolve(), Path(out).resolve()
    meta = load_prepared(prepared,verify=False)
    registry = frozen_registry(prepared,weights,meta)
    if baseline.get("weights_sha256") != sha256(weights) or Path(baseline.get("weights","")).resolve() != weights:
        raise ValueError("Baseline and diagnostics must use the exact same best.pt path/SHA256")
    if baseline.get("versions") != versions():
        raise ValueError("Evaluate diagnostics in original detector environment")
    params = dict(baseline["parameters"])
    if str(params.get("device")) != str(gpu):
        raise ValueError("Diagnostic GPU must match original evaluation device; no implicit device override")
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty, unique, predeclared diagnostic IDs")
    out.mkdir(parents=True,exist_ok=False)
    runtime_dirs(out)
    dataset = build_dataset(prepared,images,rows,out/"dataset",meta["mapping"]["names"])
    by_path = {str((out/"dataset/images/test"/r["file"]).resolve()):r for r in rows}
    predictions = {r["image_id"]:{"image_id":r["image_id"],"file":r["file"],"detections":[]} for r in rows}
    seen, effective = set(), {}
    YOLO = require_yolo()
    from ultralytics.models.yolo.detect import DetectionValidator
    class DiagnosticValidator(DetectionValidator):
        def update_metrics(self,preds,batch):
            for filename in batch["im_file"]:
                key = str(Path(filename).resolve())
                if key not in by_path or key in seen:
                    raise ValueError("Diagnostic evaluator saw unexpected/duplicate image")
                seen.add(key)
            super().update_metrics(preds,batch)

        def pred_to_json(self,predn,pbatch):
            row = by_path[str(Path(pbatch["im_file"]).resolve())]
            for box,conf,cls in zip(predn["bboxes"].tolist(),predn["conf"].tolist(),predn["cls"].tolist()):
                cls = int(cls)
                cid = meta["mapping"]["yolo_to_coco"][str(cls)]
                detection = {"bbox_xyxy":box,"confidence":conf,"class_id":cls,"category_id":cid}
                predictions[row["image_id"]]["detections"].append(detection)
                self.jdict.append({"image_id":row["image_id"],"category_id":cid,
                                   "bbox":[box[0],box[1],box[2]-box[0],box[3]-box[1]],"score":conf})

        def finalize_metrics(self):
            super().finalize_metrics()
            effective.update({key:value if isinstance(value,(str,int,float,bool,list,dict,type(None))) else str(value)
                              for key,value in vars(self.args).items()})
    model = YOLO(str(weights),task="detect")
    model.model.eval().requires_grad_(False)
    metric = model.val(validator=DiagnosticValidator,data=str(dataset),split="test",save_json=True,
                       save_txt=False,project=str(out),name="validator",exist_ok=False,**params)
    if seen != set(by_path) or sha256(weights) != registry["weights_sha256"]:
        raise ValueError("Diagnostic coverage/weights changed")
    box = metric.box
    result = {"condition":condition,"synthetic":meta["synthetic"],"test_count":len(rows),
        "image_ids":[r["image_id"] for r in rows],"weights":str(weights),"weights_sha256":sha256(weights),
        "parameters":params,"effective_validator_args":effective,"versions":versions(),
        "precision":float(box.mp),"recall":float(box.mr),"map50":float(box.map50),"map50_95":float(box.map),
        "scope":"whole official test" if len(rows)==len(meta["splits"]["test"]) else "diagnostic subset ONLY; not comparable to whole-test AP",
        "per_class":[{"class_id":int(cls),"name":meta["mapping"]["names"][int(cls)],"precision":float(box.p[i]),
                      "recall":float(box.r[i]),"map50":float(box.ap50[i]),"map50_95":float(box.ap[i])}
                     for i,cls in enumerate(box.ap_class_index)]}
    write_json(out/f"metrics_{condition}.json",result)
    write_json(out/"per_image_predictions.json",list(predictions.values()))
    return result
