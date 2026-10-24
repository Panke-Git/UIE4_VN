"""P0-C: immutable preselected IDs, native-resolution replay, explicit OOM evidence."""
from __future__ import annotations

from contextlib import contextmanager
import gc
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from ..utils import load_prepared, versions
from ..model_inference_v16 import load_v16, enhance_tensor, starts
from .common import *
from .evaluator import frozen_evaluate


def fixed_sample(rows, boxes, count=32, seed=3520):
    if seed != 3520 or not 1 <= count <= 64:
        raise ValueError("Diagnostic sampling fixes seed=3520 and supports at most 64 images")
    rng = random.Random(seed)
    ids = sorted(r["image_id"] for r in rows)
    by_id = {r["image_id"]:r for r in rows}
    chosen = []
    def add(candidates):
        available = sorted(set(candidates)-set(chosen))
        if available and len(chosen) < min(count,len(ids)):
            chosen.append(rng.choice(available))
    # Coverage first, using GT presence/size only. No detector prediction is consulted.
    for cls in sorted({b[0] for values in boxes.values() for b in values}):
        add([iid for iid in ids if any(b[0] == cls for b in boxes[iid])])
    sizes = sorted({(r["width"],r["height"]) for r in rows},key=lambda wh:wh[0]*wh[1])
    for size in sizes:
        add([iid for iid in ids if (by_id[iid]["width"],by_id[iid]["height"]) == size])
    remaining = sorted(set(ids)-set(chosen))
    rng.shuffle(remaining)
    chosen += remaining[:min(count,len(ids))-len(chosen)]
    return sorted(chosen)


def freeze_sample(path, payload):
    path = Path(path)
    # Never silently select another set or rewrite an existing sample file.
    with path.open("x",encoding="utf-8") as stream:
        json.dump(payload,stream,indent=2)
        stream.write("\n")
    path.with_suffix(".sha256").write_text(sha256(path)+"\n")


def boundary_statistics(raw, enhanced, tile, overlap, band=2):
    h,w = raw.shape[:2]
    rows = []
    if tile <= 0:
        return {"status":"not_applicable_no_tiles"}
    for axis,length in ((0,h),(1,w)):
        positions = sorted({p for start in starts(length,tile,overlap) for p in (start,min(start+tile,length)) if 0 < p < length})
        a = np.abs(np.diff(raw.astype(np.float32),axis=axis)).mean(axis=2)
        b = np.abs(np.diff(enhanced.astype(np.float32),axis=axis)).mean(axis=2)
        mask = np.zeros(a.shape,dtype=bool)
        for pos in positions:
            sl = [slice(None),slice(None)]
            sl[axis] = slice(max(0,pos-band),min(length-1,pos+band))
            mask[tuple(sl)] = True
        diff = np.abs(b-a)
        rows.append({"axis":"y" if axis == 0 else "x","tile_start_end_positions":positions,
            "boundary_pixel_gaps":int(mask.sum()),
            "raw_boundary_gradient_uint8":float(a[mask].mean()) if mask.any() else None,
            "enhanced_boundary_gradient_uint8":float(b[mask].mean()) if mask.any() else None,
            "boundary_gradient_change_uint8":float(diff[mask].mean()) if mask.any() else None,
            "nonboundary_gradient_change_uint8":float(diff[~mask].mean()) if (~mask).any() else None})
    return {"status":"measured","band_pixels":band,"axes":rows,
            "caveat":"Tile start/end bands overlap blending regions; compare with original-image natural edges and nonboundary gaps. These statistics do not establish visible seams or causation."}


@contextmanager
def conditioning_trace(model):
    trace = []
    current = {}
    handles = []
    def image_hook(module,args):
        current.clear()
        current["uicf_input_bchw"] = list(args[0].shape)
    def spatial_hook(module,args,output):
        current["coordinate_hw"] = [int(args[1]),int(args[2])]
        current["coordinate_extent"] = "linspace(-1,1) independently for this call, source verified"
    def anchor_hook(module,args,output):
        anchor,g = output
        trace.append({**current,"anchor_b":anchor.detach().float().cpu().tolist(),
                      "global_gap_g":g.detach().float().cpu().tolist()})
    try:
        handles = [model.uicf.register_forward_pre_hook(image_hook),
                   model.uicf.spatial_encoding.register_forward_hook(spatial_hook),
                   model.uicf.chromatic_anchor.register_forward_hook(anchor_hook)]
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def infer_one(model,tensor,path,device,tile,overlap):
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    record = {"status":"unknown","input_hw":list(tensor.shape[-2:]),"tile_size":tile,
              "overlap":overlap if tile else 0,"dtype":"float32","amp":False}
    array = None
    try:
        with conditioning_trace(model) as trace:
            prediction = enhance_tensor(model,tensor,device,tile,overlap)
            array = save_tensor(prediction,path)
        record.update(status="ok",output=str(Path(path).resolve()),sha256=sha256(path),conditioning_trace=trace)
    except torch.OutOfMemoryError as error:
        record.update(status="oom",error=str(error),traceback=traceback.format_exc(),
                      conditioning_trace=trace if "trace" in locals() else [])
        print(f"Native/tiled OOM: input={record['input_hw']} tile={tile}: {error}",flush=True)
    except Exception as error:
        record.update(status="error",error=repr(error),traceback=traceback.format_exc())
    finally:
        if cuda:
            record.update(gpu_name=torch.cuda.get_device_name(device),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                          gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory)
        if "prediction" in locals():
            del prediction
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    return array,record


def infer_comparison(prepared,enhanced,out,audit_result,identity_result,uie_run_dir,gpu="0",count=32):
    if audit_result["status"] != "pass" or identity_result.get("status") != "pass":
        raise ValueError("P0-C is forbidden until protocol AND Identity gates pass")
    prepared,enhanced,out = Path(prepared).resolve(),Path(enhanced).resolve(),Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    runtime_dirs(out)
    result = {"status":"running","complete":False,"records":[],"native_success_ids":[],"native_oom_ids":[],
              "versions":versions(),"precision":"float32, inference_mode, eval, no AMP, no resize",
              "manual_visual_review":"pending: inspect all sampled grids and crops; automated bands do not prove visible seams"}
    try:
        original_versions = audit_result["enhancement"]["versions"]
        for key in ("torch","numpy","Pillow"):
            if not isinstance(original_versions,dict) or versions().get(key) != original_versions.get(key):
                raise ValueError(f"Run this phase in original V16 environment; {key} differs from enhancement manifest")
        if gpu != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: native inference NOT evaluated; no silent CPU/resize fallback")
        device = "cpu" if gpu == "cpu" else f"cuda:{gpu}"
        model,provenance = load_v16(uie_run_dir,device=device)
        if provenance["checkpoint_sha256"] != audit_result["uie_checkpoint_sha256"]:
            raise ValueError("Full/tiled replay checkpoint differs from original enhancement")
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Expected FP32 weights matching original no-AMP source")
        result["provenance"] = provenance
        meta = load_prepared(prepared,verify=False)
        doc = read_json(Path(audit_result["paths"]["data_root"])/"annotations/instances_test.json")
        _,boxes,_ = expected_coco(doc,meta)
        ids = fixed_sample(meta["splits"]["test"],boxes,count)
        by_id = {r["image_id"]:r for r in meta["splits"]["test"]}
        sample = {"seed":3520,"requested_count":count,"image_ids":ids,"dataset_digest":meta["dataset_digest"],
                  "method":"Seeded class-presence coverage, exact width/height coverage where capacity permits, then seeded fill. No detection results used. Not a representative test estimate.",
                  "images":[by_id[iid] for iid in ids]}
        freeze_sample(out/"sample_ids.json",sample)
        result["sample_sha256"] = sha256(out/"sample_ids.json")
        result["sample_ids"] = ids
        tile,overlap = audit_result["enhancement"]["tile_size"],audit_result["enhancement"]["overlap"]
        if not isinstance(tile,int) or tile <= 0 or not isinstance(overlap,int) or not 0 < overlap < tile:
            raise ValueError("Original tiled parameters unavailable or invalid")
        result.update(tile_size=tile,overlap=overlap,coordinate_conditioning_risk="Source confirms every independent tile recalculates local normalized coordinates, GAP g and anchor b; Color-Query context is tile-local too. Non-equivalence alone is NOT evidence that it caused the whole-test AP drop.")
        for iid in ids:
            row = by_id[iid]
            raw = rgb(prepared/"images/test_raw"/row["file"])
            old = rgb(enhanced/"images/test_full"/row["file"])
            tensor = rgb_tensor(raw)
            replay,replay_record = infer_one(model,tensor,out/"recomputed_tiled"/row["file"],device,tile,overlap)
            native,native_record = infer_one(model,tensor,out/"native"/row["file"],device,0,0)
            record = {"image_id":iid,"file":row["file"],"width":row["width"],"height":row["height"],
                      "tile_count":len(starts(raw.shape[0],tile,overlap))*len(starts(raw.shape[1],tile,overlap)),
                      "tiled_replay":replay_record,"full_native":native_record,
                      "existing_tiled_statistics":image_statistics(old),"raw_statistics":image_statistics(raw),
                      "boundary_statistics":boundary_statistics(raw,old,tile,overlap)}
            if replay is not None:
                record["existing_vs_replayed_tiled"] = pixel_difference(old,replay)
                anchors = np.array([t["anchor_b"][0] for t in replay_record["conditioning_trace"]])
                if len(anchors):
                    record["tile_anchor_b_range_rgb"] = np.ptp(anchors,axis=0).tolist()
            if native is not None:
                result["native_success_ids"].append(iid)
                record["existing_tiled_vs_native"] = pixel_difference(old,native)
                record["native_statistics"] = image_statistics(native)
                if replay is not None:
                    record["replayed_tiled_vs_native"] = pixel_difference(replay,native)
                comparison = [raw,old,native]
                titles = [f"Raw id={iid}","Existing tiled V16","Full native V16"]
            else:
                if native_record["status"] == "oom":
                    result["native_oom_ids"].append(iid)
                comparison = [raw,old]
                titles = [f"Raw id={iid}",f"Tiled; native {native_record['status']} (not evaluated)"]
            preview_grid(comparison,titles,out/"figures"/f"{iid}_overview.png")
            # A fixed center crop plus a tile-border crop, not selected by positive results.
            h,w = raw.shape[:2]
            sx = starts(w,tile,overlap)
            cx = sx[1] if len(sx)>1 else w//2
            crop = (max(0,cx-64),max(0,h//2-64),min(w,cx+64),min(h,h//2+64))
            preview_grid(comparison,titles,out/"figures"/f"{iid}_tile_border_crop.png",width=384,crop=crop)
            overlay = ((raw.astype(np.uint16)+old.astype(np.uint16))/2).round().astype(np.uint8)
            preview_grid([raw,old,overlay],["Raw","Tiled","50% overlay Raw/Tiled"],out/"figures"/f"{iid}_raw_tiled_overlay.png")
            result["records"].append(record)
            write_json(out/"full_vs_tiled.json",result)
            print(f"Full-vs-tiled id={iid}: native={native_record['status']}, replay={replay_record['status']}",flush=True)
        result["complete"] = True
        result["native_success_multitile_ids"] = [r["image_id"] for r in result["records"]
                                                   if r["full_native"]["status"] == "ok" and r["tile_count"] > 1]
        result["status"] = "measured" if result["native_success_ids"] else "native_not_evaluated"
        if any(r["tiled_replay"]["status"] != "ok" or r["full_native"]["status"] == "error" for r in result["records"]):
            result["status"] = "inference_errors"
        if sha256(out/"sample_ids.json") != result["sample_sha256"] or sha256(provenance["checkpoint"]) != provenance["checkpoint_sha256"]:
            raise ValueError("Sample list or checkpoint changed during diagnosis")
    except Exception as error:
        result.update(status="not_evaluated",error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(out/"full_vs_tiled.json",result)
        lines = ["# P0-C native / tiled", "",f"Status: {result['status']}","",
                 "PSNR compares two predictions, not restoration GT. No input is resized.","",
                 "| ID | Width x height | Tiles | Native status | Peak allocated bytes | Tiled-native MAE uint8 | Prediction PSNR dB |",
                 "|---|---|---|---|---|---|---|"]
        for record in result["records"]:
            native = record["full_native"]
            diff = record.get("existing_tiled_vs_native",{})
            lines.append(f"| {record['image_id']} | {record['width']} x {record['height']} | {record['tile_count']} | {native['status']} | {native.get('peak_allocated_bytes','CPU / unavailable')} | {diff.get('mae_uint8','not evaluated')} | {diff.get('psnr_between_predictions_db','not evaluated')} |")
        lines += ["", "Full per-image errors/tracebacks/conditioning b/g/coordinate hooks/boundary controls are in full_vs_tiled.json.",
                  "All selected grids, raw/tiled overlays and border crops are in figures/. Manual seam inspection is pending.",
                  "Native OOM images have no substitute output and are excluded only from the explicitly reported native-success common subset.",
                  "Subset AP, when available, is recorded separately in subset_metrics.json and never replaces whole-test AP."]
        (out/"full_vs_tiled.md").write_text("\n".join(lines)+"\n")
    return result


def evaluate_comparison(prepared,weights,enhanced,out,audit_result,identity_result,gpu="0"):
    if audit_result["status"] != "pass" or identity_result.get("status") != "pass":
        raise ValueError("Cannot evaluate P0-C after an Identity/protocol alarm")
    out,prepared,enhanced = Path(out),Path(prepared),Path(enhanced)
    result = read_json(out/"full_vs_tiled.json")
    if not result.get("complete") or sha256(out/"sample_ids.json") != result["sample_sha256"]:
        raise ValueError("Incomplete inference or changed fixed sample list")
    meta = load_prepared(prepared,verify=False)
    by_id = {r["image_id"]:r for r in meta["splits"]["test"]}
    baseline = audit_result["existing_metrics"]["raw"]
    planned,common = result["sample_ids"],result["native_success_ids"]
    evaluation = {"planned_ids":planned,"native_common_ids":common,"sets":{},
                  "caveat":"AP here is diagnostic subset AP only. OOM-success intersection is hardware-selected and cannot estimate the entire official test effect."}
    for scope,ids in (("planned",planned),("native_common",common)):
        if not ids:
            evaluation["sets"][scope] = {"status":"not_evaluated","reason":"No full-native successful image"}
            continue
        if scope == "native_common" and ids == planned:
            evaluation["sets"][scope] = evaluation["sets"]["planned"]
            continue
        rows = [by_id[iid] for iid in ids]
        conditions = {"raw":prepared/"images/test_raw","tiled":enhanced/"images/test_full"}
        if ids == common:
            conditions["native"] = out/"native"
        values = {}
        for condition,images in conditions.items():
            values[condition] = frozen_evaluate(prepared,weights,images,rows,out/"evaluation"/scope/condition,
                                                condition,baseline,gpu)
        evaluation["sets"][scope] = {"status":"measured","metrics":values,"image_ids":ids}
    write_json(out/"subset_metrics.json",evaluation)
    return evaluation


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--phase",choices=["infer","evaluate"],required=True)
    parser.add_argument("--audit-json",required=True)
    parser.add_argument("--identity-json",required=True)
    parser.add_argument("--sample-count",type=int,default=32)
    args = parser.parse_args(argv)
    if not Path(args.output).resolve().is_relative_to(DIAGNOSTICS.resolve()):
        raise ValueError("Output must be below diagnostics root")
    audit_result,identity_result = read_json(args.audit_json),read_json(args.identity_json)
    if args.phase == "infer":
        if not args.uie_run_dir:
            parser.error("--uie-run-dir required for infer")
        infer_comparison(args.prepared,args.enhanced,args.output,audit_result,identity_result,args.uie_run_dir,args.gpu,args.sample_count)
    else:
        evaluate_comparison(args.prepared,args.weights,args.enhanced,args.output,audit_result,identity_result,args.gpu)


if __name__ == "__main__":
    main()
