"""P0-B: same enhancement I/O with identity(x)=x; whole-test frozen YOLO control."""
from __future__ import annotations

from pathlib import Path

import torch

from ..utils import load_prepared, versions
from ..model_inference_v16 import enhance_test
from .common import *
from .evaluator import frozen_evaluate


def identity_gate(metrics, raw, pixel_exact, tolerance=0.005):
    delta = {key:metrics[key]-raw[key] for key in METRICS}
    ap_ok = abs(delta["map50"]) <= tolerance and abs(delta["map50_95"]) <= tolerance
    return {"status":"pass" if pixel_exact and ap_ok else "stop",
            "delta_identity_minus_raw":delta,"pixel_exact":pixel_exact,"ap_engineering_threshold":tolerance,
            "note":"Engineering alarm only, not statistical significance. A stop does not establish a specific cause; no further tiling/domain attribution is allowed."}


def run_identity(prepared,weights,enhanced,out,audit_result,gpu="0"):
    if audit_result["status"] != "pass":
        raise ValueError("Identity requires successful per-image/protocol audit")
    prepared, out = Path(prepared).resolve(),Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    meta = load_prepared(prepared,verify=False)
    source = Path(audit_result["paths"]["data_root"])
    tile = audit_result["enhancement"]["tile_size"]
    overlap = audit_result["enhancement"]["overlap"]
    if not isinstance(tile,int) or not isinstance(overlap,int):
        raise ValueError("Original tile/overlap unknown: cannot claim identical identity pipeline")
    # No V16 build/load/forward is called. Reuse old exact decode/tensor/pad/tile/save/label code.
    result = {"status":"running","synthetic":meta["synthetic"],"identity_model":"torch.nn.Identity; no ICAR weights loaded",
              "io_method":"unmodified src.downstream_duo.model_inference_v16.enhance_test with identity model",
              "versions":versions(),"original_uie_versions":audit_result["enhancement"]["versions"],
              "tile_size":tile,"overlap":overlap,"pixels":[]}
    write_json(out/"identity_control.json",result)
    try:
        torch.set_num_threads(min(4,torch.get_num_threads()))
        identity = torch.nn.Identity().eval().requires_grad_(False)
        print(f"Identity export: {len(meta['splits']['test'])} images, no V16 call. Progress: {out/'test_identity/enhancement_progress.jsonl'}",flush=True)
        enhance_test(prepared,out/"test_identity",identity,{"diagnostic":"identity_without_uie"},
                     device="cpu",tile_size=tile,overlap=overlap)
        # The original B input is verified lossless prepared Raw PNG. Compare to both it
        # and the original COCO-resolved source image; file bytes may of course differ.
        by_id = {r["image_id"]:r for r in audit_result["images"]}
        for row in meta["splits"]["test"]:
            iid = row["image_id"]
            actual = rgb(out/"test_identity/images/test_full"/row["file"])
            original = rgb(by_id[iid]["paths"]["original"])
            raw = rgb(prepared/"images/test_raw"/row["file"])
            label = out/"test_identity/labels/test_full"/Path(row["file"]).with_suffix(".txt")
            if sha256(label) != row["label_sha256"]:
                raise ValueError("Identity label bytes differ")
            result["pixels"].append({"image_id":iid,"file":row["file"],
                "versus_source":pixel_difference(actual,original),"versus_prepared_raw":pixel_difference(actual,raw)})
        pixel_exact = all(r["versus_prepared_raw"]["identical"] for r in result["pixels"])
        result["pixel_exact"] = pixel_exact
        result["pixel_reference"] = "Canonical prepared Raw PNG actually evaluated by the frozen baseline detector"
        result["max_pixel_error_uint8"] = max(r["versus_prepared_raw"]["max_error_uint8"] for r in result["pixels"])
        result["max_source_decode_error_uint8"] = max(r["versus_source"]["max_error_uint8"] for r in result["pixels"])
        baseline = audit_result["existing_metrics"]["raw"]
        metric = frozen_evaluate(prepared,weights,out/"test_identity/images/test_full",meta["splits"]["test"],
                                 out/"evaluation","identity",baseline,gpu)
        write_json(out/"metrics_identity.json",metric)
        result["metrics"] = metric
        result["gate"] = identity_gate(metric,baseline,pixel_exact)
        result["status"] = result["gate"]["status"]
    except Exception as error:
        result.update(status="stop",error=repr(error))
        raise
    finally:
        write_json(out/"identity_control.json",result)
        lines = ["# P0-B Identity control", "", f"Status: {result['status']}", "",
                 "No ICAR-UIE model/weights were called. Images and labels are new copies only.", "",
                 "```json",json.dumps({k:v for k,v in result.items() if k != "pixels"},indent=2),"```"]
        (out/"identity_control.md").write_text("\n".join(lines)+"\n")
    return result


def main(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--audit-json",required=True)
    args = parser.parse_args(argv)
    out = fresh_output(args.output,(args.prepared,args.weights,args.enhanced))
    result = run_identity(args.prepared,args.weights,args.enhanced,out/"identity",read_json(args.audit_json),args.gpu)
    print(result["status"])
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
