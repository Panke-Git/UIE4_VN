"""P0-D: honest decision gates; never infer an unmeasured full-test native result."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import *


def decision(audit,identity,comparison,subset):
    if audit.get("status") != "pass":
        return {"code":"stop_protocol_or_availability", "conclusion":"Protocol/data/model audit failed or is unavailable. No model-quality/tiling attribution is justified.","next":"Resolve the recorded failed/unknown checks using new diagnostic outputs; preserve old experiments."}
    if identity.get("status") != "pass":
        return {"code":"stop_identity_pipeline", "conclusion":"Identity control is missing or raised an engineering alarm. Investigate I/O, mapping, export and evaluation before other attribution.","next":"Inspect per-image Identity pixel errors and frozen evaluation arguments. Do not tune test thresholds."}
    if not comparison.get("complete") or not comparison.get("native_success_ids"):
        return {"code":"native_unavailable_cause_undetermined", "conclusion":"Identity passed, but full-native comparison is unavailable. The cause of the whole-test drop is undetermined.","next":"Inspect OOM/error evidence and sampled images. A larger-memory native experiment would require a separately fixed protocol."}
    records = [r for r in comparison["records"] if "existing_tiled_vs_native" in r]
    replay = [r.get("existing_vs_replayed_tiled") for r in comparison["records"]]
    if any(r is None or r["max_error_uint8"] > 1 for r in replay):
        return {"code":"original_tiled_replay_mismatch", "conclusion":"Existing tiled outputs were not reproduced to within one uint8 level on all planned IDs. Implementation/environment/history must be reconciled first.","next":"Compare stored checkpoint hash, source snapshot, versions, precision and existing-versus-replay pixel statistics."}
    scope = subset.get("sets",{}).get("native_common",{})
    values = scope.get("metrics",{})
    if not {"raw","tiled","native"}.issubset(values):
        return {"code":"subset_ap_not_evaluated", "conclusion":"Pixel/native diagnostics exist, but same-ID three-way AP is incomplete. Whole-test attribution remains unresolved.","next":"Run the diagnostic subset evaluation phase in the unchanged YOLO environment."}
    mean_mae = sum(r["existing_tiled_vs_native"]["mae_uint8"] for r in records)/len(records)
    delta_nt = {key:values["native"][key]-values["tiled"][key] for key in METRICS}
    delta_nr = {key:values["native"][key]-values["raw"][key] for key in METRICS}
    evidence = {"eligible_image_count":len(records),"planned_count":len(comparison["sample_ids"]),
                "native_success_multitile_ids":comparison.get("native_success_multitile_ids",[]),
                "mean_tiled_native_mae_uint8":mean_mae,"delta_native_minus_tiled_same_subset":delta_nt,
                "delta_native_minus_raw_same_subset":delta_nr,
                "engineering_descriptive_thresholds":{"mean_mae_uint8":1.0,"absolute_ap_difference":0.005},
                "limitations":"Thresholds describe engineering contrasts, not significance or causal attribution. Native-success images are hardware-selected; no inference about full 1111-image native AP."}
    if not comparison.get("native_success_multitile_ids"):
        return {"code":"native_evidence_only_single_tile_images",
                "conclusion":"Native success was limited to images fitting in one tile. Their tiled/native equivalence cannot test multi-tile global-conditioning inconsistency on larger OOM images. Whole-test cause remains undetermined.",
                "next":"Inspect the limited native results separately. Multi-tile native evidence is still required before ruling out tiling for high-resolution DUO images.","evidence":evidence}
    if mean_mae > 1 or any(abs(delta_nt[key]) > .005 for key in ("map50","map50_95")):
        return {"code":"tiling_difference_supported_on_subset","conclusion":"Identity passed and tiled/native differ on the same eligible subset. Prioritize tiling/global conditioning investigation; this does not establish that tiling explains the whole 28pp drop.",
                "next":"Inspect native/tiled grids, boundary controls, b/g traces and same-ID AP. Validate any alternative in a new run without overwriting existing results.","evidence":evidence}
    if delta_nr["map50"] < -.005 and delta_nr["map50_95"] < -.005:
        return {"code":"consistent_with_domain_or_task_information_change_on_subset",
                "conclusion":"Identity passed, native/tiled are close by the declared checks, and native remains below Raw on the same eligible subset. Evidence is consistent with input-domain or task-information changes; these two mechanisms are not separated.",
                "next":"Retain the negative result and inspect color/texture/failure cases. P1 requires explicit user approval after reviewing this report.","evidence":evidence}
    return {"code":"cause_undetermined","conclusion":"Checks completed within the measured subset but do not isolate the reason for the whole-test drop.","next":"Document the unresolved cause; do not expand into broad test-set parameter searches.","evidence":evidence}


def write_report(out):
    out = Path(out)
    def optional(relative):
        path = out/relative
        return read_json(path) if path.is_file() else {}
    audit = optional("audit/audit_protocol.json")
    identity = optional("identity/identity_control.json")
    comparison = optional("full_vs_tiled/full_vs_tiled.json")
    subset = optional("full_vs_tiled/subset_metrics.json")
    guard = optional("input_integrity_after.json")
    outcome = decision(audit,identity,comparison,subset)
    if guard.get("status") == "fail":
        outcome = {"code":"stop_input_integrity_failure","conclusion":"Protected inputs changed: results require investigation.","next":"Inspect input_integrity_after.json; original experiment must remain read-only."}
    summary = {"synthetic":audit.get("synthetic"),"given_negative_results":GIVEN_RESULTS,
        "given_results_provenance":"User-provided real full-test results; preserved, never substituted with synthetic or subset metrics",
        "measured_existing_results":{c:{k:m.get(k) for k in METRICS} for c,m in audit.get("existing_metrics",{}).items()},
        "audit_status":audit.get("status","not_run"),"identity":{k:v for k,v in identity.items() if k not in ("pixels",)},
        "full_vs_tiled":{"status":comparison.get("status","not_run"),"sample_ids":comparison.get("sample_ids"),
             "native_success_ids":comparison.get("native_success_ids"),"native_oom_ids":comparison.get("native_oom_ids"),
             "subset_metrics":subset,"manual_visual_review":comparison.get("manual_visual_review","not_run")},
        "decision":outcome,"input_integrity":guard or {"status":"unknown"},"p1_executed":False,
        "availability":optional("availability.json"),"runner":optional("runner_status.json")}
    write_json(out/"diagnosis_summary.json",summary)
    def fmt(metric):
        return f"{metric['map50']:.6f} / {metric['map50_95']:.6f}" if metric and all(isinstance(metric.get(k),(int,float)) for k in ("map50","map50_95")) else "not evaluated"
    lines = ["# DUO frozen-detector P0 diagnosis", "",
        "**SYNTHETIC SOFTWARE TEST — NOT REAL DUO DIAGNOSIS**" if summary["synthetic"] else "Existing negative main results retained unchanged.",
        "", "| Test | Result mAP50 / mAP50-95 | Scope / limit | Next |", "|---|---|---|---|",
        f"| Existing Raw 1111 (user-given) | {fmt(GIVEN_RESULTS['raw'])} | Original baseline preserved | Keep |",
        f"| Existing Tiled Full 1111 (user-given) | {fmt(GIVEN_RESULTS['full'])} | Negative result preserved | Keep |",
        f"| Protocol/coordinate/color audit | {summary['audit_status']} | Per-ID source/GT/hash checks; aggregate counts alone insufficient | Inspect audit/audit_protocol.md |",
        f"| Identity whole test | {fmt(identity.get('metrics'))}; {identity.get('status','not_run')} | Pixels and AP engineering threshold; no statistical significance claim | {'Continue only if pass' if identity.get('status') == 'pass' else 'STOP other attribution'} |",
        f"| Native vs tiled subset | {comparison.get('status','not_run')} | Successful native IDs: {comparison.get('native_success_ids',[])}; OOM IDs: {comparison.get('native_oom_ids',[])} | Read below and inspect all fixed sample figures |", ""]
    for scope,value in subset.get("sets",{}).items():
        lines += [f"## Subset: {scope}", "",f"IDs: {value.get('image_ids',[])}", ""]
        for condition,metric in value.get("metrics",{}).items():
            lines.append(f"- {condition}: {fmt(metric)} (diagnostic subset ONLY)")
        lines.append("")
    lines += ["## Decision", "",f"**{outcome['code']}**", "",outcome["conclusion"], "",outcome["next"], "",
              "```json",json.dumps(outcome.get("evidence",{}),indent=2),"```", "",
              "## Evidence and limits", "",
              "- Actual source text: audit/source_snapshot/; source_inventory.json records paths/SHA256.",
              "- audit/audit_protocol.json contains all per-image mappings, boxes/checks, hashes, pixel statistics and unknown historical fields.",
              "- Identity images: identity/test_identity/images/test_full/. No ICAR model was called.",
              "- full_vs_tiled/full_vs_tiled.json records each native OOM/error, GPU peak VRAM and conditioning hooks.",
              "- full_vs_tiled/figures/ contains every selected image/crop/overlay. Manual visual inspection is pending until actually performed.",
              "- PSNR compares two model predictions, not restoration ground truth. Band statistics alone do not establish visible seams.",
              "- Planned/subset/native-success AP must never replace the given entire-test numbers.",
              "- P1 is not executed. Enhanced-domain training needs separate explicit approval after P0 review.","",
              f"Input integrity: {summary['input_integrity'].get('status')}",
              "Runner/availability details: runner_status.json and availability.json."]
    (out/"diagnosis_summary.md").write_text("\n".join(lines)+"\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    args = parser.parse_args(argv)
    if not Path(args.run_dir).resolve().is_relative_to(DIAGNOSTICS.resolve()):
        raise ValueError("Report output must stay under diagnostics")
    print(write_report(args.run_dir)["decision"])


if __name__ == "__main__":
    main()
