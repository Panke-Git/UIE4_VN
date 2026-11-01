"""Paired image-level audit of a frozen model's saved intervention results."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import statistics

from . import METRICS, VARIANTS, MEAN_ATOL
from .provenance import verify_identity, valid_sha256
from .statistics import bootstrap, check_rows, compare_saved, describe, number, paired_gains, equal_number


def audit_diagnostics(audit,directory,protocol,expected):
    section="intervention:"+str(directory)
    path=directory/"diagnostics.csv"
    rows=audit.read(path,"csv",section=section)
    if rows is None:
        return
    check_rows(audit,section,rows,expected,path,allow_variants=True)
    groups={}
    for row in rows:
        groups.setdefault(row.get("sample_id"),{})[row.get("variant")]=row
    dtypes={g.get("sample_id"):g.get("dtypes",{}).get("R") for g in protocol.get("sample_geometry",[])}
    for sample_id,group in groups.items():
        original=group.get("original")
        if not original:
            audit.check(section,"diagnostics_original","UNVERIFIED",sample_id,[path])
            continue
        tolerance=4*(2**-10 if dtypes.get(sample_id)=="torch.float16" else 2**-23)
        for variant,row in group.items():
            try:
                for key,value in row.items():
                    if key not in ("sample_id","filename","sample_index","variant"):
                        number(value)
                for color in "rgb":
                    if not equal_number(row.get("b_"+color),original.get("b_"+color)):
                        raise ValueError(f"{variant} changed b_{color} or anchor diagnostic is missing")
                permutation={"channel_cycle_1":(1,2,0),"channel_cycle_2":(2,0,1)}.get(variant,(0,1,2))
                if variant in ("spatial_mean","shift_xy","shift_y","shift_x","channel_cycle_1","channel_cycle_2"):
                    for c,index in zip("rgb",permutation):
                        a,b=number(row.get("mean_R_"+c)),number(original.get("mean_R_"+"rgb"[index]))
                        if a is None or b is None:
                            raise ValueError("Missing per-channel mean diagnostic")
                        if abs(a-b)>tolerance+tolerance*abs(b):
                            raise ValueError(f"{variant}: mean_R_{c} not preserved/permuted")
                if variant in ("zero_field","original"):
                    keys=(("mean_abs_R","std_R","mean_abs_delta_Ic_variant") if variant=="zero_field"
                          else ("mean_abs_Ic_variant_minus_Ic_original","mean_abs_Dout_variant_minus_Dout_original"))
                    for key in keys:
                        if not equal_number(row.get(key),0):
                            raise ValueError(f"{variant}: nonzero or missing {key}")
                if variant=="spatial_mean":
                    for color in "rgb":
                        value=number(row.get("std_R_"+color))
                        if value is None or abs(value)>tolerance:
                            raise ValueError(f"spatial_mean: spatial std_R_{color} nonzero/missing")
            except ValueError as error:
                audit.check(section,"diagnostics_invariant","FAIL",f"{sample_id}: {error}",[path],"P0")
    present=Counter(row.get("variant") for row in rows)
    for variant in VARIANTS:
        audit.check(section,"diagnostics_count_"+variant,"PASS" if present[variant]==428 else "UNVERIFIED",
                    f"{present[variant]}/428",[path])


def protocol_checks(audit,directory,protocol,summary,identity):
    section="intervention:"+str(directory)
    path=directory/"protocol.json"
    verify_identity(audit,protocol,identity,section,path)
    for key,expected in (("script_id","v16_uicf_field_intervention"),("status","PASS"),
                         ("debug_partial",False),("complete_formal_evaluation",True),
                         ("successful_sample_count",428),("failed_sample_count",0),
                         ("requested_sample_count",428),("evaluation_mode","in_domain"),
                         ("protected_files_unchanged",True)):
        value=protocol.get(key)
        state="UNVERIFIED" if value is None else ("PASS" if value==expected else (
            "UNVERIFIED" if key in ("debug_partial","complete_formal_evaluation","requested_sample_count","status") else "FAIL"))
        audit.check(section,"protocol_"+key,state,f"{key}={value!r}; expected={expected!r}",[path],"P0")
    for key in ("evaluation","metrics"):
        actual=protocol.get(key)
        expected=identity["config"].get(key)
        audit.check(section,"protocol_"+key,"UNVERIFIED" if actual is None or expected is None else (
            "PASS" if actual==expected else "FAIL"),f"{actual!r}; frozen={expected!r}",[path],"P0")
    for key,expected in (("batch_size",1),("metric_input","Dout.float().clamp(0,1) before PNG; no Ic clamp"),
                         ("metric_aggregation","per-image metric then equally weighted image arithmetic mean"),
                         ("psnr_ssim_implementation","src.v16.metrics.batch_metrics (unchanged)")):
        value=protocol.get(key)
        audit.check(section,"metric_protocol_"+key,"UNVERIFIED" if value is None else (
            "PASS" if value==expected else "UNVERIFIED"),f"{value!r}; 已审阅定义={expected!r}",[path])
    for key,expected in (("lab_illuminant","D65"),("lab_observer","2 degrees"),("kL",1),("kC",1),("kH",1),("crop_border",0)):
        value=protocol.get("e00_protocol",{}).get(key)
        audit.check(section,"e00_protocol_"+key,"UNVERIFIED" if value is None else (
            "PASS" if value==expected else "FAIL"),f"{value!r}; expected={expected!r}",[path],"P0")
    definition=protocol.get("interventions",{})
    expected_definitions={
        "Ic":"I + R_variant * (I - original b)",
        "Dout":"unchanged model.backbone(Ic_variant)",
        "transformed":"R only; never I, b, GT, tokens or R*(I-b)",
        "permutations_new_RGB_from_old_indices":{"channel_cycle_1":[1,2,0],"channel_cycle_2":[2,0,1]},
        "no_multi_shift_aggregate":True,
    }
    for key,expected in expected_definitions.items():
        value=definition.get(key)
        audit.check(section,"intervention_definition_"+key,"UNVERIFIED" if value is None else (
            "PASS" if value==expected else "UNVERIFIED"),
            f"记录={value!r}; 当前已审阅定义={expected!r}; 文本差异需要审阅",[path])
    before,after=protocol.get("frozen_model_digest_before"),protocol.get("frozen_model_digest_after")
    audit.check(section,"frozen_parameters_and_buffers","UNVERIFIED" if not before or not after else (
        "FAIL" if not valid_sha256(before) or not valid_sha256(after) else
        "PASS" if before==after else "FAIL"),f"before={before}; after={after}",[path],"P0")
    normal=protocol.get("normal_forward_consistency",{})
    audit.check(section,"original_full_forward_equivalence","PASS" if normal.get("status")=="PASS" and normal.get("samples") else (
        "FAIL" if normal.get("status")=="FAIL" else "UNVERIFIED"),str(normal.get("status")),[path],"P0")
    consistency_errors=[]
    for sample in normal.get("samples",[]):
        for key in ("normal_vs_details","backbone_vs_details","program_original"):
            check=sample.get("checks",{}).get(key)
            state="UNVERIFIED" if not check else ("PASS" if check.get("status")=="PASS" else "FAIL")
            audit.check(section,"normal_consistency_"+key,state,f"sample_id={sample.get('sample_id')}; {check}",[path],"P0")
            if check:
                try:
                    value=number(check.get("max_abs_error"))
                    if value is None or value<0:
                        raise ValueError("缺少/非法 max_abs_error")
                    consistency_errors.append(value)
                except ValueError as error:
                    audit.check(section,"normal_consistency_error","FAIL",str(error),[path],"P0")
    if consistency_errors:
        compare_saved(audit,section,"normal_consistency_max_error",max(consistency_errors),normal.get("max_abs_error"),path)
    geometries=protocol.get("sample_geometry",[])
    ids=[g.get("sample_id") for g in geometries]
    expected_ids={r[0] for r in identity["splits"]["test"] or []}
    geometry_ok=len(ids)==len(set(ids))==428 and set(ids)==expected_ids
    audit.check(section,"geometry_membership","PASS" if geometry_ok else "UNVERIFIED",
                f"geometry={len(ids)}, unique={len(set(ids))}",[path])
    bad=[]
    for item in geometries:
        h,w,dy,dx=(item.get(k) for k in ("height","width","shift_dy","shift_dx"))
        if (h,w,dy,dx)!=(256,256,64,64):
            bad.append(item.get("sample_id"))
    audit.check(section,"quarter_image_circular_shift","FAIL" if bad else (
        "PASS" if geometries else "UNVERIFIED"),f"256x256 时 dy=dx=64; 冲突={bad[:8]}",[path],"P0")
    bootstrap_protocol=protocol.get("bootstrap",{})
    audit.check(section,"bootstrap_unit","UNVERIFIED" if "unit" not in bootstrap_protocol else (
        "PASS" if bootstrap_protocol["unit"]=="image" else "FAIL"),
        f"{bootstrap_protocol}; 固定模型图像级、未校正的 pointwise percentile 95% CI",[path])
    for key,expected in (("interval","percentile 2.5%,97.5%; unadjusted pointwise"),
                         ("seed_stream","seed + 3*nonoriginal_variant_index + metric_index")):
        value=bootstrap_protocol.get(key)
        audit.check(section,"bootstrap_definition_"+key,"UNVERIFIED" if value is None else (
            "PASS" if value==expected else "UNVERIFIED"),f"{value!r}; 已审阅定义={expected!r}",[path])
    complete=audit.read(directory/"COMPLETED.json","json",section=section)
    if complete:
        audit.check(section,"completion_marker","PASS" if complete.get("status")=="PASS" and complete.get("paper_ready") is True else "UNVERIFIED",
                    str(complete),[directory/"COMPLETED.json"])
    for key,expected in (("paper_ready",True),("debug_partial",False),("failed_sample_count",0),("successful_sample_count",428)):
        value=summary.get(key)
        audit.check(section,"summary_"+key,"UNVERIFIED" if value is None else (
            "PASS" if value==expected else "FAIL"),f"{value!r}",[directory/"summary.json"],"P0")
    if summary.get("checkpoint_sha256") is not None:
        audit.check(section,"summary_protocol_checkpoint_hash","PASS" if summary["checkpoint_sha256"]==protocol.get("checkpoint_sha256") else "FAIL",
                    "summary/protocol checkpoint SHA256", [path,directory/"summary.json"],"P0")


def audit_intervention(audit,directory,identity):
    directory=Path(directory)
    section="intervention:"+str(directory)
    csv_path,summary_path,protocol_path=(directory/n for n in ("per_sample_metrics.csv","summary.json","protocol.json"))
    rows=audit.read(csv_path,"csv",section=section)
    summary=audit.read(summary_path,"json",section=section) or {}
    protocol=audit.read(protocol_path,"json",section=section) or {}
    failures=audit.read(directory/"failed_samples.csv","csv",section=section)
    if protocol:
        protocol_checks(audit,directory,protocol,summary,identity)
    check_rows(audit,section,rows,identity["splits"]["test"],csv_path,allow_variants=True)
    if failures:
        audit.check(section,"failed_samples","UNVERIFIED",f"失败样本 {len(failures)}",[directory/"failed_samples.csv"])
    groups={variant:[] for variant in VARIANTS}
    for row in rows or []:
        variant=row.get("variant")
        if variant not in groups:
            audit.check(section,"unknown_variant","UNVERIFIED",f"{variant!r}",[csv_path])
            continue
        groups[variant].append(row)
    originals=groups["original"]
    original_ids={r["sample_id"] for r in originals}
    result=[]
    gains_by_key={}
    for index,variant in enumerate(VARIANTS):
        members=groups[variant]
        ids={r["sample_id"] for r in members}
        audit.check(section,"count_"+variant,"PASS" if len(members)==len(ids)==428 else "UNVERIFIED",
                    f"{len(members)} rows / {len(ids)} unique IDs / expected 428",[csv_path])
        audit.check(section,"pairing_"+variant,"PASS" if ids==original_ids and ids else "UNVERIFIED",
                    f"与 original 的对称差={len(ids^original_ids)}",[csv_path])
        saved=summary.get("variants",{}).get(variant,{})
        if rows is not None:
            compare_saved(audit,section,variant+"_saved_count",len(members),saved.get("sample_count"),summary_path,atol=0)
        for stream,metric in enumerate(METRICS):
            stat=describe(members,metric)
            state="FAIL" if stat["nonfinite_or_malformed_count"] else (
                "PASS" if members and stat["valid_count"]==len(members)==428 else "UNVERIFIED")
            audit.check(section,variant+"_"+metric+"_validity",state,str(stat),[csv_path],"P0")
            if metric in ("e00","ssim"):
                out_of_range=[]
                for row in members:
                    try:
                        value=number(row.get(metric))
                    except ValueError:
                        continue
                    if value is not None and (value<0 if metric=="e00" else not -1<=value<=1):
                        out_of_range.append(row["sample_id"])
                if out_of_range:
                    audit.check(section,variant+"_"+metric+"_range","FAIL",str(out_of_range[:8]),[csv_path],"P0")
            compare_saved(audit,section,variant+"_"+metric+"_mean",stat["mean"],saved.get("mean_"+metric),summary_path)
            values,paired_ids=paired_gains(originals,members,metric)
            gain=math.fsum(values)/len(values) if values else None
            record=dict(analysis_dir=str(directory),variant=variant,metric=metric,
                observed_mean=stat["mean"],saved_mean=saved.get("mean_"+metric),
                valid_count=stat["valid_count"],invalid_count=stat["invalid_count"],
                original_mean=describe(originals,metric)["mean"],paired_count=len(values),
                paired_gain=None if variant=="original" else gain,
                gain_direction="intervention-original" if metric=="e00" else "original-intervention",
                ci95_low=None,ci95_high=None,numeric_status=state,source_csv=str(csv_path))
            if variant!="original":
                saved_gain=summary.get("paired_gains",{}).get(variant,{}).get(metric,{})
                compare_saved(audit,section,variant+"_"+metric+"_paired_mean",gain,saved_gain.get("mean_difference"),summary_path)
                for sample_id,value in zip(paired_ids,values):
                    gains_by_key[(sample_id,variant,metric)]=value
                audit.check(section,variant+"_"+metric+"_paired_completeness",
                    "PASS" if len(paired_ids)==len(set(paired_ids))==428 else "UNVERIFIED",
                    f"有效配对数={len(paired_ids)}；按 sample_id 配对，不按行序",[csv_path])
                b=protocol.get("bootstrap",{})
                seed=b.get("seed")
                stream_seed=seed+(index-1)*3+stream if isinstance(seed,int) else None
                for key,expected in (("bootstrap_seed",stream_seed),("bootstrap_resamples",b.get("samples")),
                                     ("resampling_unit","image"),("positive_means","original_better"),("std_ddof",0)):
                    actual=saved_gain.get(key)
                    audit.check(section,variant+"_"+metric+"_"+key,"UNVERIFIED" if actual is None or expected is None else (
                        "PASS" if actual==expected else "FAIL"),f"保存={actual!r}; 期望={expected!r}",[summary_path])
                if values:
                    for key,value in (("std",statistics.pstdev(values)),("median",statistics.median(values)),
                                      ("win_fraction",sum(v>0 for v in values)/len(values))):
                        compare_saved(audit,section,variant+"_"+metric+"_"+key,value,saved_gain.get(key),summary_path)
                    try:
                        ci=bootstrap(values,b.get("samples"),stream_seed)
                    except (ValueError,TypeError,ImportError) as error:
                        audit.check(section,variant+"_"+metric+"_bootstrap","UNVERIFIED",str(error),[summary_path])
                    else:
                        for key,value in ci.items():
                            compare_saved(audit,section,variant+"_"+metric+"_bootstrap_"+key,value,saved_gain.get(key),summary_path)
                        record.update(ci95_low=ci["ci95_low"],ci95_high=ci["ci95_high"])
                        actual=saved_gain.get("ci_includes_zero")
                        includes=ci["ci95_low"]<=0<=ci["ci95_high"]
                        audit.check(section,variant+"_"+metric+"_ci_includes_zero","UNVERIFIED" if actual is None else (
                            "PASS" if actual==includes else "FAIL"),f"保存={actual!r}; 重算={includes}",[summary_path])
            result.append(record)
            audit.annotate(csv_path,[metric,"sample_id","variant"],"图像等权均值；配对正差统一表示 original 更好")
    paired_path=directory/"paired_differences.csv"
    paired=audit.read(paired_path,"csv",section=section)
    if paired is not None and rows is not None:
        keys=[(r.get("sample_id"),r.get("variant")) for r in paired]
        expected_keys={(sample_id,variant) for sample_id,variant,metric in gains_by_key}
        audit.check(section,"saved_pair_keys","PASS" if len(keys)==len(set(keys)) and set(keys)==expected_keys else "FAIL",
                    f"保存键={len(keys)}，重算键={len(expected_keys)}",[paired_path],"P0")
        bad=[]
        for row in paired:
            for metric in METRICS:
                key=(row.get("sample_id"),row.get("variant"),metric)
                if key not in gains_by_key or not equal_number(row.get("gain_"+metric),gains_by_key[key]):
                    bad.append(key)
        audit.check(section,"saved_pair_values","FAIL" if bad else "PASS",f"冲突={bad[:8]}",[paired_path,csv_path],"P0")
    audit_diagnostics(audit,directory,protocol,identity["splits"]["test"])
    # Never subtract official results from interventions: use this experiment's
    # own original. Differences below only diagnose baseline reproduction.
    comparison={}
    official_by_id={r["sample_id"]:r for r in identity["rows"] or []}
    for metric in METRICS:
        original_mean=describe(originals,metric)["mean"]
        official_mean=identity["observed"][metric]["mean"]
        errors=[]
        for row in originals:
            if row["sample_id"] not in official_by_id:
                continue
            try:
                a,b=number(row.get(metric)),number(official_by_id[row["sample_id"]].get(metric))
                if a is not None and b is not None:
                    errors.append(abs(a-b))
            except ValueError:
                continue
        tolerance=({"psnr":1e-3,"ssim":1e-4,"e00":1e-3} if protocol.get("amp_enabled") else
                   {"psnr":1e-5,"ssim":1e-6,"e00":1e-5})[metric]
        delta=None if original_mean is None or official_mean is None else original_mean-official_mean
        comparison[metric]=dict(original_mean=original_mean,official_mean=official_mean,
            original_minus_official=delta,max_per_image_error=max(errors) if errors else None,
            paired_count=len(errors),native_tool_tolerance=tolerance)
        state="UNVERIFIED" if delta is None or len(errors)!=428 else (
            "PASS" if abs(delta)<=tolerance and max(errors)<=tolerance else "UNVERIFIED")
        audit.check(section,"original_vs_official_"+metric,state,
            f"{comparison[metric]}; 超容差需人工核查 AMP、batch、环境及指标协议；不覆盖或混用原结果。",
            [csv_path,identity["run"]/"result/test_metrics.csv"])
    metadata_keys=("checkpoint_sha256","test_manifest_sha256","amp_enabled","evaluation","metrics")
    missing=[key for key in metadata_keys if key not in identity["official"]]
    conflicts=[key for key in metadata_keys if key in identity["official"] and
               identity["official"][key]!=protocol.get(key)]
    audit.check(section,"official_protocol_comparability","UNVERIFIED" if missing or conflicts else "PASS",
                f"旧官方摘要缺字段={missing}; 明确协议差异={conflicts}; 配对差始终使用本实验 original。",
                [protocol_path,identity["run"]/"result/test_summary.json"])
    return dict(directory=str(directory),section=section,records=result,protocol=protocol,
                summary=summary,official_comparison=comparison,status=audit.status(section))
