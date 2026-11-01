"""Audit saved spatial statistics; never reconstruct maps or rerun inference."""
from __future__ import annotations

import math
from pathlib import Path

from . import MEAN_ATOL
from .provenance import verify_identity, _nested
from .statistics import bootstrap, check_rows, compare_saved, describe, number, equal_number

RESPONSE_SPECS = (
    ("anchor_deviation", "anchor_spearman_rgb", "anchor_top20_iou_rgb",
     ("anchor_deviation_control","metrics"), "||I-b||_2"),
    ("image_gradient", "gradient_spearman_rgb", "gradient_top20_iou_rgb",
     ("image_gradient_control","metrics"), "sqrt(sum_RGB(gx^2+gy^2)); RGB Sobel /8, reflect"),
    ("shifted_raw_field", "raw_field_null_spearman_rgb_mean", "raw_field_null_top20_iou_rgb_mean",
     None, "image-wise mean over circular shifts of MeanPool(||R||_2), not functional intervention"),
    ("raw_field", "raw_field_spearman_rgb", "raw_field_top20_iou_rgb",
     ("raw_field_representation","metrics"), "MeanPool(||R||_2) vs MeanPool(||Y-I||_2)"),
    ("actual_correction", "spearman_rgb", "top20_iou_rgb",
     ("metrics",), "MeanPool(||Ic-I||_2) vs MeanPool(||Y-I||_2)"),
)
COMPARISONS = ("raw_minus_anchor_spearman_rgb","raw_minus_gradient_spearman_rgb",
               "raw_minus_anchor_top20_iou_rgb","raw_minus_gradient_top20_iou_rgb",
               "raw_minus_anchor_spearman_e00","raw_minus_gradient_spearman_e00")
BOOTSTRAP_SPECS = (
    ("spearman_gain","spearman_gain_over_null"),
    ("iou_gain","top20_iou_gain_over_null"),
    ("direction_gain","direction_gain_over_null"),
    ("raw_field_spearman_gain","raw_field_spearman_gain_over_null"),
    ("raw_field_iou_gain","raw_field_top20_iou_gain_over_null"),
    ("raw_field_spearman_e00_gain","raw_field_spearman_e00_gain_over_null"),
    ("anchor_spearman_gain","anchor_spearman_gain_over_null"),
    ("anchor_iou_gain","anchor_top20_iou_gain_over_null"),
    ("anchor_spearman_e00_gain","anchor_spearman_e00_gain_over_null"),
    ("gradient_spearman_gain","gradient_spearman_gain_over_null"),
    ("gradient_iou_gain","gradient_top20_iou_gain_over_null"),
    ("gradient_spearman_e00_gain","gradient_spearman_e00_gain_over_null"),
    *((name,name) for name in COMPARISONS),
)


def audit_bootstrap(audit, section, rows, summary, protocol, source):
    saved = summary.get("bootstrap", {})
    samples, seed = saved.get("resamples"), saved.get("seed")
    if not saved:
        audit.check(section,"bootstrap","UNVERIFIED","没有保存图像级 Bootstrap 记录",[source])
        return
    audit.check(section,"bootstrap_unit","PASS" if saved.get("resampling_unit") == "image" else (
        "UNVERIFIED" if saved.get("resampling_unit") is None else "FAIL"),
        f"unit={saved.get('resampling_unit')}; 固定 checkpoint，不能解释为跨训练种子稳定性",[source])
    for key, actual, expected in (("samples",samples,protocol.get("bootstrap_samples")),
                                  ("seed",seed,protocol.get("bootstrap_seed"))):
        compare_saved(audit,section,"bootstrap_"+key,actual,expected,source,atol=0)
    for stream, (name, field) in enumerate(BOOTSTRAP_SPECS):
        if name not in saved:
            audit.check(section,"bootstrap_"+name,"UNVERIFIED","没有保存该配对区间",[source])
            continue
        values = []
        bad = False
        for row in rows:
            try:
                value = number(row.get(field))
                if value is not None:
                    values.append(value)
            except ValueError:
                bad = True
        if bad:
            audit.check(section,"bootstrap_"+name,"FAIL","配对差含无效值",[source],"P0")
            continue
        if not values:
            audit.check(section,"bootstrap_"+name,"UNVERIFIED","无有效图像配对差",[source])
            continue
        try:
            result = bootstrap(values,samples,seed+stream)
        except (ValueError, TypeError, ImportError) as error:
            audit.check(section,"bootstrap_"+name,"UNVERIFIED",str(error),[source])
            continue
        for key, value in result.items():
            compare_saved(audit,section,"bootstrap_"+name+"_"+key,value,saved[name].get(key),source)


def check_numeric_rows(audit, section, rows, source):
    stats = {}
    if not rows:
        return stats
    ignored = {"sample_index","sample_id","filename","variant"}
    for field in rows[0]:
        if field in ignored:
            continue
        values = describe(rows,field)
        stats[field] = values
        if values["nonfinite_or_malformed_count"]:
            audit.check(section,"invalid_"+field,"FAIL",str(values),[source],"P0")
        for row in rows:
            try:
                value = number(row.get(field))
            except ValueError:
                continue
            if value is None:
                continue
            correlation = "spearman" in field or "pearson" in field or field == "direction_cosine_top20_gt"
            if correlation and not any(x in field for x in ("gain","minus","std","count")) and not -1-1e-12 <= value <= 1+1e-12:
                audit.check(section,"range_"+field,"FAIL",f"{row.get('sample_id')}: {value}",[source],"P0")
            if ("iou" in field or "precision" in field or "recall" in field) and not any(
                    x in field for x in ("gain","minus","std","count")) and not -1e-12 <= value <= 1+1e-12:
                audit.check(section,"range_"+field,"FAIL",f"{row.get('sample_id')}: {value}",[source],"P0")
            if field == "e00" and value < 0:
                audit.check(section,"range_e00","FAIL",f"{row.get('sample_id')}: {value}",[source],"P0")
    return stats


def check_saved_statistics(audit, section, rows, summary, source):
    paths = [("metrics",),("quality_metrics",),("raw_field_representation","metrics"),
             ("anchor_deviation_control","metrics"),("image_gradient_control","metrics"),
             ("actual_correction_effect","metrics"),("paired_representation_comparisons",)]
    for path in paths:
        saved = _nested(summary,*path)
        if not isinstance(saved,dict):
            audit.check(section,"summary_section_"+".".join(path),"UNVERIFIED","缺少统计区段",[source])
            continue
        for field, record in saved.items():
            if not isinstance(record,dict):
                audit.check(section,"summary_record_"+field,"FAIL","统计记录不是对象",[source],"P0")
                continue
            result = describe(rows,field)
            for key in ("mean","std","median","valid_count","invalid_count"):
                if key in record:
                    compare_saved(audit,section,".".join((*path,field,key)),result[key],record[key],source)
    for path, prefix in ((("null_controls",),""),(("raw_field_representation","null_controls"),"raw_field_"),
                         (("anchor_deviation_control","null_controls"),"anchor_"),
                         (("image_gradient_control","null_controls"),"gradient_")):
        saved = _nested(summary,*path)
        if not isinstance(saved,dict):
            continue
        for metric, record in saved.items():
            actual_field = prefix+metric
            null_field = prefix + ("null_direction_cosine_mean" if metric == "direction_cosine_top20_gt"
                                   else "null_"+metric+"_mean")
            pairs = []
            for row in rows:
                try:
                    a,b=number(row.get(actual_field)),number(row.get(null_field))
                    if a is not None and b is not None:
                        pairs.append((a,b))
                except ValueError:
                    continue
            n=len(pairs)
            result=dict(valid_pair_count=n,mean_real=math.fsum(a for a,b in pairs)/n if n else None,
                        mean_null=math.fsum(b for a,b in pairs)/n if n else None,
                        mean_real_minus_null=math.fsum(a-b for a,b in pairs)/n if n else None,
                        fraction_real_greater_than_null=sum(a>b for a,b in pairs)/n if n else None)
            for key,value in result.items():
                if key in record:
                    compare_saved(audit,section,".".join((*path,metric,key)),value,record[key],source)


def check_derived_fields(audit, section, rows, source):
    specs = {}
    for prefix in ("","raw_field_","anchor_","gradient_"):
        for metric in ("spearman_rgb","top20_iou_rgb","spearman_e00"):
            gain = prefix + {"spearman_rgb":"spearman_gain_over_null",
                             "top20_iou_rgb":"top20_iou_gain_over_null",
                             "spearman_e00":"spearman_e00_gain_over_null"}[metric]
            specs[gain]=(prefix+metric,prefix+"null_"+metric+"_mean")
    for name in COMPARISONS:
        tail = name.split("_",3)
        control = tail[2]
        metric = tail[3]
        specs[name]=("raw_field_"+metric,control+"_"+metric)
    for field,(first,second) in specs.items():
        if not rows or field not in rows[0]:
            continue
        for row in rows:
            try:
                a,b,g=number(row.get(first)),number(row.get(second)),number(row.get(field))
            except ValueError:
                continue
            if a is not None and b is not None and (g is None or abs((a-b)-g)>MEAN_ATOL):
                audit.check(section,"derived_"+field,"FAIL",f"{row['sample_id']}: {a}-{b} != {g}",[source],"P0")


def audit_alignment(audit, directory, identity, contract):
    directory=Path(directory)
    section="alignment:"+str(directory)
    summary_path,protocol_path,csv_path=(directory/name for name in ("summary.json","protocol.json","per_sample_metrics.csv"))
    summary=audit.read(summary_path,"json",section=section) or {}
    protocol=audit.read(protocol_path,"json",section=section) or {}
    rows=audit.read(csv_path,"csv",section=section)
    failures=audit.read(directory/"failed_samples.csv","csv",section=section)
    if protocol:
        verify_identity(audit,protocol,identity,section,protocol_path)
        for key,expected in (("patch_size",16),("top_fraction",0.2),("evaluation_resize",True),
                             ("evaluation_size",256),("field_variant","implicit"),
                             ("use_spatial_conditioning",True),("use_global_field_conditioning",True),
                             ("use_learned_anchor",True),("evaluation_mode","in_domain")):
            actual=protocol.get(key)
            audit.check(section,"protocol_"+key,"UNVERIFIED" if actual is None else (
                "PASS" if actual == expected else "FAIL"),f"{actual!r}; expected={expected!r}",[protocol_path],"P0")
        for key,expected in (
            ("script","tools/analyze_uicf_alignment.py"),
            ("raw_field_definition","D_R(p) = ||R(p)||_2"),
            ("reference_rgb_demand","D_ref_rgb(p) = ||Y(p)-I(p)||_2"),
            ("normalization_policy","no visualization normalization is used for quantitative metrics"),
            ("gradient_control","per-channel RGB Sobel magnitude with 3x3 kernels /8 and reflect padding"),
        ):
            actual=protocol.get(key)
            audit.check(section,"definition_"+key,"UNVERIFIED" if actual is None else (
                "PASS" if actual == expected else "UNVERIFIED"),
                f"保存={actual!r}; 已审阅实现={expected!r}; 定义文字变化需人工核对，不猜测等价性。",[protocol_path])
        for key in ("num_null_shifts","null_seed","bootstrap_samples","bootstrap_seed","script_version","command"):
            audit.check(section,"recorded_"+key,"PASS" if protocol.get(key) is not None else "UNVERIFIED",
                        f"{key}={protocol.get(key)!r}",[protocol_path])
        shifts=protocol.get("num_null_shifts")
        if isinstance(shifts,int) and not isinstance(shifts,bool) and shifts>0:
            for row in rows or []:
                for field in ("raw_field_null_spearman_valid_shift_count","raw_field_null_top20_valid_shift_count",
                              "null_valid_shift_count","null_top20_valid_shift_count"):
                    try:
                        value=number(row.get(field))
                    except ValueError:
                        continue
                    if value is not None and (value!=int(value) or not 0<=value<=shifts):
                        audit.check(section,"null_shift_count_range","FAIL",
                                    f"{row.get('sample_id')}: {field}={value}; requested={shifts}",[csv_path,protocol_path],"P0")
        else:
            audit.check(section,"null_shift_count_protocol","UNVERIFIED" if shifts is None else "FAIL",
                        f"num_null_shifts={shifts!r}",[protocol_path])
    check_rows(audit,section,rows,identity["splits"]["test"],csv_path)
    if rows is not None:
        for key,actual in (("successful_sample_count",len(rows)),("failed_sample_count",len(failures or [])),
                           ("processed_sample_count",len(rows)+len(failures or []))):
            compare_saved(audit,section,key,actual,summary.get(key),summary_path,atol=0)
        for key in ("test_count","total_test_samples"):
            compare_saved(audit,section,key,428,summary.get(key),summary_path,atol=0)
    if failures:
        audit.check(section,"failed_samples","UNVERIFIED",f"失败样本={len(failures)}，不能声称完整 428 张",[directory/"failed_samples.csv"])
        failed_ids=[r.get("sample_id") for r in failures]
        if len(failed_ids)!=len(set(failed_ids)) or set(failed_ids)&{r.get("sample_id") for r in rows or []}:
            audit.check(section,"failure_accounting","FAIL","失败样本重复或同时出现在成功 CSV",[csv_path,directory/"failed_samples.csv"],"P0")
        expected_ids={r[0] for r in identity["splits"]["test"] or []}
        if expected_ids and set(failed_ids)-expected_ids:
            audit.check(section,"unknown_failed_sample","FAIL","失败清单含不属于冻结测试集的 sample_id",
                        [directory/"failed_samples.csv"],"P0")
    stats=check_numeric_rows(audit,section,rows or [],csv_path)
    if rows is not None:
        check_saved_statistics(audit,section,rows,summary,summary_path)
        check_derived_fields(audit,section,rows,csv_path)
        audit_bootstrap(audit,section,rows,summary,protocol,summary_path)
    # Optional duplicate exports are reconciled, never substituted for the primary CSV.
    by_id={r.get("sample_id"):r for r in rows or []}
    for name in ("representation_baseline_summary.csv","raw_field_null_control_summary.csv","null_control_summary.csv"):
        path=directory/name
        secondary=audit.read(path,"csv",section=section,required=False)
        if secondary is None:
            continue
        check_rows(audit,section,secondary,identity["splits"]["test"],path)
        if rows is None:
            audit.check(section,"secondary_without_primary","UNVERIFIED","主 CSV 缺失，不能用次级导出替代",[path,csv_path])
            continue
        mismatches=[]
        for row in secondary:
            original=by_id.get(row["sample_id"],{})
            for field,value in row.items():
                if field in ("sample_id","filename","sample_index") or field not in original:
                    continue
                if not equal_number(value,original[field]):
                    mismatches.append((row["sample_id"],field))
        audit.check(section,"secondary_"+name,"FAIL" if mismatches else "PASS",
                    f"与主 CSV 冲突={mismatches[:8]}",[path,csv_path],"P0")
    result=[]
    for response,rho_field,iou_field,path,definition in RESPONSE_SPECS:
        for metric,field in (("spearman",rho_field),("iou",iou_field)):
            value=describe(rows or [],field)
            saved = (_nested(summary,*path,field,"mean") if path else
                     _nested(summary,"raw_field_representation","null_controls",
                             "spearman_rgb" if metric=="spearman" else "top20_iou_rgb","mean_null"))
            if value["nonfinite_or_malformed_count"]:
                state="FAIL"
            elif value["valid_count"] != 428 or value["mean"] is None or saved is None:
                state="UNVERIFIED"
            else:
                state="PASS" if equal_number(value["mean"],saved) else "FAIL"
            audit.check(section,"response_"+response+"_"+metric,state,
                        f"统计字段={field}; {value}; 保存均值={saved!r}",[csv_path,summary_path],"P0")
            provided=contract.get("user_reference",{}).get(response,{}).get(metric)
            tolerance=(0.5e-6+1e-12 if response=="actual_correction" else 0.0005+1e-12)
            table_state="UNVERIFIED" if value["mean"] is None or provided is None else (
                "PASS" if abs(value["mean"]-provided)<=tolerance else "FAIL")
            audit.check(section,"provided_table_"+response+"_"+metric,table_state,
                        f"重算={value['mean']}; 任务提供的参考记录={provided}; tolerance={tolerance}; 不是已读取的论文原文。",
                        [csv_path])
            precise=contract.get("historical_precise_reference",{}).get(response,{}).get(metric)
            if precise is not None:
                compare_saved(audit,section,"provided_precise_"+response+"_"+metric,value["mean"],precise,
                              csv_path,atol=0.5e-6+1e-12)
            audit.annotate(csv_path,[field],definition+"; per-image equal arithmetic mean")
            result.append(dict(analysis_dir=str(directory),response=response,metric=metric,field=field,
                observed_mean=value["mean"],saved_mean=saved,valid_count=value["valid_count"],
                missing_count=value["missing_count"],invalid_count=value["nonfinite_or_malformed_count"],
                expected_count=428,provided_table_value=provided,table_atol=tolerance,
                numeric_status=state,table_status=table_state,definition=definition,source_csv=str(csv_path)))
    return dict(directory=str(directory),section=section,protocol=protocol,summary=summary,
                records=result,all_numeric_statistics=stats,status=audit.status(section))
