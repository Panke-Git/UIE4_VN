"""Synthetic numeric fixtures only: no real LSUI, checkpoint loading or inference."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pytest
import yaml

from tools.audit_uicf_evidence_chain import main
from tools.uicf_evidence_audit import BASELINE_NAME, COUNTS, METRICS, REPORT_FILES, VARIANTS
from tools.uicf_evidence_audit.alignment import audit_alignment, check_numeric_rows
from tools.uicf_evidence_audit.intervention import audit_intervention
from tools.uicf_evidence_audit.io import Audit, ROOT, export_zip, new_output, safe_output_parent, sha256
from tools.uicf_evidence_audit.provenance import baseline, review_sources, verify_identity
from tools.uicf_evidence_audit.report import manuscript
from tools.uicf_evidence_audit.statistics import (
    bootstrap, check_rows, compare_saved, describe, equal_number, number, paired_gains)


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Keep even a default pytest invocation's synthetic files inside UIE4_VN.
    parent=tmp_path_factory.getbasetemp()
    if not parent.resolve().is_relative_to(ROOT):
        parent=ROOT/"analysis/uicf_evidence_chain_audit/software_tests"
        parent.mkdir(parents=True,exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="synthetic_evidence_",dir=parent))


def source_functions(relative,names,namespace=None):
    """Execute only selected pure numeric definitions, not GPU tool imports/main."""
    tree=ast.parse((ROOT/relative).read_text())
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected}==set(names)
    future=ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,*selected],type_ignores=[]))
    scope=dict(np=np,math=math,dataclass=dataclass,Path=Path,__name__=__name__,
               Sequence=Sequence,Mapping=Mapping,Any=Any)
    scope.update(namespace or {})
    exec(compile(module,str(ROOT/relative),"exec"),scope)
    return scope


@pytest.fixture(scope="module")
def spatial_functions():
    return source_functions("tools/analyze_uicf_alignment.py",[
        "OptionalMetric","_one_dimensional_finite_pair","average_rank","pearson_correlation",
        "spearman_correlation","exact_top_fraction_mask","top_fraction_overlap","patch_average_pool",
        "generate_null_shifts","descriptive_statistics","paired_null_statistics",
        "bootstrap_mean_difference","build_summary"])


def put_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2))


def put_csv(path,rows,fields=None):
    import csv
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def evidence(tmp_path,spatial_functions):
    """428 tiny numeric rows, not images; labelled and confined to pytest temp."""
    run=tmp_path/"SYNTHETIC_EVIDENCE_NOT_EXPERIMENT"/BASELINE_NAME
    run.mkdir(parents=True)
    config=yaml.safe_load((ROOT/"configs/config_v16.yaml").read_text())
    config["data"]["root"]=str(tmp_path/"SYNTHETIC_NO_IMAGE_DIRECTORY")
    (run/"config_resolved.yaml").write_text(yaml.safe_dump(config))
    for split in COUNTS:
        target=run/"split_snapshot"/f"{split}.tsv"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/f"split/lsui19/{split}.tsv",target)
    tests=[line.split("\t") for line in (run/"split_snapshot/test.tsv").read_text().splitlines()]
    cp=run/"best/best_psnr.pt"
    cp.parent.mkdir()
    cp.write_bytes(b"SYNTHETIC BYTES, NOT A SERIALIZED NEURAL CHECKPOINT")
    digest=sha256(cp)
    info=dict(version="v16",seed=3520,synthetic_fixture=True,
              **{name+"_sample_count":count for name,count in COUNTS.items()})
    put_json(run/"run_info.json",info)
    put_json(run/"status.json",dict(status="completed",last_epoch=200))
    put_json(run/"best/best_psnr.json",dict(epoch=195,checkpoint_sha256=digest))
    quality=dict(psnr=28.388665,ssim=.928637,e00=5.204765)
    official=[dict(sample_id=row[0],filename=Path(row[1]).name,**quality) for row in tests]
    put_csv(run/"result/test_metrics.csv",official)
    e00=dict(lab_illuminant="D65",lab_observer="2 degrees",kL=1.,kC=1.,kH=1.,crop_border=0)
    summary=dict(sample_count=428,checkpoint=str(cp),checkpoint_epoch=195,checkpoint_sha256=digest,
                 **{"mean_"+metric:value for metric,value in quality.items()},
                 test_manifest_sha256=sha256(run/"split_snapshot/test.tsv"),amp_enabled=True,
                 evaluation=config["evaluation"],metrics=config["metrics"],e00_protocol=e00)
    put_json(run/"result/test_summary.json",summary)
    contract=json.loads((ROOT/"tools/uicf_evidence_audit/source_contract.json").read_text())
    code={"source_sha256":{name:record["sha256"] for name,record in contract["sources"].items()}}
    common=dict(run_dir=str(run),checkpoint_run=str(run),checkpoint=str(cp),checkpoint_selector="best_psnr",
        checkpoint_epoch=195,checkpoint_sha256=digest,dataset="LSUI19",checkpoint_dataset="LSUI19",
        evaluation_dataset="LSUI19",evaluation_mode="in_domain",
        **{name+"_count":count for name,count in COUNTS.items()},
        test_manifest_sha256=sha256(run/"split_snapshot/test.tsv"),code=code,
        command="SYNTHETIC SOFTWARE FIXTURE; NO EXPERIMENT EXECUTED")
    alignment=run/"result/uicf_representation_alignment"
    records=[]
    tree=ast.parse((ROOT/"tools/analyze_uicf_alignment.py").read_text())
    fields=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=="PER_SAMPLE_FIELDS" for t in node.targets))
    for index,row in enumerate(tests):
        record={name:0. for name in fields}
        record.update(sample_id=row[0],filename=Path(row[1]).name,sample_index=index,**quality,
            spearman_rgb=-.061790,top20_iou_rgb=.199392,
            raw_field_spearman_rgb=.534424,raw_field_top20_iou_rgb=.324506,
            anchor_spearman_rgb=-.377,anchor_top20_iou_rgb=.120,
            gradient_spearman_rgb=.209,gradient_top20_iou_rgb=.167,
            raw_field_null_spearman_rgb_mean=-.048,raw_field_null_top20_iou_rgb_mean=.106,
            b_r=.2,b_g=.4,b_b=.6)
        for field in fields:
            if "count" in field:
                record[field]=20
        for prefix in ("","raw_field_","anchor_","gradient_"):
            for metric in ("spearman_rgb","top20_iou_rgb","spearman_e00"):
                gain=prefix+{"spearman_rgb":"spearman_gain_over_null","top20_iou_rgb":"top20_iou_gain_over_null",
                             "spearman_e00":"spearman_e00_gain_over_null"}[metric]
                if gain in record:
                    record[gain]=record.get(prefix+metric,0)-record.get(prefix+"null_"+metric+"_mean",0)
        for field in [f for f in fields if f.startswith("raw_minus_")]:
            control,metric=field.split("_",3)[2:]
            record[field]=record["raw_field_"+metric]-record[control+"_"+metric]
        records.append(record)
    put_csv(alignment/"per_sample_metrics.csv",records)
    put_csv(alignment/"failed_samples.csv",[],["sample_index","sample_id","filename","error_type","error"])
    alignment_summary=spatial_functions["build_summary"](records,[],dataset_name="LSUI19",split_counts=COUNTS,
        data_root=Path(config["data"]["root"]),test_manifest=run/"split_snapshot/test.tsv",bootstrap_samples=31,bootstrap_seed=3520)
    put_json(alignment/"summary.json",alignment_summary)
    alignment_protocol=dict(common,script="tools/analyze_uicf_alignment.py",script_version="2.0",
        config_resolved_yaml_sha256=sha256(run/"config_resolved.yaml"),
        evaluation_resize=True,evaluation_size=256,patch_size=16,top_fraction=.2,
        field_variant="implicit",use_spatial_conditioning=True,use_global_field_conditioning=True,use_learned_anchor=True,
        num_null_shifts=20,null_seed=3520,bootstrap_samples=31,bootstrap_seed=3520,
        raw_field_definition="D_R(p) = ||R(p)||_2",reference_rgb_demand="D_ref_rgb(p) = ||Y(p)-I(p)||_2",
        normalization_policy="no visualization normalization is used for quantitative metrics",
        gradient_control="per-channel RGB Sobel magnitude with 3x3 kernels /8 and reflect padding")
    put_json(alignment/"protocol.json",alignment_protocol)
    intervention=run/"result/uicf_field_intervention"
    metrics,diagnostics,geometries=[],[],[]
    for index,row in enumerate(tests):
        geometries.append(dict(sample_id=row[0],filename=Path(row[1]).name,sample_index=index,height=256,width=256,
            shift_dy=64,shift_dx=64,dtypes={"R":"torch.float16"}))
        for v,variant in enumerate(VARIANTS):
            drop=v*.01
            metrics.append(dict(sample_id=row[0],filename=Path(row[1]).name,sample_index=index,variant=variant,
                psnr=quality["psnr"]-drop,ssim=quality["ssim"]-drop*.01,e00=quality["e00"]+drop))
            means={"channel_cycle_1":[.2,.3,.1],"channel_cycle_2":[.3,.1,.2],"zero_field":[0,0,0]}.get(variant,[.1,.2,.3])
            diagnostic=dict(sample_id=row[0],filename=Path(row[1]).name,sample_index=index,variant=variant,
                mean_abs_R=0. if variant=="zero_field" else .2,std_R=0. if variant=="zero_field" else .1,
                mean_abs_delta_Ic_original=.1,mean_abs_delta_Ic_variant=0. if variant=="zero_field" else .1,
                mean_abs_Ic_variant_minus_Ic_original=0. if variant=="original" else .1,
                mean_abs_Dout_variant_minus_Dout_original=0. if variant=="original" else .01,
                fraction_clipped_final=0.)
            for color,mean,b in zip("rgb",means,[.2,.4,.6]):
                diagnostic["mean_R_"+color]=mean
                diagnostic["std_R_"+color]=0. if variant in ("spatial_mean","zero_field") else .01
                diagnostic["b_"+color]=b
            diagnostics.append(diagnostic)
    core=source_functions("tools/uicf_intervention/core.py",["paired_differences","summarize"],
        dict(VARIANTS=VARIANTS,METRICS=METRICS,INTERPRETATION="Synthetic fixture",
             bootstrap_mean_difference=spatial_functions["bootstrap_mean_difference"]))
    intervention_summary=core["summarize"](metrics,samples=31,seed=3520)
    intervention_summary.update(status="PASS",paper_ready=True,debug_partial=False,failed_sample_count=0,
                                successful_sample_count=428,checkpoint_sha256=digest)
    put_json(intervention/"summary.json",intervention_summary)
    intervention_protocol=dict(common,script="tools/analyze_uicf_intervention.py",
        script_id="v16_uicf_field_intervention",script_version="1.0",status="PASS",debug_partial=False,
        complete_formal_evaluation=True,successful_sample_count=428,failed_sample_count=0,requested_sample_count=428,
        protected_files_unchanged=True,config_resolved_sha256=sha256(run/"config_resolved.yaml"),
        evaluation=config["evaluation"],metrics=config["metrics"],amp_enabled=True,e00_protocol=e00,
        batch_size=1,metric_input="Dout.float().clamp(0,1) before PNG; no Ic clamp",
        metric_aggregation="per-image metric then equally weighted image arithmetic mean",
        psnr_ssim_implementation="src.v16.metrics.batch_metrics (unchanged)",
        interventions=dict(Ic="I + R_variant * (I - original b)",Dout="unchanged model.backbone(Ic_variant)",
            transformed="R only; never I, b, GT, tokens or R*(I-b)",no_multi_shift_aggregate=True,
            permutations_new_RGB_from_old_indices={"channel_cycle_1":[1,2,0],"channel_cycle_2":[2,0,1]}),
        frozen_model_digest_before="a"*64,frozen_model_digest_after="a"*64,
        normal_forward_consistency={"status":"PASS","max_abs_error":0.,"samples":[{"sample_id":tests[0][0],
            "checks":{name:{"status":"PASS","max_abs_error":0.} for name in
                      ("normal_vs_details","backbone_vs_details","program_original")}}]},
        sample_geometry=geometries,bootstrap={"samples":31,"seed":3520,"unit":"image",
            "interval":"percentile 2.5%,97.5%; unadjusted pointwise",
            "seed_stream":"seed + 3*nonoriginal_variant_index + metric_index"})
    put_json(intervention/"protocol.json",intervention_protocol)
    put_json(intervention/"COMPLETED.json",dict(status="PASS",paper_ready=True))
    put_csv(intervention/"per_sample_metrics.csv",metrics)
    put_csv(intervention/"diagnostics.csv",diagnostics)
    put_csv(intervention/"paired_differences.csv",core["paired_differences"](metrics))
    put_csv(intervention/"failed_samples.csv",[],["sample_id","filename","variant","error_type","error"])
    return dict(run=run,alignment=alignment,intervention=intervention,records=records,
                metrics=metrics,contract=contract,protocol=common)


def setup_audit(evidence):
    audit=Audit()
    contract=review_sources(audit)
    identity=baseline(audit,evidence["run"],contract)
    return audit,identity,contract


def test_csv_macro_mean_is_recomputed_without_weights():
    result=describe([{"x":"1","pixels":1000},{"x":"3","pixels":1}],"x")
    assert result["mean"]==2
    assert result["valid_count"]==2


@pytest.mark.parametrize("value",["NaN","inf","-Infinity","bad",True])
def test_invalid_number_rejected(value):
    with pytest.raises(ValueError):
        number(value)


def test_blank_is_missing_not_zero():
    stats=describe([{"x":""},{"x":"2"}],"x")
    assert stats["mean"]==2 and stats["missing_count"]==1
    assert not equal_number(None,0)


def test_correlations_average_ties_and_constant_maps(spatial_functions):
    f=spatial_functions
    np.testing.assert_array_equal(f["average_rank"]([1,1,3]),[1.5,1.5,3])
    assert f["spearman_correlation"]([1,1,3],[1,2,3]).value==pytest.approx(math.sqrt(3)/2)
    assert f["spearman_correlation"]([1,2,3],[3,2,1]).value==pytest.approx(-1)
    assert f["spearman_correlation"]([1,1,1],[1,2,3]).valid is False


def test_top20_exact_cardinality_ties_and_iou(spatial_functions):
    f=spatial_functions
    mask=f["exact_top_fraction_mask"](np.ones((16,16)),.2)
    assert mask.sum()==52
    assert np.flatnonzero(mask).tolist()==list(range(52))
    iou,precision,recall=f["top_fraction_overlap"](np.arange(10),np.arange(9,-1,-1),.2)
    assert (iou,precision,recall)==(0,0,0)


def test_norm_before_pool_is_not_pool_before_norm(spatial_functions):
    field=np.zeros((3,2,2))
    field[0]=[[1,-1],[1,-1]]
    f=spatial_functions["patch_average_pool"]
    a=f(np.linalg.norm(field,axis=0),2,sample_id="synthetic")
    b=np.linalg.norm(f(field,2,sample_id="synthetic"),axis=0)
    assert a.item()==1 and b.item()==0


def test_null_shifts_nonzero_large_and_deterministic(spatial_functions):
    function=spatial_functions["generate_null_shifts"]
    first=function(16,16,20,3520,3)
    assert first==function(16,16,20,3520,3)
    assert len(first)==20 and len(set(first))==20
    assert all((dy,dx)!=(0,0) and (abs(dy)>=4 or abs(dx)>=4) for dy,dx in first)


def test_bootstrap_matches_original_pure_implementation(spatial_functions):
    values=[-.2,.1,.4,1.,.6]
    assert bootstrap(values,97,3520)==spatial_functions["bootstrap_mean_difference"](values,samples=97,seed=3520)


def test_pairing_uses_ids_and_gain_directions():
    original=[dict(sample_id="a",psnr=10,e00=2),dict(sample_id="b",psnr=20,e00=4)]
    changed=[dict(sample_id="b",psnr=18,e00=7),dict(sample_id="a",psnr=9,e00=3)]
    assert paired_gains(original,changed,"psnr")==([1.,2.],["a","b"])
    assert paired_gains(original,changed,"e00")==([1.,3.],["a","b"])
    assert paired_gains(original,changed[:1],"psnr")==([2.],["b"])


def test_duplicate_and_wrong_filename_are_failures(tmp_path):
    source=tmp_path/"data.csv"
    audit=Audit()
    rows=[dict(sample_id="a",filename="wrong.jpg")]*2
    check_rows(audit,"x",rows,[["a","input/a.jpg","gt/a.jpg"]],source)
    assert audit.status("x")=="FAIL"


def test_nan_is_not_silently_excluded(tmp_path):
    audit=Audit()
    check_numeric_rows(audit,"x",[dict(sample_id="a",raw_field_spearman_rgb="nan")],tmp_path/"x.csv")
    assert audit.status("x")=="FAIL"


def test_saved_mean_tolerance_is_distinct_from_paper_rounding(tmp_path):
    audit=Audit()
    compare_saved(audit,"precise","mean",.3245058437,.325,tmp_path/"summary.json")
    compare_saved(audit,"rounded","table",.3245058437,.325,tmp_path/"paper.json",atol=.0005+1e-12)
    assert audit.status("precise")=="FAIL" and audit.status("rounded")=="PASS"


def test_missing_comparison_is_unverified(tmp_path):
    audit=Audit()
    compare_saved(audit,"x","mean",None,1,tmp_path/"missing.json")
    assert audit.status("x")=="UNVERIFIED"


def test_source_contract_is_matched():
    audit=Audit()
    contract=review_sources(audit)
    assert contract["sources"] and audit.status("code_review")=="PASS"


def test_alignment_reaggregation_and_actual_coefficient_separation(evidence):
    audit,identity,contract=setup_audit(evidence)
    result=audit_alignment(audit,evidence["alignment"],identity,contract)
    assert result["status"]=="PASS", [c for c in audit.checks if c["section"]==result["section"] and c["status"]!="PASS"]
    rho={r["response"]:r["observed_mean"] for r in result["records"] if r["metric"]=="spearman"}
    assert rho["raw_field"]==pytest.approx(.534424)
    assert rho["actual_correction"]==pytest.approx(-.061790)


def test_intervention_reaggregation_all_variants_and_bootstraps(evidence):
    audit,identity,contract=setup_audit(evidence)
    result=audit_intervention(audit,evidence["intervention"],identity)
    assert result["status"]=="PASS", [c for c in audit.checks if c["section"]==result["section"] and c["status"]!="PASS"]
    assert len(result["records"])==8*3
    assert all(r["paired_gain"]>0 for r in result["records"] if r["variant"]!="original")
    assert all(r["paired_count"]==428 for r in result["records"])


def test_summary_mismatch_is_failure(evidence):
    path=evidence["alignment"]/"summary.json"
    summary=json.loads(path.read_text())
    summary["raw_field_representation"]["metrics"]["raw_field_spearman_rgb"]["mean"]+=.01
    put_json(path,summary)
    audit,identity,contract=setup_audit(evidence)
    assert audit_alignment(audit,evidence["alignment"],identity,contract)["status"]=="FAIL"


@pytest.mark.parametrize("kind",["alignment","intervention"])
def test_missing_primary_csv_is_unverified_not_an_invented_zero(evidence,kind):
    (evidence[kind]/"per_sample_metrics.csv").unlink()
    audit,identity,contract=setup_audit(evidence)
    result=(audit_alignment(audit,evidence[kind],identity,contract) if kind=="alignment" else
            audit_intervention(audit,evidence[kind],identity))
    assert result["status"]=="UNVERIFIED"
    assert all(r["observed_mean"] is None for r in result["records"])


def test_duplicate_intervention_sample_cannot_pass(evidence):
    rows=[*evidence["metrics"],evidence["metrics"][0]]
    put_csv(evidence["intervention"]/"per_sample_metrics.csv",rows)
    audit,identity,contract=setup_audit(evidence)
    assert audit_intervention(audit,evidence["intervention"],identity)["status"]=="FAIL"


def test_wrong_paired_e00_sign_is_failure(evidence):
    path=evidence["intervention"]/"paired_differences.csv"
    audit=Audit()
    rows=audit.read(path,"csv",section="fixture")
    rows[0]["gain_e00"]=str(-float(rows[0]["gain_e00"]))
    put_csv(path,rows)
    audit,identity,contract=setup_audit(evidence)
    result=audit_intervention(audit,evidence["intervention"],identity)
    assert result["status"]=="FAIL"
    assert any(c["check"]=="saved_pair_values" and c["status"]=="FAIL" for c in audit.checks)


def test_null_shift_count_cannot_exceed_protocol(evidence):
    rows=deepcopy(evidence["records"])
    rows[0]["raw_field_null_spearman_valid_shift_count"]=21
    put_csv(evidence["alignment"]/"per_sample_metrics.csv",rows)
    audit,identity,contract=setup_audit(evidence)
    assert audit_alignment(audit,evidence["alignment"],identity,contract)["status"]=="FAIL"


def test_failed_normal_forward_check_cannot_hide_behind_pass(evidence):
    path=evidence["intervention"]/"protocol.json"
    protocol=json.loads(path.read_text())
    protocol["normal_forward_consistency"]["samples"][0]["checks"]["normal_vs_details"]["status"]="FAIL"
    put_json(path,protocol)
    audit,identity,contract=setup_audit(evidence)
    assert audit_intervention(audit,evidence["intervention"],identity)["status"]=="FAIL"


def test_official_discrepancy_is_reported_without_mixing_paired_baseline(evidence):
    path=evidence["run"]/"result/test_metrics.csv"
    audit=Audit()
    rows=audit.read(path,"csv",section="fixture")
    for row in rows:
        row["psnr"]=str(float(row["psnr"])+.1)
    put_csv(path,rows)
    audit,identity,contract=setup_audit(evidence)
    result=audit_intervention(audit,evidence["intervention"],identity)
    assert result["official_comparison"]["psnr"]["original_minus_official"]==pytest.approx(-.1)
    assert next(r for r in result["records"] if r["variant"]=="spatial_mean" and r["metric"]=="psnr")["paired_gain"]==pytest.approx(.01)


def test_missing_historical_source_is_unverified_not_assumed_current(evidence):
    path=evidence["alignment"]/"protocol.json"
    protocol=json.loads(path.read_text())
    protocol.pop("code")
    put_json(path,protocol)
    audit,identity,contract=setup_audit(evidence)
    result=audit_alignment(audit,evidence["alignment"],identity,contract)
    assert result["status"]=="UNVERIFIED"
    assert all(r["numeric_status"]=="PASS" for r in result["records"])


def test_wrong_frozen_manifest_is_failure(evidence):
    path=evidence["run"]/"split_snapshot/test.tsv"
    path.write_text(path.read_text()+"extra\tinput/extra.jpg\tGT/extra.jpg\n")
    audit,identity,contract=setup_audit(evidence)
    assert audit.status("splits")=="FAIL"


def test_missing_variant_not_assumed_complete(evidence):
    changed=[row for row in evidence["metrics"] if row["variant"]!="zero_field"]
    put_csv(evidence["intervention"]/"per_sample_metrics.csv",changed)
    audit,identity,contract=setup_audit(evidence)
    result=audit_intervention(audit,evidence["intervention"],identity)
    row=next(r for r in result["records"] if r["variant"]=="zero_field" and r["metric"]=="psnr")
    assert row["observed_mean"] is None and row["numeric_status"]=="UNVERIFIED"
    assert result["status"]!="PASS"


@pytest.mark.parametrize("mutation,expected",[
    ({"checkpoint_sha256":None},"UNVERIFIED"),
    ({"checkpoint_sha256":"f"*64},"FAIL"),
    ({"checkpoint":"/moved/"+BASELINE_NAME+"/best/best_ssim.pt"},"FAIL"),
    ({"checkpoint":"/moved/"+BASELINE_NAME+"/best/best_psnr.pt"},"PASS"),
])
def test_checkpoint_identity_relocation_and_missing_hash(evidence,mutation,expected):
    audit,identity,contract=setup_audit(evidence)
    protocol={**evidence["protocol"],**mutation}
    verify_identity(audit,protocol,identity,"intervention:identity",evidence["intervention"]/"protocol.json")
    names=("checkpoint_selector","checkpoint_content_identity","checkpoint_run_label")
    states=[c["status"] for c in audit.checks if c["section"]=="intervention:identity" and c["check"] in names]
    from tools.uicf_evidence_audit import combined_status
    assert combined_status(states)==expected


def test_optional_binary_hash_is_streamed_without_loading(evidence,monkeypatch):
    import builtins
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if name=="torch" or name.startswith("torch."):
            raise AssertionError("Auditor must never import torch")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guard)
    audit=Audit()
    result=baseline(audit,evidence["run"],evidence["contract"],hash_checkpoint=True)
    assert result["sha256"]==sha256(evidence["run"]/"best/best_psnr.pt")
    assert result["hash_basis"]=="fresh_binary_hash"


def test_safe_output_rejects_source_and_symlink(tmp_path):
    project=tmp_path/"project"
    (project/"analysis").mkdir(parents=True)
    with pytest.raises(ValueError):
        safe_output_parent(project/"experiments",project_root=project)
    alias=project/"analysis/alias"
    alias.symlink_to(project/"analysis",target_is_directory=True)
    with pytest.raises(ValueError):
        safe_output_parent(alias/"nested",project_root=project)
    with pytest.raises(ValueError):
        safe_output_parent(project/"analysis/input/nested",[project/"analysis/input"],project_root=project)


def test_repeated_outputs_do_not_overwrite(tmp_path):
    first=new_output(tmp_path/"audits")
    (first/"sentinel").write_text("keep")
    second=new_output(tmp_path/"audits")
    assert first!=second and (first/"sentinel").read_text()=="keep"


def test_source_integrity_detects_external_change(tmp_path):
    source=tmp_path/"file.json"
    put_json(source,{"x":1})
    audit=Audit()
    audit.read(source,"json",section="x")
    put_json(source,{"x":2})
    audit.unchanged()
    assert audit.status("source_integrity")=="FAIL"


def test_duplicate_json_keys_and_nonfinite_json_fail(tmp_path):
    path=tmp_path/"bad.json"
    path.write_text('{"x":1,"x":2}')
    audit=Audit()
    assert audit.read(path,"json",section="x") is None and audit.status("x")=="FAIL"
    path2=tmp_path/"nan.json"
    path2.write_text('{"x":NaN}')
    assert audit.read(path2,"json",section="n") is not None and audit.status("n")=="FAIL"


def test_manuscript_absence_is_pending_not_no_revision():
    audit=Audit()
    result=manuscript(audit,None,[])
    assert result["status"]=="PENDING_MANUSCRIPT" and audit.status("manuscript")=="UNVERIFIED"


def test_manuscript_numeric_mismatch_keeps_real_quote(tmp_path):
    path=tmp_path/"paper.json"
    put_json(path,dict(section_4_4="EXACT PROVIDED TEXT",table_4_rows=[dict(response="raw_field",spearman=.8,iou=.325)]))
    audit=Audit()
    result=manuscript(audit,path,[dict(response="raw_field",metric="spearman",observed_mean=.534424),
                                dict(response="raw_field",metric="iou",observed_mean=.324506)])
    assert result["suggestions"][0]["severity"]=="P0"
    assert "EXACT PROVIDED TEXT" in result["text"] and audit.status("manuscript")=="FAIL"


def test_cli_missing_files_generates_all_outputs_without_torch(tmp_path,monkeypatch,capsys):
    import builtins
    real_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split(".")[0] in ("torch","PIL","matplotlib"):
            raise AssertionError("No model/image dependency may be imported")
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    missing=tmp_path/"missing"/BASELINE_NAME
    parent=tmp_path/"audits"
    assert main(["--run-dir",str(missing),"--output-dir",str(parent),"--export-zip"])==2
    output=next(p for p in parent.glob("task2_*") if p.is_dir())
    assert set(p.name for p in output.iterdir())==set(REPORT_FILES)
    summary=json.loads((output/"audit_summary.json").read_text())
    assert summary["status"]=="UNVERIFIED"
    assert summary["policy"]["inference_run"] is False
    assert summary["alignment"][0]["records"][0]["observed_mean"] is None
    assert summary["manuscript"]["status"]=="PENDING_MANUSCRIPT"
    with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
        assert len(archive.namelist())==6
        assert all(not name.endswith((".pt",".png",".npy")) for name in archive.namelist())
    with pytest.raises(FileExistsError):
        export_zip(output)
    capsys.readouterr()


def test_full_numeric_fixture_cli_is_read_only_and_explicitly_synthetic(evidence,tmp_path,capsys):
    before={p:sha256(p) for p in evidence["run"].rglob("*") if p.is_file()}
    parent=tmp_path/"audits"
    assert main(["--run-dir",str(evidence["run"]),"--output-dir",str(parent)])==2
    output=next(parent.glob("task2_*"))
    summary=json.loads((output/"audit_summary.json").read_text())
    assert summary["status"]=="UNVERIFIED"
    assert not summary["explicit_errors"],summary["explicit_errors"]
    assert summary["section_status"]["spatial_correspondence"]=="PASS"
    assert summary["section_status"]["functional_intervention"]=="PASS"
    assert all(sha256(path)==value for path,value in before.items())
    assert (output/"EVIDENCE_CHAIN_AUDIT_CN.md").is_file()
    capsys.readouterr()


def test_cli_help_runs_in_a_fresh_process():
    result=subprocess.run([sys.executable,str(ROOT/"tools/audit_uicf_evidence_chain.py"),"--help"],
                          capture_output=True,text=True,check=False)
    assert result.returncode==0 and "--hash-checkpoint" in result.stdout
