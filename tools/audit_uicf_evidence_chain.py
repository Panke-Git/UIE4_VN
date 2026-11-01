#!/usr/bin/env python3
"""只读核验已有 UICF 实验证据；不导入 Torch、不加载模型、不运行推理。"""
from __future__ import annotations

import argparse
from datetime import datetime
from importlib import metadata
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools.uicf_evidence_audit import combined_status
from tools.uicf_evidence_audit.alignment import audit_alignment
from tools.uicf_evidence_audit.intervention import audit_intervention
from tools.uicf_evidence_audit.io import Audit, export_zip, is_within, new_output, safe_output_parent
from tools.uicf_evidence_audit.provenance import baseline, review_sources
from tools.uicf_evidence_audit.report import claim_matrix, manuscript, render


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True,help="指定原始 V16 基线目录")
    parser.add_argument("--output-dir",default="analysis/uicf_evidence_chain_audit",
                        help="当前工程 analysis/ 内的输出父目录；每次新建 task2_<timestamp>")
    parser.add_argument("--alignment-dir",action="append",help="已有 alignment 目录；可重复指定，默认自动识别")
    parser.add_argument("--intervention-dir",action="append",help="已有 intervention 目录；可重复指定")
    parser.add_argument("--paper-claims-file",help="当前论文文字 .md/.txt/.tex 或含 table_4_rows 的 JSON")
    parser.add_argument("--hash-checkpoint",action="store_true",
                        help="可选：只流式 SHA256 checkpoint 文件，不反序列化或加载网络")
    parser.add_argument("--export-zip",action="store_true",help="另建 ZIP，仅包含六个审计交付文件")
    return parser.parse_args(argv)


def project_input(value):
    path=Path(value).expanduser()
    path=(ROOT/path if not path.is_absolute() else path).resolve()
    if not is_within(path,ROOT):
        raise ValueError(f"输入必须位于当前 UIE4_VN 工程内：{path}")
    return path


def discover(audit,run,kind,explicit):
    if explicit:
        return list(dict.fromkeys(project_input(value) for value in explicit))
    found=[]
    folder=run/"result"
    if folder.is_dir():
        for directory in sorted(folder.iterdir()):
            if not directory.is_dir() or not (directory/"protocol.json").is_file():
                continue
            # Inspect metadata, not the directory label, to classify an existing run.
            protocol=audit.read(directory/"protocol.json","json",section="discovery",required=False) or {}
            script=str(protocol.get("script","")).replace("\\","/").rsplit("/",1)[-1]
            if script==("analyze_uicf_alignment.py" if kind=="alignment" else "analyze_uicf_intervention.py"):
                found.append(directory.resolve())
    if found:
        return list(dict.fromkeys(found))
    # Known locations become explicit missing/unrecognized evidence, never scores.
    names=("uicf_alignment","uicf_representation_alignment") if kind=="alignment" else ("uicf_field_intervention",)
    return [folder/name for name in names]


def _audit_one(audit,directory,identity,contract,kind):
    section=kind+":"+str(directory)
    try:
        result=(audit_alignment(audit,directory,identity,contract) if kind=="alignment"
                else audit_intervention(audit,directory,identity))
    except (KeyError,TypeError,ValueError,OverflowError,AttributeError) as error:
        audit.check(section,"malformed_schema","FAIL",f"{type(error).__name__}: {error}",[directory],"P0")
        result=dict(directory=str(directory),section=section,records=[],status="FAIL",
                    error=f"{type(error).__name__}: {error}")
    return result


def main(argv=None):
    args=parse_args(argv)
    run=project_input(args.run_dir)
    parent_value=Path(args.output_dir).expanduser()
    if not parent_value.is_absolute():
        parent_value=ROOT/parent_value
    explicit=[project_input(value) for value in [*(args.alignment_dir or []),*(args.intervention_dir or [])]]
    paper_path=project_input(args.paper_claims_file) if args.paper_claims_file else None
    # Reject unsafe output paths before reading inputs, including symlink aliases.
    parent=safe_output_parent(parent_value,[run,*explicit,*([paper_path] if paper_path else [])])
    audit=Audit()
    contract=review_sources(audit)
    identity=baseline(audit,run,contract,hash_checkpoint=args.hash_checkpoint)
    if identity["info"].get("synthetic_fixture") or "SYNTHETIC" in str(run):
        audit.check("baseline","synthetic_evidence","UNVERIFIED",
                    "人工元数据仅用于软件测试，不是真实模型实验结果。",[run/"run_info.json"])
    alignment_dirs=discover(audit,run,"alignment",args.alignment_dir)
    intervention_dirs=discover(audit,run,"intervention",args.intervention_dir)
    parent=safe_output_parent(parent,[run,*alignment_dirs,*intervention_dirs,*audit.extra_protected,
                                     *([paper_path] if paper_path else [])])
    alignments=[_audit_one(audit,directory,identity,contract,"alignment") for directory in alignment_dirs]
    interventions=[_audit_one(audit,directory,identity,contract,"intervention") for directory in intervention_dirs]
    # Multiple outputs are all retained. Never silently choose the best correlation.
    spatial=[r for item in alignments for r in item.get("records",[])]
    for response in ("raw_field","actual_correction","anchor_deviation","image_gradient","shifted_raw_field"):
        for metric in ("spearman","iou"):
            members=[r for r in spatial if r["response"]==response and r["metric"]==metric and r["observed_mean"] is not None]
            if len(members)>1:
                spread=max(r["observed_mean"] for r in members)-min(r["observed_mean"] for r in members)
                audit.check("cross_artifact",response+"_"+metric,"PASS" if spread<=1e-9 else "UNVERIFIED",
                            f"多个产物均值差范围={spread}; 值={[r['observed_mean'] for r in members]}; "
                            "不择优合并；差异需按各自协议与来源解释。",[r["source_csv"] for r in members])
    paper=manuscript(audit,paper_path,spatial)
    # Record auditor source bytes as its code version; Git/network are not required.
    for path in [Path(__file__),*sorted((ROOT/"tools/uicf_evidence_audit").glob("*.py"))]:
        audit.read(path,"text",section="auditor_version")
        audit.annotate(path,definition="本次审计工具版本的源码 SHA256")
    audit.unchanged()
    for item in [*alignments,*interventions]:
        item["status"]=audit.status(item["section"])
        for record in item.get("records",[]):
            record["audit_status"]=item["status"]
    claims=claim_matrix(alignments,interventions)
    section_status=dict(
        baseline=audit.status("baseline"),splits=audit.status("splits"),official_test=audit.status("official"),
        reviewed_code=audit.status("code_review"),
        spatial_correspondence=combined_status(r["status"] for r in alignments),
        functional_intervention=combined_status(r["status"] for r in interventions),
        actual_correction_boundary=combined_status([r["numeric_status"] for r in spatial if r["response"]=="actual_correction"]
                                                   +[r["status"] for r in alignments]),
        manuscript=paper["status"],source_integrity=audit.status("source_integrity"))
    status=combined_status(c["status"] for c in audit.checks)
    output=new_output(parent)
    software={"python":sys.version}
    for distribution in ("numpy","PyYAML"):
        try:
            software[distribution]=metadata.version(distribution)
        except metadata.PackageNotFoundError:
            software[distribution]=None
    summary=dict(schema_version=1,created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        status=status,run_dir=str(run),output_dir=str(output),software=software,
        policy=dict(read_only=True,model_loaded=False,inference_run=False,gpu_required=False,
                    checkpoint_hashed=args.hash_checkpoint,mean_absolute_tolerance=1e-9,
                    missing_reference_values_imputed=False),
        baseline=dict(checkpoint_epoch=identity["epoch"],checkpoint_sha256=identity["sha256"],
                      checkpoint_hash_basis=identity["hash_basis"],official_recomputed=identity["observed"]),
        section_status=section_status,alignment=alignments,intervention=interventions,
        manuscript=paper,evidence_claim_matrix=claims,checks=audit.checks,
        missing_evidence=[c for c in audit.checks if c["status"]=="UNVERIFIED"],
        explicit_errors=[c for c in audit.checks if c["status"]=="FAIL"])
    render(output,audit,summary,alignments,interventions,paper,claims)
    archive=export_zip(output) if args.export_zip else None
    print(f"AUDIT_STATUS={status}")
    print(f"OUTPUT_DIR={output}")
    if archive:
        print(f"ZIP_FILE={archive}")
    return {"PASS":0,"FAIL":1,"UNVERIFIED":2}[status]


if __name__=="__main__":
    raise SystemExit(main())
