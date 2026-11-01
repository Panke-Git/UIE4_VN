"""Metadata identity and reviewed source contracts, without checkpoint loading."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
from pathlib import Path
import re

from . import BASELINE_NAME, COUNTS, METRICS
from .io import ROOT, sha256
from .statistics import check_rows, compare_saved, describe


def function_nodes(text):
    result = {}
    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                result[name] = child
                visit(child, name + ".")
            else:
                visit(child, prefix)
    visit(ast.parse(text))
    return result


def fingerprint(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def valid_sha256(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-fA-F]{64}",value) is not None


def review_sources(audit):
    contract_path = Path(__file__).with_name("source_contract.json")
    contract = audit.read(contract_path, "json", section="code_review") or {}
    for relative, record in contract.get("sources", {}).items():
        path = ROOT / relative
        text = audit.read(path, "text", section="code_review")
        if text is None:
            continue
        try:
            nodes = function_nodes(text)
            matches = {name: name in nodes and fingerprint(nodes[name]) == digest
                       for name, digest in record["functions"].items()}
        except (ValueError, SyntaxError) as error:
            audit.check("code_review", "source_parse", "FAIL", f"{relative}: {error}", [path], "P0")
            continue
        same_file = hashlib.sha256(text.encode("utf-8")).hexdigest() == record["sha256"]
        audit.check("code_review", "reviewed_source_contract", "PASS" if same_file and all(matches.values()) else "UNVERIFIED",
            f"{relative}: full_source_match={same_file}; function_matches={matches}; "
            "当前源码匹配只证明已审阅实现未变化，不证明历史实验必然运行了当前版本。",
            [path])
        audit.annotate(path, record["functions"], "审阅过的 AST 及源码 SHA256；不导入原 GPU 分析模块")
    return contract


def _nested(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def baseline(audit, run, contract, *, hash_checkpoint=False):
    section = "baseline"
    run = Path(run)
    audit.check(section, "specified_run", "PASS" if run.name == BASELINE_NAME else "FAIL",
                f"run={run}; 指定唯一基线={BASELINE_NAME}", severity="P0")
    config_path = run/"config_resolved.yaml"
    config = audit.read(config_path, "yaml", section=section) or {}
    info = audit.read(run/"run_info.json", "json", section=section) or {}
    status = audit.read(run/"status.json", "json", section=section) or {}
    official = audit.read(run/"result/test_summary.json", "json", section=section) or {}
    best = audit.read(run/"best/best_psnr.json", "json", section=section) or {}
    reference = audit.read(ROOT/"configs/config_v16.yaml", "yaml", section=section) or {}
    official_checkpoint=str(official.get("checkpoint","")).replace("\\","/")
    audit.check(section,"official_checkpoint_selector","UNVERIFIED" if not official_checkpoint else (
        "PASS" if official_checkpoint.endswith("/best/best_psnr.pt") else "FAIL"),
        official_checkpoint,[run/"result/test_summary.json"],"P0")
    for name, actual, expected in (
        ("model_version", _nested(config,"experiment","version"), "v16"),
        ("seed", _nested(config,"experiment","seed"), 3520),
        ("epochs", _nested(config,"training","epochs"), 200),
        ("evaluation", config.get("evaluation"), reference.get("evaluation")),
        ("baseline_model", config.get("model"), reference.get("model")),
        ("training_complete", status.get("status"), "completed"),
        ("last_epoch", status.get("last_epoch"), 200),
    ):
        audit.check(section, name, "UNVERIFIED" if actual is None or expected is None else (
            "PASS" if actual == expected else "FAIL"), f"记录={actual!r}; 期望={expected!r}", [config_path, run/"status.json"])
    for field, expected in (("version","v16"),("seed",3520)):
        actual = info.get(field)
        audit.check(section, "run_info_"+field, "UNVERIFIED" if actual is None else (
            "PASS" if actual == expected else "FAIL"), f"{actual!r}", [run/"run_info.json"])
    for label,declared in (("config",_nested(config,"data","dataset")),("run_info",info.get("dataset"))):
        if declared is not None:
            audit.check(section,label+"_dataset","PASS" if declared=="LSUI19" else "FAIL",
                        f"{declared!r}; LSUI 身份还须由冻结 manifest 哈希核验",[config_path,run/"run_info.json"],"P0")
    for key,expected in (("lab_illuminant","D65"),("lab_observer","2 degrees"),("kL",1),("kC",1),("kH",1),("crop_border",0)):
        actual=official.get("e00_protocol",{}).get(key)
        audit.check("official","e00_protocol_"+key,"UNVERIFIED" if actual is None else (
            "PASS" if actual==expected else "FAIL"),f"{actual!r}; expected={expected!r}",[run/"result/test_summary.json"],"P0")
    for name in ("optimizer", "loss", "scheduler"):
        actual, expected = config.get(name), reference.get(name)
        audit.check(section, "recipe_"+name, "UNVERIFIED" if actual is None or expected is None else (
            "PASS" if actual == expected else "FAIL"), f"{name} 与本地原 V16 配置比较", [config_path])
    audit.extra_protected += [Path(config["data"]["root"])] if _nested(config,"data","root") else []
    splits = {}
    for split, expected in COUNTS.items():
        path = run/"split_snapshot"/f"{split}.tsv"
        rows = audit.read(path, "tsv", section="splits")
        splits[split] = rows
        if rows is None:
            continue
        for column in range(3):
            values = [r[column] for r in rows]
            duplicates = sum(n-1 for n in Counter(values).values() if n > 1)
            audit.check("splits", f"{split}_unique_column_{column}", "FAIL" if duplicates else "PASS",
                        f"重复={duplicates}", [path], "P0")
        audit.check("splits", split+"_count", "PASS" if len(rows) == expected else "FAIL",
                    f"{len(rows)} / {expected}", [path], "P0")
        expected_hash = contract.get("split_sha256", {}).get(split)
        actual_hash = audit.sources[str(path.absolute())]["sha256"]
        audit.check("splits", split+"_frozen_hash", "UNVERIFIED" if not expected_hash else (
            "PASS" if actual_hash == expected_hash else "FAIL"), f"{actual_hash}; pinned={expected_hash}", [path], "P0")
        unsafe = [r for r in rows if any(Path(v).is_absolute() or ".." in Path(v).parts for v in r[1:])]
        audit.check("splits", split+"_relative_paths", "FAIL" if unsafe else "PASS",
                    f"非法相对路径={len(unsafe)}", [path], "P0")
        for count, label in ((info.get(split+"_sample_count"), "run_info"),
                             (_nested(config,"data","expected_counts",split), "config")):
            if count is not None:
                compare_saved(audit, "splits", f"{split}_{label}_count", len(rows), count, path, atol=0)
    names = tuple(COUNTS)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            if splits[first] is None or splits[second] is None:
                continue
            for column in range(3):
                overlap = {r[column] for r in splits[first]} & {r[column] for r in splits[second]}
                audit.check("splits", f"{first}_{second}_leakage_{column}", "FAIL" if overlap else "PASS",
                            f"交集={len(overlap)}", [run/"split_snapshot"/f"{first}.tsv",run/"split_snapshot"/f"{second}.tsv"], "P0")
    csv_path = run/"result/test_metrics.csv"
    rows = audit.read(csv_path, "csv", section="official")
    check_rows(audit, "official", rows, splits["test"], csv_path)
    observed = {}
    for metric in METRICS:
        stats = describe(rows or [], metric)
        observed[metric] = stats
        if rows is not None:
            audit.check("official", metric+"_validity",
                "FAIL" if stats["nonfinite_or_malformed_count"] else ("PASS" if stats["valid_count"] == len(rows) and rows else "UNVERIFIED"),
                str(stats), [csv_path], "P0")
            compare_saved(audit,"official",metric+"_saved_mean",stats["mean"],official.get("mean_"+metric),run/"result/test_summary.json")
            reference_value = contract.get("user_reference",{}).get("official",{}).get(metric)
            compare_saved(audit,"official",metric+"_provided_reference",stats["mean"],reference_value,
                          Path(__file__).with_name("source_contract.json"),atol=0.5e-6+1e-12)
    if rows is not None:
        compare_saved(audit,"official","sample_count",len(rows),official.get("sample_count"),run/"result/test_summary.json",atol=0)
    epoch = best.get("epoch")
    compare_saved(audit,section,"official_best_epoch",epoch,official.get("checkpoint_epoch"),run/"result/test_summary.json",atol=0)
    checkpoint = run/"best/best_psnr.pt"
    digest = None
    if checkpoint.is_file():
        digest = sha256(checkpoint) if hash_checkpoint else None
        audit.sources[str(checkpoint.absolute())] = dict(path=str(checkpoint.absolute()),
            resolved_path=str(checkpoint.resolve()), sha256=digest, bytes=checkpoint.stat().st_size,
            fields=["binary SHA256 only; no deserialization"] if hash_checkpoint else ["existence/size only"],
            definitions=["默认不读取权重；--hash-checkpoint 只流式计算文件 SHA256，不加载模型"],
            sample_count=None,status="PASS" if digest else "UNVERIFIED")
    else:
        audit.sources[str(checkpoint.absolute())] = dict(path=str(checkpoint.absolute()),
            resolved_path=str(checkpoint.resolve()), sha256=None, bytes=None,
            fields=["existence/size only"], sample_count=None, status="UNVERIFIED",
            definitions=["checkpoint 缺失；不加载、不猜测哈希或 epoch"])
        audit.check(section,"checkpoint_exists","UNVERIFIED",f"未找到 {checkpoint}",[checkpoint])
    recorded_hashes = [value for value in (official.get("checkpoint_sha256"),best.get("checkpoint_sha256"),
                                          info.get("checkpoint_sha256")) if value]
    if any(not valid_sha256(value) for value in recorded_hashes):
        audit.check(section,"baseline_hash_format","FAIL","保存的 checkpoint 哈希不是 SHA256",severity="P0")
    if len(set(recorded_hashes)) > 1:
        audit.check(section,"baseline_hash_conflict","FAIL","基线元数据中的 checkpoint 哈希互相冲突",severity="P0")
    if recorded_hashes and digest and digest != recorded_hashes[0]:
        audit.check(section,"baseline_binary_hash","FAIL","文件 SHA256 与记录不一致",[checkpoint], "P0")
    identity = dict(run=run, config=config, info=info, official=official, best=best,
                    checkpoint=checkpoint, epoch=epoch, sha256=digest or (recorded_hashes[0] if recorded_hashes else None),
                    hash_basis="fresh_binary_hash" if digest else ("saved_baseline_metadata" if recorded_hashes else "none"),
                    splits=splits, observed=observed, rows=rows)
    return identity


def verify_identity(audit, protocol, identity, section, source):
    checkpoint = str(protocol.get("checkpoint", "")).replace("\\","/")
    selector = protocol.get("checkpoint_selector")
    label_ok = checkpoint.endswith("/best/best_psnr.pt") and selector in ("best_psnr", None)
    if not checkpoint:
        state = "UNVERIFIED"
    else:
        state = "PASS" if label_ok else "FAIL"
    audit.check(section,"checkpoint_selector",state,f"{checkpoint}; selector={selector}",[source],"P0")
    old_run = protocol.get("checkpoint_run", protocol.get("run_dir"))
    run_parts = checkpoint.strip("/").split("/")
    label=(str(old_run).replace("\\","/").rstrip("/").rsplit("/",1)[-1] if old_run
           else run_parts[-3] if len(run_parts)>=3 and run_parts[-2]=="best" else None)
    wrong_run=bool(label and label.startswith("v16_") and label != BASELINE_NAME)
    audit.check(section,"checkpoint_run_label","FAIL" if wrong_run else (
        "PASS" if label==BASELINE_NAME else "UNVERIFIED"),
        f"历史绝对路径可以迁移；运行标识={old_run}; basename={label}",[source],"P0")
    compare_saved(audit,section,"checkpoint_epoch",identity["epoch"],protocol.get("checkpoint_epoch"),source,atol=0)
    actual_hash, expected_hash = protocol.get("checkpoint_sha256"),identity["sha256"]
    hash_status = ("FAIL" if actual_hash and not valid_sha256(actual_hash) else
                   "UNVERIFIED" if not actual_hash or not expected_hash else
                   "PASS" if actual_hash == expected_hash else "FAIL")
    audit.check(section,"checkpoint_content_identity",hash_status,
        f"分析记录={actual_hash}; 基线={expected_hash}; basis={identity['hash_basis']}; "
        "仅文件名与 epoch 一致不能独立确证权重同一性。",[source,identity["checkpoint"]],"P0")
    manifest = identity["run"]/"split_snapshot/test.tsv"
    expected = audit.sources.get(str(manifest.absolute()),{}).get("sha256")
    compare = protocol.get("test_manifest_sha256")
    audit.check(section,"test_manifest_hash","UNVERIFIED" if not compare or not expected else (
        "PASS" if compare == expected else "FAIL"),f"分析={compare}; 冻结={expected}",[source,manifest],"P0")
    for name, expected in COUNTS.items():
        value = protocol.get(name+"_count", protocol.get("test_manifest_sample_count") if name == "test" else None)
        compare_saved(audit,section,name+"_count",expected,value,source,atol=0)
    for name in ("dataset","checkpoint_dataset","evaluation_dataset"):
        value = protocol.get(name)
        audit.check(section,name,"UNVERIFIED" if value is None else ("PASS" if value == "LSUI19" else "FAIL"),
                    f"{name}={value!r}",[source],"P0")
    config_hash = protocol.get("config_resolved_yaml_sha256",protocol.get("config_resolved_sha256"))
    expected = audit.sources.get(str((identity["run"]/"config_resolved.yaml").absolute()),{}).get("sha256")
    audit.check(section,"config_content_identity","UNVERIFIED" if not config_hash or not expected else (
        "PASS" if config_hash == expected else "FAIL"),f"分析={config_hash}; 冻结={expected}",[source],"P0")
    hashes = protocol.get("code",{}).get("source_sha256",{})
    relevant = [("tools/analyze_uicf_alignment.py","src/shared/uicf_inr.py") if section.startswith("alignment")
                else ("tools/analyze_uicf_intervention.py","tools/uicf_intervention/core.py","src/shared/uicf_inr.py")][0]
    for relative in relevant:
        path = ROOT/relative
        current = audit.sources.get(str(path.absolute()),{}).get("sha256")
        recorded = hashes.get(relative)
        audit.check(section,"historical_source_"+relative,"UNVERIFIED" if not recorded else (
            "PASS" if current == recorded else "UNVERIFIED"),
            f"历史={recorded}; 当前={current}; 不一致时需要审阅历史源码，不能以当前实现代替历史运行证明。",[source,path])
