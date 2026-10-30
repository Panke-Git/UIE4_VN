#!/usr/bin/env python3
"""Read existing native V16 test evidence; never load weights or run inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.uicf_parameterization_ablations import VARIANTS, version

BASELINE_RUN = "/home/paxton/pro/UIE4_VN/experiments/v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528"
PROVIDED_BASELINE = dict(psnr=28.388665, ssim=0.928637, e00=5.204765)
LABELS = dict(direct_residual="Direct Residual", without_anchor="w/o Anchor", fixed_anchor="Fixed Anchor (0.5)", full_uicf="Full UICF")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_csv(path, summary, entries):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [row["sample_id"] for row in rows]
    if len(rows) != len(entries) or summary.get("sample_count") != len(entries) or len(set(ids)) != len(ids) or set(ids) != set(entries):
        raise ValueError(f"Incomplete/duplicate test samples: {path}")
    for row in rows:
        if row["filename"] != entries[row["sample_id"]]:
            raise ValueError(f"Test filename mismatch: {path}")
    scores = {}
    for metric in ("psnr", "ssim", "e00"):
        values = [float(row[metric]) for row in rows]
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"Non-finite {metric}: {path}")
        scores[metric] = sum(values)/len(values)
        if not math.isclose(scores[metric], float(summary["mean_"+metric]), abs_tol=1e-6, rel_tol=0):
            raise ValueError(f"Summary/CSV {metric} mean mismatch: {path}")
    return scores


def read_result(run: Path, variant: str):
    if not run.is_dir():
        raise FileNotFoundError(run)
    cfg = load_config(run/"config_resolved.yaml")
    if variant == "full_uicf" and run.name != Path(BASELINE_RUN).name:
        raise ValueError("仅使用用户指定的原始 V16 基线运行；不是任意 Full UICF 运行")
    if variant != "full_uicf" and cfg.get("ablation", {}).get("smoke_test", False):
        raise ValueError("Synthetic smoke results must never enter the paper comparison")
    expected_version = "v16" if variant == "full_uicf" else version(variant)
    if cfg["experiment"]["version"] != expected_version:
        raise ValueError(f"Wrong model version in {run}")
    if variant != "full_uicf" and cfg["model"].get("parameterization") != variant:
        raise ValueError(f"Wrong parameterization in {run}")
    baseline_cfg = load_config(ROOT/"configs/config_v16.yaml")
    for section in ("training", "optimizer", "scheduler", "loss", "metrics", "evaluation", "checkpoint", "test"):
        if cfg[section] != baseline_cfg[section]:
            raise ValueError(f"Original V16 recipe mismatch in {section}: {run}")
    for key in ("batch_size", "patch_size", "augmentation", "pad_if_smaller", "pad_mode"):
        if cfg["data"][key] != baseline_cfg["data"][key]:
            raise ValueError(f"V16 data recipe mismatch: {key}: {run}")
    if cfg["experiment"]["seed"] != 3520:
        raise ValueError(f"Seed mismatch: {run}")
    info = load_json(run/"run_info.json")
    if info.get("seed") != 3520 or info.get("version") != expected_version:
        raise ValueError(f"Run metadata version/seed mismatch: {run}")
    if variant != "full_uicf" and info.get("variant") != variant:
        raise ValueError(f"Run metadata variant mismatch: {run}")
    for split, count in (("train", 3466), ("validation", 385), ("test", 428)):
        if info.get(split+"_sample_count") != count:
            raise ValueError(f"Run metadata sample count mismatch: {split}: {run}")
    model = deepcopy(cfg["model"])
    if variant != "full_uicf":
        if model["type"] != "uicf_parameterization_pre_backbone" or model["uicf"].get("use_learned_anchor") is not False:
            raise ValueError(f"Not the implemented parameterization control: {run}")
        model.pop("parameterization")
        model["type"] = baseline_cfg["model"]["type"]
        model["uicf"].pop("use_learned_anchor")
    if model != baseline_cfg["model"]:
        raise ValueError(f"Backbone/QGRD/field architecture mismatch: {run}")
    pinned = load_json(ROOT/"src/uicf_parameterization_ablations/protection_snapshot.json")["split_sha256"]
    test_entries = {}
    for split in ("train", "validation", "test"):
        p = run/"split_snapshot"/f"{split}.tsv"
        if hashlib.sha256(p.read_bytes()).hexdigest() != pinned[split]:
            raise ValueError(f"LSUI split SHA256 mismatch: {p}")
        if split == "test":
            with p.open(newline="") as f:
                entries = list(csv.reader(f, delimiter="\t"))
            test_entries = {x[0]: Path(x[1]).name for x in entries}
            if len(test_entries) != 428:
                raise ValueError("Expected 428 distinct held-out LSUI samples")
    status = load_json(run/"status.json")
    summary_path = run/"result/test_summary.json"
    if status.get("status") != "completed" or status.get("last_epoch") != 200 or not summary_path.is_file():
        return None, "训练或测试尚未完成", cfg
    summary = load_json(summary_path)
    if summary.get("split", "test") != "test" or summary.get("smoke_test", False):
        raise ValueError("Validation/synthetic metrics cannot fill test results")
    if summary.get("checkpoint_selector", "best_psnr") != "best_psnr" or Path(summary["checkpoint"]).parts[-2:] != ("best", "best_psnr.pt"):
        raise ValueError("Use the best validation PSNR checkpoint")
    if not (run/"best/best_psnr.pt").is_file():
        raise FileNotFoundError(run/"best/best_psnr.pt")
    if variant != "full_uicf" and summary.get("variant") != variant:
        raise ValueError("Test summary variant mismatch")
    if variant != "full_uicf":
        if summary.get("metric_input") != "float_rgb_before_png_save" or summary.get("evaluation") != cfg["evaluation"] or summary.get("metrics_config") != cfg["metrics"]:
            raise ValueError("Test metric precision/resolution protocol mismatch")
    e00 = summary.get("e00_protocol", {})
    for key, value in (("lab_illuminant", "D65"), ("lab_observer", "2 degrees"), ("kL", 1.0), ("kC", 1.0), ("kH", 1.0), ("crop_border", 0)):
        if e00.get(key) != value:
            raise ValueError(f"CIEDE2000 protocol mismatch: {key}")
    scores = check_csv(run/"result/test_metrics.csv", summary, test_entries)
    return scores, str(summary_path), cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", type=Path, default=Path(BASELINE_RUN))
    p.add_argument("--use-provided-baseline", action="store_true", help="Explicit Mac-only fallback to user-provided values; mark WSL verification pending")
    for variant in VARIANTS:
        p.add_argument("--"+variant.replace("_", "-")+"-run", type=Path)
    p.add_argument("--parameter-counts", type=Path, default=ROOT/"analysis/uicf_parameterization/parameter_counts.json")
    p.add_argument("--output-dir", type=Path, default=ROOT/"analysis/uicf_parameterization")
    args = p.parse_args(argv)
    records = {}
    if (args.baseline_run/"result/test_summary.json").is_file():
        records["full_uicf"] = read_result(args.baseline_run, "full_uicf")
        if records["full_uicf"][0] is None:
            raise ValueError("指定 Full UICF 基线尚无完整测试证据")
    elif args.use_provided_baseline:
        records["full_uicf"] = (PROVIDED_BASELINE, "用户提供，待 WSL 验证；未读取 checkpoint/测试文件", None)
    else:
        raise FileNotFoundError("指定基线不可读；WSL 应提供真实路径，Mac 仅可显式使用 --use-provided-baseline")
    for variant in VARIANTS:
        path = getattr(args, variant+"_run")
        records[variant] = read_result(path, variant) if path is not None else (None, "未提供正式运行，测试指标缺失", None)
    capacity = load_json(args.parameter_counts)["variants"] if args.parameter_counts.is_file() else {}
    rows = []
    for variant in (*VARIANTS, "full_uicf"):
        scores, source, cfg = records[variant]
        cap = capacity.get(variant, {})
        path = args.baseline_run if variant == "full_uicf" else getattr(args, variant+"_run")
        if path is not None and (path/"run_info.json").is_file():
            info = load_json(path/"run_info.json")
            actual = info.get("parameter_counts", {})
            if actual:
                cap = dict(cap, **actual)
            if "model_total_params" in info and cap.get("total") not in (None, info["model_total_params"]):
                raise ValueError(f"Registered parameter count differs from measured architecture: {path}")
        row = dict(variant=LABELS[variant], **{m: scores[m] if scores else None for m in ("psnr", "ssim", "e00")},
                   **{"delta_"+m: scores[m]-records["full_uicf"][0][m] if scores else None for m in ("psnr", "ssim", "e00")},
                   total_parameters=cap.get("total"), trainable_parameters=cap.get("trainable"),
                   forward_participating_parameters=cap.get("forward_participating", cap.get("output_connected")),
                   not_output_connected_parameters=cap.get("not_output_connected"), parameter_difference_from_full=cap.get("difference_from_full"),
                   evidence=source, run_dir=str(path) if path else None)
        rows.append(row)
    # Complete validation before writing; existing reports survive rejected input.
    output = args.output_dir.resolve()
    if any(path and (output == path.resolve() or path.resolve() in output.parents) for path in [args.baseline_run, *[getattr(args, v+"_run") for v in VARIANTS]]):
        raise ValueError("Do not write reports inside experiment runs")
    output.mkdir(parents=True, exist_ok=True)
    with (output/"results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def fmt(x):
        return "—" if x is None else f"{x:.6f}"
    lines = ["# UICF 校正参数化消融结果", "", "PSNR/SSIM 越高越好；ΔE00 越低越好。差值均为 Variant − Full UICF，因此负 ΔE00 表示改善。", "",
             "| Variant | PSNR ↑ | SSIM ↑ | ΔE00 ↓ | ΔPSNR | ΔSSIM | ΔΔE00 |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines += ["| "+" | ".join([row["variant"], *[fmt(row[k]) for k in ("psnr", "ssim", "e00", "delta_psnr", "delta_ssim", "delta_e00")]])+" |" for row in rows]
    lines += ["", "## 模型容量", "", "参数来自本地标准维度模型的实际输出图检查或正式 run_info；不是未完成实验的恢复性能。", "",
              "| Variant | 注册总量 | 可训练 | 参与输出前向 | 不参与输出 | 相对 Full 差异 |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| "+" | ".join([row["variant"], *[str(row[k]) if row[k] is not None else "—" for k in ("total_parameters", "trainable_parameters", "forward_participating_parameters", "not_output_connected_parameters", "parameter_difference_from_full")]])+" |")
    lines += ["", "## 证据与限制", ""]
    lines += [f"- {row['variant']}: {row['evidence']}; run: `{row['run_dir']}`." for row in rows]
    lines += ["", "Fixed Anchor 的常数为 (0.5,0.5,0.5)，不是无锚点，也不是训练集均值。三个变体移除 Anchor MLP，不能宣称与 Full 严格参数匹配。",
              "未完成实验保留缺失值；不读取验证指标代替测试，不加载 checkpoint，不执行模型推理。", ""]
    (output/"results.md").write_text("\n".join(lines), encoding="utf-8")
    print(output/"results.md")
    return rows


if __name__ == "__main__":
    main()
