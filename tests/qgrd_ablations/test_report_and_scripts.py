from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import yaml
from src.shared.e00 import e00_protocol
from .helpers import ROOT, VARIANTS, config


def report_module():
    spec = importlib.util.spec_from_file_location("qgrd_report", ROOT / "tools/qgrd_ablation_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def report_runs(tmp_path, monkeypatch):
    report = report_module()
    monkeypatch.setattr(report, "PROTOCOLS", {"lsui": {"dataset": "LSUI19",
                        "counts": {"train": 2, "validation": 1, "test": 2}}})
    runs = {}
    for role in (*VARIANTS, "v16"):
        run = tmp_path / role
        for child in ("result", "split_snapshot"):
            (run / child).mkdir(parents=True)
        cfg = config(role) if role != "v16" else yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text())
        # Synthetic successful comparison uses one declared precision for all
        # roles. This changes no real baseline/A1/A3/A4 files or their runners.
        cfg["evaluation"]["precision"] = "fp32"
        cfg["data"]["expected_counts"] = report.PROTOCOLS["lsui"]["counts"]
        (run / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))
        info = {"version": cfg["experiment"]["version"], "resolved_config": cfg}
        protocol = {
            "precision": "fp32", "autocast_enabled": False,
            "input_dtype": "torch.float32", "prediction_dtype": "torch.float32",
            "loss_dtype": "torch.float32", "model_weights_changed_for_evaluation": False,
        }
        info.update(evaluation_precision="fp32", evaluation_forward_protocol=protocol)
        for split, count in cfg["data"]["expected_counts"].items():
            manifest = run / f"split_snapshot/{split}.tsv"
            manifest.write_text("".join(f"{split}_{i}\t{split}/input/{split}_{i}.png\t{split}/GT/{split}_{i}.png\n"
                                        for i in range(count)))
            info[split + "_tsv_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            info[split + "_sample_count"] = count
        (run / "run_info.json").write_text(json.dumps(info))
        offset = 0 if role == "v16" else -0.25
        rows = [{"sample_id": f"test_{i}", "filename": f"test_{i}.png",
                 "psnr": 25 + i + offset, "ssim": 0.9, "e00": 5 + i} for i in range(2)]
        with (run / "result/test_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"checkpoint": str(run / "best/best_psnr.pt"), "checkpoint_epoch": 1, "sample_count": 2,
                   "mean_psnr": 25.5 + offset, "mean_ssim": 0.9, "mean_e00": 5.5,
                   "e00_protocol": e00_protocol(cfg["metrics"])}
        summary.update(evaluation_precision="fp32", evaluation_forward_protocol=protocol)
        (run / "result/test_summary.json").write_text(json.dumps(summary))
        runs[role] = run
    return report, runs


def rewrite_json(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_report_audit_deltas_and_no_experiment_mutation(report_runs, tmp_path):
    report, runs = report_runs
    before = {p: p.read_bytes() for run in runs.values() for p in run.rglob("*") if p.is_file()}
    rows, deltas, audit = report.compare(runs, "lsui")
    assert audit["status"] == "PASS"
    assert len(rows) == 5 and len(deltas) == 8
    assert rows[0]["delta_psnr"] == -0.25
    assert rows[-1]["variant"] == "Full V16" and rows[-1]["delta_psnr"] == 0
    output = tmp_path / "reports/lsui"
    report.write_report(output, rows, deltas, audit)
    assert {p.name for p in output.iterdir()} == {"comparison.csv", "comparison.json", "comparison.md",
                                                "per_image_deltas.csv", "protocol_audit.json"}
    assert before == {p: p.read_bytes() for run in runs.values() for p in run.rglob("*") if p.is_file()}
    report.write_report(output, rows, deltas, audit)
    assert len(list(output.iterdir())) == 5


@pytest.mark.parametrize("field,value", [
    ("dataset", "UIEB"), ("sample_count", 90), ("checkpoint", "/other/best/best_ssim.pt"),
    ("metric_input", "png8"), ("metric_config", {"crop_border": 1}),
    ("evaluation", {"resize": False}), ("mean_e00", 100),
])
def test_report_refuses_summary_protocol_mismatch(report_runs, field, value):
    report, runs = report_runs
    rewrite_json(runs["shared_query"] / "result/test_summary.json", lambda s: s.update({field: value}))
    with pytest.raises(ValueError, match="refused"):
        report.compare(runs, "lsui")


def test_report_refuses_cross_run_manifest_hash_even_when_each_is_internally_consistent(report_runs):
    report, runs = report_runs
    run = runs["single_stage"]
    manifest = run / "split_snapshot/train.tsv"
    manifest.write_text(manifest.read_text().replace("train_0", "train_changed"))
    rewrite_json(run / "run_info.json", lambda i: i.update(train_tsv_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    with pytest.raises(ValueError, match="split_sha256"):
        report.compare(runs, "lsui")


def test_report_refuses_different_metric_protocol_and_sample_ids(report_runs):
    report, runs = report_runs
    path = runs["no_local_refine"] / "config_resolved.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["metrics"]["ssim_sigma"] = 2.0
    path.write_text(yaml.safe_dump(cfg))
    rewrite_json(runs["no_local_refine"] / "run_info.json", lambda i: i.update(resolved_config=cfg))
    with pytest.raises(ValueError, match="metrics"):
        report.compare(runs, "lsui")
    csv_path = runs["shared_query"] / "result/test_metrics.csv"
    csv_path.write_text(csv_path.read_text().replace("test_0", "missing_id"))
    with pytest.raises(ValueError, match="sample ID"):
        report.compare(runs, "lsui")


def test_report_does_not_import_models():
    import ast
    tree = ast.parse((ROOT / "tools/qgrd_ablation_report.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("src.") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("src.")


def test_report_refuses_mixed_legacy_amp_and_new_fp32(report_runs):
    report, runs = report_runs
    path = runs["v16"]
    cfg = yaml.safe_load((path / "config_resolved.yaml").read_text())
    cfg["evaluation"].pop("precision")
    (path / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))
    def legacy_info(record):
        record["resolved_config"] = cfg
        record.pop("evaluation_precision")
        record.pop("evaluation_forward_protocol")
    rewrite_json(path / "run_info.json", legacy_info)
    def legacy_summary(record):
        record.pop("evaluation_precision")
        record.pop("evaluation_forward_protocol")
    rewrite_json(path / "result/test_summary.json", legacy_summary)
    with pytest.raises(ValueError, match="evaluation_precision"):
        report.compare(runs, "lsui")


def test_report_refuses_precision_metadata_relabeling(report_runs):
    report, runs = report_runs
    rewrite_json(runs["shared_query"] / "result/test_summary.json",
                 lambda record: record.update(evaluation_precision="native_amp"))
    with pytest.raises(ValueError, match="evaluation_precision"):
        report.compare(runs, "lsui")


@pytest.mark.parametrize("dataset", ("lsui", "uieb"))
def test_wrapper_scripts_dry_run_explicit_sequential_commands(dataset, tmp_path):
    environment = {**os.environ, "PYTHON_BIN": sys.executable}
    training = subprocess.run(["bash", str(ROOT / f"scripts/qgrd_ablations/run_{dataset}_all.sh"), "--dry-run"],
                              capture_output=True, text=True, check=True, env=environment)
    lines = training.stdout.splitlines()
    assert len(lines) == 4
    for line, variant in zip(lines, VARIANTS):
        assert f"src.qgrd_ablations.{variant}.train" in line
        assert f"configs/ablations/qgrd/{dataset}/{variant}.yaml" in line
    testing = subprocess.run(["bash", str(ROOT / f"scripts/qgrd_ablations/test_{dataset}_all.sh"),
                              *(str(tmp_path / v) for v in VARIANTS), "--dry-run"],
                             capture_output=True, text=True, check=True, env=environment)
    assert len(testing.stdout.splitlines()) == 4
    for line, variant in zip(testing.stdout.splitlines(), VARIANTS):
        assert f"src.qgrd_ablations.{variant}.test" in line
        assert "best_psnr" in line and str(tmp_path / variant) in line


def test_training_script_stops_on_first_nonzero_exit(tmp_path):
    failed = subprocess.run(["bash", str(ROOT / "scripts/qgrd_ablations/run_lsui_all.sh")],
                            capture_output=True, text=True, env={**os.environ, "PYTHON_BIN": "/usr/bin/false"})
    assert failed.returncode != 0
    assert len(failed.stdout.splitlines()) == 1
