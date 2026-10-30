from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from src.v16.utils import PROJECT_ROOT, load_yaml, save_yaml
from src.v16.train import _torch_load
from src.uicf_parameterization_ablations import VARIANTS
from src.uicf_parameterization_ablations.models import build_model
from src.uicf_parameterization_ablations.protocol import verify_checkpoint
from src.uicf_parameterization_ablations.smoke import prepare_smoke
from src.uicf_parameterization_ablations.train import main as train
from src.uicf_parameterization_ablations.test import main as evaluate


@pytest.mark.parametrize("variant", VARIANTS)
def test_real_entrypoints_train_resume_test_and_exact_recovery(tmp_path, variant):
    source = prepare_smoke(tmp_path/variant, variant)
    run = train(["--config", str(source), "--variant", variant, "--smoke-test", "--seed", "3520", "--gpu", "0", "--stop-after-epoch", "1"])
    assert json.loads((run/"status.json").read_text())["status"] == "interrupted"
    assert not (run/"result/test_summary.json").exists()
    checkpoint = _torch_load(run/"checkpoint/last.pt", torch.device("cpu"))
    assert checkpoint["variant"] == variant and not checkpoint["training_status"]["completed"]
    for key in ("resolved_config", "seed", "split_sha256", "code_version", "model_state_dict", "optimizer_state_dict", "amp_scaler_state_dict", "rng_state", "dataloader_generator_state"):
        assert key in checkpoint
    cfg = load_yaml(run/"config_resolved.yaml")
    restored = build_model(cfg["model"]).eval()
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    other = build_model(cfg["model"]).eval()
    other.load_state_dict(checkpoint["model_state_dict"], strict=True)
    probe = torch.rand(1, 3, 32, 32)
    assert torch.equal(restored(probe), other(probe))
    bad = dict(checkpoint, variant="another_variant")
    with pytest.raises(ValueError, match="variant"):
        verify_checkpoint(bad, cfg)
    with pytest.raises(ValueError, match="Complete"):
        evaluate(["--run-dir", str(run)])
    before = (run/"status.json").read_bytes()
    wrong_variant = next(v for v in VARIANTS if v != variant)
    with pytest.raises(ValueError, match="variant"):
        train(["--resume", str(run/"checkpoint/last.pt"), "--variant", wrong_variant])
    assert before == (run/"status.json").read_bytes()
    train(["--resume", str(run/"checkpoint/last.pt"), "--variant", variant])
    assert json.loads((run/"status.json").read_text())["status"] == "completed"
    summary = evaluate(["--run-dir", str(run), "--variant", variant, "--checkpoint", "best_psnr", "--data-root", cfg["data"]["root"], "--gpu", "0"])
    assert summary["variant"] == variant and summary["smoke_test"] is True
    assert summary["sample_count"] == 1 and summary["split"] == "test"
    assert summary["checkpoint_selector"] == "best_psnr"
    for key in ("mean_psnr", "mean_ssim", "mean_e00"):
        assert torch.isfinite(torch.tensor(summary[key]))
    with (run/"result/test_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and set(rows[0]) == {"filename", "sample_id", "psnr", "ssim", "e00"}
    with pytest.raises(ValueError, match="variant"):
        evaluate(["--run-dir", str(run), "--variant", wrong_variant])
    # A second full tiny run must exactly match the interrupted/resumed trajectory.
    uninterrupted_cfg = load_yaml(source)
    uninterrupted_cfg["experiment"]["output_root"] = str(tmp_path/"continuous"/variant)
    continuous_source = tmp_path/(variant+"_continuous.yaml")
    save_yaml(continuous_source, uninterrupted_cfg)
    continuous = train(["--config", str(continuous_source), "--smoke-test"])
    a = _torch_load(run/"checkpoint/last.pt", torch.device("cpu"))
    b = _torch_load(continuous/"checkpoint/last.pt", torch.device("cpu"))
    for key in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][key], b["model_state_dict"][key]), key
    for index, values in a["optimizer_state_dict"]["state"].items():
        for key, value in values.items():
            assert torch.equal(value, b["optimizer_state_dict"]["state"][index][key])
    assert torch.equal(a["rng_state"]["torch"], b["rng_state"]["torch"])
    assert torch.equal(a["dataloader_generator_state"], b["dataloader_generator_state"])
    # Corrupt only a disposable synthetic snapshot; never touch real LSUI splits.
    path = run/"split_snapshot/test.tsv"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="SHA256"):
        evaluate(["--run-dir", str(run)])


@pytest.mark.parametrize("variant", VARIANTS)
def test_formal_cli_dry_run_has_exact_original_counts_and_no_run(tmp_path, variant, capsys):
    result = train(["--variant", variant, "--seed", "3520", "--gpu", "0", "--data-root", "/LOCAL_NOT_A_DATASET", "--output-dir", str(tmp_path/"never_created"), "--dry-run"])
    assert result is None and not (tmp_path/"never_created").exists()
    assert json.loads(capsys.readouterr().out)["sample_counts"] == dict(train=3466, validation=385, test=428)


def report_module():
    spec = importlib.util.spec_from_file_location("uicf_parameterization_report", PROJECT_ROOT/"tools/uicf_parameterization_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", VARIANTS)
def test_small_checkpoint_preserves_actual_forward(tmp_path, variant):
    from .test_models import small
    config = small(variant)
    original = build_model(config).eval()
    with torch.no_grad():
        original.uicf.field_mlp.net[-1].bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    image = torch.rand(1, 3, 32, 32)
    expected = original(image)
    path = tmp_path/(variant+".pt")
    torch.save({"variant": variant, "model_config": config, "state": original.state_dict()}, path)
    saved = _torch_load(path, torch.device("cpu"))
    assert saved["variant"] == variant
    restored = build_model(saved["model_config"]).eval()
    restored.load_state_dict(saved["state"], strict=True)
    assert torch.equal(restored(image), expected)


def test_report_preserves_missing_metrics_and_explicit_baseline_provenance(tmp_path):
    report = report_module()
    with pytest.raises(FileNotFoundError):
        report.main(["--baseline-run", str(tmp_path/"absent")])
    rows = report.main(["--baseline-run", str(tmp_path/"absent"), "--use-provided-baseline", "--output-dir", str(tmp_path/"report")])
    assert all(row["psnr"] is None and row["delta_e00"] is None for row in rows[:3])
    assert rows[3]["psnr"] == 28.388665 and "待 WSL 验证" in rows[3]["evidence"]
    source = prepare_smoke(tmp_path/"synthetic_run", "direct_residual")
    folder = tmp_path/"synthetic_result"
    folder.mkdir()
    (folder/"config_resolved.yaml").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="smoke"):
        report.read_result(folder, "direct_residual")


def test_report_independently_checks_per_image_evidence(tmp_path):
    report = report_module()
    path = tmp_path/"synthetic_unit_metrics.csv"
    path.write_text("filename,sample_id,psnr,ssim,e00\na.png,a,10,0.5,2\nb.png,b,20,0.7,4\n")
    summary = dict(sample_count=2, mean_psnr=15, mean_ssim=0.6, mean_e00=3)
    assert report.check_csv(path, summary, dict(a="a.png", b="b.png"))["e00"] == 3
    with pytest.raises(ValueError, match="mean mismatch"):
        report.check_csv(path, dict(summary, mean_psnr=999), dict(a="a.png", b="b.png"))
    with pytest.raises(ValueError, match="Incomplete"):
        report.check_csv(path, dict(summary, sample_count=3), dict(a="a.png", b="b.png"))


@pytest.mark.parametrize("variant", (*VARIANTS, "full_uicf"))
def test_metadata_reader_accepts_native_evidence_and_rejects_validation(tmp_path, variant):
    """Disposable metadata-only fixture; never publish its invented scores."""
    from src.shared.e00 import e00_protocol
    from src.uicf_parameterization_ablations.protocol import config_path
    report = report_module()
    name = Path(report.BASELINE_RUN).name if variant == "full_uicf" else variant
    run = tmp_path/"SYNTHETIC_METADATA_FIXTURE_NOT_A_TRAINING_RUN"/name
    for folder in ("split_snapshot", "result", "best"):
        (run/folder).mkdir(parents=True)
    cfg = load_yaml(PROJECT_ROOT/"configs/config_v16.yaml" if variant == "full_uicf" else config_path(variant))
    if variant == "full_uicf":
        # Legacy baseline configs need not contain the newer dataset/count fields.
        cfg["data"].pop("dataset")
        cfg["data"].pop("expected_counts")
    save_yaml(run/"config_resolved.yaml", cfg)
    info = dict(seed=3520, version=cfg["experiment"]["version"], variant=variant,
                train_sample_count=3466, validation_sample_count=385, test_sample_count=428)
    (run/"run_info.json").write_text(json.dumps(info))
    (run/"status.json").write_text(json.dumps(dict(status="completed", last_epoch=200)))
    for split in ("train", "validation", "test"):
        shutil.copyfile(PROJECT_ROOT/f"split/lsui19/{split}.tsv", run/f"split_snapshot/{split}.tsv")
    (run/"best/best_psnr.pt").write_bytes(b"SYNTHETIC UNIT FIXTURE: never load this sentinel")
    with (run/"split_snapshot/test.tsv").open() as f:
        entries = list(csv.reader(f, delimiter="\t"))
    with (run/"result/test_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("filename", "sample_id", "psnr", "ssim", "e00"))
        writer.writeheader()
        writer.writerows(dict(filename=Path(x[1]).name, sample_id=x[0], psnr=10, ssim=0.5, e00=2) for x in entries)
    summary = dict(sample_count=428, checkpoint=str(run/"best/best_psnr.pt"), checkpoint_epoch=1, mean_psnr=10,
                   mean_ssim=0.5, mean_e00=2, variant=variant, e00_protocol=e00_protocol(cfg["metrics"]),
                   evaluation=cfg["evaluation"], metrics_config=cfg["metrics"], metric_input="float_rgb_before_png_save")
    path = run/"result/test_summary.json"
    path.write_text(json.dumps(summary))
    assert report.read_result(run, variant)[0] == dict(psnr=10, ssim=0.5, e00=2)
    path.write_text(json.dumps(dict(summary, split="validation")))
    with pytest.raises(ValueError, match="Validation"):
        report.read_result(run, variant)


@pytest.mark.parametrize("module", ("train", "test", "smoke", "capacity"))
def test_module_help_is_executable(module):
    p = subprocess.run([sys.executable, "-m", "src.uicf_parameterization_ablations."+module, "--help"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "usage:" in p.stdout
