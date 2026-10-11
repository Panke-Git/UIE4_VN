"""Isolation, model equivalence, frozen membership, and train/resume/Val integration."""
from __future__ import annotations

import ast
import copy
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.v16.models import build_model as legacy_model
from src.v16_19 import protocol, train, utils
from src.v16_19.dataset import validate_split_protocol
from src.v16_19.models import build_model
from src.v16_19.validate import export_validation

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(key="lsui19"):
    config = yaml.safe_load((PROJECT / f"configs/v16_19/{key}.yaml").read_text())
    model = config["model"]
    model["base_channels"] = 4
    model["color_query"].update(token_dim=8, num_heads=2, num_color_queries=2)
    model["uicf"].update(feat_dim=4, num_frequencies=2, mlp_hidden_dim=8, mlp_hidden_layers=1,
                         anchor_hidden_dim=4, query_chunk_size=512)
    config["data"].update(patch_size=32, batch_size=2, num_workers=0)
    config["evaluation"]["size"] = 32
    config["training"].update(epochs=1, deterministic=True, amp=False)
    config["validation"]["visualization"].update(num_samples=2, cell_width=32, cell_height=32)
    config["logging"]["console"] = False
    return config


def _data(root):
    random = np.random.default_rng(2026)
    for folder, count in [("Train", 4), ("Val", 2)]:
        for role in ("input", "GT"):
            (root / folder / role).mkdir(parents=True)
        for i in range(count):
            array = random.integers(0, 256, size=(35+i, 39+i, 3), dtype=np.uint8)
            name = f"{folder.lower()}_{i}.png"
            Image.fromarray(array).save(root / folder / "input" / name)
            Image.fromarray(np.clip(array.astype(np.int16)+9, 0, 255).astype(np.uint8)).save(root / folder / "GT" / name)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(protocol.COUNTS, "lsui19", {"train": 4, "validation": 2})
    monkeypatch.setitem(protocol.COUNTS, "uieb19", {"train": 4, "validation": 2})
    monkeypatch.setattr(train, "select_device", lambda gpu: torch.device("cpu"))
    import src.v16_19.validate as validation
    monkeypatch.setattr(validation, "select_device", lambda gpu: torch.device("cpu"))
    root = tmp_path / "data"
    _data(root)
    def make(key="lsui19", name="smoke", epochs=1):
        protocol_dir = tmp_path / "split/v16_19" / key
        protocol.freeze_dataset(root, key, protocol_dir)
        config = _config(key)
        config["experiment"]["name"] = name
        config["data"].update(root=str(root), expected_counts={"train": 4, "validation": 2})
        config["training"]["epochs"] = epochs
        path = tmp_path / f"{key}_{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path, config
    return tmp_path, root, make


def test_full_default_model_and_recipe_are_preserved():
    old = yaml.safe_load((PROJECT / "configs/config_v16.yaml").read_text())
    for key in ("lsui19", "uieb19"):
        new = yaml.safe_load((PROJECT / f"configs/v16_19/{key}.yaml").read_text())
        for section in ("model", "loss", "optimizer", "scheduler", "training", "checkpoint", "evaluation", "metrics"):
            assert new[section] == old[section]
        utils.validate_config(new)
        assert "test" not in new and "test_manifest" not in new["data"]
    torch.manual_seed(123)
    before = legacy_model(old["model"])
    torch.manual_seed(123)
    after = build_model(old["model"])
    assert before.state_dict().keys() == after.state_dict().keys()
    for name, tensor in before.state_dict().items():
        assert torch.equal(tensor, after.state_dict()[name]), name


def test_forward_details_and_gradients_equal_legacy_with_nonzero_uicf():
    config = _config()["model"]
    torch.manual_seed(73)
    before = legacy_model(config)
    torch.manual_seed(73)
    after = build_model(config)
    with torch.no_grad():
        for name, parameter in before.named_parameters():
            if name.startswith("uicf.") and torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.02)
    after.load_state_dict(before.state_dict(), strict=True)
    before.eval(); after.eval()
    a = torch.rand(2, 3, 33, 39, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    old_output, old_details = before.forward_with_uicf_details(a)
    new_output, new_details = after.forward_with_uicf_details(b)
    assert old_details.correction_field.abs().max() > 0
    torch.testing.assert_close(old_output, new_output, rtol=0, atol=0)
    torch.testing.assert_close(old_details.correction_field, new_details.correction_field, rtol=0, atol=0)
    old_output.square().mean().backward(); new_output.square().mean().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    for (name, p), (new_name, q) in zip(before.named_parameters(), after.named_parameters()):
        assert name == new_name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_no_legacy_runtime_imports():
    for path in (PROJECT / "src/v16_19").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert not node.module.startswith("src."), (path, node.module)
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith("src.") for alias in node.names), path
    assert not (PROJECT / "src/v16_19/test.py").exists()


def test_protocol_preserves_duplicates_and_detects_mutations(experiment):
    tmp, root, make = experiment
    # Different ids, same input: report content leakage without repartitioning.
    source = root / "Train/input/train_0.png"
    target = root / "Val/input/val_0.png"
    target.write_bytes(source.read_bytes())
    (root / "Val/GT/val_0.png").write_bytes((root / "Train/GT/train_0.png").read_bytes())
    _, config = make()
    directory = tmp / config["data"]["protocol_dir"]
    quality = json.loads((directory / "data_quality.json").read_text())
    assert quality["cross_split_duplicate_groups"] == 1
    assert protocol.verify_dataset(root, directory)["counts"] == {"train": 4, "validation": 2}
    assert not (directory / "test.tsv").exists()
    original = source.read_bytes()
    source.write_bytes(original+b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        protocol.freeze_dataset(root, "lsui19", directory)
    source.write_bytes(original)
    (directory / "train.tsv").write_text("corrupted")
    with pytest.raises(ValueError, match="Frozen protocol file changed"):
        protocol.verify_dataset(root, directory)


def test_pair_name_and_two_split_guards(experiment):
    tmp, root, make = experiment
    _, config = make()
    directory = tmp / config["data"]["protocol_dir"]
    manifests = {k: directory / f"{k}.tsv" for k in ("train", "validation")}
    assert len(validate_split_protocol(manifests)["validation"]) == 2
    with pytest.raises(ValueError):
        validate_split_protocol({"train": manifests["train"], "validation": manifests["validation"], "test": manifests["validation"]})
    (root / "Val/GT/val_0.png").rename(root / "Val/GT/misnamed.png")
    with pytest.raises(ValueError, match="Unpaired"):
        protocol.collect_dataset(root, "lsui19")


@pytest.mark.parametrize("key", ["lsui19", "uieb19"])
def test_train_and_complete_best_checkpoint_val_export(experiment, key):
    tmp, root, make = experiment
    path, config = make(key)
    train.main(["--config", str(path)])
    run = next((tmp / config["experiment"]["output_root"]).iterdir())
    status = json.loads((run / "status.json").read_text())
    summary = status["final_validation"]
    assert status["status"] == "completed" and summary["split"] == "validation"
    assert summary["sample_count"] == 2 and summary["checkpoint_selector"] == "best_psnr"
    assert summary["checkpoint_sha256"] == protocol.file_sha256(run / "best/best_psnr.pt")
    assert not any(p.name.startswith("test") for p in run.rglob("*"))
    result = run / "result/val/best_psnr"
    assert len(list((result / "enhanced").glob("*.png"))) == 2
    assert len(list((result / "samples").glob("*.png"))) == 6
    assert (result / "val_grid_2x3.png").exists() and (result / "training_curves.png").exists()
    with (result / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["sample_id"] for r in rows} == {"val_0", "val_1"}
    assert summary["mean_psnr"] == pytest.approx(sum(float(r["psnr"]) for r in rows)/2)
    best = json.loads((run / "best/best_psnr.json").read_text())
    assert summary["mean_psnr"] == pytest.approx(best["psnr"], abs=1e-6)
    assert summary["mean_ssim"] == pytest.approx(best["ssim"], abs=1e-6)
    assert summary["mean_e00"] == pytest.approx(best["e00"], abs=1e-6)
    # Evaluation uses the run snapshot even if repository-side manifests change.
    (tmp / config["data"]["protocol_dir"] / "validation.tsv").write_text("changed after run")
    second = export_validation(run, "best_ssim")
    assert second["sample_count"] == 2 and (run / "result/val/best_psnr/summary.json").exists()
    # A bad export leaves the prior successful report intact.
    checkpoint = torch.load(run / "checkpoint/last.pt", weights_only=False)
    checkpoint["version"] = "v16"
    torch.save(checkpoint, run / "checkpoint/legacy.pt")
    with pytest.raises(ValueError, match="version or dataset"):
        export_validation(run, "checkpoint/legacy.pt")
    assert (result / "summary.json").exists()


def test_native_resolution_export_and_data_relocation(experiment):
    tmp, root, make = experiment
    path, config = make()
    config["evaluation"]["resize"] = False
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    train.main(["--config", str(path)])
    run = next((tmp / config["experiment"]["output_root"]).iterdir())
    result = run / "result/val/best_psnr/enhanced"
    for i in range(2):
        with Image.open(result / f"val_{i}_enhanced.png") as output, Image.open(root / f"Val/input/val_{i}.png") as original:
            assert output.size == original.size
    import shutil
    moved = tmp / "relocated"
    shutil.copytree(root, moved)
    summary = export_validation(run, data_root_override=str(moved))
    assert summary["sample_count"] == 2
    old_report = (run / "result/val/best_psnr/summary.json").read_bytes()
    image = moved / "Val/input/val_0.png"
    image.write_bytes(image.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        export_validation(run, data_root_override=str(moved))
    assert (run / "result/val/best_psnr/summary.json").read_bytes() == old_report


def test_resume_reproduces_uninterrupted_training(experiment):
    tmp, root, make = experiment
    full_path, full_config = make(name="full", epochs=2)
    train.main(["--config", str(full_path)])
    full = next((tmp / full_config["experiment"]["output_root"]).glob("*full*"))
    part_path, _ = make(name="part", epochs=1)
    train.main(["--config", str(part_path)])
    part = next((tmp / full_config["experiment"]["output_root"]).glob("*part*"))
    train.main(["--resume", str(part / "checkpoint/last.pt"), "--epochs", "2"])
    a = torch.load(full / "checkpoint/last.pt", weights_only=False)
    b = torch.load(part / "checkpoint/last.pt", weights_only=False)
    assert a["epoch"] == b["epoch"] == 2
    for name, value in a["model_state_dict"].items():
        torch.testing.assert_close(value, b["model_state_dict"][name], rtol=0, atol=0)
    history = json.loads((part / "log/metrics_history.json").read_text())["epochs"]
    assert [r["epoch"] for r in history] == [1, 2]
    # Re-running a completed resume only re-exports Val; it adds no epoch.
    train.main(["--resume", str(part / "checkpoint/last.pt")])
    assert len(json.loads((part / "log/metrics_history.json").read_text())["epochs"]) == 2


def test_legacy_configs_and_output_roots_rejected():
    config = _config()
    bad = copy.deepcopy(config); bad["experiment"]["version"] = "v16"
    with pytest.raises(ValueError, match="legacy"):
        utils.validate_config(bad)
    bad = copy.deepcopy(config); bad["experiment"]["output_root"] = "experiments"
    with pytest.raises(ValueError, match="old results"):
        utils.validate_config(bad)
    bad = copy.deepcopy(config); bad["data"]["test_manifest"] = "split/lsui19/test.tsv"
    with pytest.raises(ValueError, match="test split"):
        utils.validate_config(bad)


def test_checkpoint_recipe_guard_allows_only_location_and_epoch_budget():
    config = _config()
    checkpoint = {"seed": config["experiment"]["seed"], "resolved_config": copy.deepcopy(config)}
    relocated = copy.deepcopy(config)
    relocated["data"]["root"] = "/different/server/path"
    relocated["training"]["epochs"] = 300
    utils.validate_checkpoint_config(checkpoint, relocated)
    relocated["optimizer"]["learning_rate"] = 0.001
    with pytest.raises(ValueError, match="run recipe"):
        utils.validate_checkpoint_config(checkpoint, relocated)


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    from src.v16_19.engine import _save_checkpoint
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous successful checkpoint")
    def fail(payload, target):
        target.write_bytes(b"partial write")
        raise OSError("simulated disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        _save_checkpoint({}, path)
    assert path.read_bytes() == b"previous successful checkpoint"
    assert not path.with_suffix(".pt.tmp").exists()
