from __future__ import annotations
from copy import deepcopy
import csv
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import yaml
from .helpers import VARIANTS, config, module, small_config


class Interrupted(Exception):
    pass


@pytest.fixture
def tiny_protocol(tmp_path, monkeypatch):
    # Only test-local protocol constants/path resolution change; production
    # configuration and fixed manifest guards remain strict and are tested separately.
    root = tmp_path / "data"
    paths = {}
    counts = {"train": 2, "validation": 1, "test": 1}
    random = np.random.default_rng(3520)
    for split, count in counts.items():
        rows = []
        for i in range(count):
            sample = f"{split}_{i}"
            inp, gt = f"{split}/input/{sample}.png", f"{split}/GT/{sample}.png"
            for relative in (inp, gt):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
            array = random.integers(0, 240, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(array).save(root / inp)
            Image.fromarray(array + 10).save(root / gt)
            rows.append(f"{sample}\t{inp}\t{gt}\n")
        paths[split] = tmp_path / f"{split}.tsv"
        paths[split].write_text("".join(rows))
    hashes = {s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in paths.items()}

    def prepare(variant, dataset):
        cfg = config(variant, dataset)
        cfg["model"] = small_config(variant)["model"]
        cfg["data"].update(root=str(root), expected_counts=counts, batch_size=2,
                           num_workers=0, patch_size=32)
        cfg["evaluation"]["size"] = 32
        cfg["training"].update(epochs=2, save_every=1)
        cfg["test"]["visualization"].update(num_samples=1, cell_width=32, cell_height=32)
        cfg["logging"]["console"] = False
        output_root = str(tmp_path / variant / dataset / "runs")
        cfg["experiment"]["output_root"] = output_root
        guard = module(variant, "provenance")
        protocols = deepcopy(guard.PROTOCOLS)
        protocols[cfg["data"]["dataset"]].update(counts=counts, hashes=hashes)
        monkeypatch.setattr(guard, "PROTOCOLS", protocols)
        outputs = dict(guard.OUTPUT_ROOTS)
        outputs[cfg["data"]["dataset"]] = output_root
        monkeypatch.setattr(guard, "OUTPUT_ROOTS", outputs)
        train, test, experiment = (module(variant, n) for n in ("train", "test", "experiment"))
        monkeypatch.setattr(train, "manifest_path", lambda p: paths[Path(p).stem])
        monkeypatch.setattr(experiment, "split_paths", lambda cfg: paths)
        monkeypatch.setattr(train, "select_device", lambda gpu: torch.device("cpu"))
        monkeypatch.setattr(test, "select_device", lambda gpu: torch.device("cpu"))
        source = tmp_path / f"{variant}_{dataset}.yaml"
        source.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return cfg, source, Path(output_root)
    return prepare


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dataset", ("lsui", "uieb"))
def test_full_runner_interruption_resume_validation_and_test(tiny_protocol, monkeypatch, variant, dataset):
    cfg, source, output = tiny_protocol(variant, dataset)
    train, test, engine = (module(variant, n) for n in ("train", "test", "engine"))
    original_status = engine.update_status
    def interrupt_after_saved_epoch(run_dir, state, **updates):
        state = original_status(run_dir, state, **updates)
        if updates.get("last_epoch") == 1:
            raise Interrupted("simulate interruption after epoch 1 checkpoint and history")
        return state
    monkeypatch.setattr(engine, "update_status", interrupt_after_saved_epoch)
    with pytest.raises(Interrupted):
        train.main(["--config", str(source)])
    run = next(output.iterdir())
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
    assert not (run / "result/test_summary.json").exists()
    saved = train._torch_load(run / "checkpoint/last.pt", torch.device("cpu"))
    assert saved["epoch"] == 1
    assert saved["version"] == f"qgrd_{variant}"
    reloaded = module(variant, "models").build_model(cfg["model"])
    reloaded.load_state_dict(saved["model_state_dict"], strict=True)
    # Invalid resume identity must not alter any existing experiment file.
    wrong = deepcopy(saved)
    wrong["variant"] = "another_variant"
    bad = run / "checkpoint/bad.pt"
    torch.save(wrong, bad)
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="variant/version"):
        train.main(["--resume", str(bad)])
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}

    monkeypatch.setattr(engine, "update_status", original_status)
    train.main(["--resume", str(run / "checkpoint/last.pt")])
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    assert [x["epoch"] for x in json.loads((run / "log/metrics_history.json").read_text())["epochs"]] == [1, 2]
    assert not (run / "result/test_summary.json").exists()
    test.main(["--run-dir", str(run), "--checkpoint", "best_psnr"])
    summary = json.loads((run / "result/test_summary.json").read_text())
    val = json.loads((run / "result/validation_summary.json").read_text())
    assert summary["sample_count"] == val["sample_count"] == 1
    assert summary["checkpoint_selector"] == "best_psnr"
    assert summary["dataset"] == cfg["data"]["dataset"]
    assert summary["metric_input"] == "float_rgb_before_png_save"
    for key in ("mean_psnr", "mean_ssim", "mean_e00"):
        assert np.isfinite(summary[key]) and np.isfinite(val[key])
    with (run / "result/test_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == {"filename", "sample_id", "psnr", "ssim", "e00"}
    assert len(list((run / "result/test_all_enhanced").glob("*.png"))) == 1
    # Continuous versus interrupted/resumed training must recover exact weights,
    # optimizer and RNG, rather than merely be loadable.
    train.main(["--config", str(source)])
    continuous = next(p for p in output.iterdir() if p != run)
    a = train._torch_load(run / "checkpoint/last.pt", torch.device("cpu"))
    b = train._torch_load(continuous / "checkpoint/last.pt", torch.device("cpu"))
    for key in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][key], b["model_state_dict"][key]), key
    for state_id in a["optimizer_state_dict"]["state"]:
        for key, value in a["optimizer_state_dict"]["state"][state_id].items():
            other = b["optimizer_state_dict"]["state"][state_id][key]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    assert torch.equal(a["rng_state"]["torch"], b["rng_state"]["torch"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_modified_frozen_manifest_is_rejected(tiny_protocol, variant):
    cfg, source, output = tiny_protocol(variant, "lsui")
    train = module(variant, "train")
    train.main(["--config", str(source)])
    run = next(output.iterdir())
    manifest = run / "split_snapshot/test.tsv"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest SHA256"):
        module(variant, "test").main(["--run-dir", str(run)])
    assert not (run / "result/test_summary.json").exists()
