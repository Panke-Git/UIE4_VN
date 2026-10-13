from __future__ import annotations

import ast
from copy import deepcopy
import json

import pytest
import torch
from torch import nn
import yaml

from src.qgrd_ablations.shared_query import engine, evaluation, provenance, train
from src.v16.engine import validate as reference_validate
from .helpers import ROOT, config, module
from .test_runner_smoke import tiny_protocol


class OverflowSensitiveBlock(nn.Module):
    """Finite input/weight, half Conv output overflow before a normalizer.

    A component regression, not a reproduction of the user's trained weights.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 3, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        with torch.no_grad():
            self.conv.weight.fill_(0.175)
            self.bn.weight.zero_()

    def forward(self, inputs):
        return torch.sigmoid(self.bn(self.conv(inputs)))


def test_fp32_validation_handles_representability_overflow_and_preserves_metrics(tmp_path):
    cfg = config("shared_query")
    cfg["training"]["amp"] = True
    model = OverflowSensitiveBlock().eval()
    initial = deepcopy(model.state_dict())
    inputs = torch.full((2, 64, 8, 8), 1340.0)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        legacy = model(inputs)
    assert legacy.dtype == torch.float16
    assert not torch.isfinite(legacy).all()
    batches = [
        {"input": inputs, "target": torch.rand(2, 3, 8, 8),
         "id": ["a", "b"], "filename": ["a.png", "b.png"]},
        {"input": inputs[:1], "target": torch.rand(1, 3, 8, 8),
         "id": ["c"], "filename": ["c.png"]},
    ]
    run = tmp_path / "new"
    reference_run = tmp_path / "reference"
    for path in (run, reference_run):
        (path / "result").mkdir(parents=True)
    observed = []
    def record(_module, _args, output):
        observed.append((output.dtype, torch.is_autocast_enabled("cpu"), float(output.abs().max())))
    handle = model.conv.register_forward_hook(record)
    criterion = module("shared_query", "losses").build_loss(cfg["loss"])
    # A surrounding autocast must not leak into either forward or metric math.
    with torch.autocast("cpu", dtype=torch.float16):
        actual = engine.validate(model, batches, criterion, torch.device("cpu"), cfg, run, 1)
    handle.remove()
    expected = reference_validate(model, batches, criterion, torch.device("cpu"), cfg, reference_run, 1)
    assert actual == expected
    assert observed and all(dtype == torch.float32 and not enabled for dtype, enabled, _ in observed)
    assert all(maximum > torch.finfo(torch.float16).max for _, _, maximum in observed)
    assert (run / "result/validation_metrics.csv").read_bytes() == (
        reference_run / "result/validation_metrics.csv").read_bytes()
    summary = json.loads((run / "result/validation_summary.json").read_text())
    assert summary["sample_count"] == 3
    assert summary["evaluation_precision"] == "fp32"
    assert summary["evaluation_forward_protocol"] == evaluation.evaluation_protocol(cfg)
    for name in initial:
        assert torch.equal(initial[name], model.state_dict()[name]), name


def test_training_loop_remains_exactly_v16_amp_loop():
    def function_ast(path, name):
        tree = ast.parse(path.read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    assert function_ast(ROOT / "src/qgrd_ablations/shared_query/engine.py", "train_model") == (
        function_ast(ROOT / "src/v16/engine.py", "train_model"))
    for dataset, baseline in (("lsui", "configs/config_v16.yaml"), ("uieb", "configs/uieb/config_v16_uieb.yaml")):
        cfg = config("shared_query", dataset)
        old = yaml.safe_load((ROOT / baseline).read_text())
        for section in ("data", "loss", "optimizer", "scheduler", "training", "checkpoint", "metrics", "test", "logging"):
            assert cfg[section] == old[section], section
        assert cfg["evaluation"] == {**old["evaluation"], "precision": "fp32"}
        assert "FP32Eval" in cfg["experiment"]["name"]


@pytest.mark.parametrize("value", (None, "native_amp", "fp16", "bf16"))
def test_non_fp32_or_legacy_config_is_rejected(value):
    cfg = config("shared_query")
    if value is None:
        cfg["evaluation"].pop("precision")
    else:
        cfg["evaluation"]["precision"] = value
    with pytest.raises(ValueError, match="start a new run"):
        provenance.validate_config(cfg)


def test_fp32_dtype_assertion_does_not_silently_cast_bad_output():
    with pytest.raises(TypeError, match="prediction"):
        evaluation.require_fp32(torch.ones(1, dtype=torch.float16), "prediction")
    evaluation.require_fp32(torch.ones(1, dtype=torch.float32), "prediction")


def test_legacy_resume_is_refused_without_run_writes(tmp_path, monkeypatch):
    legacy = config("shared_query")
    legacy["evaluation"].pop("precision")
    legacy["experiment"]["name"] = "UICF_ColorQuery_SharedQuery"
    run = tmp_path / "old_run"
    (run / "checkpoint").mkdir(parents=True)
    (run / "config_resolved.yaml").write_text(yaml.safe_dump(legacy))
    (run / "status.json").write_text('{"status":"failed","last_epoch":123}')
    (run / "checkpoint/last.pt").write_bytes(b"must not even deserialize this legacy checkpoint")
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("legacy checkpoint must not be loaded or migrated")
    monkeypatch.setattr(train, "_torch_load", forbidden_load)
    with pytest.raises(ValueError, match="start a new run"):
        train.main(["--resume", str(run / "checkpoint/last.pt")])
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_fresh_run_and_test_record_fp32_and_leave_existing_run_untouched(tiny_protocol, tmp_path, monkeypatch):
    cfg, source, output = tiny_protocol("shared_query", "lsui")
    old = output / "old_amp_run"
    (old / "checkpoint").mkdir(parents=True)
    (old / "checkpoint/last.pt").write_bytes(b"old checkpoint remains untouched")
    (old / "status.json").write_text('{"last_epoch":123}')
    old_files = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    train.main(["--config", str(source)])
    run = next(p for p in output.iterdir() if p != old)
    assert "FP32Eval" in run.name
    checkpoint = train._torch_load(run / "checkpoint/last.pt", torch.device("cpu"))
    assert checkpoint["epoch"] == 2 and checkpoint["evaluation_precision"] == "fp32"
    assert checkpoint["resolved_config"]["training"]["amp"] is True
    info = json.loads((run / "run_info.json").read_text())
    assert info["evaluation_forward_protocol"] == evaluation.evaluation_protocol(cfg)
    assert not (run / "result/test_summary.json").exists()

    test_module = module("shared_query", "test")
    factory = test_module.build_model
    observed = []
    def spy_factory(config):
        model = factory(config)
        def pre(_module, args):
            observed.append((args[0].dtype, torch.is_autocast_enabled("cpu"), _module.training))
        model.register_forward_pre_hook(pre)
        return model
    monkeypatch.setattr(test_module, "build_model", spy_factory)
    with torch.autocast("cpu", dtype=torch.float16):
        test_module.main(["--run-dir", str(run), "--checkpoint", "best_psnr"])
    summary = json.loads((run / "result/test_summary.json").read_text())
    assert summary["evaluation_precision"] == "fp32"
    assert summary["evaluation_forward_protocol"] == evaluation.evaluation_protocol(cfg)
    assert summary["evaluation"]["precision"] == "fp32"
    assert observed and all(dtype == torch.float32 and not enabled and not training
                            for dtype, enabled, training in observed)
    assert old_files == {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}

    # Do not silently accept a relabeled checkpoint under the new protocol.
    sources = provenance.verify_source_hashes()
    checkpoint["evaluation_precision"] = "native_amp"
    with pytest.raises(ValueError, match="evaluation precision"):
        provenance.verify_checkpoint(checkpoint, cfg, sources)
