from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tools import diagnose_qgrd_numerics as diag
from .helpers import module
from .test_runner_smoke import tiny_protocol


def test_tensor_statistics_mixed_nonfinite_and_all_nonfinite():
    stats = diag.tensor_statistics("guide3.projected_q", torch.tensor([1., -3., float("nan"), float("inf"), -float("inf")]))
    assert stats["tensor_name"] == "guide3.projected_q"
    assert stats["shape"] == [5] and stats["dtype"] == "torch.float32" and stats["device"] == "cpu"
    assert stats["num_nan"] == stats["num_posinf"] == stats["num_neginf"] == 1
    assert stats["finite_count"] == 2
    assert stats["finite_min"] == -3 and stats["finite_max"] == 1 and stats["finite_abs_max"] == 3
    assert stats["finite_mean"] == -1 and stats["finite_std"] == 2
    all_bad = diag.tensor_statistics("tokens", torch.tensor([float("nan"), float("inf"), -float("inf")]))
    assert all_bad["finite_count"] == 0
    assert all(all_bad[key] is None for key in ("finite_min", "finite_max", "finite_abs_max", "finite_mean", "finite_std"))
    json.dumps(all_bad, allow_nan=False)


def test_chunk_statistics_noncontiguous_and_empty(monkeypatch):
    monkeypatch.setattr(diag, "CHUNK_ELEMENTS", 5)
    value = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).transpose(1, 2)
    assert not value.is_contiguous()
    parts = list(diag.tensor_chunks(value))
    assert all(part.numel() <= 5 for part in parts)
    assert sum(p.numel() for p in parts) == value.numel()
    actual = diag.tensor_statistics("feature", value)
    assert actual["finite_mean"] == float(value.double().mean())
    assert actual["finite_std"] == pytest.approx(float(value.double().std(unbiased=False)), abs=1e-12)
    empty = diag.tensor_statistics("empty", torch.empty(0, 3))
    assert empty["finite_count"] == 0 and diag.all_finite(torch.empty(0, 3))


def test_hooks_leave_outputs_gradients_rng_and_state_exact():
    torch.manual_seed(3520)
    plain = nn.Sequential(nn.Linear(5, 7), nn.LayerNorm(7), nn.GELU(), nn.Linear(7, 2))
    observed = deepcopy(plain)
    x = torch.randn(3, 5)
    rng = torch.get_rng_state()
    expected = plain(x)
    with diag.BoundaryTrace("train").attach(observed) as trace:
        trace.set_batch(1, ["a", "b", "c"])
        actual = observed(x)
        assert trace.first_nonfinite is None
        actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(expected, actual)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in plain.named_parameters():
        assert torch.equal(value.grad, dict(observed.named_parameters())[name].grad), name
    assert plain.state_dict().keys() == observed.state_dict().keys()
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in observed.modules())


class HalfCast(nn.Module):
    def forward(self, x):
        return x.to(torch.float16)


def test_first_boundary_distinguishes_finite_input_and_fp16_overflow():
    model = nn.Sequential(nn.Identity(), HalfCast(), nn.Identity())
    value = torch.tensor([[1., 70000., -70000.]])
    with diag.BoundaryTrace("validation_configured_amp").attach(model) as trace:
        trace.set_batch(1, ["sample"])
        output = model(value)
    event = trace.first_nonfinite
    assert event["tensor"]["tensor_name"] == "1.output"
    assert event["tensor"]["dtype"] == "torch.float16"
    assert event["tensor"]["num_posinf"] == event["tensor"]["num_neginf"] == 1
    assert event["inputs"][0]["dtype"] == "torch.float32"
    assert event["inputs"][0]["num_nan"] == event["inputs"][0]["num_posinf"] == 0
    assert event["sample_ids"] == ["sample"]
    assert not torch.isfinite(output).all()


def test_input_nan_reports_input_name_not_generic_final_output():
    model = nn.Linear(2, 2)
    with diag.BoundaryTrace("train").attach(model) as trace:
        trace.set_batch(4, ["broken"])
        model(torch.tensor([[float("nan"), float("inf")]]))
    assert trace.first_nonfinite["tensor"]["tensor_name"] == "<model>.input[0]"
    assert trace.first_nonfinite["step"] == 4


def test_model_summary_includes_bn_negative_variance_and_optimizer():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    optimizer = torch.optim.AdamW(model.parameters())
    with torch.no_grad():
        model[1].running_var[0] = -0.2
    summary = diag.model_summary(model, optimizer)
    assert summary["invalid"] is True
    assert summary["model_nonfinite_tensors"] == []
    assert summary["batch_norm"][0]["negative_running_variance"] is True
    assert summary["batch_norm"][0]["running_var"]["finite_min"] == pytest.approx(-0.2)
    assert summary["model_max_abs_top30"]


@pytest.fixture
def interrupted_a2(tiny_protocol, monkeypatch):
    cfg, source, output = tiny_protocol("shared_query", "lsui")
    train, engine = (module("shared_query", name) for name in ("train", "engine"))
    original = engine.update_status
    class Interrupted(Exception):
        pass
    def interrupt(run_dir, state, **updates):
        state = original(run_dir, state, **updates)
        if updates.get("last_epoch") == 1:
            raise Interrupted()
        return state
    monkeypatch.setattr(engine, "update_status", interrupt)
    with pytest.raises(Interrupted):
        train.main(["--config", str(source)])
    monkeypatch.setattr(engine, "update_status", original)
    return cfg, next(output.iterdir()), train


def test_replay_matches_native_resume_weights_optimizer_and_rng(interrupted_a2):
    cfg, run, train = interrupted_a2
    session = diag.load_session(run, "last", None, None, True, expected_epoch=1)
    replay = diag.replay_next_epoch(session, log_every=0)
    assert replay["completed"] and replay["epoch"] == 2 and replay["sample_count"] == 2
    assert replay["first_nonfinite_boundary"] is None
    assert replay["first_post_step_state_issue"] is None
    validation = diag.compare_validation(session)
    assert all(row["output_loss_finite"] for row in validation["modes"].values())
    model_state = deepcopy(session.model.state_dict())
    optim_state = deepcopy(session.optimizer.state_dict())
    rng = session.runtime.utils.rng_state()
    generator = session.generator.get_state().clone()
    train.main(["--resume", str(run / "checkpoint/last.pt")])
    actual = train._torch_load(run / "checkpoint/last.pt", torch.device("cpu"))
    for key in model_state:
        assert torch.equal(model_state[key], actual["model_state_dict"][key]), key
    for index, state in optim_state["state"].items():
        for key, value in state.items():
            expected = actual["optimizer_state_dict"]["state"][index][key]
            assert torch.equal(value, expected) if torch.is_tensor(value) else value == expected
    assert rng["python"] == actual["rng_state"]["python"]
    assert np.array_equal(rng["numpy"][1], actual["rng_state"]["numpy"][1])
    assert rng["numpy"][0] == actual["rng_state"]["numpy"][0]
    assert rng["numpy"][2:] == actual["rng_state"]["numpy"][2:]
    assert torch.equal(rng["torch"], actual["rng_state"]["torch"])
    assert torch.equal(generator, actual["dataloader_generator_state"])


def test_cli_replay_does_not_write_run_or_checkpoint(interrupted_a2, tmp_path):
    cfg, run, _ = interrupted_a2
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "report.json"
    code = diag.main(["--run-dir", str(run), "--checkpoint", "last", "--expect-checkpoint-epoch", "1",
                      "--replay-next-epoch", "--output-json", str(output), "--log-every", "0"])
    assert code == 0
    report = json.loads(output.read_text())
    assert report["checkpoint_epoch"] == 1 and report["replay"]["epoch"] == 2
    assert report["status"] == "NO_FAILURE_OBSERVED_CPU_OR_FP32_ONLY"
    assert report["original_run_unchanged"] and report["checkpoint_bytes_unchanged"]
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert set(report["validation"]["modes"]) == {"configured_amp", "fp32"}
    assert not any(m._forward_hooks for m in diag.load_session(run, "last", None, None, False).model.modules())


def test_cli_inspect_without_images_and_wrong_epoch_rejection(interrupted_a2, tmp_path):
    _, run, _ = interrupted_a2
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "inspect.json"
    assert diag.main(["--run-dir", str(run), "--data-root", str(tmp_path / "absent"),
                      "--output-json", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "CHECKPOINT_FINITE"
    assert diag.main(["--run-dir", str(run), "--expect-checkpoint-epoch", "123",
                      "--output-json", str(tmp_path / "error.json")]) == 1
    error = json.loads((tmp_path / "error.json").read_text())
    assert error["status"] == "ERROR" and "Expected checkpoint epoch 123" in error["exception_message"]
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_cli_injected_bn_problem_produces_report_without_original_writes(interrupted_a2, tmp_path, monkeypatch):
    _, run, _ = interrupted_a2
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    original = diag.replay_next_epoch
    def injected(session, log_every):
        result = original(session, log_every=0)
        for module in session.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_var.fill_(-1)
                break
        return result
    monkeypatch.setattr(diag, "replay_next_epoch", injected)
    path = tmp_path / "bn_report.json"
    code = diag.main(["--run-dir", str(run), "--replay-next-epoch", "--output-json", str(path)])
    assert code == 2
    report = json.loads(path.read_text())
    assert not report["checkpoint_state"]["invalid"]
    assert report["post_replay_state"]["invalid"]
    assert report["status"] == "POST_TRAIN_STATE_INVALID"
    assert report["validation"]["modes"]["fp32"]["first_nonfinite_boundary"]
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_validation_attempts_fp32_even_if_configured_mode_raises(interrupted_a2):
    _, run, _ = interrupted_a2
    session = diag.load_session(run, "last", None, None, True, expected_epoch=1)
    base_loss = session.criterion
    class FailOnce(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, prediction, target):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("test-only configured-mode runtime failure")
            return base_loss(prediction, target)
    session.criterion = FailOnce()
    result = diag.compare_validation(session)
    assert result["modes"]["configured_amp"]["first_runtime_error"]["exception_type"] == "RuntimeError"
    assert result["modes"]["fp32"]["output_loss_finite"]
    assert diag.classify({"checkpoint_state": {"invalid": False}, "validation": result}) == "VALIDATION_RUNTIME_ERROR"


@pytest.mark.parametrize("amp_finite,fp32_finite,enabled,expected", [
    (False, True, True, "AMP_ONLY_FORWARD_FAILURE"),
    (False, False, True, "FP32_FORWARD_FAILURE"),
    (True, False, True, "FP32_FORWARD_FAILURE"),
    (True, True, True, "NOT_REPRODUCED"),
    (True, True, False, "NO_FAILURE_OBSERVED_CPU_OR_FP32_ONLY"),
])
def test_precision_classifications(amp_finite, fp32_finite, enabled, expected):
    report = {"checkpoint_state": {"invalid": False}, "validation": {"modes": {
        "configured_amp": {"output_loss_finite": amp_finite, "autocast_enabled": enabled,
                           "first_nonfinite_boundary": None},
        "fp32": {"output_loss_finite": fp32_finite, "first_nonfinite_boundary": None}}}}
    assert diag.classify(report) == expected


def test_atomic_report_preserves_previous_report_on_serialization_failure(tmp_path):
    path = tmp_path / "report.json"
    diag.publish_json(path, {"status": "previous"})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        diag.publish_json(path, {"bad": float("nan")})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_output_inside_experiment_or_source_is_forbidden(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for path in (run / "result/report.json", diag.PROJECT_ROOT / "src/report.json",
                 diag.PROJECT_ROOT / "experiments/another/report.json"):
        with pytest.raises(ValueError, match="outside"):
            diag.resolve_output(path, run)


def test_scaler_metadata_rejects_nonfinite_or_zero_scale():
    diag.validate_scaler_metadata({"scale": 65536., "growth_factor": 2., "backoff_factor": 0.5})
    diag.validate_scaler_metadata({})
    for value in (float("nan"), float("inf"), 0., -1.):
        with pytest.raises(ValueError, match="scaler"):
            diag.validate_scaler_metadata({"scale": value})
