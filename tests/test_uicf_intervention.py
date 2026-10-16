from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.shared.uicf_inr import UICFINROutput
from src.v16.models import build_model
from src.v16.test import main as native_test_main
from src.v16.utils import sha256_file
import tools.analyze_uicf_intervention as intervention
from tools.uicf_intervention.artifacts import (
    audit_official_results, model_digest, safe_identity, strict_json, validate_output_path,
)
from tools.uicf_intervention.core import (
    MAIN_VARIANTS, VARIANTS, assert_field_invariants, assert_prediction_close,
    corrected_image, intervene_field, paired_differences, predict_variants,
    shift_pixels, summarize,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="module")
def small_cpu_thread_count():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def small_config() -> dict:
    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text(encoding="utf-8"))
    config["model"]["base_channels"] = 4
    config["model"]["color_query"].update(token_dim=16, num_heads=4)
    config["model"]["uicf"] = {
        "feat_dim": 4, "num_frequencies": 2, "mlp_hidden_dim": 8,
        "mlp_hidden_layers": 1, "anchor_hidden_dim": 4, "query_chunk_size": 31,
    }
    config["data"].update(num_workers=0, expected_counts={"train": 1, "validation": 1, "test": 2})
    config["evaluation"] = {"resize": True, "size": 16}
    config["logging"]["console"] = False
    config["test"]["visualization"].update(enabled=False, num_samples=1)
    return config


def nonzero_tiny_model(config: dict) -> torch.nn.Module:
    torch.manual_seed(3510)
    model = build_model(config["model"]).eval()
    # Only synthetic fixtures change weights. Test a nonidentity spatial field,
    # rather than letting all interventions trivially agree at zero initialization.
    with torch.no_grad():
        layer = model.uicf.field_mlp.net[-1]
        layer.weight.normal_(0, 0.08)
        layer.bias.copy_(torch.tensor([-0.35, 0.2, 0.5]))
    return model


@pytest.fixture
def fake_run(tmp_path: Path):
    run, data = tmp_path / "v16_run", tmp_path / "RGB_data"
    (run / "best").mkdir(parents=True)
    (run / "split_snapshot").mkdir()
    (run / "log").mkdir()
    for subdir in ("input", "gt"):
        (data / subdir).mkdir(parents=True)
    ids = {"train": ["train_id"], "validation": ["validation_id"], "test": ["test_a", "test_b"]}
    for offset, sample_id in enumerate(sum(ids.values(), [])):
        yy, xx = np.mgrid[:19, :23]
        base = 25 + offset * 15 + xx * 2 + yy
        rgb = np.stack((base, base + 13, base + 29), -1).astype(np.uint8)
        Image.fromarray(rgb).save(data / "input" / f"{sample_id}.png")
        Image.fromarray(rgb + 7).save(data / "gt" / f"{sample_id}.png")
    for split, members in ids.items():
        (run / "split_snapshot" / f"{split}.tsv").write_text(
            "".join(f"{member}\tinput/{member}.png\tgt/{member}.png\n" for member in members), encoding="utf-8",
        )
    config = small_config()
    config["data"]["root"] = str(data)
    (run / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    model = nonzero_tiny_model(config)
    checkpoint = {"version": "v16", "epoch": 5, "resolved_config": deepcopy(config),
                  "model_state_dict": model.state_dict()}
    torch.save(checkpoint, run / "best/best_psnr.pt")
    return run, data, config


def run_args(run: Path, *extra: str) -> list[str]:
    return ["--run-dir", str(run), "--num-workers", "0", "--bootstrap-samples", "40",
            "--viz-count", "1", *extra]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_pure_fields_mean_shift_channel_order_and_immutability(dtype):
    field = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).to(dtype) / 16 - 4
    original = field.clone()
    for variant in VARIANTS:
        result = intervene_field(field, variant)
        assert result.shape == field.shape and result.dtype == dtype and result.device == field.device
        assert result.data_ptr() != field.data_ptr()
        assert_field_invariants(field, result, variant)
        assert torch.equal(field, original)
    mean = intervene_field(field, "spatial_mean")
    torch.testing.assert_close(mean, field.mean((-2, -1), keepdim=True).expand_as(field), rtol=0, atol=0)
    assert not torch.equal(mean[:, 0], mean[:, 1])
    assert shift_pixels(5, 7) == (1, 1)
    assert torch.equal(intervene_field(field, "shift_xy"), torch.roll(field, (1, 1), (-2, -1)))
    assert torch.equal(intervene_field(field, "shift_y"), torch.roll(field, 1, -2))
    assert torch.equal(intervene_field(field, "shift_x"), torch.roll(field, 1, -1))
    for variant in ("shift_xy", "shift_y", "shift_x"):
        changed = intervene_field(field, variant)
        torch.testing.assert_close(field.double().mean((-2, -1)), changed.double().mean((-2, -1)), rtol=0, atol=0)
        torch.testing.assert_close(field.double().var((-2, -1)), changed.double().var((-2, -1)), rtol=1e-12, atol=1e-12)
    assert intervene_field(field, "shift_xy")[0, 0, 0, 0] == field[0, 0, -1, -1]
    for name, order in (("channel_cycle_1", [1, 2, 0]), ("channel_cycle_2", [2, 0, 1])):
        assert torch.equal(intervene_field(field, name), field[:, order])
    if torch.cuda.is_available():
        for variant in VARIANTS:
            assert torch.equal(intervene_field(field.cuda(), variant).cpu(), intervene_field(field, variant))


def test_shift_uses_actual_non_square_dimensions_and_nonzero_small_strategy():
    assert shift_pixels(20, 28) == (5, 7)
    assert shift_pixels(2, 3) == (1, 1)
    for height, width in ((1, 7), (7, 1), (0, 2)):
        with pytest.raises(ValueError, match="H and W"):
            shift_pixels(height, width)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_field_is_rejected_without_cleaning(bad):
    field = torch.zeros(1, 3, 5, 7)
    field[0, 0, 1, 1] = bad
    with pytest.raises(FloatingPointError, match="R: non-finite"):
        intervene_field(field, "spatial_mean")


def test_bad_shapes_and_invariant_corruption_fail():
    with pytest.raises(ValueError, match="B,3,H,W"):
        intervene_field(torch.zeros(1, 4, 5, 7), "original")
    with pytest.raises(ValueError, match="Unknown"):
        intervene_field(torch.zeros(1, 3, 5, 7), "bad")
    field = torch.arange(105).reshape(1, 3, 5, 7).float()
    with pytest.raises(RuntimeError, match="multisets"):
        assert_field_invariants(field, torch.zeros_like(field), "shift_xy")
    with pytest.raises(RuntimeError, match="channel order"):
        assert_field_invariants(field, field, "channel_cycle_1")


def test_reconstruction_changes_only_R_not_I_anchor_or_delta_and_does_not_clamp():
    inputs = torch.linspace(0.05, 0.95, 105).reshape(1, 3, 5, 7)
    field = torch.linspace(-3, 4, 105).reshape_as(inputs)
    anchor = torch.tensor([[0.2, 0.4, 0.8]])
    enhanced = inputs + field * (inputs - anchor[:, :, None, None])
    details = UICFINROutput(enhanced, field, anchor, torch.ones(1, 4))
    for variant in VARIANTS:
        changed = intervene_field(field, variant)
        corrected = corrected_image(inputs, details, changed)
        assert torch.equal(corrected, inputs + changed * (inputs - anchor[:, :, None, None]))
    shifted = corrected_image(inputs, details, intervene_field(field, "shift_xy"))
    assert not torch.allclose(shifted - inputs, torch.roll(enhanced - inputs, (1, 1), (-2, -1)))
    assert not torch.allclose(shifted, inputs + intervene_field(field, "shift_xy"))
    assert (shifted < 0).any() or (shifted > 1).any()
    assert torch.equal(corrected_image(inputs, details, torch.zeros_like(field)), inputs)
    assert torch.equal(anchor, torch.tensor([[0.2, 0.4, 0.8]]))


def test_real_model_original_path_single_extraction_and_frozen_state():
    model = nonzero_tiny_model(small_config())
    inputs = torch.rand(1, 3, 17, 23)
    before = model_digest(model)
    calls = []
    hook = model.uicf.register_forward_hook(lambda *args: calls.append(1))
    rng = torch.get_rng_state().clone()
    predictions, fields, corrected, details, checks = predict_variants(model, inputs, amp=False, check_normal=False)
    assert len(calls) == 1 and checks == {}
    assert torch.equal(rng, torch.get_rng_state())
    with torch.inference_mode():
        assert torch.equal(predictions["original"], model.backbone(corrected["original"]))
    assert torch.equal(fields["original"], details.correction_field)
    assert details.correction_field.std() > 0
    calls.clear()
    _, _, _, _, checks = predict_variants(model, inputs, amp=False, check_normal=True)
    assert len(calls) == 2  # One shared extraction plus independent full-forward audit.
    hook.remove()
    assert all(check["status"] == "PASS" for check in checks.values())
    assert model_digest(model) == before


def test_consistency_gate_catches_deliberate_wrong_forward(monkeypatch):
    model = nonzero_tiny_model(small_config())
    forward = model.forward
    monkeypatch.setattr(model, "forward", lambda image: forward(image) + 0.01)
    with pytest.raises(RuntimeError, match="forward equivalence FAILED"):
        predict_variants(model, torch.rand(1, 3, 16, 16), amp=False, check_normal=True)
    with pytest.raises(RuntimeError, match="equivalence FAILED"):
        assert_prediction_close(torch.zeros(1, 3, 5, 7), torch.ones(1, 3, 5, 7), amp=True, label="injected")


def test_all_variants_run_full_guidance_and_use_original_anchor():
    model = nonzero_tiny_model(small_config())
    counts = {stage: 0 for stage in (4, 3, 2, 1)}
    ic_inputs = []
    hooks = [model.backbone.register_forward_pre_hook(lambda module, args: ic_inputs.append(args[0].clone()))]
    for stage in counts:
        def count(module, args, result, current_stage=stage):
            counts[current_stage] += 1
        hooks.append(getattr(model.backbone, f"guide{stage}").register_forward_hook(count))
    image = torch.rand(1, 3, 17, 23)
    _, fields, corrected, details, _ = predict_variants(model, image, amp=False, check_normal=False)
    for hook in hooks:
        hook.remove()
    assert all(value == len(VARIANTS) for value in counts.values())
    for index, variant in enumerate(VARIANTS):
        assert torch.equal(ic_inputs[index], corrected[variant])
        assert torch.equal(corrected[variant], image + fields[variant] * (image - details.chromatic_anchor[:, :, None, None]))


def test_formula_gate_rejects_corrupted_details():
    model = nonzero_tiny_model(small_config())
    def corrupt(module, args, result):
        assert isinstance(result, UICFINROutput)
        return UICFINROutput(result.enhanced + .01, result.correction_field,
                             result.chromatic_anchor, result.global_feature)
    hook = model.uicf.register_forward_hook(corrupt)
    with pytest.raises(RuntimeError, match="reconstruction check failed"):
        predict_variants(model, torch.rand(1, 3, 16, 16), amp=False, check_normal=False)
    hook.remove()


def test_directed_gain_sign_image_unit_and_bootstrap_seed():
    gains = [2.0, -1.0, 0.0, 3.0]
    rows = []
    for index, gain in enumerate(gains):
        for variant in VARIANTS:
            value = 0 if variant == "original" else gain
            rows.append({"sample_id": str(index), "filename": f"{index}.png", "variant": variant,
                         "psnr": 30 - value, "ssim": .9 - value / 100, "e00": 5 + value})
    paired = paired_differences(rows)
    assert len(paired) == 4 * (len(VARIANTS) - 1)
    first = paired[0]
    assert first["gain_psnr"] == 2 and first["gain_ssim"] == pytest.approx(.02) and first["gain_e00"] == 2
    summary = summarize(rows, samples=100, seed=3520)
    assert summary == summarize(rows, samples=100, seed=3520)
    for variant in VARIANTS[1:]:
        stats = summary["paired_gains"][variant]["e00"]
        assert stats["mean_difference"] == 1
        assert stats["valid_count"] == 4 and stats["resampling_unit"] == "image"
        assert stats["win_fraction"] == .5 and stats["median"] == 1
        assert stats["std"] == pytest.approx(np.std(gains))
        assert stats["ci95_low"] <= stats["mean_difference"] <= stats["ci95_high"]
    with pytest.raises(ValueError, match="Duplicate"):
        paired_differences([*rows, rows[0]])


@pytest.mark.parametrize("dataset", ["LSUI19", "UIEB", "legacy_LSUI"])
def test_end_to_end_complete_protocol_fields_native_equivalence_and_no_old_writes(fake_run, dataset):
    run, data, config = fake_run
    if dataset != "LSUI19":
        if dataset == "legacy_LSUI":
            del config["data"]["dataset"]
        else:
            config["data"]["dataset"] = dataset
            checkpoint_path = run / "best/best_psnr.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            checkpoint["resolved_config"]["data"]["dataset"] = dataset
            torch.save(checkpoint, checkpoint_path)
        (run / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    native_test_main(["--run-dir", str(run)])
    protected = {path: sha256_file(path) for path in run.rglob("*") if path.is_file()}
    cpu_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    intervention.main(run_args(run, "--save-all-outputs", "--consistency-every", "1"))
    assert torch.equal(cpu_rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    assert all(sha256_file(path) == value for path, value in protected.items())
    output = run / "result/uicf_field_intervention"
    protocol = json.loads((output / "protocol.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert protocol["status"] == "PASS" and protocol["complete_formal_evaluation"]
    assert protocol["dataset"] == ("LSUI19" if dataset == "legacy_LSUI" else dataset)
    assert protocol["test_count"] == 2 and protocol["successful_sample_count"] == 2
    assert protocol["frozen_model_digest_before"] == protocol["frozen_model_digest_after"]
    assert protocol["protected_files_unchanged"]
    assert not protocol["amp_enabled"]  # CPU never uses AMP, even with training.amp=true.
    assert len(protocol["normal_forward_consistency"]["samples"]) == 2
    assert protocol["normal_forward_consistency"]["max_abs_error"] < 1e-6
    assert summary["paper_ready"] and summary["failed_sample_count"] == 0
    assert summary["official_test_audit"]["status"] == "NUMERIC_MATCH_LEGACY_METADATA_INCOMPLETE"
    for error in summary["official_test_audit"]["errors"].values():
        assert error["max_per_image_error"] == 0
    rows = read_csv(output / "per_sample_metrics.csv")
    assert len(rows) == 2 * len(VARIANTS)
    assert len(read_csv(output / "paired_differences.csv")) == 2 * (len(VARIANTS) - 1)
    assert len(read_csv(output / "diagnostics.csv")) == len(rows)
    assert len(read_csv(output / "paper_table.csv")) == 4
    assert (output / "paper_table.md").is_file()
    assert read_csv(output / "failed_samples.csv") == []
    for sample_id in ("test_a", "test_b"):
        enhanced = output / "all_outputs" / safe_identity(sample_id) / "original.png"
        native = run / "result/test_all_enhanced" / f"{sample_id}_enhanced.png"
        assert enhanced.read_bytes() == native.read_bytes()
    selection = json.loads((output / "visualizations/selection.json").read_text())
    assert selection["selection_before_inference"] and len(selection["samples"]) == 1
    panel = output / "visualizations" / selection["samples"][0]["directory"] / "comparison.png"
    with Image.open(panel) as image:
        assert image.width > image.height and image.info["dpi"][0] == pytest.approx(300, abs=.1)


def test_explicit_selection_keeps_full_evaluation_and_debug_has_no_paper_results(fake_run):
    run, _, _ = fake_run
    intervention.main(run_args(run, "--max-samples", "1", "--sample-ids", "test_a"))
    output = run / "result/uicf_field_intervention"
    assert not (output / "summary.json").exists()
    assert not (output / "paper_table.csv").exists()
    summary = json.loads((output / "debug_summary.json").read_text())
    assert summary["debug_partial"] and not summary["paper_ready"]
    assert summary["successful_sample_count"] == 1 and summary["full_test_sample_count"] == 2
    intervention.main(run_args(run, "--overwrite", "--sample-ids", "test_b"))
    assert json.loads((output / "summary.json").read_text())["successful_sample_count"] == 2


@pytest.mark.parametrize("mutation,match", [
    ("config_version", "only accepts"), ("checkpoint_version", "version is incompatible"),
    ("architecture", "architecture differs"), ("strict", "state_dict"),
    ("empty", "count is"), ("duplicate", "Duplicate"), ("bad_resize", "evaluation.size"),
    ("frozen_evaluation", "frozen evaluation"), ("path_escape", "escapes data.root"),
    ("checkpoint_dataset", "Checkpoint dataset identity"),
])
def test_preflight_boundaries_refuse_invalid_experiment(fake_run, mutation, match):
    run, _, config = fake_run
    checkpoint_path = run / "best/best_psnr.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if mutation == "config_version":
        config["experiment"]["version"] = "v16_19"
    elif mutation == "checkpoint_version":
        checkpoint["version"] = "v19"
    elif mutation == "architecture":
        checkpoint["resolved_config"]["model"]["base_channels"] = 8
    elif mutation == "strict":
        del checkpoint["model_state_dict"]["backbone.base_queries"]
    elif mutation == "empty":
        (run / "split_snapshot/test.tsv").write_text("", encoding="utf-8")
        match = "Manifest is empty"
    elif mutation == "duplicate":
        path = run / "split_snapshot/test.tsv"
        first = path.read_text().splitlines()[0]
        path.write_text(first + "\n" + first + "\n", encoding="utf-8")
    elif mutation == "bad_resize":
        config["evaluation"]["size"] = 0
    elif mutation == "frozen_evaluation":
        config["evaluation"]["size"] = 32
    elif mutation == "path_escape":
        (run / "split_snapshot/test.tsv").write_text(
            "test_a\t../outside/test_a.png\tgt/test_a.png\ntest_b\tinput/test_b.png\tgt/test_b.png\n")
    elif mutation == "checkpoint_dataset":
        checkpoint["resolved_config"]["data"]["dataset"] = "UIEB"
    (run / "config_resolved.yaml").write_text(yaml.safe_dump(config))
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises((ValueError, RuntimeError), match=match):
        intervention.main(run_args(run))
    assert not (run / "result/uicf_field_intervention").exists()


def test_failed_load_records_all_samples_and_leaves_no_formal_result(fake_run):
    run, data, _ = fake_run
    (data / "input/test_a.png").write_bytes(b"invalid PNG")
    with pytest.raises(RuntimeError, match="Incomplete experiment"):
        intervention.main(run_args(run))
    assert not (run / "result/uicf_field_intervention").exists()
    failed = list((run / "result").glob(".uicf_field_intervention.failed-*"))
    assert len(failed) == 1
    assert len(read_csv(failed[0] / "failed_samples.csv")) == 1
    assert read_csv(failed[0] / "failed_samples.csv")[0]["sample_id"] == "test_a"
    protocol = json.loads((failed[0] / "protocol.json").read_text())
    assert protocol["successful_sample_count"] == 1 and protocol["failed_sample_count"] == 1
    assert not (failed[0] / "paper_table.md").exists() and not (failed[0] / "summary.json").exists()


def test_overwrite_is_recoverable_and_failure_does_not_replace_success(fake_run, monkeypatch):
    run, _, _ = fake_run
    intervention.main(run_args(run))
    output = run / "result/uicf_field_intervention"
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        intervention.main(run_args(run))
    original_predict = intervention.predict_variants
    monkeypatch.setattr(intervention, "predict_variants", lambda *args, **kwargs: (_ for _ in ()).throw(
        FloatingPointError("variant=shift_xy injected nonfinite")))
    with pytest.raises(RuntimeError, match="Incomplete experiment"):
        intervention.main(run_args(run, "--overwrite"))
    assert all((output / path).read_bytes() == value for path, value in before.items())
    monkeypatch.setattr(intervention, "predict_variants", original_predict)
    intervention.main(run_args(run, "--overwrite"))
    archives = list(output.parent.glob(".uicf_field_intervention.previous-*"))
    assert len(archives) == 1
    assert all((archives[0] / path).read_bytes() == value for path, value in before.items())


def test_output_protection_symlinks_and_unmarked_custom_directory(fake_run, tmp_path):
    run, data, _ = fake_run
    for path in (run, run / "result", run / "best", data, ROOT / "src", tmp_path / "anything"):
        with pytest.raises(ValueError):
            validate_output_path(path, run_dir=run, data_root=data, overwrite=True)
    output = run / "result/uicf_field_intervention"
    output.mkdir(parents=True)
    marker = output / "do_not_delete.txt"
    marker.write_text("old")
    with pytest.raises(ValueError, match="unmarked"):
        intervention.main(run_args(run, "--overwrite"))
    assert marker.read_text() == "old"
    symlink = tmp_path / "uicf_field_intervention"
    symlink.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        validate_output_path(symlink, run_dir=run, data_root=data, overwrite=True)
    assert safe_identity("a/b") != safe_identity("a_b")


def test_official_modern_audit_passes_and_catches_wrong_metrics(fake_run):
    run, _, config = fake_run
    native_test_main(["--run-dir", str(run)])
    intervention.main(run_args(run))
    output = run / "result/uicf_field_intervention"
    protocol = json.loads((output / "protocol.json").read_text())
    rows = [{**row, **{metric: float(row[metric]) for metric in ("psnr", "ssim", "e00")}}
            for row in read_csv(output / "per_sample_metrics.csv")]
    summary_path = run / "result/test_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update({key: protocol[key] for key in (
        "checkpoint_sha256", "test_manifest_sha256", "amp_enabled", "evaluation", "metrics")})
    strict_json(summary_path, summary)
    audit = audit_official_results(run, rows, config, protocol, debug_partial=False)
    assert audit["status"] == "PASS"
    summary["mean_psnr"] += 1
    strict_json(summary_path, summary)
    with pytest.raises(RuntimeError, match="reproduction FAILED"):
        intervention.main(run_args(run, "--overwrite"))
    assert json.loads((output / "protocol.json").read_text())["status"] == "PASS"


def test_mismatched_official_protocol_is_explicit_not_false_pass(fake_run):
    run, _, config = fake_run
    native_test_main(["--run-dir", str(run)])
    intervention.main(run_args(run))
    output = run / "result/uicf_field_intervention"
    protocol = json.loads((output / "protocol.json").read_text())
    summary_path = run / "result/test_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["evaluation"] = {"resize": False, "size": 16}
    strict_json(summary_path, summary)
    audit = audit_official_results(run, [], config, protocol, debug_partial=False)
    assert audit["status"] == "NOT_COMPARABLE" and "Different evaluation" in audit["reasons"]


def test_strict_json_rejects_nan_and_no_partial_file(tmp_path):
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        strict_json(path, {"value": float("nan")})
    assert not path.exists()


def test_per_image_official_drift_is_detected_even_when_dataset_mean_is_unchanged(fake_run):
    run, _, config = fake_run
    native_test_main(["--run-dir", str(run)])
    intervention.main(run_args(run))
    output = run / "result/uicf_field_intervention"
    protocol = json.loads((output / "protocol.json").read_text())
    rows = [{**row, **{metric: float(row[metric]) for metric in ("psnr", "ssim", "e00")}}
            for row in read_csv(output / "per_sample_metrics.csv")]
    path = run / "result/test_metrics.csv"
    old = read_csv(path)
    old[0]["psnr"] = str(float(old[0]["psnr"]) + .1)
    old[1]["psnr"] = str(float(old[1]["psnr"]) - .1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(old[0]))
        writer.writeheader()
        writer.writerows(old)
    audit = audit_official_results(run, rows, config, protocol, debug_partial=False)
    assert audit["status"] == "FAIL"
    assert audit["errors"]["psnr"]["max_per_image_error"] == pytest.approx(.1)
    assert audit["errors"]["psnr"]["dataset_mean_error"] < 1e-12


def test_live_manifest_symlink_is_not_accepted_as_a_frozen_snapshot(fake_run, tmp_path):
    run, _, _ = fake_run
    path = run / "split_snapshot/test.tsv"
    target = tmp_path / "live_test.tsv"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="real TSV snapshots"):
        intervention.main(run_args(run))
