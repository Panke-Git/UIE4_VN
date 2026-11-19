import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.downstream_duo.smoke import tiny_v16_config
from src.downstream_usod.config import V16_CHECKPOINTS, Paths, Protocol
from src.downstream_usod.dataset import manifest, read_input, scan_test
from src.downstream_usod.evaluation import evaluate_method
from src.downstream_usod.inference import ImageStage, enhance, saliency
from src.downstream_usod import pipeline
from src.downstream_usod.pipeline import claim, verify_output_root
from src.downstream_usod.tc_usod_adapter import TCUSOD
from src.downstream_usod.v16_adapter import V16Enhancer, v16_identity
from src.downstream_usod.visualization import make_figure
from src.v16.models.network import build_model

DIRS = {"rgb": "rgb", "gt": "masks", "depth": "depth"}


@pytest.fixture
def test_data(tmp_path):
    root = tmp_path / "test"
    ids = ("00002", "00012", "00022")
    for directory in DIRS.values():
        (root / directory).mkdir(parents=True)
    rng = np.random.default_rng(7)
    for identifier, (w, h) in zip(ids, [(49, 33), (49, 33), (39, 45)], strict=True):
        visible = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        height_map = rng.integers(0, 512, (h, w), dtype=np.uint16)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[3:h // 2, 3:w // 2] = 255
        for key, image in (("rgb", visible), ("depth", height_map), ("gt", mask)):
            Image.fromarray(image).save(root / DIRS[key] / f"{identifier}.png")
    return root, ids


def test_membership_geometry_and_relocation(test_data, tmp_path):
    root, ids = test_data
    expected = scan_test(root, DIRS, ids, official=False)
    assert [sample.sample_id for sample in expected] == list(ids)
    relocated = root.rename(tmp_path / "portable-test")
    assert manifest(scan_test(relocated, DIRS, ids, official=False), official=False) == manifest(
        expected, official=False)
    with pytest.raises(ValueError, match="author-published"):
        scan_test(relocated, DIRS, ids)
    with pytest.raises(ValueError, match="outside|relative path"):
        scan_test(relocated, {**DIRS, "rgb": "../rgb"}, ids, official=False)
    path = relocated / DIRS["depth"] / f"{ids[0]}.png"
    Image.new("L", (1, 2)).save(path)
    with pytest.raises(ValueError, match="changed after manifest freeze"):
        read_input(relocated, expected[0], "depth")
    with pytest.raises(ValueError, match="align at native size"):
        scan_test(relocated, DIRS, ids, official=False)
    path.unlink()
    with pytest.raises(ValueError, match="membership differs"):
        scan_test(relocated, DIRS, ids, official=False)


def test_duplicate_membership(test_data):
    root, ids = test_data
    Image.new("RGB", (49, 33)).save(root / "rgb" / f"{ids[0]}.jpg")
    with pytest.raises(ValueError, match="duplicate sample ID"):
        scan_test(root, DIRS, ids, official=False)


def test_path_template_and_frozen_v16_identity(tmp_path):
    config = tmp_path / "host.yml"
    config.write_text("usod10k:\n  test_root: assets/test\noutput_root: runs/new\n")
    paths = Paths.load(config)
    assert paths.test_root == tmp_path / "assets/test"
    assert paths.output == tmp_path / "runs/new"
    verify_output_root(paths.output, paths)
    with pytest.raises(ValueError, match="overlaps"):
        verify_output_root(paths.test_root / "results", paths)

    paths.output.mkdir(parents=True)
    original = paths.output / "old-result.txt"
    original.write_text("previous experiment")
    with pytest.raises(ValueError, match="already in use"):
        claim(paths.output)
    assert original.read_text() == "previous experiment"

    bogus = tmp_path / "synthetic.pt"
    bogus.write_bytes(b"not a trained V16")
    with pytest.raises(ValueError, match=V16_CHECKPOINTS["lsui"][0]):
        v16_identity(config, bogus, "lsui")
    assert Protocol.load(Path("configs/downstream_usod/tc_usod_usod10k.yaml")).saliency_batch == 1
    assert Paths.load(Path("configs/downstream_usod/paths.example.yaml")).tc_asset()[1]


class MockTC(torch.nn.Module):
    def forward(self, rgb, depth):
        prediction = torch.sigmoid((rgb[:, :1] + depth[:, :1]) / 4)
        return (prediction,) * 11


@pytest.mark.parametrize("tile_size,overlap", [(0, 0), (32, 8)])
def test_cpu_pipeline_geometry_and_resume(test_data, tmp_path, tile_size, overlap):
    root, ids = test_data
    samples = scan_test(root, DIRS, ids, official=False)
    destination = tmp_path / "synthetic-smoke"
    settings = {"synthetic": True, "tile_size": tile_size}
    model = build_model(tiny_v16_config()["model"]).eval().requires_grad_(False)
    v16 = V16Enhancer(model, "cpu", tile_size, overlap)
    enhanced = ImageStage(destination, "enhanced", "icar_uie", samples, settings, "RGB")
    tc = TCUSOD(MockTC().eval(), "cpu")
    raw = ImageStage(destination, "predictions", "raw", samples, settings, "L")
    candidate = ImageStage(destination, "predictions", "icar_uie", samples, settings, "L")

    def interrupted(images):
        if images[0].size == samples[-1].size:
            raise RuntimeError("deliberate interruption for fixture")
        return v16(images)

    with pytest.raises(RuntimeError, match="1 failed"):
        enhance(enhanced, interrupted, root, batch_size=2)
    first = enhanced.image_path(samples[0])
    previous = first.read_bytes(), first.stat().st_mtime_ns
    assert "deliberate interruption" in enhanced.log.read_text()
    assert not (enhanced.audit / "complete.json").exists()
    assert enhance(enhanced, v16, root, batch_size=2)["count"] == 3
    assert (first.read_bytes(), first.stat().st_mtime_ns) == previous

    # Both inference stages operate using RGB and depth even when the GT is unavailable.
    original_gt = root / DIRS["gt"]
    hidden_gt = original_gt.rename(root / "held-out")
    saliency(raw, tc, root, batch_size=2)
    saliency(candidate, tc, root, batch_size=2, enhanced=enhanced)
    hidden_gt.rename(original_gt)

    for sample in samples:
        for stage in (raw, candidate):
            receipt = json.loads(stage.receipt_path(sample).read_text())
            assert receipt["inputs"]["depth"] == sample.hashes["depth"]
            assert stage.load(sample).size == sample.size
            assert "gt" not in receipt["inputs"]

    predictors = {"raw": raw, "icar_uie": candidate}
    measures = {key: evaluate_method(samples, root, stage.load) for key, stage in predictors.items()}
    for key, rows in measures.items():
        assert rows.ids == ids and np.isfinite(rows.values).all()
        rows.save(destination / "metrics" / key, {"synthetic": True})
    vis = destination / "visualizations"
    selection = make_figure(samples, root, measures, predictors, vis, 1)
    assert {row["sample_id"] for row in selection} == set(ids)
    assert (vis / "saliency_examples.png").is_file()

    first.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        enhanced.complete()


def test_phase_orchestration_uses_smoke_and_preserves_primary_results(test_data, tmp_path, monkeypatch):
    data, ids = test_data
    samples = scan_test(data, DIRS, ids, official=False)
    destination = tmp_path / "phases"
    destination.mkdir()
    claim(destination)
    spec = tmp_path / "paths.yml"
    fake_v16 = {"config": "fixture.yaml", "checkpoint": "fixture.pt"}
    spec.write_text(yaml.safe_dump({
        "usod10k": {"test_root": str(data)}, "device": "cpu",
        "tc_usod": {"repository": ".", "checkpoint": "fixture.pth"},
        "v16": {"lsui": fake_v16, "uieb": fake_v16},
    }))
    monkeypatch.setattr(pipeline, "verify_model_source", lambda repo: None)
    monkeypatch.setattr(pipeline, "verify_hash", lambda path, checksum: checksum)
    monkeypatch.setattr(pipeline, "environment", lambda *args, **kwargs: {"device": "cpu"})
    monkeypatch.setattr(pipeline, "prepare", lambda paths, root: (samples, "synthetic-test"))
    monkeypatch.setattr(pipeline, "v16_identity", lambda config, weight, source: {"source": source})
    monkeypatch.setattr(pipeline, "load_v16", lambda *args: torch.nn.Identity().eval())
    monkeypatch.setattr(pipeline, "load_tc", lambda *args: MockTC().eval())
    runtime = pipeline.Experiment(Paths.load(spec), destination, Protocol(0, 0, 2, 2, 12, 0, 1, 2),
                                  "synthetic-fixture")
    with pytest.raises(FileNotFoundError, match="p0 smoke"):
        runtime.run("p0")
    assert runtime.smoke()["status"] == "passed"
    assert not list((destination / "predictions").glob("*"))
    assert runtime.run("p0")["status"] == "complete"
    original_report = (destination / "reports/p0_summary.json").read_bytes()

    extension = runtime.run("p1", ("waternet",))
    assert extension["status"] == "partial"
    assert extension["fx"]["waternet"]["status"] == "unavailable"
    assert [row["method"] for row in extension["metrics"]] == ["raw", "icar_uie"]
    with pytest.raises(FileNotFoundError, match="p2 smoke"):
        runtime.run("p2")
    runtime.smoke("p2")
    assert runtime.run("p2")["metrics"][-1]["method"] == "icar_uie_uieb"
    assert (destination / "reports/p0_summary.json").read_bytes() == original_report
