"""Temporary synthetic software tests, never substituted for DUO research results."""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.downstream_duo.inspect_duo import EXPECTED, audit_dataset
from src.downstream_duo.prepare import coco_to_yolo, prepare, split_train
from src.downstream_duo.model_inference_v16 import (enhance_tensor, enhance_test, load_v16,
                                                    pad_tensor, validate_enhanced)
from src.downstream_duo.smoke import synthetic_coco, synthetic_v16_run
from src.downstream_duo.utils import load_prepared, read_json, sha256, write_json


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    counts = synthetic_coco(root)
    return root, counts


def change(root, split, fn):
    path = root / "annotations" / f"instances_{split}.json"
    doc = read_json(path)
    fn(doc)
    write_json(path, doc)


def test_parse_mapping_paths_and_normalization(source):
    root, counts = source
    # JSON may include a root-relative or split-relative prefix.
    change(root, "train", lambda d: d["images"][0].update(file_name="images/train/frame_000.png"))
    data, mapping, report = audit_dataset(root, counts)
    assert report["valid"]
    assert mapping["coco_to_yolo"] == {"7": 0, "19": 1, "30": 2, "42": 3}
    assert mapping["yolo_to_coco"]["2"] == 30
    assert Path(data["train"]["images"][101]["source"]).name == "frame_000.png"
    assert list(map(float, coco_to_yolo([10, 20, 30, 40], 100, 200, 2).split())) == [2, .25, .2, .3, .2]
    assert EXPECTED == {"train": 6671, "test": 1111}
    # Numeric IDs and basenames overlap; differing RGB pixels are not leakage.
    assert report["overlapping_numeric_ids"]
    assert not report["cross_official_split_identical_rgb"]


@pytest.mark.parametrize("mutation", [
    lambda d: d["annotations"][0].update(bbox=[-1, 0, 10, 10]),
    lambda d: d["annotations"][0].update(bbox=[70, 0, 11, 10]),
    lambda d: d["annotations"][0].update(bbox=[0, 0, 0, 10]),
    lambda d: d["annotations"][0].update(bbox=[float("inf"), 0, 10, 10]),
    lambda d: d["annotations"][0].update(image_id=999),
    lambda d: d["annotations"][0].update(category_id=999),
    lambda d: d["annotations"][0].update(iscrowd=1),
    lambda d: d["annotations"].append(dict(d["annotations"][0])),
    lambda d: d["annotations"].append({**d["annotations"][0], "id": 999}),
])
def test_invalid_annotations_reported(source, mutation):
    root, counts = source
    # json.dumps default allows Inf for exercising strict finite validation.
    import json
    path = root / "annotations/instances_train.json"
    doc = read_json(path)
    mutation(doc)
    path.write_text(json.dumps(doc))
    _, _, report = audit_dataset(root, counts)
    assert not report["valid"]
    assert not report["conversion_allowed"]


def test_duplicate_ids_and_ambiguous_path(source):
    root, counts = source
    change(root, "train", lambda d: d["images"].append(dict(d["images"][0])))
    with pytest.raises(ValueError, match="duplicate image ID"):
        audit_dataset(root, counts)


def test_duplicate_categories(source):
    root, counts = source
    change(root, "train", lambda d: d["categories"].append(dict(d["categories"][0])))
    with pytest.raises(ValueError, match="category IDs"):
        audit_dataset(root, counts)


def test_missing_corrupt_ambiguous_images(source):
    root, counts = source
    image = root / "images/train/frame_000.png"
    (image.parent / "nested").mkdir()
    (image.parent / "nested" / image.name).write_bytes(image.read_bytes())
    (image.parent / "frame_001.png").write_bytes(b"not an image")
    (image.parent / "frame_002.png").unlink()
    _, _, report = audit_dataset(root, counts)
    assert not report["valid"]
    assert any("Duplicate basename" in e for e in report["errors"])
    assert any("Missing or ambiguous" in e for e in report["errors"])
    assert any("cannot identify image" in e for e in report["errors"])


def test_explicit_clipping_keeps_source_and_logs(source, tmp_path):
    root, counts = source
    change(root, "train", lambda d: d["annotations"][0].update(bbox=[70, 10, 11, 15]))
    original = sha256(root / "annotations/instances_train.json")
    _, _, strict = audit_dataset(root, counts)
    assert not strict["conversion_allowed"]
    meta = prepare(root, tmp_path / "clipped", expected=counts, synthetic=True, bbox_policy="clip")
    corrections = read_json(tmp_path / "clipped/bbox_corrections.json")
    assert corrections[0]["original_bbox"] == [70, 10, 11, 15]
    assert corrections[0]["derived_bbox"] == [70, 10, 10, 15]
    assert sha256(root / "annotations/instances_train.json") == original
    assert meta["bbox_policy"] == "clip"


def test_zero_intersection_explicit_drop_keeps_image_and_original(source, tmp_path):
    root, counts = source
    change(root, "train", lambda d: d["annotations"][0].update(bbox=[4, 64, 13, 1]))
    original = sha256(root / "annotations/instances_train.json")
    _, _, report = audit_dataset(root, counts, bbox_policy="clip")
    assert not report["conversion_allowed"]
    _, _, report = audit_dataset(root, counts, bbox_policy="clip", empty_after_clip="drop")
    assert report["conversion_allowed"]
    assert report["bbox_corrections"][0]["action"] == "drop_empty_after_clip"
    assert report["bbox_corrections"][0]["derived_bbox"] is None
    meta = prepare(root, tmp_path / "dropped", expected=counts, synthetic=True,
                   bbox_policy="clip", empty_after_clip="drop")
    assert meta["counts"]["train"] + meta["counts"]["val"] == counts["train"]
    assert sha256(root / "annotations/instances_train.json") == original


def test_deterministic_split_snapshots_empty_labels_and_readonly_preview(source, tmp_path):
    root, counts = source
    hashes = {str(p): sha256(p) for p in root.rglob("*") if p.is_file()}
    a = prepare(root, tmp_path / "a", expected=counts, synthetic=True)
    b = prepare(root, tmp_path / "b", expected=counts, synthetic=True)
    assert a["splits"] == b["splits"]
    assert a["dataset_digest"] == b["dataset_digest"]
    for split in ("train", "val", "test"):
        assert sha256(tmp_path / "a/manifests" / f"{split}.tsv") == sha256(tmp_path / "b/manifests" / f"{split}.tsv")
        assert sha256(tmp_path / "a/manifests" / f"{split}.json") == sha256(tmp_path / "b/manifests" / f"{split}.json")
    assert all(r["official_split"] == "train" for s in ("train", "val") for r in a["splits"][s])
    assert all(r["official_split"] == "test" for r in a["splits"]["test"])
    assert len(a["splits"]["test"]) == counts["test"]
    assert any(p.stat().st_size == 0 for p in (tmp_path / "a/labels").rglob("*.txt"))
    assert list((tmp_path / "a/gt_previews").glob("*.png"))
    assert all(sha256(path) == value for path, value in hashes.items())
    load_prepared(tmp_path / "a")
    with pytest.raises(FileExistsError):
        prepare(root, tmp_path / "a", expected=counts)


def test_grouping_only_train_and_exact_duplicates():
    images = {i: {"pixel_sha256": str(i)} for i in range(20)}
    images[1]["pixel_sha256"] = images[2]["pixel_sha256"]
    result = split_train(images, groups={2: "scene", 3: "scene"})
    assert len(result["val"]) > 0
    assert all((i in result["train"]) == (1 in result["train"]) for i in (2, 3))
    assert not set(result["train"]) & set(result["val"])
    assert sorted(result["train"] + result["val"]) == list(range(20))


def test_full_official_train_count_split_and_wrong_expected_rejected(source):
    images = {i: {"pixel_sha256": str(i)} for i in range(6671)}
    a, b = split_train(images), split_train(images)
    assert a == b and len(a["train"]) == 6004 and len(a["val"]) == 667
    assert set(a["train"]).isdisjoint(a["val"])
    root, _ = source
    _, _, report = audit_dataset(root)  # 8/3 synthetic cannot masquerade as 6671/1111.
    assert not report["conversion_allowed"]
    assert any("expected 6671" in error for error in report["errors"])


def test_prepared_tamper_detected(source, tmp_path):
    root, counts = source
    prepare(root, tmp_path / "prepared", expected=counts, synthetic=True)
    (tmp_path / "prepared/manifests/test.tsv").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        load_prepared(tmp_path / "prepared")


@pytest.fixture
def v16(tmp_path):
    torch.set_num_threads(2)
    run = tmp_path / "run"
    synthetic_v16_run(run)
    return run


def test_strict_v16_load_and_arbitrary_geometry(v16):
    model, provenance = load_v16(v16)
    assert provenance["selector"] == "best_psnr"
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for h, w in [(1, 1), (9, 13), (32, 48), (37, 51)]:
        output = enhance_tensor(model, torch.rand(1, 3, h, w))
        assert output.shape == (1, 3, h, w)
        assert output.isfinite().all() and output.min() >= 0 and output.max() <= 1
    tiled = enhance_tensor(model, torch.rand(1, 3, 37, 51), tile_size=32, overlap=8)
    assert tiled.shape == (1, 3, 37, 51) and tiled.isfinite().all()
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("mode", ["version", "architecture", "state", "haze"])
def test_checkpoint_rejections(v16, mode):
    path = v16 / "best/best_psnr.pt"
    checkpoint = torch.load(path, weights_only=False)
    if mode == "version":
        checkpoint["version"] = "haze4k"
    elif mode == "architecture":
        checkpoint["resolved_config"]["model"]["base_channels"] = 99
    elif mode == "state":
        checkpoint["model_state_dict"].pop(next(iter(checkpoint["model_state_dict"])))
    else:
        checkpoint["resolved_config"]["data"]["dataset"] = "Haze4K"
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_v16(v16)


def test_padding_tile_regression_and_invalid_output():
    identity = torch.nn.Identity().eval().requires_grad_(False)
    source = torch.rand(1, 3, 43, 67)
    assert pad_tensor(source).shape[-2:] == (48, 80)
    assert torch.equal(pad_tensor(source)[..., :43, :67], source)
    assert pad_tensor(torch.rand(1, 3, 1, 2)).shape[-2:] == (16, 16)
    assert torch.allclose(enhance_tensor(identity, source, tile_size=24, overlap=7), source, atol=3e-7)
    assert torch.equal(enhance_tensor(identity, source), source)
    with pytest.raises(ValueError):
        enhance_tensor(identity, source, tile_size=24, overlap=24)
    class Bad(torch.nn.Module):
        def forward(self, x):
            return x * float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        enhance_tensor(Bad().eval(), source)


def test_enhance_cardinality_gt_identity_failure_gate(source, tmp_path):
    root, counts = source
    prepared, out = tmp_path / "prepared", tmp_path / "full"
    meta = prepare(root, prepared, expected=counts, synthetic=True)
    identity = torch.nn.Identity().eval()
    enhance_test(prepared, out, identity, {"synthetic_untrained": True})
    validate_enhanced(prepared, out, meta)
    report = read_json(out / "enhancement_manifest.json")
    assert len(report["records"]) == counts["test"]
    for row in meta["splits"]["test"]:
        label = Path(row["file"]).with_suffix(".txt")
        assert (prepared / "labels/test_raw" / label).read_bytes() == (out / "labels/test_full" / label).read_bytes()
        assert (prepared / "images/test_raw" / row["file"]).read_bytes() == (out / "images/test_full" / row["file"]).read_bytes()
    report["complete"] = False
    write_json(out / "enhancement_manifest.json", report)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_enhanced(prepared, out, meta)
    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("simulated OOM")
    with pytest.raises(RuntimeError):
        enhance_test(prepared, tmp_path / "failure", Broken().eval(), {})
    failure = read_json(tmp_path / "failure/enhancement_manifest.json")
    assert not failure["complete"] and failure["records"][0]["status"] == "failed"


@pytest.mark.skipif(os.getenv("DUO_RUN_SMOKE") != "1", reason="Enable explicit one-epoch integration smoke")
def test_one_epoch_smoke_and_frozen_weight_guards(tmp_path):
    from src.downstream_duo.smoke import run_smoke
    from src.downstream_duo.evaluate import frozen_registry
    from src.downstream_duo.train_detector import train
    from src.downstream_duo.compare import validate_pair, compare
    from src.downstream_duo.utils import load_config
    out = run_smoke(tmp_path / "integration")
    raw = read_json(out / "eval_raw/metrics_raw.json")
    full = read_json(out / "eval_full/metrics_full.json")
    assert raw["synthetic"] and full["synthetic"]
    assert raw["weights"] == full["weights"] and raw["weights_sha256"] == full["weights_sha256"]
    validate_pair(raw, full)
    original = read_json(out / "comparison/comparison.json")
    repeated = compare(out / "eval_raw/metrics_raw.json", out / "eval_full/metrics_full.json", out / "comparison_repeat")
    assert original == repeated
    modified = deepcopy(full)
    modified["parameters"]["iou"] = 0.2
    with pytest.raises(ValueError, match="parameters"):
        validate_pair(raw, modified)
    other = out / "other_best.pt"
    other.write_bytes(Path(raw["weights"]).read_bytes())
    with pytest.raises(ValueError, match="path AND SHA256"):
        frozen_registry(out / "prepared", other, load_prepared(out / "prepared"))
    original_bytes = Path(raw["weights"]).read_bytes()
    try:
        with Path(raw["weights"]).open("ab") as stream:
            stream.write(b"tamper")
        with pytest.raises(ValueError, match="path AND SHA256"):
            frozen_registry(out / "prepared", raw["weights"], load_prepared(out / "prepared"))
    finally:
        Path(raw["weights"]).write_bytes(original_bytes)
    # No second training invocation, even when output directory differs.
    with pytest.raises(FileExistsError):
        train(out / "prepared", out / "second_training", load_config(), model_path="yolo11s.yaml", synthetic=True)
