from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.shared.uicf_inr import UICFINROutput
from src.shared.uicf_models import UICFPreBackbone
from src.v16.dataset import ManifestEntry
from src.v16.models import build_model
from tools.visualize_v16_uicf import (
    CapturedSample,
    assert_uicf_consistency,
    build_and_load_v16_model,
    build_metadata,
    main as visualization_main,
    save_visualization_sample,
    select_sample_indices,
    split_correction_channels,
    symmetric_heatmap_range,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_model_config() -> dict:
    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text(encoding="utf-8"))
    model_config = deepcopy(config["model"])
    model_config.update({"base_channels": 4})
    model_config["color_query"] = {
        **model_config["color_query"],
        "token_dim": 16,
        "num_heads": 4,
    }
    model_config["uicf"] = {
        "feat_dim": 4,
        "num_frequencies": 2,
        "mlp_hidden_dim": 8,
        "mlp_hidden_layers": 1,
        "anchor_hidden_dim": 4,
        "query_chunk_size": 31,
    }
    return model_config


def _sample() -> CapturedSample:
    inputs = torch.linspace(0.0, 1.0, 3 * 8 * 10).reshape(3, 8, 10)
    field = torch.stack(
        (
            torch.linspace(-2.0, 1.0, 80).reshape(8, 10),
            torch.linspace(-1.0, 3.0, 80).reshape(8, 10),
            torch.linspace(-4.0, 2.0, 80).reshape(8, 10),
        )
    )
    anchor = torch.tensor([0.2, 0.4, 0.6])
    corrected = inputs + field * (inputs - anchor[:, None, None])
    return CapturedSample(
        index=7,
        sample_id="sample/007",
        filename="007.png",
        input_tensor=inputs,
        target_tensor=inputs.flip(0),
        prediction=corrected.sigmoid(),
        enhanced=corrected,
        correction_field=field,
        chromatic_anchor=anchor,
        global_feature=torch.linspace(-1.0, 1.0, 6),
    )


def test_correction_field_channels_are_split_without_value_changes() -> None:
    field = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    channels = split_correction_channels(field)
    assert len(channels) == 3
    for channel_index, channel in enumerate(channels):
        np.testing.assert_array_equal(channel, field[channel_index])
    with pytest.raises(ValueError, match=r"\[3,H,W\]"):
        split_correction_channels(np.zeros((1, 3, 4, 5), dtype=np.float32))


def test_symmetric_range_is_global_zero_centered_and_optionally_robust() -> None:
    first = np.array([[[-4.0, 1.0]], [[2.0, 3.0]], [[-1.0, 0.0]]], dtype=np.float32)
    second = np.array([[[6.0, -2.0]], [[0.5, 1.5]], [[-5.0, 2.0]]], dtype=np.float32)
    assert symmetric_heatmap_range([first, second]) == (-6.0, 6.0)
    vmin, vmax = symmetric_heatmap_range([first, second], robust_percentile=75.0)
    expected = float(np.percentile(np.abs(np.concatenate((first.ravel(), second.ravel()))), 75.0))
    assert vmin == -expected and vmax == expected
    assert vmin == -vmax


def test_ic_reconstruction_and_normal_forward_equivalence_checks() -> None:
    inputs = torch.rand(2, 3, 5, 7)
    field = torch.randn_like(inputs) * 0.2
    anchor = torch.rand(2, 3)
    enhanced = inputs + field * (inputs - anchor[:, :, None, None])
    details = UICFINROutput(
        enhanced=enhanced,
        correction_field=field,
        chromatic_anchor=anchor,
        global_feature=torch.rand(2, 9),
    )
    prediction = enhanced.square()
    manual = assert_uicf_consistency(inputs, prediction, details, prediction.clone())
    torch.testing.assert_close(manual, enhanced, rtol=0, atol=0)
    bad_details = UICFINROutput(
        enhanced=enhanced + 0.01,
        correction_field=field,
        chromatic_anchor=anchor,
        global_feature=details.global_feature,
    )
    with pytest.raises(RuntimeError, match="reconstruction check failed"):
        assert_uicf_consistency(inputs, prediction, bad_details)
    with pytest.raises(RuntimeError, match="Diagnostics forward changed"):
        assert_uicf_consistency(inputs, prediction, details, prediction + 0.01)


def test_metadata_is_plain_json_serializable() -> None:
    metadata = build_metadata(
        _sample(),
        checkpoint_path=Path("/tmp/best_psnr.pt"),
        checkpoint_selector="best_psnr",
        checkpoint_epoch=17,
        heatmap_vmin=-4.0,
        heatmap_vmax=4.0,
        robust_percentile=None,
    )
    serialized = json.dumps(metadata)
    assert serialized
    assert metadata["sample_index"] == 7
    assert metadata["chromatic_anchor_b"] == pytest.approx([0.2, 0.4, 0.6])
    assert metadata["correction_field"]["R_b"]["min"] == pytest.approx(-4.0)


def test_all_sample_artifacts_and_panel_are_generated(tmp_path: Path) -> None:
    sample = _sample()
    metadata = save_visualization_sample(
        sample,
        tmp_path,
        checkpoint_path=Path("/tmp/best_psnr.pt"),
        checkpoint_selector="best_psnr",
        checkpoint_epoch=17,
        heatmap_vmin=-4.0,
        heatmap_vmax=4.0,
        robust_percentile=None,
    )
    expected = {
        "input.png",
        "uicf_corrected_Ic.png",
        "v16_final.png",
        "gt.png",
        "R_r.npy",
        "R_g.npy",
        "R_b.npy",
        "correction_field_rgb.npy",
        "R_r_heatmap.png",
        "R_g_heatmap.png",
        "R_b_heatmap.png",
        "global_feature.npy",
        "uicf_metadata.json",
        "uicf_panel_1x5.png",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    np.testing.assert_array_equal(
        np.load(tmp_path / "correction_field_rgb.npy"),
        sample.correction_field.numpy(),
    )
    assert json.loads((tmp_path / "uicf_metadata.json").read_text()) == metadata
    with Image.open(tmp_path / "uicf_panel_1x5.png") as panel:
        assert panel.mode == "RGB"
        assert panel.width > panel.height
        assert panel.info["dpi"] == pytest.approx((300.0, 300.0), abs=0.1)


def test_sample_selection_modes_are_deterministic_and_unambiguous() -> None:
    entries = [ManifestEntry(f"id{index}", f"in/{index}.png", f"gt/{index}.png") for index in range(8)]
    first = select_sample_indices(
        entries, indices=None, sample_ids=None, num_samples=3, random_seed=3407
    )
    second = select_sample_indices(
        entries, indices=None, sample_ids=None, num_samples=3, random_seed=3407
    )
    assert first == second and len(set(first)) == 3
    assert select_sample_indices(
        entries, indices=[5, 1], sample_ids=None, num_samples=3, random_seed=0
    ) == [5, 1]
    assert select_sample_indices(
        entries, indices=None, sample_ids=["id6", "id2"], num_samples=3, random_seed=0
    ) == [6, 2]
    with pytest.raises(ValueError, match="not both"):
        select_sample_indices(
            entries, indices=[1], sample_ids=["id1"], num_samples=3, random_seed=0
        )


def test_v16_checkpoint_strict_load_and_diagnostics_forward_match() -> None:
    model_config = _small_model_config()
    torch.manual_seed(3520)
    source = build_model(model_config)
    checkpoint = {
        "version": "v16",
        "epoch": 1,
        "resolved_config": {"model": model_config},
        "model_state_dict": source.state_dict(),
    }
    loaded = build_and_load_v16_model(
        {"model": model_config}, checkpoint, torch.device("cpu")
    )
    assert isinstance(loaded, UICFPreBackbone)
    assert loaded.training is False
    assert loaded.state_dict().keys() == source.state_dict().keys()
    inputs = torch.rand(1, 3, 16, 16)
    with torch.inference_mode():
        prediction, details = loaded.forward_with_uicf_details(inputs)
        normal = loaded(inputs)
    assert_uicf_consistency(inputs, prediction, details, normal)


def test_cli_flow_runs_end_to_end_with_synthetic_v16_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "v16_synthetic_run"
    data_root = tmp_path / "data"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "split_snapshot").mkdir()
    for directory in (data_root / "input", data_root / "gt"):
        directory.mkdir(parents=True)
    split_ids = {"train": "train_id", "validation": "validation_id", "test": "test_id"}
    for offset, sample_id in enumerate(split_ids.values()):
        array = np.full((16, 16, 3), 40 + offset * 40, dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(data_root / "input" / f"{sample_id}.png")
        Image.fromarray(array + 5, mode="RGB").save(data_root / "gt" / f"{sample_id}.png")
    for split, sample_id in split_ids.items():
        (run_dir / "split_snapshot" / f"{split}.tsv").write_text(
            f"{sample_id}\tinput/{sample_id}.png\tgt/{sample_id}.png\n",
            encoding="utf-8",
        )

    config = yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text(encoding="utf-8"))
    config["model"] = _small_model_config()
    config["data"] = {
        **config["data"],
        "root": str(data_root),
        "expected_counts": {"train": 1, "validation": 1, "test": 1},
        "num_workers": 0,
    }
    config["evaluation"] = {"resize": True, "size": 16}
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    model = build_model(config["model"])
    torch.save(
        {
            "version": "v16",
            "epoch": 3,
            "resolved_config": {"model": config["model"]},
            "model_state_dict": model.state_dict(),
        },
        run_dir / "best" / "best_psnr.pt",
    )

    visualization_main(
        [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best_psnr",
            "--indices",
            "0",
        ]
    )
    output = run_dir / "result" / "uicf_visualization"
    summary = json.loads((output / "visualization_summary.json").read_text())
    assert summary["sample_count"] == 1
    assert summary["formula_sanity_check"].startswith("PASS")
    sample_directory = output / summary["samples"][0]["directory"]
    assert (sample_directory / "uicf_panel_1x5.png").is_file()
    assert (sample_directory / "correction_field_rgb.npy").is_file()
