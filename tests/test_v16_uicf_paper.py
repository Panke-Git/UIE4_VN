"""Software checks; synthetic fixtures never become delivered paper samples."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from src.v16.models import build_model
from tools.v16_uicf_paper import (
    ZERO_EPSILON, capture_once, full_model_config, preprocess, render_figures,
)
from tools.visualize_v16_uicf import build_and_load_v16_model

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / "configs/config_v16.yaml").read_text())


def test_rejects_ablation_and_other_version():
    full = config()
    full_model_config(full)
    for key in ("use_spatial_conditioning", "use_global_field_conditioning", "use_learned_anchor"):
        bad = deepcopy(full)
        bad["model"]["uicf"][key] = False
        with pytest.raises(ValueError, match="full implicit"):
            full_model_config(bad)
    bad = deepcopy(full)
    bad["experiment"]["version"] = "v16_19"
    with pytest.raises(ValueError, match="canonical v16"):
        full_model_config(bad)


def test_one_cpu_forward_preserves_unpadded_uicf_and_backbone_crop():
    conf = config()
    conf["model"]["base_channels"] = 4
    conf["model"]["color_query"].update(token_dim=16, num_heads=4)
    conf["model"]["uicf"].update(feat_dim=4, num_frequencies=2, mlp_hidden_dim=8,
                                 mlp_hidden_layers=1, anchor_hidden_dim=4, query_chunk_size=31)
    torch.manual_seed(3520)
    model = build_model(conf["model"]).eval().requires_grad_(False)
    # Force nonzero signed coefficients to exercise reconstruction, not identity.
    with torch.no_grad():
        model.uicf.field_mlp.net[-1].bias.copy_(torch.tensor([0.2, -0.3, 0.4]))
    image = Image.fromarray(np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3))
    conf["evaluation"]["resize"] = False
    inputs, geometry = preprocess(image, conf)
    arrays, error, calls, padding = capture_once(model, inputs, torch.device("cpu"), False)
    assert calls == {"uicf": 1, "backbone": 1, "field_mlp": 1}
    assert padding == [0, 13, 0, 15]
    assert arrays["R"].shape == arrays["I"].shape == arrays["I_hat"].shape == (1, 3, 17, 19)
    assert arrays["backbone_padded_input"].shape == (1, 3, 32, 32)
    assert error == 0
    assert geometry["preprocessing_matches_existing_dataset_exactly"]
    checkpoint = {"version": "v16", "resolved_config": {"model": conf["model"]},
                  "model_state_dict": dict(model.state_dict())}
    checkpoint["model_state_dict"].pop(next(iter(checkpoint["model_state_dict"])))
    with pytest.raises(RuntimeError, match="Missing key"):
        build_and_load_v16_model(conf, checkpoint, torch.device("cpu"))


@pytest.mark.parametrize("zero", [False, True])
def test_render_uses_same_signed_scale_and_keeps_raw_arrays(tmp_path, zero):
    field = np.zeros((1, 3, 6, 9), dtype=np.float32)
    if not zero:
        field[0, 0, 0, 0], field[0, 1, 1, 1], field[0, 2, 2, 2] = -3, 1, 2
    raw = field.copy()
    arrays = {"R": field, "b": np.array([[0.2, 0.4, 0.6]], dtype=np.float32)}
    plot = render_figures(arrays, tmp_path, 60)
    np.testing.assert_array_equal(field, raw)
    assert plot["all_zero"] is zero
    assert plot["vmax"] == (ZERO_EPSILON if zero else 3)
    assert plot["vmin"] == -plot["vmax"]
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    norm = Normalize(plot["vmin"], plot["vmax"])
    for index, channel in enumerate("rgb"):
        with Image.open(tmp_path / f"uicf_field_{channel}.png") as image:
            assert image.size == (9, 6)
            expected = colormaps["RdBu_r"](norm(field[0, index]), bytes=True)[..., :3]
            np.testing.assert_array_equal(np.asarray(image), expected)
