from __future__ import annotations
from copy import deepcopy
import importlib
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("no_qgrd", "shared_query", "single_stage", "no_local_refine")


def module(variant, name):
    return importlib.import_module(f"src.qgrd_ablations.{variant}.{name}")


def config(variant, dataset="lsui"):
    return yaml.safe_load((ROOT / f"configs/ablations/qgrd/{dataset}/{variant}.yaml").read_text())


def small_config(variant):
    cfg = config(variant)
    cfg["model"]["base_channels"] = 8
    cfg["model"]["color_query"].update(num_color_queries=4, token_dim=32)
    cfg["model"]["uicf"].update(feat_dim=8, num_frequencies=2, mlp_hidden_dim=16,
                               mlp_hidden_layers=2, anchor_hidden_dim=8)
    return cfg


def pair(variant):
    from src.v16.models import build_model
    cfg = small_config(variant)["model"]
    reference = deepcopy(cfg)
    reference["type"] = "plain_unet_color_query_uicf_pre_backbone"
    torch.manual_seed(3520)
    v16 = build_model(reference)
    torch.manual_seed(3520)
    model = module(variant, "models").build_model(cfg)
    return v16, model, cfg


def assert_common_exact(v16, model):
    expected, actual = v16.state_dict(), model.state_dict()
    assert actual.keys() <= expected.keys()
    for key, value in actual.items():
        assert torch.equal(value, expected[key]), key
    assert any(k.startswith("uicf.") for k in actual)


def assert_forward_backward_reload(variant):
    v16, model, cfg = pair(variant)
    assert_common_exact(v16, model)
    model.eval()
    x = torch.rand(1, 3, 63, 79)
    y, shapes = model.forward_with_shapes(x)
    assert y.shape == x.shape and shapes["final_output"] == tuple(x.shape)
    assert torch.isfinite(y).all()
    (y - torch.rand_like(y)).square().mean().backward()
    # No dead trainable parameters may remain after physical module deletion.
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    replacement = module(variant, "models").build_model(cfg)
    replacement.load_state_dict(model.state_dict(), strict=True)
    assert replacement.state_dict().keys() == model.state_dict().keys()
    return v16, model


def trace(backbone, x):
    events, token_outputs, token_inputs, guide_tokens = [], {}, {}, {}
    handles = []
    for stage in ("4", "3", "2", "1"):
        if hasattr(backbone, "token_refinement"):
            def token_hook(_module, inputs, output, s=stage):
                events.append("token" + s)
                token_inputs[s], token_outputs[s] = inputs[0], output
            handles.append(backbone.token_refinement[stage].register_forward_hook(token_hook))
        guide = getattr(backbone, "guide" + stage, None)
        if guide is not None:
            def guide_hook(_module, inputs, _output, s=stage):
                events.append("guide" + s)
                guide_tokens[s] = inputs[1]
            handles.append(guide.register_forward_hook(guide_hook))
        refine = getattr(backbone, "refine" + stage, None)
        if refine is not None:
            def refine_hook(_module, _inputs, _output, s=stage):
                events.append("refine" + s)
            handles.append(refine.register_forward_hook(refine_hook))
    with torch.no_grad():
        backbone(x)
    for handle in handles:
        handle.remove()
    if token_outputs:
        for current, previous in (("3", "4"), ("2", "3"), ("1", "2")):
            assert token_inputs[current] is token_outputs[previous]
        assert torch.equal(token_inputs["4"], backbone.base_queries.expand(x.shape[0], -1, -1))
    return events, token_outputs, guide_tokens
