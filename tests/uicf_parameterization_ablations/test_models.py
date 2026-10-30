from copy import deepcopy
import hashlib
import json

import pytest
import torch

from src.v16.models import build_model as build_full
from src.v16.utils import PROJECT_ROOT, load_yaml
from src.uicf_parameterization_ablations import VARIANTS
from src.uicf_parameterization_ablations.capacity import measure
from src.uicf_parameterization_ablations.models import build_model, parameter_counts
from src.uicf_parameterization_ablations.protocol import config_path, validate_config


def small(variant):
    cfg = load_yaml(config_path(variant))["model"]
    cfg["base_channels"] = 4
    cfg["color_query"].update(num_color_queries=2, token_dim=16)
    cfg["uicf"].update(feat_dim=4, num_frequencies=2, mlp_hidden_dim=8, anchor_hidden_dim=8, query_chunk_size=256)
    return cfg


@pytest.mark.parametrize("variant", VARIANTS)
def test_formula_and_residual_semantics(variant):
    field = build_model(small(variant)).uicf
    with torch.no_grad():
        field.field_mlp.net[-1].bias.copy_(torch.tensor([-0.2, 0.3, 0.7]))
    image = torch.linspace(0, 1, 2*3*5*7).reshape(2, 3, 5, 7)
    prediction = torch.tensor([-0.2, 0.3, 0.7]).reshape(1, 3, 1, 1).expand_as(image)
    details = field(image, return_details=True)
    expected_delta = prediction if variant == "direct_residual" else prediction * (image if variant == "without_anchor" else image-0.5)
    torch.testing.assert_close(details.enhanced, image+expected_delta, rtol=0, atol=1e-7)
    torch.testing.assert_close(details.actual_correction, expected_delta, rtol=0, atol=1e-7)
    if variant == "direct_residual":
        assert details.correction_field is None and details.reconstruction_anchor is None
        assert torch.equal(details.predicted_residual, prediction)
    else:
        assert details.predicted_residual is None and torch.equal(details.correction_field, prediction)
        assert torch.equal(details.reconstruction_anchor, torch.full((2, 3), 0.0 if variant == "without_anchor" else 0.5))


@pytest.mark.parametrize("variant", VARIANTS)
def test_initial_identity_and_backbone_interface(variant):
    model = build_model(small(variant)).eval()
    image = torch.rand(1, 3, 31, 47)
    output, details = model.forward_with_uicf_details(image)
    assert torch.equal(image, details.enhanced)
    assert output.shape == image.shape and torch.isfinite(output).all()
    assert torch.equal(output, model.backbone(image))
    assert not any("anchor_mlp" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("variant", VARIANTS)
def test_two_step_gradients_reach_field_backbone_and_qgrd(variant):
    torch.manual_seed(3520)
    model = build_model(small(variant)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    image, target = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(image)
        loss = torch.sqrt((prediction-target).square()+0.001**2).mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.requires_grad and param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
        # At step zero, a zero output layer can block nonzero upstream gradients.
        if step == 1:
            for prefix in ("uicf.encoder", "uicf.field_mlp", "backbone.enc", "backbone.token_refinement", "backbone.guide", "backbone.refine"):
                gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
                assert gradients and sum(float(g.abs().sum()) for g in gradients) > 0, prefix
        optimizer.step()


def test_standard_dimensions_share_exact_original_v16_initial_weights():
    cfg = load_yaml(PROJECT_ROOT/"configs/config_v16.yaml")
    torch.manual_seed(3520)
    full = build_full(cfg["model"])
    expected = full.state_dict()
    full_total = sum(p.numel() for p in full.parameters())
    for variant in VARIANTS:
        torch.manual_seed(3520)
        model = build_model(load_yaml(config_path(variant))["model"])
        actual = model.state_dict()
        assert actual.keys() < expected.keys()
        assert {k for k in expected if k not in actual} == {k for k in expected if "uicf.chromatic_anchor.anchor_mlp" in k}
        for key, tensor in actual.items():
            assert torch.equal(tensor, expected[key]), key
        counts = parameter_counts(model)
        assert counts["full_uicf_total"] == full_total == 38878217
        assert counts["total"] == counts["trainable"] == 38874886
        assert counts["removed_anchor_predictor"] == 3331
        del actual, model


@pytest.mark.parametrize("variant", VARIANTS)
def test_actual_output_graph_parameter_inventory(variant):
    model = build_model(small(variant))
    measured, declared = measure(model), parameter_counts(model)
    assert measured["total"] == measured["output_connected"] == declared["total"]
    assert measured["not_output_connected"] == 0


def test_original_v16_anchor_relative_formula_and_initialization():
    cfg = small("direct_residual")
    cfg.pop("parameterization")
    cfg["type"] = "plain_unet_color_query_uicf_pre_backbone"
    cfg["uicf"].pop("use_learned_anchor")
    model = build_full(cfg)
    image = torch.rand(2, 3, 5, 7)
    assert torch.equal(model.uicf(image), image)
    with torch.no_grad():
        model.uicf.field_mlp.net[-1].bias.fill_(0.25)
    details = model.uicf(image, return_details=True)
    torch.testing.assert_close(details.enhanced, image+details.correction_field*(image-details.chromatic_anchor[:, :, None, None]), rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_configs_pin_original_recipe_and_reject_extra_variables(variant):
    cfg = load_yaml(config_path(variant))
    assert validate_config(cfg) == variant
    for section, key, value in (("data", "batch_size", 8), ("training", "epochs", 10), ("loss", "epsilon", 0.01)):
        wrong = deepcopy(cfg)
        wrong[section][key] = value
        with pytest.raises(ValueError):
            validate_config(wrong)
    with pytest.raises(ValueError):
        validate_config(cfg, next(v for v in VARIANTS if v != variant))


def test_all_preexisting_tracked_files_unchanged():
    snapshot = json.loads((PROJECT_ROOT/"src/uicf_parameterization_ablations/protection_snapshot.json").read_text())
    for name, digest in snapshot["tracked_files"].items():
        assert hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest() == digest, name
