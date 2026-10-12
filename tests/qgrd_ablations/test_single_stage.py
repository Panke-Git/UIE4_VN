import torch
from src.shared.color_query_unet import SpatialTokenGuidance, ResidualLocalRefine
from .helpers import assert_forward_backward_reload, trace


def test_single_stage_keeps_all_refines_and_query_gradients():
    v16, model = assert_forward_backward_reload("single_stage")
    backbone = model.backbone
    for stage in (4, 3, 2):
        assert not hasattr(backbone, f"guide{stage}")
        assert not any(k.startswith(f"backbone.guide{stage}.") for k in model.state_dict())
    assert sum(isinstance(m, SpatialTokenGuidance) for m in backbone.modules()) == 1
    assert sum(isinstance(m, ResidualLocalRefine) for m in backbone.modules()) == 4
    expected_removed = sum(p.numel() for s in (4, 3, 2) for p in getattr(v16.backbone, f"guide{s}").parameters())
    assert sum(p.numel() for p in v16.parameters()) - sum(p.numel() for p in model.parameters()) == expected_removed
    events, tokens, guides = trace(backbone.eval(), torch.rand(1, 3, 32, 32))
    assert events == ["token4", "token3", "token2", "token1", "refine4", "refine3", "refine2", "guide1", "refine1"]
    assert guides["1"] is tokens["1"]
    for stage in ("4", "3", "2", "1"):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in backbone.token_refinement[stage].parameters())
