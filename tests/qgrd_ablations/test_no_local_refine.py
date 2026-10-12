import torch
from src.shared.color_query_unet import SpatialTokenGuidance, ResidualLocalRefine
from .helpers import assert_forward_backward_reload, trace


def test_no_local_refine_preserves_guidance_and_internal_residual():
    v16, model = assert_forward_backward_reload("no_local_refine")
    backbone = model.backbone
    assert sum(isinstance(m, SpatialTokenGuidance) for m in backbone.modules()) == 4
    assert not any(isinstance(m, ResidualLocalRefine) for m in backbone.modules())
    assert not any(k.startswith("backbone.refine") for k in model.state_dict())
    removed = sum(p.numel() for s in (4, 3, 2, 1) for p in getattr(v16.backbone, f"refine{s}").parameters())
    assert sum(p.numel() for p in v16.parameters()) - sum(p.numel() for p in model.parameters()) == removed
    events, tokens, guides = trace(backbone.eval(), torch.rand(1, 3, 32, 32))
    assert events == ["token4", "token3", "token2", "token1", "guide4", "guide3", "guide2", "guide1"]
    for stage in ("4", "3", "2", "1"):
        assert guides[stage] is tokens[stage]
        guide = getattr(backbone, "guide" + stage)
        assert type(guide) is SpatialTokenGuidance
        # Zero attention output projection proves the inherited spatial residual remains.
        with torch.no_grad():
            guide.output_projection.weight.zero_()
            guide.output_projection.bias.zero_()
            spatial = torch.rand(1, guide.spatial_channels, 4, 5)
            assert torch.equal(guide(spatial, tokens[stage]), spatial)
