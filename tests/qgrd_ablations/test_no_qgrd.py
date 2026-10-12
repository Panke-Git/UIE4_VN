import torch
from src.shared.color_query_unet import SpatialTokenGuidance, ResidualLocalRefine, ColorTokenRefinementBlock
from src.v4.models.unet import PlainUNet
from .helpers import assert_forward_backward_reload, trace


def test_no_qgrd_structure_initialization_and_backward():
    v16, model = assert_forward_backward_reload("no_qgrd")
    assert not any(isinstance(m, (SpatialTokenGuidance, ResidualLocalRefine, ColorTokenRefinementBlock))
                   for m in model.backbone.modules())
    assert not hasattr(model.backbone, "base_queries")
    removed = sum(p.numel() for n, p in v16.named_parameters() if n not in dict(model.named_parameters()))
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in v16.parameters()) - removed
    assert all(not any(s in key for s in ("guide", "refine", "feature_to_token", "base_queries"))
               for key in model.state_dict())


def test_no_qgrd_is_exact_plain_unet_after_nonzero_uicf():
    _, model = assert_forward_backward_reload("no_qgrd")
    model.eval()
    plain = PlainUNet(base_channels=8).eval()
    plain.load_state_dict(model.backbone.state_dict(), strict=True)
    # Use a genuinely nonzero UICF correction, not only its identity initialization.
    with torch.no_grad():
        model.uicf.field_mlp.net[-1].bias.fill_(0.03)
        x = torch.rand(1, 3, 63, 79)
        corrected = model.uicf(x, return_details=False)
        assert not torch.equal(corrected, x)
        assert torch.equal(model(x), plain(corrected))
        assert trace(model.backbone, corrected)[0] == []
