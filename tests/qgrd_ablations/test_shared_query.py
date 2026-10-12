import torch
from .helpers import assert_forward_backward_reload, trace


def test_shared_query_state_capacity_and_behavior():
    v16, model = assert_forward_backward_reload("shared_query")
    assert model.state_dict().keys() == v16.state_dict().keys()
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in v16.parameters())
    events, tokens, guides = trace(model.backbone.eval(), torch.rand(1, 3, 32, 32))
    assert events == ["token4", "token3", "token2", "token1",
                      "guide4", "refine4", "guide3", "refine3",
                      "guide2", "refine2", "guide1", "refine1"]
    for stage in ("4", "3", "2", "1"):
        assert guides[stage] is tokens["1"]
