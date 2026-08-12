import copy

import torch

from src.v2.models.point_inr import PointINR


def make_module(chunk: int) -> PointINR:
    return PointINR(
        channels=8,
        hidden_dim=16,
        num_frequencies=3,
        depth=3,
        include_raw_coordinate=True,
        query_chunk=chunk,
        residual=True,
    )


def test_point_inr_shape_finite_backward_and_pe_dimension() -> None:
    module = make_module(7)
    assert module.encoding.output_dim == 2 + 4 * 3
    features = torch.randn(2, 8, 5, 7, requires_grad=True)
    output = module(features)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_point_inr_identity_start() -> None:
    module = make_module(7)
    features = torch.randn(2, 8, 5, 7)
    torch.testing.assert_close(module(features), features, rtol=0, atol=0)


def test_point_inr_query_chunk_equivalence() -> None:
    large = make_module(10_000)
    # Move the correction head away from identity-start so this test exercises
    # non-zero chunked computation rather than comparing E with itself.
    with torch.no_grad():
        large.mlp.net[-1].weight.normal_(mean=0.0, std=0.05)
        large.mlp.net[-1].bias.normal_(mean=0.0, std=0.05)
    small = make_module(5)
    small.load_state_dict(copy.deepcopy(large.state_dict()))
    features = torch.randn(2, 8, 4, 6)
    torch.testing.assert_close(large(features), small(features), rtol=1e-6, atol=1e-6)
