import copy

import torch

from src.v3.models.glinr import GLINR


def make_module(chunk: int) -> GLINR:
    return GLINR(
        channels=8,
        latent_dim=6,
        hidden_dim=12,
        latent_stride=2,
        global_num_frequencies=3,
        local_num_frequencies=0,
        include_raw_absolute_coordinate=True,
        include_raw_relative_coordinate=True,
        local_depth=3,
        global_depth=3,
        fusion_depth=3,
        query_chunk=chunk,
        residual=True,
    )


def test_four_neighbor_indices_boundaries_and_weights() -> None:
    coordinates, indices, relative, weights = GLINR.query_geometry(
        7, 9, 4, 5, torch.device("cpu"), torch.float32
    )
    assert coordinates.shape == (63, 2)
    assert indices.shape == (63, 4, 2)
    assert relative.shape == (63, 4, 2)
    assert weights.shape == (63, 4)
    assert indices[..., 0].min() >= 0 and indices[..., 0].max() < 4
    assert indices[..., 1].min() >= 0 and indices[..., 1].max() < 5
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(63), rtol=0, atol=1e-6)
    assert (weights >= 0).all()


def _expected_cell_relative(
    query_height: int,
    query_width: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    coordinates, _, _, _ = GLINR.query_geometry(
        query_height,
        query_width,
        latent_height,
        latent_width,
        torch.device("cpu"),
        torch.float32,
    )
    gx = (coordinates[:, 0] + 1.0) * latent_width / 2.0 - 0.5
    gy = (coordinates[:, 1] + 1.0) * latent_height / 2.0 - 0.5
    x0_raw, y0_raw = torch.floor(gx), torch.floor(gy)
    x1_raw, y1_raw = x0_raw + 1.0, y0_raw + 1.0
    return torch.stack(
        (
            torch.stack((gx - x0_raw, gy - y0_raw), dim=-1),
            torch.stack((gx - x1_raw, gy - y0_raw), dim=-1),
            torch.stack((gx - x0_raw, gy - y1_raw), dim=-1),
            torch.stack((gx - x1_raw, gy - y1_raw), dim=-1),
        ),
        dim=1,
    )


def test_relative_coordinates_are_in_latent_cell_units() -> None:
    for latent_height, latent_width in ((8, 8), (16, 16)):
        _, _, relative, _ = GLINR.query_geometry(
            32,
            32,
            latent_height,
            latent_width,
            torch.device("cpu"),
            torch.float32,
        )
        expected = _expected_cell_relative(32, 32, latent_height, latent_width)
        torch.testing.assert_close(relative, expected, rtol=0, atol=0)

    # Interior query (row=12, col=10) at 32 -> 16 has latent location
    # (gx, gy)=(4.75, 5.75), so deltas are measured in whole cell units.
    _, _, relative_16, _ = GLINR.query_geometry(
        32, 32, 16, 16, torch.device("cpu"), torch.float32
    )
    index = 12 * 32 + 10
    expected_interior = torch.tensor(
        [[0.75, 0.75], [-0.25, 0.75], [0.75, -0.25], [-0.25, -0.25]]
    )
    torch.testing.assert_close(relative_16[index], expected_interior, rtol=0, atol=0)


def test_glinr_shape_finite_and_backpropagation() -> None:
    module = make_module(11)
    features = torch.randn(2, 8, 7, 9, requires_grad=True)
    output = module(features)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_glinr_identity_start() -> None:
    module = make_module(11)
    features = torch.randn(2, 8, 7, 9)
    torch.testing.assert_close(module(features), features, rtol=0, atol=0)


def test_glinr_query_chunk_equivalence() -> None:
    large = make_module(100_000)
    # Move the correction head away from identity-start so chunk equivalence
    # covers the complete local/global/fusion computation.
    with torch.no_grad():
        large.fusion.net[-1].weight.normal_(mean=0.0, std=0.05)
        large.fusion.net[-1].bias.normal_(mean=0.0, std=0.05)
    small = make_module(7)
    small.load_state_dict(copy.deepcopy(large.state_dict()))
    features = torch.randn(2, 8, 5, 7)
    torch.testing.assert_close(large(features), small(features), rtol=1e-6, atol=1e-6)
