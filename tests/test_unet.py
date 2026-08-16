from __future__ import annotations

import torch
from torch import nn

from src.v4.models import DoubleConv, PlainUNet


def test_double_conv_is_plain_conv_batchnorm_relu_sequence() -> None:
    block = DoubleConv(3, 64, use_batch_norm=True)
    layers = list(block.layers)
    assert [type(layer) for layer in layers] == [
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU,
    ]
    convolutions = [layer for layer in layers if isinstance(layer, nn.Conv2d)]
    assert all(layer.kernel_size == (3, 3) for layer in convolutions)
    assert all(layer.padding == (1, 1) for layer in convolutions)
    assert all(layer.groups == 1 for layer in convolutions)


def test_small_unet_forward_backward_is_finite_and_bounded() -> None:
    model = PlainUNet(base_channels=8, output_activation="sigmoid")
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = model(inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert float(output.detach().min()) >= 0.0
    assert float(output.detach().max()) <= 1.0
    output.mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_odd_native_size_is_padded_and_cropped_back() -> None:
    model = PlainUNet(base_channels=8).eval()
    inputs = torch.randn(1, 3, 37, 53)
    with torch.inference_mode():
        output, shapes = model.forward_with_shapes(inputs)
    assert output.shape == inputs.shape
    assert shapes["e1"] == (1, 8, 48, 64)
    assert shapes["bottleneck"] == (1, 128, 3, 4)
    assert shapes["final_output"] == tuple(inputs.shape)


def test_zeroed_unet_predicts_sigmoid_half_without_global_residual() -> None:
    model = PlainUNet(base_channels=4).eval()
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    inputs = torch.rand(1, 3, 17, 19)
    with torch.inference_mode():
        output = model(inputs)
    assert torch.equal(output, torch.full_like(output, 0.5))
    assert not torch.equal(output, inputs)
