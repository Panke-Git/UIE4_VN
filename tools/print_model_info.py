#!/usr/bin/env python3
"""Build any isolated version and report parameter and feature shapes."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    version = config["experiment"]["version"]
    build_model = importlib.import_module(f"src.{version}.models").build_model
    model = build_model(config["model"])
    model.eval()
    total = parameter_count(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if version in {"v4", "v5", "v6"}:
        test_input = torch.randn(
            1, int(config["model"]["in_channels"]), args.height, args.width
        )
        captured: dict[str, tuple[int, ...]] = {}
        bottleneck_module = getattr(model, "bottleneck_module", None)
        latent_projection = getattr(bottleneck_module, "latent_projection", None)
        hook = None
        if latent_projection is not None:
            hook = latent_projection.register_forward_hook(
                lambda _module, _inputs, output: captured.update(latent_grid=tuple(output.shape))
            )
        with torch.no_grad():
            output, shapes = model.forward_with_shapes(test_input)
        if hook is not None:
            hook.remove()
        bottleneck_name = type(bottleneck_module).__name__ if bottleneck_module else "Identity"
        bottleneck_params = parameter_count(bottleneck_module) if bottleneck_module else 0
        print(f"model: {config['experiment']['name']}")
        print("model family: Plain U-Net")
        print(f"structure: four-level encoder -> {bottleneck_name} -> concat-skip decoder")
        print("skip method: channel-wise torch.cat")
        print(f"output activation: {config['model']['output_activation']}")
        print("global image residual: disabled")
        print(f"total params: {total}")
        print(f"trainable params: {trainable}")
        print(f"common Plain U-Net params: {total - bottleneck_params}")
        print(f"{bottleneck_name} params: {bottleneck_params}")
        print(f"input shape: {shapes['input']}")
        for level in range(1, 5):
            print(f"E{level} shape: {shapes[f'e{level}']}")
        print(f"bottleneck shape: {shapes['bottleneck']}")
        if "module_input" in shapes:
            print(f"bottleneck module: {bottleneck_name}")
            print(f"module input shape: {shapes['module_input']}")
            if "latent_grid" in captured:
                print(f"GL-INR latent grid shape: {captured['latent_grid']}")
            print(f"module output shape: {shapes['module_output']}")
        print(f"decoder output shape: {shapes['decoder_output']}")
        print(f"final output shape: {tuple(output.shape)}")
        return
    encoder_params = sum(
        parameter_count(module) for module in (model.intro, model.encoders, model.downs)
    )
    decoder_params = sum(
        parameter_count(module) for module in (model.ups, model.decoders, model.ending)
    )
    bottleneck_params = parameter_count(model.bottleneck_module)
    bottleneck_name = type(model.bottleneck_module).__name__
    test_input = torch.randn(1, int(config["model"]["img_channel"]), args.height, args.width)
    captured: dict[str, tuple[int, ...]] = {}
    hook = None
    latent_projection = getattr(model.bottleneck_module, "latent_projection", None)
    if latent_projection is not None:
        hook = latent_projection.register_forward_hook(
            lambda _module, _inputs, output: captured.update(latent_grid=tuple(output.shape))
        )
    with torch.no_grad():
        output, shapes = model.forward_with_shapes(test_input)
    if hook is not None:
        hook.remove()
    print(f"model: {config['experiment']['name']}")
    print(f"structure: Encoder -> {bottleneck_name} -> Decoder")
    print(f"middle NAF blocks: {len(model.middle_blks)}")
    print(f"total params: {total}")
    print(f"trainable params: {trainable}")
    print(f"common encoder params: {encoder_params}")
    print(f"common decoder params: {decoder_params}")
    print(f"{bottleneck_name} params: {bottleneck_params}")
    print(f"input shape: {tuple(test_input.shape)}")
    print(f"encoder output shape: {shapes['encoder_output']}")
    print(f"bottleneck: {bottleneck_name}")
    print(f"bottleneck input shape: {shapes['module_input']}")
    if "latent_grid" in captured:
        print(f"GL-INR latent grid shape: {captured['latent_grid']}")
    print(f"bottleneck output shape: {shapes['module_output']}")
    print(f"decoder output shape: {shapes['decoder_output']}")
    print(f"final output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
