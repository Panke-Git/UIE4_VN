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
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
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
    print(f"total params: {total}")
    print(f"trainable params: {trainable}")
    print(f"input test shape: {tuple(test_input.shape)}")
    print(f"bottleneck shape: {shapes['bottleneck']}")
    print(f"{version} module input shape: {shapes['module_input']}")
    if "latent_grid" in captured:
        print(f"GL-INR latent grid shape: {captured['latent_grid']}")
    print(f"{version} module output shape: {shapes['module_output']}")
    print(f"final output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
