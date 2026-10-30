"""Tiny synthetic CPU workflow, clearly separated from scientific experiments."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.v16.utils import load_yaml, project_path, save_yaml
from . import VARIANTS
from .protocol import config_path


def prepare_smoke(directory: Path, variant: str) -> Path:
    """Create 2/1/1 artificial pairs; never read or rewrite real LSUI manifests."""
    directory.mkdir(parents=True, exist_ok=False)
    cfg = deepcopy(load_yaml(config_path(variant)))
    data_root = directory/"SYNTHETIC_DATA_NOT_LSUI"
    rng = np.random.default_rng(3520)
    for split, count in dict(train=2, validation=1, test=1).items():
        rows = []
        for index in range(count):
            sample = f"{split}_{index}"
            inp, gt = f"{split}/input/{sample}.png", f"{split}/GT/{sample}.png"
            array = rng.integers(0, 220, (32, 32, 3), dtype=np.uint8)
            for relative, pixels in ((inp, array), (gt, array+20)):
                path = data_root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
            rows.append(f"{sample}\t{inp}\t{gt}\n")
        manifest = directory/f"{split}.tsv"
        manifest.write_text("".join(rows))
        cfg["data"][split+"_manifest"] = str(manifest)
    cfg["data"].update(root=str(data_root), expected_counts=dict(train=2, validation=1, test=1),
                       patch_size=32, batch_size=2, num_workers=0, pin_memory=False)
    cfg["model"]["base_channels"] = 4
    cfg["model"]["color_query"].update(num_color_queries=2, token_dim=16)
    cfg["model"]["uicf"].update(feat_dim=4, num_frequencies=2, mlp_hidden_dim=8, anchor_hidden_dim=8, query_chunk_size=256)
    cfg["training"].update(epochs=2, save_every=1, deterministic=True)
    cfg["evaluation"]["size"] = 32
    cfg["test"]["visualization"].update(num_samples=1, cell_width=32, cell_height=32)
    cfg["logging"]["console"] = False
    cfg["experiment"]["output_root"] = str(directory/"SYNTHETIC_RUNS_NOT_PAPER_RESULTS")
    cfg["ablation"]["smoke_test"] = True
    source = directory/"smoke_config.yaml"
    save_yaml(source, cfg)
    return source


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, action="append")
    p.add_argument("--output-dir", default="analysis/uicf_parameterization/local_smoke")
    args = p.parse_args(argv)
    from .train import main as train
    from .test import main as test
    torch.set_num_threads(1)
    session = project_path(args.output_dir).resolve()/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for variant in args.variant or VARIANTS:
        source = prepare_smoke(session/variant, variant)
        run = train(["--config", str(source), "--variant", variant, "--smoke-test", "--stop-after-epoch", "1"])
        train(["--resume", str(run/"checkpoint/last.pt"), "--variant", variant])
        test(["--run-dir", str(run), "--variant", variant, "--checkpoint", "best_psnr"])
        print(f"SYNTHETIC_SMOKE_PASSED={variant}; {run}")


if __name__ == "__main__":
    main()
