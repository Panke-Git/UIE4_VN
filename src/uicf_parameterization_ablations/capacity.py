"""Measure registered, trainable and output-connected parameters on local CPU."""
from __future__ import annotations

import argparse
import gc
import json

import torch

from src.v16.models import build_model as build_full
from src.v16.utils import PROJECT_ROOT, load_yaml, project_path
from . import VARIANTS
from .models import build_model
from .protocol import config_path


def measure(model) -> dict:
    model.eval()
    parameters = list(model.parameters())
    if not all(p.requires_grad for p in parameters):
        raise ValueError("Capacity audit expects no frozen parameters")
    # A fixed small probe keeps the full configured architecture, with no AMP,
    # optimizer update, real data or checkpoint. Zero-valued gradients still
    # prove graph connectivity; only None means absent from the output graph.
    probe = torch.linspace(0, 1, 3*32*32).reshape(1, 3, 32, 32)
    grads = torch.autograd.grad(model(probe).mean(), parameters, allow_unused=True)
    connected = sum(p.numel() for p, g in zip(parameters, grads) if g is not None)
    total = sum(p.numel() for p in parameters)
    return dict(total=total, trainable=total, forward_participating=connected,
                output_connected=connected, not_output_connected=total-connected,
                method="Actual CPU autograd output-graph reachability; zero gradient is counted as connected")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="analysis/uicf_parameterization/parameter_counts.json")
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    rows = {}
    for variant in ("full_uicf", *VARIANTS):
        torch.manual_seed(3520)
        cfg = load_yaml(PROJECT_ROOT/"configs/config_v16.yaml" if variant == "full_uicf" else config_path(variant))
        model = build_full(cfg["model"]) if variant == "full_uicf" else build_model(cfg["model"])
        rows[variant] = measure(model)
        del model
        gc.collect()
    for row in rows.values():
        row["difference_from_full"] = row["output_connected"]-rows["full_uicf"]["output_connected"]
    path = project_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"architecture": "original V16 LSUI production dimensions", "variants": rows,
                               "baseline_checkpoint_read": False}, indent=2)+"\n")
    print(path)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
