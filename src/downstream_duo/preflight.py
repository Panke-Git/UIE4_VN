"""Read-only environment check; does not install/upgrade any packages."""
import argparse
import os
from pathlib import Path
import sys

from .utils import PIN, output_path, versions, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["audit", "uie", "detector"], default="detector")
    p.add_argument("--data-root")
    p.add_argument("--output")
    args = p.parse_args(argv)
    result = {"executable": sys.executable, "conda_env": os.getenv("CONDA_DEFAULT_ENV"),
              "versions": versions(), "role": args.role, "errors": []}
    try:
        import numpy as np
        from PIL import Image
        import yaml
        result["image_smoke"] = Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).size
        if args.role != "audit":
            import torch
            result.update(cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                          gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
            torch.from_numpy(np.zeros((1,), dtype=np.float32)).numpy()
        if args.role == "detector":
            from .utils import require_yolo
            require_yolo()
            import torchvision
            torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([0.5]), 0.7)
        if args.role == "uie":
            from src.v16.models.network import build_model
    except Exception as error:
        result["errors"].append(repr(error))
    if args.data_root:
        for name in ("annotations/instances_train.json", "annotations/instances_test.json", "images/train", "images/test"):
            if not (Path(args.data_root) / name).exists():
                result["errors"].append(f"Missing {name}")
    result["ok"] = not result["errors"]
    if args.output:
        path = output_path(args.output)
        if path.exists():
            raise FileExistsError(path)
        write_json(path, result)
    import json
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        print(f"Do not upgrade pax in place. Create isolated environment; detector pin={PIN}. See README.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
