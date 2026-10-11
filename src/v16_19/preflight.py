"""Verify full real data identity and one actual sample forward before training."""
import argparse
import json
from pathlib import Path

import torch

from .dataset import LSUIDataset, validate_split_protocol
from .e00 import batch_delta_e00
from .metrics import batch_metrics
from .models import build_model
from .protocol import verify_dataset
from .utils import load_yaml, parameter_counts, project_path, seed_everything, select_device, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check V16-19 full data/config and one real image forward")
    parser.add_argument("--config", default="configs/v16_19/lsui19.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_yaml(project_path(args.config))
    if args.data_root:
        config["data"]["root"] = args.data_root
    validate_config(config)
    data = config["data"]
    root, directory = project_path(data["root"]), project_path(data["protocol_dir"])
    protocol = verify_dataset(root, directory)
    if protocol["dataset"] != data["dataset_key"] or protocol["counts"] != data["expected_counts"]:
        raise ValueError("Dataset/counts do not match configuration")
    manifests = {k: directory / f"{k}.tsv" for k in ("train", "validation")}
    validate_split_protocol(manifests, data["expected_counts"])
    seed_everything(config["experiment"]["seed"], config["training"]["deterministic"])
    device = select_device(args.gpu)
    model = build_model(config["model"]).to(device).eval()
    dataset = LSUIDataset(manifests["validation"], root, "validation", data["patch_size"], data["augmentation"],
                         data["pad_if_smaller"], data["pad_mode"], config["evaluation"], verify_files=False)
    sample = dataset[0]
    inp, gt = sample["input"].unsqueeze(0).to(device), sample["target"].unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda" and config["training"]["amp"]):
        output = model(inp)
    if output.shape != inp.shape or not torch.isfinite(output).all():
        raise ValueError("Model forward shape/finiteness check failed")
    clipped = output.float().clamp(0, 1)
    psnr, ssim = batch_metrics(clipped, gt, config["metrics"])
    e00 = batch_delta_e00(clipped, gt, config["metrics"])
    if not torch.isfinite(psnr).all() or not torch.isfinite(ssim).all() or not torch.isfinite(e00).all():
        raise ValueError("Sample metric finiteness check failed")
    quality = json.loads((directory / "data_quality.json").read_text(encoding="utf-8"))
    total, trainable = parameter_counts(model)
    result = {"status": "CONFIG_DATA_AND_SAMPLE_FORWARD_PASS", "version": "v16_19", "dataset": data["dataset"],
              "counts": protocol["counts"], "dataset_fingerprint": protocol["fingerprint"], "has_test_split": False,
              "device": str(device), "pytorch_version": torch.__version__, "model_total_params": total,
              "model_trainable_params": trainable, "sample_id": sample["id"], "sample_shape": list(inp.shape),
              "metrics_finite": True, "cross_split_duplicate_input_groups": quality["cross_split_duplicate_groups"],
              "training_executed": False, "full_validation_executed": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
