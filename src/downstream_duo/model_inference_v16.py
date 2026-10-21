"""Frozen LSUI V16 inference at native resolution (or explicitly selected overlap tiles)."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

from .utils import (OUTPUT, fresh_dir, load_prepared, output_path, pair_identity,
                    read_json, sha256, versions, write_json)


def load_v16(run_dir, selector="best_psnr", device="cpu"):
    """Mirror v16/test.py version, architecture equality and strict state loading."""
    run = Path(run_dir).expanduser().resolve()
    if selector != "best_psnr":
        raise ValueError("This protocol fixes LSUI enhancer selection to best_psnr")
    config_path = run / "config_resolved.yaml"
    checkpoint_path = run / "best" / f"{selector}.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Need {config_path} and {checkpoint_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment", {}).get("version") != "v16":
        raise ValueError("Only canonical V16 is accepted; Haze4K/other versions are forbidden")
    dataset = str(config.get("data", {}).get("dataset", "")).lower()
    if not dataset.startswith("lsui") or "haze" in str(config.get("data", {})).lower():
        raise ValueError("Enhancer provenance must identify LSUI training, never Haze4K")
    if config["model"].get("output_activation") != "sigmoid":
        raise ValueError("Expected V16 RGB sigmoid output")
    uicf = config["model"].get("uicf", {})
    if uicf.get("field_variant", "implicit") not in ("implicit", "implicit_inr") or any(
        uicf.get(flag, True) is not True for flag in ("use_spatial_conditioning", "use_global_field_conditioning", "use_learned_anchor")
    ):
        raise ValueError("Full condition requires canonical implicit UICF with all conditioning enabled")
    # Only load a trusted local run checkpoint, as in the original evaluator.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != "v16":
        raise ValueError("Checkpoint version is incompatible")
    resolved = checkpoint.get("resolved_config", {})
    if resolved.get("model") != config["model"]:
        raise ValueError("Checkpoint architecture differs from config_resolved.yaml")
    if str(resolved.get("data", {}).get("dataset", "")).lower() != dataset:
        raise ValueError("Checkpoint LSUI provenance differs from run configuration")
    if "haze" in str(resolved.get("data", {})).lower():
        raise ValueError("Haze4K checkpoint is forbidden")
    from src.v16.models.network import build_model
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    factor = model.backbone.padding_factor
    if factor != 16:
        raise ValueError(f"Source architecture padding factor changed: {factor}")
    return model, {"run_dir": str(run), "checkpoint": str(checkpoint_path),
                   "checkpoint_sha256": sha256(checkpoint_path), "config_sha256": sha256(config_path),
                   "dataset": dataset, "epoch": checkpoint.get("epoch"), "selector": selector,
                   "padding_factor": factor, "model_config": config["model"]}


def pad_tensor(tensor, factor=16):
    h, w = tensor.shape[-2:]
    ph, pw = (-h) % factor, (-w) % factor
    mode = "reflect" if min(h, w) > 1 and ph < h and pw < w else "replicate"
    return F.pad(tensor, (0, pw, 0, ph), mode=mode) if ph or pw else tensor


def starts(length, tile, overlap):
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, tile - overlap))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.inference_mode()
def enhance_tensor(model, tensor, device="cpu", tile_size=0, overlap=64):
    """BCHW float RGB [0,1], CPU accumulation; never resize image content."""
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError("Expected one RGB image BCHW")
    if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
        raise ValueError("Input must be finite RGB [0,1]")
    if tile_size < 0 or (tile_size and not 0 < overlap < tile_size):
        raise ValueError("Tiles need 0 < overlap < tile_size")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Enhancer must be eval() with all parameters frozen")
    h, w = tensor.shape[-2:]
    def forward(x):
        ih, iw = x.shape[-2:]
        # Right/bottom-only padding and exact crop preserve bbox origin and scale.
        output = model(pad_tensor(x.to(device)))[..., :ih, :iw].float().cpu()
        if output.shape != x.shape or not torch.isfinite(output).all():
            raise ValueError("Enhancer produced invalid shape/nonfinite output")
        if output.min() < -1e-6 or output.max() > 1 + 1e-6:
            raise ValueError("Enhancer did not produce sigmoid RGB [0,1]")
        return output.clamp(0, 1)
    if not tile_size:
        return forward(tensor)
    output = torch.zeros_like(tensor, device="cpu")
    denominator = torch.zeros((1, 1, h, w), dtype=torch.float32)
    for y in starts(h, tile_size, overlap):
        for x in starts(w, tile_size, overlap):
            crop = tensor[..., y:y+tile_size, x:x+tile_size]
            prediction = forward(crop)
            ch, cw = crop.shape[-2:]
            wy = torch.hann_window(ch, periodic=False).clamp_min(0.01)
            wx = torch.hann_window(cw, periodic=False).clamp_min(0.01)
            weight = (wy[:, None] * wx[None, :])[None, None]
            output[..., y:y+ch, x:x+cw] += prediction * weight
            denominator[..., y:y+ch, x:x+cw] += weight
    if (denominator <= 0).any():
        raise RuntimeError("Tile stitching left uncovered pixels")
    return output.div_(denominator).clamp_(0, 1)


def enhance_test(prepared, out, model, provenance, device="cpu", tile_size=0, overlap=64):
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    meta = load_prepared(prepared)
    fresh_dir(out)
    records = []
    report = {"complete": False, "identity": pair_identity(meta), "provenance": provenance,
              "device": str(device), "tile_size": tile_size, "overlap": overlap if tile_size else 0,
              "padding": "right/bottom to factor 16, reflect or replicate, exact crop; no resize",
              "tile_note": "Tile-local UICF spatial/global conditioning and CQ attention differ from full image; blending is approximate, never claimed numerically equivalent",
              "output_encoding": "8-bit RGB PNG, round(255 * sigmoid output)",
              "versions": versions(), "records": records}
    write_json(out / "enhancement_manifest.json", report)
    try:
        for row in meta["splits"]["test"]:
            start = time.perf_counter()
            record = {"image_id": row["image_id"], "file": row["file"], "status": "failed"}
            try:
                with Image.open(prepared / "images/test_raw" / row["file"]) as image:
                    array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
                result = enhance_tensor(model, tensor, device, tile_size, overlap)
                image = Image.fromarray(result[0].mul(255).round().byte().permute(1, 2, 0).numpy())
                if image.size != (row["width"], row["height"]):
                    raise ValueError("Output geometry differs from source")
                target = out / "images/test_full" / row["file"]
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target)
                # Copy labels byte-for-byte, never transform annotation coordinates.
                label = out / "labels/test_full" / Path(row["file"]).with_suffix(".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_bytes((prepared / "labels/test_raw" / Path(row["file"]).with_suffix(".txt")).read_bytes())
                record.update(status="ok", sha256=sha256(target), label_sha256=sha256(label),
                              width=image.width, height=image.height, seconds=time.perf_counter()-start)
            except Exception as error:
                record["error"] = repr(error)
                records.append(record)
                raise
            records.append(record)
            # Incremental log survives interruption without quadratic JSON serialization.
            with (out / "enhancement_progress.jsonl").open("a", encoding="utf-8") as stream:
                import json
                stream.write(json.dumps(record) + "\n")
        if len(records) != len(meta["splits"]["test"]):
            raise ValueError("Incomplete test enhancement")
        report["complete"] = True
    finally:
        write_json(out / "enhancement_manifest.json", report)
    return report


def validate_enhanced(prepared, enhanced, meta):
    enhanced = Path(enhanced).resolve()
    report = read_json(enhanced / "enhancement_manifest.json")
    if not report["complete"] or report["identity"] != pair_identity(meta):
        raise ValueError("Incomplete enhancement or mismatched test IDs/GT")
    expected = {r["image_id"]: r for r in meta["splits"]["test"]}
    if len(report["records"]) != len(expected) or {r["image_id"] for r in report["records"]} != set(expected):
        raise ValueError("Enhancement count/ID mismatch")
    for record in report["records"]:
        row = expected[record["image_id"]]
        if record["status"] != "ok" or record["file"] != row["file"]:
            raise ValueError("Failed/mismatched enhanced image")
        path = enhanced / "images/test_full" / row["file"]
        label = enhanced / "labels/test_full" / Path(row["file"]).with_suffix(".txt")
        if sha256(path) != record["sha256"] or sha256(label) != row["label_sha256"]:
            raise ValueError("Enhanced image/GT hash mismatch")
        with Image.open(path) as im:
            if im.size != (row["width"], row["height"]) or im.mode != "RGB" or im.format != "PNG":
                raise ValueError("Enhanced output must be same-size RGB PNG")
    actual = {p.relative_to(enhanced / "images/test_full").as_posix()
              for p in (enhanced / "images/test_full").rglob("*") if p.is_file()}
    if actual != {r["file"] for r in expected.values()}:
        raise ValueError("Unexpected/missing enhanced test files")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default=str(OUTPUT / "prepared_duo"))
    p.add_argument("--output", default=str(OUTPUT / "enhanced_full"))
    p.add_argument("--uie-run-dir", required=True)
    p.add_argument("--uie-checkpoint", default="best_psnr", choices=["best_psnr"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tile-size", type=int, default=0, help="0=native; choose before examining test mAP")
    p.add_argument("--overlap", type=int, default=64)
    args = p.parse_args(argv)
    model, provenance = load_v16(args.uie_run_dir, args.uie_checkpoint, args.device)
    enhance_test(args.prepared, output_path(args.output), model, provenance, args.device, args.tile_size, args.overlap)


if __name__ == "__main__":
    main()
