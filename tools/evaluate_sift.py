#!/usr/bin/env python3
"""Evaluate V16 with aligned GT or overlapping scene pairs; see README_SIFT.md."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.sift_eval.matching import Settings, match_images, match_panel, protocol, stack_panels, summarize
from src.shared.transactional_output import transactional_output_directory
from src.shared.paper_evaluation import official_enhanced_names
from src.v16.dataset import read_manifest


@dataclass(frozen=True)
class Case:
    pair_id: str
    image_a: Path
    image_b: Path
    enhanced_a: Path | None = None
    enhanced_b: Path | None = None


def path_arg(value):
    return Path(value).expanduser().resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gt", "pairs"))
    parser.add_argument("--run-dir", required=True, type=path_arg)
    parser.add_argument("--data-root", type=path_arg, help="WSL dataset root; overrides the old server path")
    parser.add_argument("--checkpoint", help="best_psnr, best_ssim, best_loss, last, or a checkpoint path")
    parser.add_argument("--prediction-dir", type=path_arg, help="Use existing PNG outputs instead of running V16")
    parser.add_argument("--output-dir", type=path_arg)
    parser.add_argument("--manifest", type=path_arg, help="GT TSV override; default: run/split_snapshot/test.tsv")
    parser.add_argument("--pairs", type=path_arg, help="Pairs CSV: pair_id,image_a,image_b[,enhanced_a,enhanced_b]")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (WSL RTX 3060) or cpu")
    parser.add_argument("--amp", action="store_true", help="Opt-in float16 CUDA inference; recorded in summary")
    parser.add_argument("--resolution", choices=("config", "native"), default="config",
                        help="config: reproduce run evaluation resize; native: keep source size")
    parser.add_argument("--max-side", type=int, default=0,
                        help="With --resolution native, shrink long edge to this size, preserving aspect ratio")
    parser.add_argument("--geometry", choices=("homography", "fundamental"), default="homography",
                        help="Pairs only: homography for approximately planar scenes; fundamental for 3D views")
    parser.add_argument("--threshold-px", type=float, default=3.0, help="Geometric threshold at evaluation resolution")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--no-mutual", action="store_true", help="Disable reciprocal ratio matching")
    parser.add_argument("--nfeatures", type=int, default=0, help="SIFT feature cap, 0: unlimited")
    parser.add_argument("--contrast-threshold", type=float, default=0.04)
    parser.add_argument("--edge-threshold", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, help="First N manifest pairs, for smoke runs; result marked partial")
    parser.add_argument("--num-visualizations", type=int, default=6)
    parser.add_argument("--max-draw", type=int, default=0, help="Display cap only; default 0 draws all verified matches")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this tool's output directory")
    args = parser.parse_args(argv)
    if args.max_side < 0 or args.max_draw < 0 or args.num_visualizations < 0 or args.threads < 1:
        parser.error("max-side, max-draw, num-visualizations must be >= 0; threads must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.max_side and args.resolution != "native":
        parser.error("--max-side requires --resolution native")
    if args.mode == "pairs" and (args.pairs is None or args.manifest is not None):
        parser.error("pairs mode requires --pairs and does not accept --manifest")
    if args.mode == "gt" and args.pairs is not None:
        parser.error("--pairs is only available in pairs mode")
    if args.prediction_dir and (args.checkpoint or args.amp):
        parser.error("--checkpoint/--amp apply to inference; omit them when using --prediction-dir")
    return args


def load_cases(args, config):
    root = args.data_root or path_arg(config.get("data", {}).get("root", "."))
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset unavailable: {root}. Pass --data-root using a WSL path.")
    if args.mode == "gt":
        manifest = args.manifest or args.run_dir / "split_snapshot/test.tsv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing frozen test manifest: {manifest}. Restore it or pass --manifest explicitly.")
        entries = read_manifest(manifest)
        expected = config.get("data", {}).get("expected_counts", {}).get("test")
        if args.manifest is None and expected is not None and len(entries) != int(expected):
            raise ValueError(f"Frozen test manifest count {len(entries)} != expected {expected}")
        names = official_enhanced_names(entries)
        cases = [Case(e.sample_id, (root / e.input_relative).resolve(), (root / e.gt_relative).resolve(),
                      (args.prediction_dir / names[e.sample_id]) if args.prediction_dir else None)
                 for e in entries]
        if len({c.image_a for c in cases}) != len(cases):
            raise ValueError("Duplicate inputs in GT manifest")
        if len({c.image_b for c in cases}) != len(cases):
            raise ValueError("Duplicate references in GT manifest")
    else:
        manifest = args.pairs
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"pair_id", "image_a", "image_b"}
            if args.prediction_dir:
                required.update(("enhanced_a", "enhanced_b"))
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f"Pairs CSV requires columns: {sorted(required)}")
            cases = []
            for number, row in enumerate(reader, 2):
                if any(not (row.get(key) or "").strip() for key in required):
                    raise ValueError(f"{manifest}:{number}: missing required value")
                cases.append(Case(
                    row["pair_id"].strip(), (root / row["image_a"].strip()).resolve(),
                    (root / row["image_b"].strip()).resolve(),
                    (args.prediction_dir / row["enhanced_a"].strip()).resolve() if args.prediction_dir else None,
                    (args.prediction_dir / row["enhanced_b"].strip()).resolve() if args.prediction_dir else None,
                ))
        identities = [frozenset((c.image_a, c.image_b)) for c in cases]
        if any(len(identity) != 2 for identity in identities) or len(set(identities)) != len(cases):
            raise ValueError("Scene pairs must use distinct images and cannot duplicate/reverse an existing pair")
    if not cases or len({c.pair_id for c in cases}) != len(cases):
        raise ValueError("Manifest must have at least one pair and unique pair IDs")
    total = len(cases)
    cases = cases[:args.limit] if args.limit else cases
    for case in cases:
        for path in (case.image_a, case.image_b, case.enhanced_a, case.enhanced_b):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"pair={case.pair_id}: missing image {path}")
    return cases, root, manifest, total


def load_model(args, config):
    import torch
    from src.v16.models import build_model

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. In WSL check nvidia-smi and your CUDA PyTorch installation, or use --device cpu.")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    elif device.type != "cpu":
        raise ValueError("Supported devices: cpu or cuda:N")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    selector = args.checkpoint or config.get("test", {}).get("checkpoint", "best_psnr")
    if selector in {"best_psnr", "best_ssim", "best_loss", "last"}:
        checkpoint_path = args.run_dir / ("checkpoint/last.pt" if selector == "last" else f"best/{selector}.pt")
    else:
        checkpoint_path = Path(selector).expanduser()
        if not checkpoint_path.is_absolute():
            candidates = [args.run_dir / checkpoint_path, path_arg(selector)]
            checkpoint_path = next((p for p in candidates if p.is_file()), candidates[0])
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != "v16" or checkpoint.get("resolved_config", {}).get("model") != config["model"]:
        raise ValueError("Checkpoint V16 version/architecture differs from config_resolved.yaml")
    torch.manual_seed(args.seed)
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, device, {
        "checkpoint_loaded": True, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "epoch": checkpoint.get("epoch"),
        "device": str(device), "amp": args.amp, "batch_size": 1,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
    }


def resize_source(image, args, config):
    if args.resolution == "config":
        evaluation = config.get("evaluation", {})
        if evaluation.get("resize", True):
            size = int(evaluation["size"])
            return image.resize((size, size), Image.Resampling.BILINEAR)
    elif args.max_side and max(image.size) > args.max_side:
        scale = args.max_side / max(image.size)
        return image.resize(tuple(max(1, round(v * scale)) for v in image.size), Image.Resampling.BILINEAR)
    return image


def read_rgb(path):
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image, found {image.mode}: {path}")
        image.load()
        return image.copy()


class Images:
    def __init__(self, args, config, staging, model, device):
        self.args, self.config, self.staging = args, config, staging
        self.model, self.device = model, device
        self.generated, self.sources = {}, {}

    def source(self, path):
        image = read_rgb(path)
        original_size = image.size
        image = resize_source(image, self.args, self.config)
        key = str(path)
        if key not in self.sources:
            self.sources[key] = {"sha256": sha256(path), "original_size_wh": list(original_size),
                                 "evaluation_size_wh": list(image.size)}
        return image

    def enhanced(self, path, image, prediction_path):
        if prediction_path is not None:
            output = read_rgb(prediction_path)
            if output.size != image.size:
                raise ValueError(f"Prediction size {output.size} != evaluation size {image.size}: {prediction_path}. "
                                 "Select the resolution used to generate these predictions; outputs are never resized.")
            key = str(prediction_path)
            if key not in self.sources:
                self.sources[key] = {"sha256": sha256(prediction_path), "role": "provided_prediction",
                                     "evaluation_size_wh": list(output.size)}
            return output
        if path not in self.generated:
            import torch
            array = np.array(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
            try:
                with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.args.amp):
                    prediction = self.model(tensor)
                if prediction.shape != tensor.shape or not torch.isfinite(prediction).all():
                    raise ValueError(f"V16 output shape/nonfinite error: {path}")
                prediction = prediction[0].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError("CUDA out of memory. Use --resolution config (normally 256), or reduce "
                                   "--max-side with --resolution native, or opt into --amp. No automatic resizing was applied.") from error
            key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
            destination = self.staging / "enhanced" / f"{key}.png"
            destination.parent.mkdir(exist_ok=True)
            Image.fromarray(prediction).save(destination)
            self.generated[path] = destination
            self.sources[str(path)]["generated_prediction"] = {
                "path": str(destination.relative_to(self.staging)), "sha256": sha256(destination),
            }
        return read_rgb(self.generated[path])


def main(argv=None):
    args = parse_args(argv)
    settings = Settings(
        ratio=args.ratio, mutual=not args.no_mutual, nfeatures=args.nfeatures,
        contrast_threshold=args.contrast_threshold, edge_threshold=args.edge_threshold,
        sigma=args.sigma, threshold_px=args.threshold_px, seed=args.seed,
    )
    cv2.setNumThreads(args.threads)
    cv2.ocl.setUseOpenCL(False)
    config_path = args.run_dir / "config_resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment", {}).get("version") != "v16":
        raise ValueError("Expected canonical src/v16 run, not v16_19 or another model version")
    cases, root, manifest, total = load_cases(args, config)
    output = (args.output_dir or args.run_dir / "result" / f"sift_{args.mode}").resolve()
    protected = [config_path, manifest, args.run_dir, root]
    protected += [p for case in cases for p in (case.image_a, case.image_b, case.enhanced_a, case.enhanced_b)
                  if p is not None]
    if any(path.resolve().is_relative_to(output) for path in protected):
        raise ValueError("Output directory would replace source data or the model run; choose a separate result directory")
    if output.exists() and args.overwrite and not (output / "sift_evaluation.json").is_file():
        raise ValueError("--overwrite only replaces an existing SIFT result containing sift_evaluation.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output}. Choose a new path or use --overwrite for an existing SIFT result.")
    model, device = None, None
    provenance = {"checkpoint_loaded": False, "prediction_dir": str(args.prediction_dir),
                  "note": "Provided predictions; checkpoint association is supplied by the user, not verified"}
    if args.prediction_dir is None:
        model, device, provenance = load_model(args, config)
        if Path(provenance["checkpoint"]).is_relative_to(output):
            raise ValueError("Output directory would replace the selected checkpoint")
    geometry = "identity" if args.mode == "gt" else args.geometry
    count = min(args.num_visualizations, len(cases))
    visual_indices = set(map(int, np.random.default_rng(args.seed).choice(len(cases), count, replace=False)))
    rows, overview = [], []
    with transactional_output_directory(output, overwrite=args.overwrite) as staging:
        (staging / "matches").mkdir()
        (staging / "figures").mkdir()
        images = Images(args, config, staging, model, device)
        for index, case in enumerate(cases):
            raw_a, raw_b = images.source(case.image_a), images.source(case.image_b)
            if args.mode == "gt" and (raw_a.size != raw_b.size or
                    images.sources[str(case.image_a)]["original_size_wh"] !=
                    images.sources[str(case.image_b)]["original_size_wh"]):
                raise ValueError(f"Aligned GT requires equal dimensions: {case.pair_id}")
            enhanced_a = images.enhanced(case.image_a, raw_a, case.enhanced_a)
            enhanced_b = raw_b if args.mode == "gt" else images.enhanced(case.image_b, raw_b, case.enhanced_b)
            panels = []
            for condition, a, b in (("raw", raw_a, raw_b), ("v16", enhanced_a, enhanced_b)):
                result = match_images(a, b, geometry, settings)
                stem = f"{index:05d}_{condition}"
                result.save(staging / "matches" / f"{stem}.npz")
                rows.append({"pair_id": case.pair_id, "condition": condition, "geometry": geometry,
                             "image_a": str(case.image_a), "image_b": str(case.image_b),
                             "provided_enhanced_a": str(case.enhanced_a) if case.enhanced_a else "",
                             "provided_enhanced_b": str(case.enhanced_b) if case.enhanced_b else "",
                             "width_a": a.width, "height_a": a.height,
                             "width_b": b.width, "height_b": b.height,
                             "correspondences": f"matches/{stem}.npz", **result.metrics})
                if index in visual_indices:
                    label = ("Input vs GT" if condition == "raw" else "V16 vs GT") if args.mode == "gt" else condition.upper()
                    panel = match_panel(a, b, result, f"Pair {index:05d}: {label}", max_draw=args.max_draw, seed=args.seed)
                    panel.save(staging / "figures" / f"{stem}.png", dpi=(300, 300))
                    panels.append(panel)
            if panels:
                combined = stack_panels(panels)
                combined.save(staging / "figures" / f"{index:05d}_comparison.png", dpi=(300, 300))
                if len(overview) < 6:
                    thumbnail = combined.copy()
                    thumbnail.thumbnail((600, 600))
                    overview.append(thumbnail)
            print(f"[{index + 1}/{len(cases)}] {case.pair_id}: verified "
                  f"{rows[-2]['verified_matches']} -> {rows[-1]['verified_matches']}", flush=True)
        if overview:
            width, height = max(p.width for p in overview), max(p.height for p in overview)
            columns = min(3, len(overview))
            grid = Image.new("RGB", (width * columns, height * ((len(overview) + columns - 1) // columns)), "white")
            for i, panel in enumerate(overview):
                grid.paste(panel, ((i % columns) * width, (i // columns) * height))
            grid.save(staging / "figures/overview.png", dpi=(300, 300))
        with (staging / "per_pair_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        scores = summarize(rows)
        summary = {
            "complete": True, "mode": args.mode, "protocol": protocol(settings, geometry),
            "interpretation": "Aligned reference feature consistency" if args.mode == "gt" else "Overlapping-view feature matching",
            "scope": {"evaluated_pairs": len(cases), "manifest_pairs": total,
                      "partial": len(cases) < total, "manifest": str(manifest), "manifest_sha256": sha256(manifest)},
            "run_dir": str(args.run_dir), "data_root": str(root),
            "config_sha256": sha256(config_path), "inference": provenance,
            "resolution": {"policy": args.resolution, "max_side": args.max_side,
                           "run_evaluation": config.get("evaluation"), "prediction_resize": "never",
                           "output_encoding": "RGB PNG8 round(255*clamp(prediction,0,1)) for generated predictions"},
            "software": {"python": platform.python_version(), "platform": platform.platform(),
                         "opencv": cv2.__version__, "numpy": np.__version__, "pillow": Image.__version__,
                         "opencv_threads": args.threads, "opencl": False},
            "visualization": {"selected_pair_indices": sorted(visual_indices), "max_draw": args.max_draw,
                              "selection": "random seed fixed before evaluation; no score-based selection",
                              "lines": "geometrically verified matches only; display cap never changes scores"},
            "metrics": scores,
        }
        write_json(staging / "sources.json", images.sources)
        write_json(staging / "sift_evaluation.json", summary)
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")
    return summary


if __name__ == "__main__":
    main()
