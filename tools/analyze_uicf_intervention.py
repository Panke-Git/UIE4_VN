#!/usr/bin/env python3
"""Frozen canonical V16: intervene on R only, then run the complete QGRD backbone."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import random
import shlex
import sys
from typing import Any, Mapping

import skimage
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.e00 import e00_protocol
from src.shared.uicf_inr import UnderwaterImplicitCorrectionField
from src.v16.dataset import LSUIDataset, validate_split_protocol
from src.v16.test import _checkpoint_path, _torch_load
from src.v16.utils import load_yaml, project_path, select_device, sha256_file, tensor_to_image
from tools.analyze_uicf_alignment import SafeAlignmentDataset, _list_collate, resolve_dataset_name
from tools.visualize_v16_uicf import build_and_load_v16_model
from tools.uicf_intervention.artifacts import (
    OUTPUT_NAME, SCRIPT_ID, SCRIPT_VERSION, OutputTransaction, audit_official_results,
    code_provenance, model_digest, official_result_snapshot, safe_identity,
    save_visualization, strict_json, utc_now, validate_output_path, write_csv, write_paper_tables,
)
from tools.uicf_intervention.core import (
    INTERPRETATION, MAIN_VARIANTS, PERMUTATIONS, VARIANTS, check_tensor,
    intervention_diagnostics, paired_differences, predict_variants, score_prediction,
    shift_pixels, summarize,
)


METRIC_FIELDS = ["sample_index", "sample_id", "filename", "variant", "psnr", "ssim", "e00"]
DIAGNOSTIC_FIELDS = ["sample_index", "sample_id", "filename", "variant", "mean_abs_R", "std_R",
                     *[f"{name}_{color}" for color in "rgb" for name in ("mean_R", "std_R", "b")],
                     "mean_abs_delta_Ic_original", "mean_abs_delta_Ic_variant",
                     "mean_abs_Ic_variant_minus_Ic_original", "mean_abs_Dout_variant_minus_Dout_original",
                     "fraction_clipped_final"]
FAILURE_FIELDS = ["sample_index", "sample_id", "filename", "variant", "error_type", "error"]
DIFFERENCE_FIELDS = ["sample_id", "filename", "variant", "gain_psnr", "gain_ssim", "gain_e00"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Default: frozen config test.checkpoint")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None, help="Dedicated uicf_field_intervention directory only")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=3520)
    parser.add_argument("--viz-count", type=int, default=8)
    parser.add_argument("--viz-seed", type=int, default=3407)
    parser.add_argument("--sample-ids", nargs="+", default=None, help="Explicit visualization IDs only; evaluation stays full")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Default: training.amp; CUDA only, same for every variant")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Transactionally replace only this tool's marked result")
    parser.add_argument("--save-all-outputs", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="DEBUG ONLY: no paper_table or formal summary.json")
    parser.add_argument("--consistency-every", type=int, default=0,
                        help="Always check first successful image; N>0 also checks every Nth manifest image")
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1 or args.viz_count < 0 or args.consistency_every < 0:
        parser.error("bootstrap-samples >= 1, viz-count >= 0 and consistency-every >= 0 required")
    if args.bootstrap_seed < 0 or args.viz_seed < 0:
        parser.error("Seeds must be nonnegative")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("max-samples must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    return args


def validate_evaluation(config: Mapping[str, Any]) -> None:
    evaluation, metrics = config["evaluation"], config["metrics"]
    resize = evaluation.get("resize", True)
    if not isinstance(resize, bool):
        raise ValueError("evaluation.resize must be a YAML boolean")
    border = metrics["crop_border"]
    if not isinstance(border, int) or isinstance(border, bool) or border < 0:
        raise ValueError("metrics.crop_border must be a nonnegative integer")
    if float(metrics["data_range"]) != 1.0:
        raise ValueError("This sRGB evaluation requires metrics.data_range=1.0")
    if resize:
        size = evaluation.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 2 or size <= 2 * border:
            raise ValueError("evaluation.size must be an integer >=2, larger than twice crop_border")


def choose_visualizations(entries: list, args: argparse.Namespace, limit: int) -> list[dict[str, Any]]:
    available = entries[:limit]
    by_id = {entry.sample_id: index for index, entry in enumerate(available)}
    if args.sample_ids is not None:
        if len(args.sample_ids) != len(set(args.sample_ids)):
            raise ValueError("Duplicate --sample-ids")
        missing = set(args.sample_ids) - set(by_id)
        if missing:
            raise ValueError(f"Visualization IDs absent from evaluated manifest: {sorted(missing)}")
        indices = [by_id[sample_id] for sample_id in args.sample_ids]
    else:
        indices = random.Random(args.viz_seed).sample(range(limit), min(args.viz_count, limit))
    return [{"sample_index": index, "sample_id": available[index].sample_id,
             "filename": Path(available[index].input_relative).name,
             "directory": safe_identity(available[index].sample_id)} for index in indices]


def _write_progress(stage: Path, rows: list, diagnostics: list, failures: list) -> None:
    write_csv(stage / "per_sample_metrics.csv", rows, METRIC_FIELDS)
    write_csv(stage / "diagnostics.csv", diagnostics, DIAGNOSTIC_FIELDS)
    write_csv(stage / "failed_samples.csv", failures, FAILURE_FIELDS)
    write_csv(stage / "paired_differences.csv", paired_differences(rows), DIFFERENCE_FIELDS)


def _run_samples(
    model: torch.nn.Module, dataset: LSUIDataset, args: argparse.Namespace,
    config: dict, device: torch.device, stage: Path, protocol: dict,
    selection: list, rows: list, diagnostics: list, failures: list,
) -> None:
    selected = {sample["sample_index"]: sample for sample in selection}
    limit = protocol["requested_sample_count"]
    safe_data = Subset(SafeAlignmentDataset(dataset), range(limit))
    loader = DataLoader(
        safe_data, batch_size=1, shuffle=False, num_workers=protocol["num_workers"],
        pin_memory=bool(config["data"]["pin_memory"]) and device.type == "cuda",
        collate_fn=_list_collate, generator=torch.Generator().manual_seed(args.viz_seed),
    )
    checked_count = 0
    for batch in loader:
        item = batch[0]
        index = int(item["index"])
        entry = dataset.entries[index]
        identity = {"sample_index": index, "sample_id": entry.sample_id,
                    "filename": Path(entry.input_relative).name}
        variant = "load"
        try:
            if not item["ok"]:
                raise RuntimeError(f"{item['error_type']}: {item['error']}")
            # Match native DataLoader's default collate exactly. Unsqueezing the
            # noncontiguous PIL->CHW tensor instead leaves channels-last strides,
            # which can select different convolution kernels and alter rounding.
            inputs = torch.stack([item["input"]]).to(device)
            target = torch.stack([item["target"]]).to(device)
            check_tensor(inputs, "I", shape=(1, 3, *inputs.shape[-2:]), device=device, dtype=torch.float32)
            check_tensor(target, "GT", shape=inputs.shape, device=device, dtype=torch.float32)
            dy, dx = shift_pixels(*inputs.shape[-2:])
            if config["evaluation"].get("resize", True):
                size = config["evaluation"]["size"]
                if tuple(inputs.shape[-2:]) != (size, size):
                    raise RuntimeError("Dataset output does not match frozen PIL bilinear resize protocol")
            check_normal = checked_count == 0 or (
                args.consistency_every > 0 and index % args.consistency_every == 0
            )
            variant = "forward (see tensor/variant in exception)"
            predictions, fields, corrected, details, checks = predict_variants(
                model, inputs, amp=protocol["amp_enabled"], check_normal=check_normal,
            )
            per_rows, per_diagnostics, scores = [], [], {}
            for variant in VARIANTS:
                scores[variant] = score_prediction(predictions[variant], target, config["metrics"])
                per_rows.append({**identity, "variant": variant, **scores[variant]})
                diagnostic = intervention_diagnostics(
                    inputs, fields[variant], corrected[variant], predictions[variant], details, predictions["original"],
                )
                if not all(math.isfinite(value) for value in diagnostic.values()):
                    raise FloatingPointError(f"variant={variant}: non-finite intervention diagnostic")
                per_diagnostics.append({**identity, "variant": variant, **diagnostic})
            if index in selected:
                save_visualization(stage / "visualizations" / selected[index]["directory"],
                                   inputs, target, predictions, details, scores)
            if args.save_all_outputs:
                folder = stage / "all_outputs" / safe_identity(entry.sample_id)
                folder.mkdir(parents=True, exist_ok=False)
                for variant in VARIANTS:
                    tensor_to_image(predictions[variant][0]).save(folder / f"{variant}.png")
            rows.extend(per_rows)
            diagnostics.extend(per_diagnostics)
            protocol["sample_geometry"].append({
                **identity, "height": inputs.shape[-2], "width": inputs.shape[-1],
                "shift_dy": dy, "shift_dx": dx,
                "shift_dy_fraction": dy / inputs.shape[-2], "shift_dx_fraction": dx / inputs.shape[-1],
                "dtypes": {"I": str(inputs.dtype), "R": str(details.correction_field.dtype),
                           "b": str(details.chromatic_anchor.dtype), "Ic": str(details.enhanced.dtype),
                           "Dout": str(predictions["original"].dtype)},
            })
            if checks:
                protocol["normal_forward_consistency"]["samples"].append({**identity, "checks": checks})
                checked_count += 1
        except Exception as error:
            failure = {**identity, "variant": variant, "error_type": type(error).__name__, "error": str(error)}
            failures.append(failure)
            print(f"FAILED sample_id={entry.sample_id} variant={variant}: {error}", file=sys.stderr)
        protocol["successful_sample_count"] = len(rows) // len(VARIANTS)
        protocol["failed_sample_count"] = len(failures)
        _write_progress(stage, rows, diagnostics, failures)
        strict_json(stage / "protocol.json", protocol)
        print(f"intervention: {index + 1}/{limit} success={protocol['successful_sample_count']} failed={len(failures)}")
    if protocol["successful_sample_count"] + len(failures) != limit:
        raise RuntimeError("Not every requested manifest sample was accounted for")
    protocol["normal_forward_consistency"]["status"] = "PASS" if checked_count else "FAIL"
    protocol["normal_forward_consistency"]["max_abs_error"] = max(
        (check["max_abs_error"] for sample in protocol["normal_forward_consistency"]["samples"]
         for check in sample["checks"].values()), default=None,
    )


@torch.inference_mode()
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = project_path(Path(args.run_dir).expanduser()).resolve()
    config_path = run_dir / "config_resolved.yaml"
    config = load_yaml(config_path)
    if config.get("experiment", {}).get("version") != "v16":
        raise ValueError("This tool only accepts experiment.version='v16'; never v16_19/v18/v19")
    validate_evaluation(config)
    dataset_name = resolve_dataset_name(config["data"], context="frozen V16 checkpoint config")
    manifests = {split: run_dir / "split_snapshot" / f"{split}.tsv" for split in ("train", "validation", "test")}
    if (run_dir / "split_snapshot").is_symlink() or any(path.is_symlink() for path in manifests.values()):
        raise ValueError("Frozen split_snapshot must contain real TSV snapshots, not live manifest symlinks")
    entries = validate_split_protocol(manifests, config["data"].get("expected_counts"))
    if any(not group for group in entries.values()):
        raise ValueError("Every frozen train/validation/test manifest must be nonempty")
    data_root = Path(args.data_root or config["data"]["root"]).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data.root unavailable: {data_root}; provide --data-root")
    for entry in entries["test"]:
        for relative in (entry.input_relative, entry.gt_relative):
            if not (data_root / relative).resolve().is_relative_to(data_root):
                raise ValueError(f"Test manifest path escapes data.root: sample_id={entry.sample_id}")
    data = config["data"]
    dataset = LSUIDataset(
        manifests["test"], data_root, "test", int(data["patch_size"]), data["augmentation"],
        bool(data["pad_if_smaller"]), str(data["pad_mode"]), config["evaluation"], verify_files=False,
    )  # SafeAlignmentDataset records load failures for every image instead of silently skipping.
    if dataset.entries != entries["test"]:
        raise RuntimeError("Dataset differs from validated frozen test manifest")
    selector = args.checkpoint or config["test"]["checkpoint"]
    checkpoint_path = _checkpoint_path(run_dir, selector).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    output = (run_dir / "result" / OUTPUT_NAME if args.output_dir is None
              else project_path(Path(args.output_dir).expanduser()))
    output = validate_output_path(output, run_dir=run_dir, data_root=data_root, overwrite=args.overwrite)
    device = select_device(args.gpu)
    checkpoint = _torch_load(checkpoint_path, device)
    # Only the established strict V16 builder/load path is used. Constructor RNG
    # is restored afterwards; no randomly initialized weight participates in inference.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        model = build_and_load_v16_model(config, checkpoint, device)
    if not isinstance(model.uicf, UnderwaterImplicitCorrectionField) or any(
        not getattr(model.uicf, flag) for flag in (
            "use_spatial_conditioning", "use_global_field_conditioning", "use_learned_anchor")
    ):
        raise ValueError("This experiment requires canonical V16 UICF, not a field-control ablation")
    if checkpoint.get("resolved_config", {}).get("experiment", {}).get("version", "v16") != "v16":
        raise ValueError("Checkpoint resolved experiment version is not v16")
    saved_data = checkpoint.get("resolved_config", {}).get("data", {})
    saved_dataset = str(saved_data.get("dataset", "")).strip()
    if saved_dataset and saved_dataset != dataset_name:
        raise ValueError("Checkpoint dataset identity differs from frozen run dataset")
    if saved_data.get("expected_counts") is not None and saved_data["expected_counts"] != data.get("expected_counts"):
        raise ValueError("Checkpoint dataset expected_counts differ from frozen run protocol")
    for key in ("evaluation", "metrics"):
        saved = checkpoint.get("resolved_config", {}).get(key)
        if saved is not None and saved != config[key]:
            raise ValueError(f"Checkpoint frozen {key} differs from run config")
    model.requires_grad_(False)
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.is_floating_point():
            check_tensor(tensor, f"checkpoint tensor {name}")
    initial_model_digest = model_digest(model)
    protected_files = {str(path): sha256_file(path) for path in (config_path, checkpoint_path, *manifests.values())}
    old_results = official_result_snapshot(run_dir)
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    selection = choose_visualizations(entries["test"], args, limit)
    debug = args.max_samples is not None  # Even max_samples >= N stays explicitly debug.
    amp_requested = bool(config["training"]["amp"]) if args.amp is None else args.amp
    num_workers = int(data["num_workers"]) if args.num_workers is None else args.num_workers
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    protocol = {
        "script_id": SCRIPT_ID, "script_version": SCRIPT_VERSION,
        "script": "tools/analyze_uicf_intervention.py", "status": "RUNNING",
        "started_utc": utc_now(), "command": shlex.join([sys.executable, "tools/analyze_uicf_intervention.py",
                                                           *(sys.argv[1:] if argv is None else argv)]),
        "run_dir": str(run_dir), "output_dir": str(output), "code": code_provenance(),
        "checkpoint": str(checkpoint_path), "checkpoint_selector": selector,
        "checkpoint_epoch": int(checkpoint["epoch"]), "checkpoint_sha256": protected_files[str(checkpoint_path)],
        "config_resolved_yaml": str(config_path), "config_resolved_sha256": protected_files[str(config_path)],
        "manifests": {name: {"path": str(path), "sha256": protected_files[str(path)],
                             "sample_count": len(entries[name])} for name, path in manifests.items()},
        "test_manifest": str(manifests["test"]), "test_manifest_sha256": protected_files[str(manifests["test"])],
        "dataset": dataset_name, "checkpoint_dataset": dataset_name, "evaluation_dataset": dataset_name,
        "evaluation_mode": "in_domain", "data_root": str(data_root),
        "expected_counts": config["data"].get("expected_counts"),
        "train_count": len(entries["train"]), "validation_count": len(entries["validation"]),
        "test_count": len(entries["test"]), "full_expected_sample_count": len(dataset),
        "requested_sample_count": limit, "successful_sample_count": 0, "failed_sample_count": 0,
        "debug_partial": debug, "complete_formal_evaluation": False,
        "batch_size": 1, "num_workers": num_workers, "device": str(device), "device_type": device.type,
        "amp_requested": amp_requested, "amp_enabled": bool(amp_requested and device.type == "cuda"),
        "amp_dtype": str(torch.get_autocast_dtype(device.type)) if amp_requested and device.type == "cuda" else None,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "cuda": torch.version.cuda, "scikit_image": skimage.__version__,
                        "cudnn": torch.backends.cudnn.version(),
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_tf32": torch.backends.cudnn.allow_tf32},
        "evaluation": config["evaluation"], "resize_implementation": "LSUIDataset PIL RGB bilinear; input and GT",
        "metrics": config["metrics"], "e00_protocol": e00_protocol(config["metrics"]),
        "metric_input": "Dout.float().clamp(0,1) before PNG; no Ic clamp",
        "metric_aggregation": "per-image metric then equally weighted image arithmetic mean",
        "psnr_ssim_implementation": "src.v16.metrics.batch_metrics (unchanged)",
        "interventions": {"main": list(MAIN_VARIANTS), "supplementary": list(VARIANTS[4:]),
                          "original": "R unchanged; use details.enhanced and diagnostic prediction directly",
                          "spatial_mean": "R.mean((-2,-1), keepdim=True).expand_as(R), independently per B,C",
                          "shift_xy": "roll(R, (max(1,H//4),max(1,W//4)), (-2,-1))",
                          "shift_y": "roll R by (max(1,H//4),0)",
                          "shift_x": "roll R by (0,max(1,W//4))",
                          "permutations_new_RGB_from_old_indices": PERMUTATIONS,
                          "zero_field": "zeros_like(R); in-model deactivation only",
                          "Ic": "I + R_variant * (I - original b)",
                          "Dout": "unchanged model.backbone(Ic_variant)",
                          "transformed": "R only; never I, b, GT, tokens or R*(I-b)",
                          "no_multi_shift_aggregate": True},
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed,
                      "unit": "image", "interval": "percentile 2.5%,97.5%; unadjusted pointwise",
                      "seed_stream": "seed + 3*nonoriginal_variant_index + metric_index"},
        "visualization": {"seed": args.viz_seed, "selection": "pre-inference fixed random indices or explicit IDs",
                          "save_all_outputs": args.save_all_outputs},
        "sample_geometry": [], "normal_forward_consistency": {"status": "PENDING", "samples": [],
            "every": args.consistency_every, "fp32_atol": 1e-6, "fp32_rtol": 1e-5,
            "amp_atol": 1e-5, "amp_rtol": 1e-4,
            "uicf_formula_atol": 1e-6, "uicf_formula_rtol": 1e-5,
            "extra_normal_forward": "One independent full-model audit per checked sample; R/b from that audit are discarded"},
        "frozen_model_digest_before": initial_model_digest, "interpretation": INTERPRETATION,
    }
    transaction = OutputTransaction(output, run_dir=run_dir, data_root=data_root, overwrite=args.overwrite)
    stage = transaction.stage
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        (stage / "visualizations").mkdir()
        strict_json(stage / "visualizations/selection.json", {
            "seed": args.viz_seed, "selection_before_inference": True, "samples": selection,
        })
        _write_progress(stage, rows, diagnostics, failures)
        strict_json(stage / "protocol.json", protocol)
        _run_samples(model, dataset, args, config, device, stage, protocol, selection, rows, diagnostics, failures)
        protocol["frozen_model_digest_after"] = model_digest(model)
        if protocol["frozen_model_digest_after"] != initial_model_digest:
            raise RuntimeError("Frozen model parameter/buffer state changed during inference")
        if any(sha256_file(Path(path)) != digest for path, digest in protected_files.items()):
            raise RuntimeError("Checkpoint/config/manifest bytes changed during experiment")
        if official_result_snapshot(run_dir) != old_results:
            raise RuntimeError("Official test results changed during experiment")
        protocol["protected_files_unchanged"] = True
        if failures or protocol["normal_forward_consistency"]["status"] != "PASS":
            raise RuntimeError(f"Incomplete experiment: successful={len(rows)//len(VARIANTS)} failed={len(failures)}")
        official_audit = audit_official_results(run_dir, rows, config, protocol, debug_partial=debug)
        protocol["official_test_audit"] = official_audit
        strict_json(stage / "official_test_audit.json", official_audit)
        if official_audit["status"] != "PASS":
            print(f"Official test audit: {official_audit['status']}; "
                  f"reasons={official_audit.get('reasons', [])}; "
                  f"unverified_metadata={official_audit.get('unverified_metadata', [])}")
        if official_audit["status"] == "FAIL":
            raise RuntimeError(f"Official original-result reproduction FAILED: {official_audit['reasons']}")
        protocol.update({"status": "DEBUG_PARTIAL" if debug else "PASS", "finished_utc": utc_now(),
                         "complete_formal_evaluation": not debug})
        summary = {**summarize(rows, samples=args.bootstrap_samples, seed=args.bootstrap_seed),
                   "status": protocol["status"], "dataset": dataset_name, "debug_partial": debug,
                   "paper_ready": not debug, "full_test_sample_count": len(dataset),
                   "successful_sample_count": len(rows) // len(VARIANTS), "failed_sample_count": 0,
                   "official_test_audit": official_audit,
                   "checkpoint_sha256": protocol["checkpoint_sha256"]}
        strict_json(stage / ("debug_summary.json" if debug else "summary.json"), summary)
        if not debug:
            write_paper_tables(stage, summary)
        strict_json(stage / "protocol.json", protocol)
        strict_json(stage / "COMPLETED.json", {"status": protocol["status"], "paper_ready": not debug})
        transaction.commit()
    except BaseException as error:
        # No paper-ready files are emitted before all scientific gates pass. If
        # export/commit itself failed, explicitly invalidate any staged table too.
        for name in ("paper_table.csv", "paper_table.md", "COMPLETED.json", "summary.json"):
            path = stage / name
            if path.is_file():
                path.rename(stage / f"INVALID_{name}")
        protocol.update({"status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL",
                         "complete_formal_evaluation": False, "finished_utc": utc_now(),
                         "error_type": type(error).__name__, "error": str(error)})
        _write_progress(stage, rows, diagnostics, failures)
        strict_json(stage / "protocol.json", protocol)
        strict_json(stage / "failure_summary.json", {"status": protocol["status"], "paper_ready": False,
                                                     "error": str(error), "failed_sample_count": len(failures)})
        retained = transaction.retain_failure()
        print(f"FAILED; no new formal result. Audit retained at {retained}", file=sys.stderr)
        raise
    print(json.dumps({"status": protocol["status"], "output": str(output), "paper_ready": not debug,
                      "previous_result_archive": None if transaction.previous is None else str(transaction.previous)}))


if __name__ == "__main__":
    main()
