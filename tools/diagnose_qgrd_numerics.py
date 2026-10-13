#!/usr/bin/env python3
"""A2 numerical diagnostics without writes to the original run or source package.

Inspection is read-only. --replay-next-epoch explicitly performs exactly one
training epoch IN MEMORY, using the saved protocol/optimizer/scaler/RNG. No
checkpoint, experiment log, metric result, config, or status is saved.
"""
from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_VERSION = "qgrd_shared_query"
CHUNK_ELEMENTS = 1_048_576


def floating_tensors(value: Any, name: str):
    if torch.is_tensor(value):
        if value.is_floating_point():
            yield name, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from floating_tensors(child, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from floating_tensors(child, f"{name}[{index}]")


def tensor_chunks(value: Tensor):
    """Bound diagnostic temporaries, including for non-contiguous feature maps."""
    if value.numel() <= CHUNK_ELEMENTS:
        yield value.reshape(-1)
    elif value.shape[0] == 1:
        yield from tensor_chunks(value[0])
    else:
        elements_per_item = value.numel() // value.shape[0]
        step = max(1, CHUNK_ELEMENTS // elements_per_item)
        for start in range(0, value.shape[0], step):
            yield from tensor_chunks(value[start:start + step])


def all_finite(value: Tensor) -> bool:
    return all(bool(torch.isfinite(part).all()) for part in tensor_chunks(value))


@torch.no_grad()
def tensor_statistics(name: str, value: Tensor) -> dict:
    """Finite-only float64 statistics; safe even if every element is NaN/Inf."""
    result = {"tensor_name": name, "shape": list(value.shape), "dtype": str(value.dtype),
              "device": str(value.device), "num_nan": 0, "num_posinf": 0,
              "num_neginf": 0, "finite_count": 0, "finite_min": None,
              "finite_max": None, "finite_abs_max": None, "finite_mean": None,
              "finite_std": None}
    count, mean, m2 = 0, 0.0, 0.0
    finite_sums = []
    for part in tensor_chunks(value.detach()):
        # CPU double avoids reduced-precision statistics under surrounding AMP.
        array = part.to(device="cpu", dtype=torch.float64)
        result["num_nan"] += int(torch.isnan(array).sum())
        result["num_posinf"] += int(torch.isposinf(array).sum())
        result["num_neginf"] += int(torch.isneginf(array).sum())
        finite = array[torch.isfinite(array)]
        if finite.numel() == 0:
            continue
        n = finite.numel()
        finite_sums.append(float(finite.sum()))
        lo, hi = float(finite.min()), float(finite.max())
        result["finite_min"] = lo if result["finite_min"] is None else min(lo, result["finite_min"])
        result["finite_max"] = hi if result["finite_max"] is None else max(hi, result["finite_max"])
        abs_max = max(abs(lo), abs(hi))
        result["finite_abs_max"] = max(result["finite_abs_max"] or 0.0, abs_max)
        chunk_mean = float(finite.mean())
        chunk_m2 = float(((finite - chunk_mean) ** 2).sum())
        delta = chunk_mean - mean
        m2 += chunk_m2 + delta * delta * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    result["finite_count"] = count
    if count:
        result["finite_mean"] = math.fsum(finite_sums) / count
        result["finite_std"] = math.sqrt(max(0.0, m2 / count))
    return result


def tree_issues(value: Any, name: str) -> list[dict]:
    return [tensor_statistics(path, tensor) for path, tensor in floating_tensors(value, name)
            if not all_finite(tensor)]


@torch.no_grad()
def finite_max_abs(value: Tensor) -> float | None:
    maximum = None
    for part in tensor_chunks(value.detach()):
        finite = part[torch.isfinite(part)]
        if finite.numel():
            item = float(finite.abs().max())
            maximum = item if maximum is None else max(maximum, item)
    return maximum


def model_summary(model: nn.Module, optimizer=None) -> dict:
    issues = tree_issues(model.state_dict(), "model_state_dict")
    optimizer_issues = tree_issues(optimizer.state_dict(), "optimizer_state_dict") if optimizer is not None else []
    norms = []
    for name, tensor in floating_tensors(model.state_dict(), ""):
        norms.append({"tensor_name": name.lstrip("."), "max_abs": finite_max_abs(tensor),
                      "finite": all_finite(tensor)})
    norms.sort(key=lambda r: r["max_abs"] if r["max_abs"] is not None else -math.inf, reverse=True)
    bn = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            row = {"module": name, "eps": module.eps}
            for key in ("running_mean", "running_var"):
                tensor = getattr(module, key)
                row[key] = tensor_statistics(f"{name}.{key}", tensor) if tensor is not None else None
            var = row["running_var"]
            row["negative_running_variance"] = bool(var is not None and var["finite_min"] is not None
                                                    and var["finite_min"] < 0)
            bn.append(row)
    selected = {}
    for name, tensor in model.state_dict().items():
        if name == "backbone.base_queries" or any(
            name.startswith(f"backbone.guide{stage}.") for stage in (4, 3, 2, 1)
        ):
            if tensor.is_floating_point():
                selected[name] = tensor_statistics(name, tensor)
    return {"model_nonfinite_tensors": issues, "optimizer_nonfinite_tensors": optimizer_issues,
            "model_max_abs_top30": norms[:30], "batch_norm": bn,
            "base_query_and_guidance_state": selected,
            "invalid": bool(issues or optimizer_issues or any(r["negative_running_variance"] for r in bn))}


class BoundaryTrace:
    """Observational hooks only: no replacements, exceptions, or output edits.

    First detected *module boundary*, not a claim about hidden fused-kernel
    intermediates. Does not retain activations/graphs after a hook returns.
    """
    def __init__(self, phase: str):
        self.phase = phase
        self.context = {}
        self.first_nonfinite = None
        self.recent = deque(maxlen=16)
        self.active_modules = []
        self.handles = []

    def set_batch(self, step, ids):
        self.context = {"step": int(step), "sample_ids": [str(i) for i in ids]}

    def check(self, name, value, module=None, inputs=None):
        tensors = list(floating_tensors(value, name))
        if self.first_nonfinite is not None:
            return
        for path, tensor in tensors:
            if not all_finite(tensor):
                self.first_nonfinite = {
                    "phase": self.phase, **self.context,
                    "tensor": tensor_statistics(path, tensor),
                    "last_finite_boundaries": list(self.recent),
                    "active_modules": list(self.active_modules),
                }
                if inputs is not None:
                    self.first_nonfinite["inputs"] = [tensor_statistics(n, t)
                                                     for n, t in floating_tensors(inputs, name + ".inputs")]
                if module is not None:
                    self.first_nonfinite["module_type"] = type(module).__name__
                    self.first_nonfinite["module_state"] = [tensor_statistics(n, t)
                        for n, t in floating_tensors(module.state_dict(), name + ".state")]
                print("FIRST_NONFINITE_BOUNDARY " + json.dumps(self.first_nonfinite, allow_nan=False), flush=True)
                return
        self.recent.extend({"tensor_name": n, "shape": list(t.shape),
                            "dtype": str(t.dtype), "device": str(t.device)} for n, t in tensors)

    def __enter__(self):
        return self

    def attach(self, model):
        for name, module in model.named_modules():
            label = name or "<model>"
            def pre_hook(mod, inputs, label=label):
                self.active_modules.append(label)
                self.check(label + ".input", inputs, mod)
            def post_hook(mod, inputs, output, label=label):
                self.check(label + ".output", output, mod, inputs)
                if self.active_modules and self.active_modules[-1] == label:
                    self.active_modules.pop()
            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@dataclass
class Session:
    config: dict
    checkpoint: dict
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: Any
    device: torch.device
    runtime: Any
    train_loader: DataLoader | None = None
    validation_loader: DataLoader | None = None
    generator: torch.Generator | None = None

    @property
    def amp_enabled(self):
        return bool(self.config["training"]["amp"]) and self.device.type == "cuda"


def runtime():
    from types import SimpleNamespace
    prefix = "src.qgrd_ablations.shared_query"
    return SimpleNamespace(**{name: importlib.import_module(prefix + "." + name)
                              for name in ("train", "test", "dataset", "models", "losses", "utils", "provenance")})


def validate_scaler_metadata(state: dict) -> None:
    for key, value in state.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError(f"Saved AMP scaler metadata {key!r} is non-finite")
    if "scale" in state and float(state["scale"]) <= 0:
        raise ValueError("Saved AMP scaler scale must be positive")


def load_session(run_dir: Path, selector: str, gpu: int | None, data_root: str | None,
                 with_loaders: bool, expected_epoch: int | None = None) -> Session:
    rt = runtime()
    config = rt.utils.load_yaml(run_dir / "config_resolved.yaml")
    rt.provenance.validate_config(config)
    sources = rt.provenance.verify_source_hashes()
    manifests = {s: run_dir / "split_snapshot" / f"{s}.tsv" for s in ("train", "validation", "test")}
    rt.provenance.verify_run_metadata(run_dir, config, sources, manifests)
    checkpoint_path = rt.test._checkpoint_path(run_dir, selector).resolve()
    if not checkpoint_path.is_relative_to(run_dir):
        raise ValueError("Diagnostic checkpoint must belong to the selected run")
    checkpoint = rt.train._torch_load(checkpoint_path, torch.device("cpu"))
    rt.provenance.verify_checkpoint(checkpoint, config, sources)
    if expected_epoch is not None and int(checkpoint["epoch"]) != expected_epoch:
        raise ValueError(f"Expected checkpoint epoch {expected_epoch}, got {checkpoint['epoch']}")
    if data_root is not None:
        config = rt.utils.apply_overrides(config, data_root=data_root)
    rt.utils.seed_everything(int(config["experiment"]["seed"]), bool(config["training"]["deterministic"]))
    device = rt.utils.select_device(gpu) if with_loaders else torch.device("cpu")
    model = rt.models.build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = rt.losses.build_loss(config["loss"]).to(device)
    optim = config["optimizer"]
    if optim["name"].lower() != "adamw" or config["scheduler"]["name"].lower() != "none":
        raise ValueError("Diagnostic preserves only the existing AdamW/no-scheduler protocol")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim["learning_rate"]),
                                  weight_decay=float(optim["weight_decay"]),
                                  betas=tuple(float(v) for v in optim["betas"]))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    rt.train._optimizer_to(optimizer, device)
    if checkpoint.get("scheduler_state_dict") is not None:
        raise ValueError("Checkpoint scheduler differs from the saved no-scheduler protocol")
    scaler = rt.train._grad_scaler(bool(config["training"]["amp"]) and device.type == "cuda")
    validate_scaler_metadata(checkpoint.get("amp_scaler_state_dict", {}))
    scaler.load_state_dict(checkpoint.get("amp_scaler_state_dict", {}))
    session = Session(config, checkpoint, model, criterion, optimizer, scaler, device, rt)
    if with_loaders:
        rt.dataset.validate_split_protocol(manifests, config["data"].get("expected_counts"))
        data = config["data"]
        root = Path(data["root"])
        if not root.is_dir():
            raise FileNotFoundError(f"data.root unavailable: {root}; pass --data-root, never substitute splits")
        datasets = {}
        for split in ("train", "validation"):
            datasets[split] = rt.dataset.LSUIDataset(
                manifests[split], root, split, int(data["patch_size"]), data["augmentation"],
                bool(data["pad_if_smaller"]), str(data["pad_mode"]), config["evaluation"], verify_files=True)
        generator = torch.Generator()
        generator.manual_seed(int(config["experiment"]["seed"]))
        kwargs = {"batch_size": int(data["batch_size"]), "num_workers": int(data["num_workers"]),
                  "pin_memory": bool(data["pin_memory"]) and device.type == "cuda",
                  "worker_init_fn": rt.utils.seed_worker, "generator": generator}
        session.train_loader = DataLoader(datasets["train"], shuffle=True, **kwargs)
        val_kwargs = dict(kwargs)
        if not bool(config["evaluation"].get("resize", True)):
            val_kwargs["batch_size"] = 1
        session.validation_loader = DataLoader(datasets["validation"], shuffle=False, **val_kwargs)
        session.generator = generator
        if checkpoint.get("rng_state") is None or checkpoint.get("dataloader_generator_state") is None:
            raise ValueError("Exact replay requires saved RNG and DataLoader generator states")
        # Match native resume: restore only AFTER model/optimizer/loader construction.
        rt.utils.restore_rng_state(checkpoint["rng_state"])
        generator.set_state(checkpoint["dataloader_generator_state"])
    return session


def replay_next_epoch(session: Session, log_every=20) -> dict:
    epoch = int(session.checkpoint["epoch"]) + 1
    if epoch > int(session.config["training"]["epochs"]):
        raise ValueError("Checkpoint already reached the configured epoch budget")
    if session.train_loader is None:
        raise ValueError("Replay requires the saved Train/Validation loaders")
    model, optimizer, scaler, device = session.model, session.optimizer, session.scaler, session.device
    model.train()
    total_loss, count, steps = 0.0, 0, 0
    result = {"epoch": epoch, "completed": False, "first_scaled_gradient_issue": None,
              "first_post_step_state_issue": None, "scale_decreases": 0}
    with BoundaryTrace("train") as trace:
        trace.attach(model)
        for step, batch in enumerate(session.train_loader, start=1):
            steps = step
            trace.set_batch(step, batch["id"])
            inputs, targets = (batch[k].to(device, non_blocking=True) for k in ("input", "target"))
            trace.check("batch.input", inputs)
            trace.check("batch.target", targets)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=session.amp_enabled):
                prediction = model(inputs)
                loss = session.criterion(prediction, targets)
            trace.check("prediction", prediction)
            trace.check("loss", loss)
            finite = all_finite(prediction) and all_finite(loss)
            if not finite and bool(session.config["training"]["fail_on_nonfinite"]):
                result["native_forward_check_failure"] = {
                    "step": step, "sample_ids": list(batch["id"]),
                    "prediction": tensor_statistics("prediction", prediction),
                    "loss": tensor_statistics("loss", loss)}
                break
            scaler.scale(loss).backward()
            # Observe SCALED gradients without extra unscale/clip/optimizer calls.
            if result["first_scaled_gradient_issue"] is None:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not all_finite(parameter.grad):
                        result["first_scaled_gradient_issue"] = {
                            "step": step, "sample_ids": list(batch["id"]),
                            "tensor": tensor_statistics(name + ".scaled_grad", parameter.grad)}
                        break
            clip_norm = session.config["training"].get("gradient_clip_norm")
            if clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            result["scale_decreases"] += int(scale_after < scale_before)
            # Scan after every step; nonfinite gradients alone can be handled by
            # the ORIGINAL scaler. Observations never force a skip or raise.
            if result["first_post_step_state_issue"] is None:
                issues = tree_issues(model.state_dict(), "model_state_dict")
                opt_issues = tree_issues(optimizer.state_dict(), "optimizer_state_dict")
                negative_bn = []
                for name, module in model.named_modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.running_var is not None:
                        if bool((module.running_var < 0).any()):
                            negative_bn.append(tensor_statistics(name + ".running_var", module.running_var))
                if issues or opt_issues or negative_bn:
                    result["first_post_step_state_issue"] = {
                        "step": step, "sample_ids": list(batch["id"]), "model": issues,
                        "optimizer": opt_issues, "negative_bn_variance": negative_bn,
                        "scale_before": scale_before, "scale_after": scale_after}
                    print("FIRST_POST_STEP_STATE_ISSUE " + json.dumps(result["first_post_step_state_issue"], allow_nan=False), flush=True)
            batch_size = inputs.shape[0]
            total_loss += float(loss.detach()) * batch_size
            count += batch_size
            if log_every > 0 and step % log_every == 0:
                print(f"replay epoch={epoch} step={step}/{len(session.train_loader)} train_loss={total_loss / count:.8f} scaler={scale_after:g}", flush=True)
        else:
            result["completed"] = True
        result["first_nonfinite_boundary"] = trace.first_nonfinite
    mean_loss = total_loss / count if count else None
    result.update(steps=steps, sample_count=count,
                  mean_train_loss=mean_loss if mean_loss is not None and math.isfinite(mean_loss) else None,
                  mean_train_loss_finite=mean_loss is not None and math.isfinite(mean_loss))
    return result


@torch.no_grad()
def compare_validation(session: Session, max_batches=0) -> dict:
    if session.validation_loader is None:
        raise ValueError("Validation comparison requires the frozen validation loader")
    session.model.eval()
    modes = {"configured_amp": session.amp_enabled, "fp32": False}
    summaries = {name: {"autocast_enabled": enabled, "output_loss_finite": True,
                        "first_nonfinite_boundary": None, "first_output_loss_failure": None,
                        "first_runtime_error": None,
                        "loss_total": 0.0, "sample_count": 0} for name, enabled in modes.items()}
    examined = 0
    # Fetch each validation batch ONCE; both modes see identical tensors and
    # weights. Eval has no optimizer steps or running-stat updates.
    for step, batch in enumerate(session.validation_loader, start=1):
        examined = step
        inputs, targets = (batch[k].to(session.device, non_blocking=True) for k in ("input", "target"))
        for mode, enabled in modes.items():
            row = summaries[mode]
            with BoundaryTrace("validation_" + mode) as trace:
                trace.set_batch(step, batch["id"])
                trace.attach(session.model)
                trace.check("batch.input", inputs)
                trace.check("batch.target", targets)
                try:
                    with torch.autocast(device_type=session.device.type, enabled=enabled):
                        prediction = session.model(inputs)
                        loss = session.criterion(prediction, targets)
                    trace.check("prediction", prediction)
                    trace.check("loss", loss)
                    finite = all_finite(prediction) and all_finite(loss)
                except Exception as error:
                    row["first_runtime_error"] = {
                        "step": step, "sample_ids": list(batch["id"]),
                        "exception_type": type(error).__name__, "exception_message": str(error),
                        "active_modules": list(trace.active_modules), "traceback": traceback.format_exc()}
                    finite = False
                row["output_loss_finite"] &= finite
                if row["first_nonfinite_boundary"] is None:
                    row["first_nonfinite_boundary"] = trace.first_nonfinite
                if finite:
                    row["loss_total"] += float(loss) * inputs.shape[0]
                    row["sample_count"] += inputs.shape[0]
                elif row["first_output_loss_failure"] is None and row["first_runtime_error"] is None:
                    row["first_output_loss_failure"] = {
                        "step": step, "sample_ids": list(batch["id"]),
                        "nonfinite_prediction_sample_ids": [
                            str(sample) for sample, image in zip(batch["id"], prediction) if not all_finite(image)],
                        "prediction": tensor_statistics("prediction", prediction),
                        "loss": tensor_statistics("loss", loss)}
            print(f"validation mode={mode} batch={step} output_loss_finite={finite}", flush=True)
        if any(not row["output_loss_finite"] or row["first_nonfinite_boundary"] for row in summaries.values()):
            break
        if max_batches and step >= max_batches:
            break
    for row in summaries.values():
        row["mean_finite_batch_loss"] = row.pop("loss_total") / row["sample_count"] if row["sample_count"] else None
    return {"batches_examined": examined, "requested_batch_limit": max_batches, "modes": summaries}


def classify(report: dict) -> str:
    if report["checkpoint_state"]["invalid"]:
        return "CHECKPOINT_STATE_INVALID"
    replay = report.get("replay")
    if replay is not None and not replay["completed"]:
        return "TRAINING_FORWARD_FAILURE"
    if report.get("post_replay_state", {}).get("invalid"):
        return "POST_TRAIN_STATE_INVALID"
    validation = report.get("validation")
    if validation is None:
        return "CHECKPOINT_FINITE"
    modes = validation["modes"]
    if any(row.get("first_runtime_error") for row in modes.values()):
        return "VALIDATION_RUNTIME_ERROR"
    amp, fp32 = modes["configured_amp"], modes["fp32"]
    if not amp["output_loss_finite"] and fp32["output_loss_finite"] and amp["autocast_enabled"]:
        return "AMP_ONLY_FORWARD_FAILURE"
    if not fp32["output_loss_finite"]:
        return "FP32_FORWARD_FAILURE"
    traces = [row["first_nonfinite_boundary"] for row in modes.values()]
    if replay is not None:
        traces.append(replay["first_nonfinite_boundary"])
    if any(traces):
        return "INTERMEDIATE_NONFINITE_WITH_FINITE_OUTPUT"
    return "NOT_REPRODUCED" if amp["autocast_enabled"] else "NO_FAILURE_OBSERVED_CPU_OR_FP32_ONLY"


def run_file_inventory(run_dir: Path) -> dict:
    return {p.relative_to(run_dir).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in run_dir.rglob("*") if p.is_file()}


def resolve_output(path: Path, run_dir: Path) -> Path:
    path = path.resolve()
    forbidden = [run_dir, *[PROJECT_ROOT / p for p in ("src", "configs", "split", "experiments", "tests", "tools")]]
    if path.suffix != ".json":
        raise ValueError("Diagnostic output must be a .json report")
    if any(path.is_relative_to(root.resolve()) for root in forbidden):
        raise ValueError("Diagnostic report must be outside experiment/code/config/split directories")
    return path


def publish_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay-next-epoch", action="store_true")
    mode.add_argument("--validate-checkpoint", action="store_true")
    parser.add_argument("--expect-checkpoint-epoch", type=int, default=None)
    parser.add_argument("--validation-batches", type=int, default=0, help="0: all, stopping at first numerical issue")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)
    if args.validation_batches < 0 or args.log_every < 0:
        parser.error("Batch limit/log interval must be non-negative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory unavailable: {run_dir}")
    mode = "replay_next_epoch" if args.replay_next_epoch else ("validate_checkpoint" if args.validate_checkpoint else "inspect_only")
    output = resolve_output(Path(args.output_json) if args.output_json else
                            PROJECT_ROOT / "diagnostics/qgrd_numerics" / run_dir.name / f"{mode}.json", run_dir)
    before = run_file_inventory(run_dir)
    report = {"run_dir": str(run_dir), "mode": mode, "status": "ERROR",
              "diagnostic_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "report_is_not_a_formal_experiment_result": True,
              "notes": ["Observational hooks identify module boundaries, not hidden fused-kernel intermediates.",
                        "Saved deterministic=false/CUDA kernels may prevent exact reproduction across machines or runs.",
                        "Scaled nonfinite gradients alone may trigger ordinary GradScaler skips; this tool does not change them.",
                        "No checkpoint, config, run status, experiment log, or metric result is written."]}
    exit_code = 1
    try:
        session = load_session(run_dir, args.checkpoint, args.gpu, args.data_root,
                               args.replay_next_epoch or args.validate_checkpoint, args.expect_checkpoint_epoch)
        path = session.runtime.test._checkpoint_path(run_dir, args.checkpoint).resolve()
        digest_before = session.runtime.utils.sha256_file(path)
        report.update(checkpoint=str(path), checkpoint_epoch=int(session.checkpoint["epoch"]),
                      checkpoint_sha256=digest_before, version=session.checkpoint["version"],
                      saved_amp_scaler_state=session.checkpoint.get("amp_scaler_state_dict", {}),
                      resolved_config=deepcopy(session.config),
                      environment={"python": platform.python_version(), "torch": torch.__version__,
                                   "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                                   "device": str(session.device), "amp_enabled": session.amp_enabled,
                                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                   "gpu_name": torch.cuda.get_device_name(session.device) if session.device.type == "cuda" else None})
        print(f"checkpoint={path}\nepoch={session.checkpoint['epoch']} device={session.device} amp_enabled={session.amp_enabled}", flush=True)
        if (args.replay_next_epoch or args.validate_checkpoint) and session.config["training"]["amp"] and session.device.type != "cuda":
            print("[WARNING] CPU run cannot diagnose CUDA AMP failures; both validation modes are FP32.", flush=True)
        print(f"report={output}\nmode={mode}; ORIGINAL RUN WILL NOT BE WRITTEN", flush=True)
        report["checkpoint_state"] = model_summary(session.model, session.optimizer)
        if not report["checkpoint_state"]["invalid"]:
            if args.replay_next_epoch:
                print(f"replaying ONE epoch {session.checkpoint['epoch'] + 1} in memory with the saved training recipe", flush=True)
                report["replay"] = replay_next_epoch(session, args.log_every)
                report["post_replay_state"] = model_summary(session.model, session.optimizer)
            if args.validate_checkpoint or (args.replay_next_epoch and report["replay"]["completed"]):
                report["validation"] = compare_validation(session, args.validation_batches)
        report["status"] = classify(report)
        report["original_run_unchanged"] = before == run_file_inventory(run_dir)
        report["checkpoint_bytes_unchanged"] = digest_before == session.runtime.utils.sha256_file(path)
        if not report["original_run_unchanged"] or not report["checkpoint_bytes_unchanged"]:
            raise RuntimeError("Original run changed during diagnosis; avoid concurrent training/resume")
        exit_code = 0 if report["status"] in {
            "CHECKPOINT_FINITE", "NOT_REPRODUCED", "NO_FAILURE_OBSERVED_CPU_OR_FP32_ONLY"
        } else 2
    except Exception as error:
        report.update(status="ERROR", exception_type=type(error).__name__,
                      exception_message=str(error), traceback=traceback.format_exc(),
                      original_run_unchanged=before == run_file_inventory(run_dir))
        traceback.print_exc()
    publish_json(output, report)
    print("DIAGNOSIS_STATUS=" + report["status"], flush=True)
    print("REPORT_JSON=" + str(output), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
