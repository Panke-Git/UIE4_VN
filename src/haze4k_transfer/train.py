"""Independent random-initialization training; official test never used for selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import uuid

import torch
from torch.utils.data import DataLoader

from src.v16.losses import build_loss
from .dataset import HazeDataset
from .models import build_model, canonical_model_config
from .prepare import validate_bundle, verify_content, write_json
from .utils import (ROOT, METRICS, audit_environment, device_for, fairness, finite, load_yaml,
                    plot_history, project_path, restore_rng_state, rng_state, run_path_guard,
                    save_checkpoint, save_yaml, seed_everything, seed_worker, source_hashes,
                    torch_load, write_csv)


def accumulation_sample_count(dataset_size, batch_size, accumulation_steps, step, step_limit):
    """Actual images in this optimizer update, including a partial final batch/group."""
    first = ((step - 1) // accumulation_steps) * accumulation_steps
    end = min(first + accumulation_steps, step_limit)
    count = min(end * batch_size, dataset_size) - first * batch_size
    if count < 1 or not (1 <= step <= step_limit):
        raise ValueError("Invalid accumulation group")
    return count


def normalized_backward_loss(loss, batch_samples, group_samples):
    """Normalize BEFORE scaling/backward to avoid amplified FP16 intermediates."""
    if not 0 < batch_samples <= group_samples:
        raise ValueError("Invalid batch/group sample count")
    return loss * (batch_samples / group_samples)


def validate_config(c):
    if not c.get("smoke") and (c["training"]["epochs"] != 200 or c["data"]["patch_size"] != 256):
        raise ValueError("Main protocol requires 200 epochs and patch 256; use explicit smoke flags for software checks")
    if c["model"] != canonical_model_config():
        raise ValueError("Architecture must exactly match canonical V16")
    if c["metrics"] != METRICS or c["evaluation"] != {"validation": "PIL_BILINEAR_256x256", "test": "PIL_BILINEAR_256x256"}:
        raise ValueError("Changed metric/evaluation protocol")
    if c["optimizer"] != {"name": "AdamW", "lr": 0.0002, "betas": [0.9, 0.999], "weight_decay": 0.0}:
        raise ValueError("Changed canonical optimizer")
    if c["loss"] != {"name": "charbonnier", "epsilon": 0.001} or c["training"]["scheduler"] != "none":
        raise ValueError("Changed loss/scheduler")
    if c["training"]["validate_every"] != 1 or c["training"]["checkpoint_selection"] != "best_validation_PSNR":
        raise ValueError("Validation/checkpoint selection must be fixed")
    for key in ("batch_size", "patch_size"):
        if int(c["data"][key]) <= 0:
            raise ValueError(f"Invalid {key}")
    if c["training"]["accumulation_steps"] < 1 or c["training"]["epochs"] < 1 or c["data"]["num_workers"] < 0:
        raise ValueError("Invalid training dimensions")
    scale = float(c["training"].get("amp_init_scale", 65536.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("amp_init_scale must be finite and positive")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--data-root")
    p.add_argument("--split-dir")
    p.add_argument("--gpu", default="0", help="CUDA index, or explicit cpu for software tests")
    p.add_argument("--resume", help="Existing run checkpoint/last.pt; epoch-boundary resume only")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation-steps", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--amp-init-scale", type=float, help="Explicit initial CUDA GradScaler scale; apply identically to both variants")
    p.add_argument("--fp32-smoke", action="store_true", help="Disable AMP only for diagnostic --smoke-epochs runs")
    p.add_argument("--smoke-epochs", type=int, choices=(1, 2))
    p.add_argument("--smoke-steps", type=int, help="Train batches per epoch; requires --smoke-epochs; non-paper")
    p.add_argument("--synthetic-smoke", action="store_true", help="Patch 32, batch 2, workers 0; requires synthetic prepare")
    p.add_argument("--stop-after-epoch", type=int, help="Controlled epoch-boundary interruption, retaining target epochs")
    a = p.parse_args(argv)
    if a.resume and any(v is not None for v in (a.config, a.batch_size, a.accumulation_steps, a.num_workers, a.amp_init_scale, a.smoke_epochs, a.smoke_steps)):
        p.error("Resume uses saved config; only --gpu/--data-root/--stop-after-epoch may change")
    if a.resume and (a.synthetic_smoke or a.split_dir or a.fp32_smoke):
        p.error("Resume uses saved synthetic marker and split snapshot")
    if not a.resume and not a.config:
        p.error("--config is required for a new run")
    checkpoint = None
    if a.resume:
        cp_path = project_path(a.resume)
        run = run_path_guard(cp_path.parent.parent)
        if cp_path != run / "checkpoint/last.pt":
            raise ValueError("Resume only last.pt to avoid stale history/best-checkpoint branch overwrites")
        c = load_yaml(run / "config_resolved.yaml")
        checkpoint = torch_load(cp_path)
        if checkpoint["config"] != c:
            raise ValueError("Checkpoint/resolved config mismatch")
        if checkpoint["source_hashes"] != source_hashes():
            raise ValueError("Sources changed since checkpoint; refusing non-equivalent resume")
        split_dir = run / "split"
    else:
        source = project_path(a.config)
        c = load_yaml(source)
        split_dir = project_path(a.split_dir or c["split_dir"])
        if a.batch_size is not None:
            c["data"]["batch_size"] = a.batch_size
        if a.accumulation_steps is not None:
            c["training"]["accumulation_steps"] = a.accumulation_steps
        if a.num_workers is not None:
            c["data"]["num_workers"] = a.num_workers
        c["training"].setdefault("amp_init_scale", 65536.0)
        if a.amp_init_scale is not None:
            c["training"]["amp_init_scale"] = a.amp_init_scale
        if a.fp32_smoke:
            if not a.smoke_epochs:
                p.error("--fp32-smoke requires --smoke-epochs; main experiment remains AMP-enabled")
            c["training"]["amp"] = False
        if a.smoke_epochs:
            c["training"]["epochs"] = a.smoke_epochs
        if a.smoke_steps and (not a.smoke_epochs or a.smoke_steps < 1):
            p.error("Positive --smoke-steps requires --smoke-epochs")
        c["smoke_steps"] = a.smoke_steps
        c["smoke"] = bool(a.smoke_epochs or a.synthetic_smoke)
        if a.synthetic_smoke:
            if not a.smoke_epochs:
                p.error("--synthetic-smoke requires --smoke-epochs")
            c["data"].update(patch_size=32, batch_size=2, num_workers=0)
    validate_config(c)
    meta, rows, inventory = validate_bundle(split_dir)
    if int(c["seed"]) != int(meta["seed"]):
        raise ValueError("Training/split seed mismatch")
    if a.synthetic_smoke and not meta["synthetic"]:
        raise ValueError("--synthetic-smoke requires prepare --synthetic")
    c["synthetic"] = meta["synthetic"]
    if meta["synthetic"]:
        c["smoke"] = True
    data_root = Path(a.data_root or c["data_root"]).resolve()
    # Hash train/validation only. No official test image is loaded in training.
    verify_content(data_root, rows["train"] + rows["validation"], inventory)
    if not a.resume:
        c["data_root"] = str(data_root)
    seed_everything(int(c["seed"]), bool(c["training"]["deterministic"]))
    device = device_for(a.gpu)
    model = build_model(c["variant"], c["model"]).to(device)
    criterion = build_loss(c["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["optimizer"]["lr"],
                                 betas=tuple(c["optimizer"]["betas"]), weight_decay=c["optimizer"]["weight_decay"])
    amp = bool(c["training"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=float(c["training"].get("amp_init_scale", 65536.0)))
    generator = torch.Generator().manual_seed(int(c["seed"]))
    train_loader = DataLoader(HazeDataset(data_root, rows["train"], "train", c["data"]["patch_size"]),
                              batch_size=c["data"]["batch_size"], shuffle=True, drop_last=False,
                              num_workers=c["data"]["num_workers"], pin_memory=device.type == "cuda",
                              worker_init_fn=seed_worker, generator=generator, persistent_workers=False)
    # Fixed float32 single-image validation matches standalone test inference.
    val_loader = DataLoader(HazeDataset(data_root, rows["validation"], "validation"), batch_size=1, num_workers=0,
                            generator=torch.Generator().manual_seed(int(c["seed"]) + 1))
    history, start, best = [], 1, -float("inf")
    if checkpoint:
        if checkpoint["split_hashes"] != meta["hashes"]:
            raise ValueError("Checkpoint/split hashes differ")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
        scaler.load_state_dict(checkpoint["amp_scaler"])
        generator.set_state(checkpoint["loader_generator"])
        val_loader.generator.set_state(checkpoint["val_generator"])
        restore_rng_state(checkpoint["rng"])
        history, start, best = checkpoint["history"], checkpoint["epoch"] + 1, checkpoint["best_psnr"]
        if start > c["training"]["epochs"]:
            raise ValueError("Already reached target epochs")
    else:
        name = f"{c['variant']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{c['seed']}_{uuid.uuid4().hex[:6]}"
        run = run_path_guard(ROOT / "experiments/haze4k_transfer" / name)
        run.mkdir(parents=True, exist_ok=False)
        (run / "checkpoint").mkdir()
        (run / "validation").mkdir()
        shutil.copyfile(source, run / "config_source.yaml")
        save_yaml(run / "config_resolved.yaml", c)
        shutil.copytree(split_dir, run / "split")
        write_json(run / "environment.json", audit_environment(device))
        write_json(run / "model.json", {"variant": c["variant"], "factory": "src.v16.models.network.build_model" if c["variant"] == "full" else "src.v15.models.network.build_model",
                                       "parameters": sum(v.numel() for v in model.parameters()), "initialization": "random; no pretrained weights"})
    print(f"RUN_DIR={run}", flush=True)
    print(f"device={device}; AMP requested={c['training']['amp']} enabled={amp}; batch={c['data']['batch_size']}; "
          f"accumulation={c['training']['accumulation_steps']}; effective batch={c['data']['batch_size'] * c['training']['accumulation_steps']}; "
          f"synthetic={c['synthetic']}; smoke={c['smoke']}", flush=True)
    print(f"AMP initial scale={c['training'].get('amp_init_scale', 65536.0)}; current scale={scaler.get_scale()}; "
          "gradient normalization=sample mean before scaled backward", flush=True)
    write_json(run / "runtime.json", {"device": str(device), "amp_enabled": amp, "data_root_current": str(data_root),
                                     "amp_init_scale": c["training"].get("amp_init_scale", 65536.0),
                                     "gradient_normalization": "sample-weighted mean before GradScaler.scale/backward",
                                     "effective_batch_size": c["data"]["batch_size"] * c["training"]["accumulation_steps"],
                                     "tail_batch": "sample-weighted; possibly smaller effective batch", "resume": bool(checkpoint),
                                     "resume_limit": "epoch boundaries; exact worker/batch continuity requires same software/device/workers and deterministic kernels"})
    sources = source_hashes()
    try:
        from .eval import evaluate_loader
        for epoch in range(start, int(c["training"]["epochs"]) + 1):
            begun = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total, samples = 0.0, 0
            limit = min(len(train_loader), c["smoke_steps"] or len(train_loader))
            for step, batch in enumerate(train_loader, 1):
                x, y = batch["input"].to(device), batch["target"].to(device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    pred = model(x)
                    loss = criterion(pred, y)
                finite(pred, f"train output epoch={epoch} IDs={batch['sample_id']}")
                finite(loss, f"train loss epoch={epoch}")
                n = len(x)
                group_samples = accumulation_sample_count(len(train_loader.dataset), c["data"]["batch_size"],
                                                          c["training"]["accumulation_steps"], step, limit)
                scaler.scale(normalized_backward_loss(loss, n, group_samples)).backward()
                total += loss.detach().item() * n
                samples += n
                if step % c["training"]["accumulation_steps"] == 0 or step == limit:
                    scaler.unscale_(optimizer)
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if not torch.isfinite(param.grad).all():
                                diag = {"stage": "unscaled gradient", "parameter": name, "epoch": epoch,
                                        "step": step, "sample_ids": batch["sample_id"], "amp_enabled": amp,
                                        "grad_scaler_scale": scaler.get_scale(), "batch_samples": n,
                                        "accumulation_group_samples": group_samples, "loss": loss.detach().item(),
                                        "nan_count": torch.isnan(param.grad).sum().item(),
                                        "inf_count": torch.isinf(param.grad).sum().item()}
                                write_json(run / "nonfinite_diagnostic.json", diag)
                                raise FloatingPointError(
                                    f"NaN/Inf in gradient {name} epoch={epoch} step={step}; AMP={amp} "
                                    f"scale={scaler.get_scale()}; no optimizer update applied. "
                                    f"See {run / 'nonfinite_diagnostic.json'}. "
                                    "Check a fresh smoke run with an explicitly lower --amp-init-scale; "
                                    "--fp32-smoke can help distinguish AMP-specific behavior. Do not skip bad gradients."
                                )
                    scaler.step(optimizer)
                    scaler.update()
                    for name, param in model.named_parameters():
                        finite(param, f"parameter {name} epoch={epoch}")
                    optimizer.zero_grad(set_to_none=True)
                if step == limit:
                    break
            val_rows, val_summary = evaluate_loader(model, val_loader, device, criterion)
            improved = val_summary["psnr"] > best
            if improved:
                best = val_summary["psnr"]
            record = {"epoch": epoch, "train_loss": total / samples, "val_loss": val_summary["loss"],
                      "val_psnr": val_summary["psnr"], "val_ssim": val_summary["ssim"],
                      "seconds": time.perf_counter() - begun, "train_samples": samples}
            history.append(record)
            write_csv(run / "validation" / f"epoch_{epoch:04d}.csv", val_rows)
            write_json(run / "history.json", history)
            write_csv(run / "history.csv", history)
            plot_history(run, history)
            with (run / "train.log").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            # Capture RNG after plotting/logging, matching the state seen by the next epoch.
            payload = {"format": "haze4k-checkpoint-v1", "epoch": epoch, "best_psnr": best,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(), "amp_scaler": scaler.state_dict(),
                       "rng": rng_state(), "loader_generator": generator.get_state(), "val_generator": val_loader.generator.get_state(),
                       "config": c, "split_hashes": meta["hashes"], "history": history, "source_hashes": sources}
            if improved:
                save_checkpoint(run / "checkpoint/best_psnr.pt", payload)
            save_checkpoint(run / "checkpoint/last.pt", payload)
            print(json.dumps(record, allow_nan=False), flush=True)
            write_json(run / "status.json", {"status": "running", "epoch": epoch, "best_validation_psnr": best})
            if a.stop_after_epoch and epoch >= a.stop_after_epoch:
                break
        complete = history[-1]["epoch"] == int(c["training"]["epochs"])
        write_json(run / "status.json", {"status": "completed" if complete else "interrupted_at_epoch_boundary",
                                        "epoch": history[-1]["epoch"], "best_validation_psnr": best})
    except BaseException as e:
        write_json(run / "status.json", {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
        raise
    return run


if __name__ == "__main__":
    main()
