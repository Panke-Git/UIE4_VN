"""Small checkpoint-aware loop adapter; validation/loss/metrics stay in V16."""
from __future__ import annotations

import json
import time

import torch

from src.v16.engine import validate, _checkpoint_payload, _save_best, _write_history
from src.v16.experiment import update_status
from . import FORMULAS


def train_epochs(model, train_loader, validation_loader, criterion, optimizer, scaler,
                 device, config, run_dir, logger, status, *, start_epoch=1, best=None,
                 stop_after_epoch=None):
    total = int(config["training"]["epochs"])
    end = min(total, stop_after_epoch) if stop_after_epoch is not None else total
    if end < start_epoch:
        raise ValueError("stop-after-epoch precedes the next resumable epoch")
    history_path = run_dir/"log/metrics_history.json"
    history = json.loads(history_path.read_text())["epochs"] if start_epoch > 1 else []
    best = best or {"val_loss": float("inf"), "psnr": -float("inf"), "ssim": -float("inf")}
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    for epoch in range(start_epoch, end+1):
        started = time.perf_counter()
        model.train()
        total_loss, count = 0.0, 0
        lr = float(optimizer.param_groups[0]["lr"])
        for batch in train_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(inputs)
                loss = criterion(prediction, targets)
            if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training output/loss at epoch {epoch}")
            scaler.scale(loss).backward()
            clip = config["training"].get("gradient_clip_norm")
            if clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(inputs)
            count += len(inputs)
        # Original V16 validation, including its CUDA AMP/float-RGB metric path.
        val = validate(model, validation_loader, criterion, device, config, run_dir, epoch)
        improved = {key: val[source] < best[key] if key == "val_loss" else val[source] > best[key]
                    for key, source in (("val_loss", "loss"), ("psnr", "psnr"), ("ssim", "ssim"))}
        for key, source in (("val_loss", "loss"), ("psnr", "psnr"), ("ssim", "ssim")):
            if improved[key]:
                best[key] = val[source]
        payload = _checkpoint_payload(epoch, model, optimizer, None, scaler, best, config,
                                      train_loader.generator.get_state())
        variant = config["model"]["parameterization"]
        payload.update(variant=variant, formula=FORMULAS[variant],
                       split_sha256=config["provenance"]["split_sha256"],
                       code_version=config["provenance"]["code"],
                       parameter_counts=config["provenance"]["parameter_counts"],
                       training_status={"last_epoch": epoch, "configured_epochs": total,
                                        "completed": epoch == total})
        for key, kind in (("val_loss", "loss"), ("psnr", "psnr"), ("ssim", "ssim")):
            flag = "save_best_val_loss" if key == "val_loss" else "save_best_"+kind
            if improved[key] and config["checkpoint"][flag]:
                _save_best(run_dir, kind, payload, val, lr)
        if config["checkpoint"]["save_last"]:
            torch.save(payload, run_dir/"checkpoint/last.pt")
        if config["checkpoint"]["save_periodic"] and epoch % int(config["training"]["save_every"]) == 0:
            torch.save(payload, run_dir/"checkpoint"/f"epoch_{epoch:04d}.pt")
        history.append(dict(epoch=epoch, lr=lr, train_loss=total_loss/count,
                            val_loss=val["loss"], val_psnr=val["psnr"], val_ssim=val["ssim"],
                            val_e00=val["e00"], epoch_time_seconds=time.perf_counter()-started))
        _write_history(run_dir, history)
        status = update_status(run_dir, status, status="running", last_epoch=epoch,
                               variant=variant, best_psnr=best["psnr"], best_ssim=best["ssim"], best_val_loss=best["val_loss"])
        logger.info("epoch=%d/%d train_loss=%.8f val_psnr=%.6f val_ssim=%.6f val_e00=%.6f",
                    epoch, total, total_loss/count, val["psnr"], val["ssim"], val["e00"])
    return best, status
