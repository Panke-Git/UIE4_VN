"""Training/validation engine with auditable metrics and resumable checkpoints."""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .experiment import update_status
from .metrics import batch_metrics
from .utils import atomic_json, rng_state


def _write_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    atomic_json(run_dir / "log" / "metrics_history.json", {"epochs": history})
    columns = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "val_psnr",
        "val_ssim",
        "epoch_time_seconds",
    ]
    with (run_dir / "log" / "metrics_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def _checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    best: dict[str, float],
    config: dict,
    dataloader_generator_state: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "amp_scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best["val_loss"],
        "best_psnr": best["psnr"],
        "best_ssim": best["ssim"],
        "resolved_config": config,
        "seed": int(config["experiment"]["seed"]),
        "version": config["experiment"]["version"],
        "rng_state": rng_state(),
        "dataloader_generator_state": dataloader_generator_state,
    }


def _save_best(
    run_dir: Path,
    kind: str,
    payload: dict[str, Any],
    metrics: dict[str, float],
    learning_rate: float,
) -> None:
    filename = f"best_{kind}.pt"
    torch.save(payload, run_dir / "best" / filename)
    atomic_json(
        run_dir / "best" / f"best_{kind}.json",
        {
            "epoch": payload["epoch"],
            "val_loss": metrics["loss"],
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "learning_rate": learning_rate,
            "checkpoint": filename,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
    run_dir: Path,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    sample_count = 0
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    rows: list[dict[str, Any]] = []
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            prediction = model(inputs)
            loss = criterion(prediction, targets)
        if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation output/loss at epoch={epoch} ids={batch['id']}")
        clipped = prediction.float().clamp(0.0, 1.0)
        psnr, ssim = batch_metrics(clipped, targets.float(), config["metrics"])
        batch_size = inputs.shape[0]
        loss_total += float(loss) * batch_size
        sample_count += batch_size
        for index in range(batch_size):
            psnr_value, ssim_value = float(psnr[index]), float(ssim[index])
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
            rows.append(
                {
                    "filename": batch["filename"][index],
                    "sample_id": batch["id"][index],
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                }
            )
    metrics = {
        "loss": loss_total / sample_count,
        "psnr": sum(psnr_values) / len(psnr_values),
        "ssim": sum(ssim_values) / len(ssim_values),
    }
    with (run_dir / "result" / "validation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "sample_id", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        run_dir / "result" / "validation_summary.json",
        {"epoch": epoch, "sample_count": sample_count, "mean_loss": metrics["loss"], "mean_psnr": metrics["psnr"], "mean_ssim": metrics["ssim"]},
    )
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    config: dict,
    run_dir: Path,
    logger: logging.Logger,
    validation_logger: logging.Logger,
    status: dict,
    start_epoch: int = 1,
    best: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict]:
    version = config["experiment"]["version"]
    epochs = int(config["training"]["epochs"])
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    fail_nonfinite = bool(config["training"]["fail_on_nonfinite"])
    log_steps = int(config["logging"]["log_every_steps"])
    best = best or {"val_loss": float("inf"), "psnr": -float("inf"), "ssim": -float("inf")}
    history_path = run_dir / "log" / "metrics_history.json"
    history = []
    if start_epoch > 1 and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("epochs", [])

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for step, batch in enumerate(train_loader, start=1):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(inputs)
                loss = criterion(prediction, targets)
            finite = bool(torch.isfinite(prediction).all() and torch.isfinite(loss))
            if not finite and fail_nonfinite:
                raise FloatingPointError(
                    f"Non-finite train output/loss: epoch={epoch} step={step} ids={batch['id']} loss={float(loss)}"
                )
            scaler.scale(loss).backward()
            clip_norm = config["training"].get("gradient_clip_norm")
            if clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
            scaler.step(optimizer)
            scaler.update()
            batch_size = inputs.shape[0]
            train_loss_total += float(loss.detach()) * batch_size
            train_samples += batch_size
            if log_steps > 0 and step % log_steps == 0:
                logger.info("epoch=%d step=%d train_loss=%.6f", epoch, step, float(loss))

        train_loss = train_loss_total / train_samples
        val_metrics = {"loss": math.nan, "psnr": math.nan, "ssim": math.nan}
        if epoch % int(config["training"]["validate_every"]) == 0:
            val_metrics = validate(model, validation_loader, criterion, device, config, run_dir, epoch)
            validation_logger.info(
                "epoch=%d val_loss=%.8f val_psnr=%.6f val_ssim=%.6f",
                epoch, val_metrics["loss"], val_metrics["psnr"], val_metrics["ssim"],
            )
            improved_loss = val_metrics["loss"] < best["val_loss"]
            improved_psnr = val_metrics["psnr"] > best["psnr"]
            improved_ssim = val_metrics["ssim"] > best["ssim"]
            if improved_loss:
                best["val_loss"] = val_metrics["loss"]
            if improved_psnr:
                best["psnr"] = val_metrics["psnr"]
            if improved_ssim:
                best["ssim"] = val_metrics["ssim"]
        else:
            improved_loss = improved_psnr = improved_ssim = False

        if scheduler is not None:
            scheduler.step()
        loader_generator = getattr(train_loader, "generator", None)
        generator_state = loader_generator.get_state() if loader_generator is not None else None
        payload = _checkpoint_payload(
            epoch, model, optimizer, scheduler, scaler, best, config, generator_state
        )
        checkpoint_config = config["checkpoint"]
        if improved_loss and checkpoint_config["save_best_val_loss"]:
            _save_best(run_dir, "loss", payload, val_metrics, learning_rate)
        if improved_psnr and checkpoint_config["save_best_psnr"]:
            _save_best(run_dir, "psnr", payload, val_metrics, learning_rate)
        if improved_ssim and checkpoint_config["save_best_ssim"]:
            _save_best(run_dir, "ssim", payload, val_metrics, learning_rate)
        if checkpoint_config["save_last"]:
            torch.save(payload, run_dir / "checkpoint" / "last.pt")
        if (
            checkpoint_config["save_periodic"]
            and epoch % int(config["training"]["save_every"]) == 0
        ):
            torch.save(payload, run_dir / "checkpoint" / f"epoch_{epoch:04d}.pt")

        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "lr": learning_rate,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "epoch_time_seconds": epoch_seconds,
        }
        history.append(record)
        _write_history(run_dir, history)
        status = update_status(
            run_dir,
            status,
            status="running",
            last_epoch=epoch,
            best_psnr=best["psnr"],
            best_ssim=best["ssim"],
            best_val_loss=best["val_loss"],
        )
        logger.info(
            "[%s][Epoch %03d/%03d] lr=%.6e train_loss=%.6f val_loss=%.6f val_psnr=%.4f val_ssim=%.4f time=%.1fs",
            version, epoch, epochs, learning_rate, train_loss, val_metrics["loss"],
            val_metrics["psnr"], val_metrics["ssim"], epoch_seconds,
        )
    return best, status
