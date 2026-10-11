"""Train isolated V16-19 on physical Train/Val and export the selected checkpoint on Val."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import LSUIDataset, validate_split_protocol
from .engine import train_model
from .experiment import initialize_run, load_status, update_status
from .protocol import read_protocol, verify_dataset
from .validate import export_validation
from .losses import build_loss
from .models import build_model
from .utils import (
    PROJECT_ROOT,
    apply_overrides,
    load_yaml,
    project_path,
    restore_rng_state,
    seed_everything,
    seed_worker,
    select_device,
    setup_logger,
    save_yaml,
    atomic_json,
    runtime_code_hashes,
    validate_config,
    validate_checkpoint_config,
)


EXPECTED_VERSION = "v16_19"
DEFAULT_CONFIG = "configs/v16_19/lsui19.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train {EXPECTED_VERSION}")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args(argv)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_config(args: argparse.Namespace) -> tuple[dict, Path, Path | None]:
    if args.resume:
        checkpoint_path = project_path(args.resume).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
        run_dir = checkpoint_path.parent.parent
        resolved_path = run_dir / "config_resolved.yaml"
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Resume run lacks config_resolved.yaml: {run_dir}")
        config = load_yaml(resolved_path)
        if args.config is not None:
            raise ValueError("Resume uses its saved configuration; omit --config")
        if args.epochs is not None:
            if args.epochs < int(config["training"]["epochs"]):
                raise ValueError("Resume cannot reduce the saved epoch budget")
            config["training"]["epochs"] = args.epochs
        for key, argument in [("batch_size", args.batch_size), ("num_workers", args.num_workers)]:
            if argument is not None and argument != int(config["data"][key]):
                raise ValueError(f"Resume cannot change {key}; RNG continuity is preserved")
        if args.seed is not None and args.seed != int(config["experiment"]["seed"]):
            raise ValueError("Changing seed while resuming would invalidate RNG continuity")
        if args.name is not None and args.name != config["experiment"]["name"]:
            raise ValueError("Changing experiment name while resuming is not supported")
        config = apply_overrides(config, data_root=args.data_root)
        return config, run_dir / "config_source.yaml", checkpoint_path

    source = project_path(args.config or DEFAULT_CONFIG).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Config does not exist: {source}")
    config = apply_overrides(load_yaml(source), args.seed, args.data_root, args.name)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    return config, source, None


def load_status_history(path: Path) -> list[dict]:
    import json
    return json.loads(path.read_text(encoding="utf-8"))["epochs"]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir: Path | None = None
    status: dict | None = None
    logger = None
    try:
        config, source_config, resume_path = _load_config(args)
        validate_config(config)
        seed = int(config["experiment"]["seed"])
        seed_everything(seed, bool(config["training"]["deterministic"]))
        device = select_device(args.gpu)
        data_root = project_path(config["data"]["root"]).resolve()
        config["data"]["root"] = str(data_root)
        if resume_path is None:
            protocol_dir = project_path(config["data"]["protocol_dir"]).resolve()
            if not protocol_dir.is_relative_to((PROJECT_ROOT / "split/v16_19").resolve()):
                raise ValueError("Protocol must be inside split/v16_19")
        else:
            protocol_dir = resume_path.parent.parent / "split_snapshot"
        protocol = verify_dataset(data_root, protocol_dir)
        if protocol["dataset"] != config["data"]["dataset_key"] or protocol["counts"] != config["data"]["expected_counts"]:
            raise ValueError("Configuration dataset/counts differ from the frozen protocol")
        current_hashes = runtime_code_hashes()
        if resume_path is not None:
            if config["data"].get("fingerprint") != protocol["fingerprint"]:
                raise ValueError("Resume dataset fingerprint mismatch")
            if config.get("runtime_code_sha256") != current_hashes:
                raise ValueError("V16-19 source changed since the run; resume refused")
        config["data"]["fingerprint"] = protocol["fingerprint"]
        config["runtime_code_sha256"] = current_hashes
        manifests = {split: protocol_dir / f"{split}.tsv" for split in ("train", "validation")}
        entries = validate_split_protocol(manifests, config["data"]["expected_counts"])
        model = build_model(config["model"]).to(device)
        if resume_path is None:
            run_dir, status = initialize_run(config, source_config, model, device, protocol_dir, protocol, [sys.executable, *sys.argv])
            # Training and later evaluation both use the immutable run snapshot.
            manifests = {split: run_dir / "split_snapshot" / f"{split}.tsv" for split in ("train", "validation")}
            append = False
        else:
            run_dir = resume_path.parent.parent
            status = load_status(run_dir)
            append = True

        logger = setup_logger(
            f"{EXPECTED_VERSION}.train.{run_dir.name}",
            run_dir / "log" / "train.log",
            append=append,
            console=bool(config["logging"]["console"]),
        )
        validation_logger = setup_logger(
            f"{EXPECTED_VERSION}.val.{run_dir.name}",
            run_dir / "log" / "val.log",
            append=append,
            console=False,
        )
        logger.info("device=%s amp_requested=%s amp_enabled=%s", device, config["training"]["amp"], device.type == "cuda" and config["training"]["amp"])
        if bool(config["training"]["deterministic"]):
            logger.warning("Deterministic algorithms requested; unsupported operations will emit PyTorch warnings")

        data_root = Path(config["data"]["root"])
        if not data_root.is_dir():
            raise FileNotFoundError(
                f"data.root is unavailable: {data_root}. Manifests were validated; set --data-root on the data server."
            )
        data = config["data"]
        train_dataset = LSUIDataset(
            manifests["train"], data_root, "train", int(data["patch_size"]),
            data["augmentation"], bool(data["pad_if_smaller"]), str(data["pad_mode"]),
            config["evaluation"], verify_files=True,
        )
        validation_dataset = LSUIDataset(
            manifests["validation"], data_root, "validation", int(data["patch_size"]),
            data["augmentation"], bool(data["pad_if_smaller"]), str(data["pad_mode"]),
            config["evaluation"], verify_files=True,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {
            "batch_size": int(data["batch_size"]),
            "num_workers": int(data["num_workers"]),
            "pin_memory": bool(data["pin_memory"]) and device.type == "cuda",
            "worker_init_fn": seed_worker,
            "generator": generator,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        validation_loader_kwargs = dict(loader_kwargs)
        if not bool(config["evaluation"].get("resize", True)):
            # Native-resolution samples may have different H/W and therefore cannot be stacked.
            validation_loader_kwargs["batch_size"] = 1
        validation_loader = DataLoader(validation_dataset, shuffle=False, **validation_loader_kwargs)

        criterion = build_loss(config["loss"]).to(device)
        optimizer_config = config["optimizer"]
        if optimizer_config["name"].lower() != "adamw":
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
        )
        if config["scheduler"]["name"].lower() != "none":
            raise ValueError("V16-19 supports scheduler.name=none")
        scheduler = None
        scaler = _grad_scaler(bool(config["training"]["amp"]) and device.type == "cuda")
        start_epoch = 1
        best = None
        if resume_path is not None:
            checkpoint = _torch_load(resume_path, device)
            if checkpoint.get("version") != EXPECTED_VERSION:
                raise ValueError("Resume checkpoint version is incompatible")
            if checkpoint.get("dataset_fingerprint") != config["data"]["fingerprint"] or checkpoint.get("runtime_code_sha256") != current_hashes:
                raise ValueError("Resume checkpoint data/code identity is incompatible")
            if checkpoint.get("resolved_config", {}).get("model") != config["model"]:
                raise ValueError("Resume checkpoint model architecture is incompatible")
            validate_checkpoint_config(checkpoint, config)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _optimizer_to(optimizer, device)
            if checkpoint.get("scheduler_state_dict") is not None:
                raise ValueError("Checkpoint contains a scheduler but resolved config requests none")
            scaler.load_state_dict(checkpoint.get("amp_scaler_state_dict", {}))
            start_epoch = int(checkpoint["epoch"]) + 1
            best = {
                "val_loss": float(checkpoint["best_val_loss"]),
                "psnr": float(checkpoint["best_psnr"]),
                "ssim": float(checkpoint["best_ssim"]),
            }
            restore_rng_state(checkpoint.get("rng_state"))
            if checkpoint.get("dataloader_generator_state") is not None:
                generator.set_state(checkpoint["dataloader_generator_state"])
            logger.info("resumed checkpoint=%s next_epoch=%d", resume_path, start_epoch)
            del checkpoint
        if resume_path is not None and start_epoch <= int(config["training"]["epochs"]):
            history = run_dir / "log/metrics_history.json"
            if not history.is_file() or max((r["epoch"] for r in load_status_history(history)), default=0) != start_epoch - 1:
                raise ValueError("Resume must use the latest committed epoch; history would otherwise be duplicated")
        status = update_status(run_dir, status, status="running", end_time=None)
        save_yaml(run_dir / "config_resolved.yaml", config)
        atomic_json(run_dir / "config.json", config)
        if start_epoch <= int(config["training"]["epochs"]):
            best, status = train_model(
                model, train_loader, validation_loader, criterion, optimizer, scheduler, scaler,
                device, config, run_dir, logger, validation_logger, status,
                start_epoch=start_epoch, best=best,
            )
        status = update_status(run_dir, status, status="finalizing_validation")
        # Release optimizer/model GPU storage before reconstructing the selected model.
        del model, optimizer, scaler, criterion
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        summary = export_validation(run_dir, gpu=args.gpu, data_root_override=args.data_root)
        update_status(
            run_dir, status, status="completed",
            end_time=datetime.now().astimezone().isoformat(timespec="seconds"),
            best_psnr=best["psnr"], best_ssim=best["ssim"], best_val_loss=best["val_loss"],
            final_validation=summary,
        )
        logger.info("training and final Val export completed: %s", summary["result_directory"])
        print(f"RUN_DIR={run_dir}")
    except Exception as error:
        if logger is not None:
            logger.error("training failed\n%s", traceback.format_exc())
        else:
            traceback.print_exc()
        if run_dir is not None and status is not None:
            update_status(
                run_dir, status, status="failed",
                end_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                exception_type=type(error).__name__, exception_message=str(error),
            )
        raise


if __name__ == "__main__":
    main()
