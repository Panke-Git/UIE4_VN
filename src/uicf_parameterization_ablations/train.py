"""Train an isolated correction parameterization with the unchanged V16 recipe."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from src.v16.dataset import LSUIDataset, validate_split_protocol
from src.v16.experiment import initialize_run, load_status, update_status, now_iso
from src.v16.losses import build_loss
from src.v16.train import _torch_load, _grad_scaler, _optimizer_to
from src.v16.utils import (atomic_json, load_yaml, manifest_path, project_path, restore_rng_state,
                          seed_everything, seed_worker, select_device, setup_logger)
from . import FORMULAS, VARIANTS
from .engine import train_epochs
from .models import build_model, parameter_counts
from .protocol import (SPLITS, code_version, config_path, manifest_hashes, same_config,
                       validate_config, verify_checkpoint, verify_saved_run)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--seed", type=int)
    p.add_argument("--gpu", type=int)
    p.add_argument("--data-root")
    p.add_argument("--output-dir", help="Parent directory for a new, non-overwriting run")
    p.add_argument("--resume", help="Own run's latest checkpoint/last.pt; never a Full V16 checkpoint")
    p.add_argument("--smoke-test", action="store_true", help="Accept only explicitly marked tiny synthetic config")
    p.add_argument("--dry-run", action="store_true", help="Validate config/manifests without building a model or writing a run")
    p.add_argument("--stop-after-epoch", type=int, help="Operational interruption after saving a complete epoch; total schedule stays unchanged")
    return p.parse_args(argv)


def load_configuration(args):
    checkpoint = None
    if args.resume:
        path = project_path(args.resume).resolve()
        if path.name != "last.pt" or path.parent.name != "checkpoint":
            raise ValueError("Resume requires the run's latest checkpoint/last.pt")
        run = path.parent.parent
        config = load_yaml(run/"config_resolved.yaml")
        smoke = bool(config.get("ablation", {}).get("smoke_test", False))
        validate_config(config, args.variant, allow_smoke=smoke)
        paths = verify_saved_run(run, config)
        if args.config:
            requested = load_yaml(project_path(args.config))
            original = deepcopy(config)
            original.pop("provenance", None)
            for value in (requested, original):
                value["experiment"]["output_root"] = str(project_path(value["experiment"]["output_root"]).resolve())
            if not same_config(requested, original):
                raise ValueError("Explicit resume config differs from saved recipe")
        for value, saved, label in ((args.seed, config["experiment"]["seed"], "seed"),
                                    (args.output_dir, config["experiment"]["output_root"], "output-dir")):
            if label == "output-dir" and value is not None:
                value, saved = str(project_path(value).resolve()), str(project_path(saved).resolve())
            if value is not None and value != saved:
                raise ValueError(f"Cannot change {label} during resume")
        checkpoint = _torch_load(path, torch.device("cpu"))
        verify_checkpoint(checkpoint, config)
        history = json.loads((run/"log/metrics_history.json").read_text())["epochs"]
        if not history or history[-1]["epoch"] != checkpoint["epoch"]:
            raise ValueError("Resume checkpoint/history epoch mismatch")
        if checkpoint["epoch"] >= config["training"]["epochs"]:
            raise ValueError("Run already reached its configured total epochs")
        source = run/"config_source.yaml"
    else:
        if args.config is None and args.variant is None:
            raise ValueError("Provide --config or --variant")
        source = project_path(args.config) if args.config else config_path(args.variant)
        config = load_yaml(source)
        if args.seed is not None:
            config["experiment"]["seed"] = args.seed
        if args.output_dir:
            config["experiment"]["output_root"] = str(project_path(args.output_dir).resolve())
        validate_config(config, args.variant, allow_smoke=args.smoke_test)
        resolve = project_path if config.get("ablation", {}).get("smoke_test", False) else manifest_path
        paths = {s: resolve(config["data"][s+"_manifest"]).resolve() for s in SPLITS}
        run = None
    if args.data_root:
        config["data"]["root"] = str(Path(args.data_root).expanduser().resolve())
    hashes = manifest_hashes(config, paths)
    entries = validate_split_protocol(paths, config["data"]["expected_counts"])
    return config, source, paths, entries, hashes, run, checkpoint


def main(argv=None):
    args = parse_args(argv)
    config, source, paths, entries, hashes, saved_run, checkpoint = load_configuration(args)
    variant = config["model"]["parameterization"]
    if args.dry_run:
        print(json.dumps({"variant": variant, "formula": FORMULAS[variant], "split_sha256": hashes,
                          "sample_counts": {s: len(x) for s, x in entries.items()},
                          "data_root": config["data"]["root"], "dry_run": True}, indent=2))
        return None
    if not Path(config["data"]["root"]).is_dir():
        raise FileNotFoundError("Set --data-root to the actual paired LSUI root; historical AutoDL paths are not assumed")
    smoke = config.get("ablation", {}).get("smoke_test", False)
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("Formal 200-epoch runs require the intended CUDA environment; use synthetic --smoke-test locally")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["training"]["deterministic"]))
    device = torch.device("cpu") if smoke else select_device(args.gpu)
    data = config["data"]
    datasets = {s: LSUIDataset(paths[s], Path(data["root"]), s, int(data["patch_size"]),
                              data["augmentation"], bool(data["pad_if_smaller"]), data["pad_mode"],
                              config["evaluation"], verify_files=True) for s in ("train", "validation")}
    # Exactly the V16 shared train/validation loader generator and worker seeding.
    generator = torch.Generator().manual_seed(seed)
    kwargs = dict(batch_size=int(data["batch_size"]), num_workers=int(data["num_workers"]),
                  pin_memory=bool(data["pin_memory"]) and device.type == "cuda",
                  worker_init_fn=seed_worker, generator=generator)
    train_loader = DataLoader(datasets["train"], shuffle=True, **kwargs)
    val_loader = DataLoader(datasets["validation"], shuffle=False, **kwargs)
    model = build_model(config["model"]).to(device)
    criterion = build_loss(config["loss"]).to(device)
    opt = config["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt["learning_rate"]),
                                 weight_decay=float(opt["weight_decay"]), betas=tuple(opt["betas"]))
    scaler = _grad_scaler(bool(config["training"]["amp"]) and device.type == "cuda")
    start, best = 1, None
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _optimizer_to(optimizer, device)
        scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        generator.set_state(checkpoint["dataloader_generator_state"])
        start = checkpoint["epoch"]+1
        best = {"psnr": checkpoint["best_psnr"], "ssim": checkpoint["best_ssim"], "val_loss": checkpoint["best_val_loss"]}
    if args.stop_after_epoch is not None and args.stop_after_epoch < start:
        raise ValueError("stop-after-epoch precedes next epoch")
    run_dir = None
    status = None
    logger = None
    try:
        if saved_run is None:
            config["provenance"] = dict(split_sha256=hashes, code=code_version(),
                                         parameter_counts=parameter_counts(model), formula=FORMULAS[variant])
            run_dir, status = initialize_run(config, source, model, device,
                                             {s: len(x) for s, x in entries.items()}, [sys.executable, *sys.argv])
            info = json.loads((run_dir/"run_info.json").read_text())
            info.update(variant=variant, formula=FORMULAS[variant], parameter_counts=parameter_counts(model),
                        smoke_test=bool(smoke), initialization="random, shared canonical V16 RNG order; no pretrained checkpoint")
            atomic_json(run_dir/"run_info.json", info)
        else:
            run_dir, status = saved_run, load_status(saved_run)
        status = update_status(run_dir, status, status="running", end_time=None, variant=variant, smoke_test=bool(smoke))
        logger = setup_logger(f"{config['experiment']['version']}.{run_dir.name}", run_dir/"log/train.log",
                              append=checkpoint is not None, console=bool(config["logging"]["console"]))
        logger.info("variant=%s formula=%s device=%s AMP_requested=%s AMP_enabled=%s", variant,
                    FORMULAS[variant], device, config["training"]["amp"], scaler.is_enabled())
        best, status = train_epochs(model, train_loader, val_loader, criterion, optimizer, scaler,
                                    device, config, run_dir, logger, status, start_epoch=start, best=best,
                                    stop_after_epoch=args.stop_after_epoch)
        completed = status["last_epoch"] == config["training"]["epochs"]
        update_status(run_dir, status, status="completed" if completed else "interrupted", end_time=now_iso())
        logger.info("Training %s; held-out test was NOT run", "completed" if completed else "interrupted after saved epoch")
        print(f"RUN_DIR={run_dir}")
        return run_dir
    except BaseException as exc:
        if run_dir is not None and status is not None:
            update_status(run_dir, status, status="failed", exception_type=type(exc).__name__, exception_message=str(exc), end_time=now_iso())
        raise
    finally:
        if logger:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    main()
