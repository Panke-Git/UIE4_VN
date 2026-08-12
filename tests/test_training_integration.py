import json
import logging
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.v1.dataset import LSUIDataset
from src.v1.engine import train_model
from src.v1.experiment import update_status
from src.v1.losses import CharbonnierLoss
from src.v1.models import build_model


def _write_pairs(root: Path, split: str, sample_ids: list[str]) -> Path:
    rows = []
    input_dir = root / split / "input"
    gt_dir = root / split / "GT"
    input_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    generator = np.random.default_rng(3520 + len(sample_ids))
    for sample_id in sample_ids:
        input_array = generator.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        gt_array = np.clip(input_array.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        input_relative = Path(split) / "input" / f"{sample_id}.png"
        gt_relative = Path(split) / "GT" / f"{sample_id}.png"
        Image.fromarray(input_array, mode="RGB").save(root / input_relative)
        Image.fromarray(gt_array, mode="RGB").save(root / gt_relative)
        rows.append(f"{sample_id}\t{input_relative.as_posix()}\t{gt_relative.as_posix()}\n")
    manifest = root / f"{split}.tsv"
    manifest.write_text("".join(rows), encoding="utf-8")
    return manifest


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


def _disabled_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


def test_one_epoch_synthetic_training_pipeline(tmp_path) -> None:
    data_root = tmp_path / "data"
    train_manifest = _write_pairs(data_root, "train", ["0", "1", "2", "3"])
    validation_manifest = _write_pairs(data_root, "validation", ["10", "11"])
    augmentation = {"hflip": True, "vflip": True, "rot90": True}
    evaluation = {"resize": True, "size": 32}
    train_dataset = LSUIDataset(
        train_manifest,
        data_root,
        "train",
        patch_size=32,
        augmentation=augmentation,
        pad_if_smaller=True,
        pad_mode="reflect",
        evaluation=evaluation,
        verify_files=True,
    )
    validation_dataset = LSUIDataset(
        validation_manifest,
        data_root,
        "validation",
        patch_size=32,
        augmentation=augmentation,
        pad_if_smaller=True,
        pad_mode="reflect",
        evaluation=evaluation,
        verify_files=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample = train_dataset[0]
    assert sample["input"].shape == sample["target"].shape == (3, 32, 32)
    assert not any("not writable" in str(item.message).lower() for item in caught)

    generator = torch.Generator().manual_seed(3520)
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0, generator=generator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=2, shuffle=False, num_workers=0
    )
    config = {
        "experiment": {"version": "v1", "seed": 3520},
        "training": {
            "epochs": 1,
            "amp": False,
            "validate_every": 1,
            "save_every": 1,
            "gradient_clip_norm": None,
            "fail_on_nonfinite": True,
        },
        "checkpoint": {
            "save_best_psnr": True,
            "save_best_ssim": True,
            "save_best_val_loss": True,
            "save_last": True,
            "save_periodic": True,
        },
        "metrics": {
            "data_range": 1.0,
            "crop_border": 0,
            "ssim_window_size": 11,
            "ssim_sigma": 1.5,
        },
        "logging": {"log_every_steps": 0},
        "model": {
            "type": "nafnet_small",
            "img_channel": 3,
            "width": 8,
            "enc_blk_nums": [1],
            "middle_blk_num": 1,
            "dec_blk_nums": [1],
        },
    }
    torch.manual_seed(3520)
    model = build_model(config["model"])
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    criterion = CharbonnierLoss(1e-3)

    run_dir = tmp_path / "run"
    for directory in ("best", "checkpoint", "log", "result"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    state = update_status(
        run_dir,
        {"status": "running", "last_epoch": 0},
        status="running",
        last_epoch=0,
    )
    best, state = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        scaler=_disabled_scaler(),
        device=torch.device("cpu"),
        config=config,
        run_dir=run_dir,
        logger=_logger("synthetic.train"),
        validation_logger=_logger("synthetic.validation"),
        status=state,
    )
    state = update_status(
        run_dir,
        state,
        status="completed",
        best_psnr=best["psnr"],
        best_ssim=best["ssim"],
        best_val_loss=best["val_loss"],
    )

    assert state["status"] == "completed"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, model.parameters())
    )
    assert (run_dir / "checkpoint" / "last.pt").is_file()
    assert (run_dir / "checkpoint" / "epoch_0001.pt").is_file()
    assert (run_dir / "log" / "metrics_history.json").is_file()
    assert (run_dir / "result" / "validation_summary.json").is_file()
    history = json.loads((run_dir / "log" / "metrics_history.json").read_text())
    validation = json.loads((run_dir / "result" / "validation_summary.json").read_text())
    assert len(history["epochs"]) == 1
    assert validation["epoch"] == 1
    assert validation["sample_count"] == 2
