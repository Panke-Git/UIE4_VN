"""Recipe, source, fixed split and checkpoint identity checks. No model mutation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

from src.v16.utils import PROJECT_ROOT, load_yaml, project_path, sha256_file
from . import FORMULAS, VARIANTS, version

SPLITS = ("train", "validation", "test")
COUNTS = dict(train=3466, validation=385, test=428)
SPLIT_HASHES = {
    "train": "5cf9be63b7ed565ad3190936c61efe56c7b27c1e0cb7d8b0c9266ef62f87c6ab",
    "validation": "e81c35ae694ce9c0e2ba656ad5dece093ae7804d4ffd711eeb357103686f9c18",
    "test": "cee6a22aeb2903f1cd053f641eab3aa1733f55a394682e257cb4ab4b27b0373c",
}


def config_path(variant: str) -> Path:
    version(variant)
    return PROJECT_ROOT / f"configs/ablations/uicf_parameterization/{variant}.yaml"


def validate_config(config: dict, expected_variant: str | None = None, *, allow_smoke: bool = False) -> str:
    variant = config["model"].get("parameterization")
    version(variant)
    if expected_variant is not None and variant != expected_variant:
        raise ValueError("Requested variant differs from saved config variant")
    if config["experiment"]["version"] != version(variant):
        raise ValueError("Config version/variant mismatch")
    if config["model"]["type"] != "uicf_parameterization_pre_backbone":
        raise ValueError("Config is not a correction-parameterization model")
    if config.get("ablation", {}).get("smoke_test", False):
        if not allow_smoke:
            raise ValueError("Synthetic smoke config requires --smoke-test; never a paper experiment")
        if config["training"]["epochs"] > 2 or config["data"]["patch_size"] > 32 or config["model"]["base_channels"] > 8:
            raise ValueError("Smoke mode permits only tiny models and at most two epochs")
        if config["data"]["expected_counts"] != dict(train=2, validation=1, test=1):
            raise ValueError("Smoke split must be 2/1/1")
    else:
        baseline = load_yaml(PROJECT_ROOT / "configs/config_v16.yaml")
        for section in ("optimizer", "loss", "scheduler", "training", "checkpoint", "evaluation", "metrics", "test"):
            if config[section] != baseline[section]:
                raise ValueError(f"Original V16 recipe differs in {section}")
        for key, value in baseline["data"].items():
            if key != "root" and config["data"].get(key) != value:
                raise ValueError(f"Original V16 data protocol differs: {key}")
        if config["experiment"]["seed"] != baseline["experiment"]["seed"]:
            raise ValueError("This controlled LSUI protocol requires seed 3520")
        model = deepcopy(config["model"])
        model.pop("parameterization")
        model["type"] = baseline["model"]["type"]
        # All three variants deliberately discard image-adaptive anchoring.
        model["uicf"].pop("use_learned_anchor", None)
        if model != baseline["model"]:
            raise ValueError("Backbone/QGRD/field architecture differs from original V16")
        for split in SPLITS:
            if config["data"][f"{split}_manifest"] != f"split/lsui19/{split}.tsv":
                raise ValueError("Use the original project-local LSUI split, never regenerate it")
    field = config["model"]["uicf"]
    if field.get("use_learned_anchor", False) is not False:
        raise ValueError("Parameterization ablations must disable the learned anchor")
    if any(field.get(flag, True) is not True for flag in ("use_spatial_conditioning", "use_global_field_conditioning")):
        raise ValueError("Keep spatial and global field conditioning enabled")
    return variant


def manifest_hashes(config: dict, paths: dict[str, Path]) -> dict:
    hashes = {split: sha256_file(path) for split, path in paths.items()}
    expected = config.get("provenance", {}).get("split_sha256")
    if not config.get("ablation", {}).get("smoke_test", False):
        expected = SPLIT_HASHES
    if expected is not None and hashes != expected:
        raise ValueError("Frozen manifest SHA256 mismatch")
    return hashes


def source_hashes() -> dict[str, str]:
    paths = []
    for folder in ("src/uicf_parameterization_ablations", "src/v16", "src/v15", "src/v4"):
        paths.extend((PROJECT_ROOT/folder).rglob("*.py"))
    paths.extend(PROJECT_ROOT/p for p in ("src/shared/uicf_inr.py", "src/shared/uicf_models.py",
                 "src/shared/color_query_unet.py", "src/shared/e00.py", "configs/config_v16.yaml"))
    return {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sorted(set(paths))}


def code_version() -> dict:
    def git(*args):
        p = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
            "git_status": git("status", "--short"), "source_sha256": source_hashes()}


def same_config(a: dict, b: dict) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict) or "data" not in a or "data" not in b:
        return False
    a, b = deepcopy(a), deepcopy(b)
    # The only allowed saved-config difference is relocating identical data.
    a["data"].pop("root", None)
    b["data"].pop("root", None)
    return a == b


def verify_checkpoint(checkpoint: dict, config: dict) -> None:
    variant = config["model"]["parameterization"]
    if checkpoint.get("version") != version(variant) or checkpoint.get("variant") != variant:
        raise ValueError("Checkpoint variant/version mismatch; Full V16 checkpoints cannot initialize these experiments")
    if checkpoint.get("formula") != FORMULAS[variant] or not same_config(checkpoint.get("resolved_config", {}), config):
        raise ValueError("Checkpoint config/formula mismatch")
    for key in ("model_state_dict", "optimizer_state_dict", "amp_scaler_state_dict", "rng_state", "dataloader_generator_state", "training_status"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint lacks required state: {key}")
    if checkpoint.get("seed") != config["experiment"]["seed"]:
        raise ValueError("Checkpoint seed mismatch")
    if checkpoint.get("scheduler_state_dict") is not None:
        raise ValueError("Original V16 protocol has no scheduler")
    if checkpoint.get("split_sha256") != config["provenance"]["split_sha256"] or checkpoint.get("code_version") != config["provenance"]["code"]:
        raise ValueError("Checkpoint source/split provenance mismatch")
    state = checkpoint["training_status"]
    if state.get("last_epoch") != checkpoint.get("epoch") or state.get("configured_epochs") != config["training"]["epochs"]:
        raise ValueError("Checkpoint training status/epoch mismatch")


def verify_saved_run(run: Path, config: dict) -> dict[str, Path]:
    info = json.loads((run/"run_info.json").read_text())
    if info.get("variant") != config["model"]["parameterization"] or not same_config(info["resolved_config"], config):
        raise ValueError("Run metadata/config variant mismatch")
    if config["provenance"]["code"]["source_sha256"] != source_hashes():
        raise ValueError("Runtime source SHA256 differs from the saved run; deploy the matching code")
    paths = {split: run/"split_snapshot"/f"{split}.tsv" for split in SPLITS}
    manifest_hashes(config, paths)
    return paths
