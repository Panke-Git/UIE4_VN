"""Local source/protocol/checkpoint guards, independent of every other variant."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .utils import PROJECT_ROOT, sha256_file
from .models import EXPECTED_TYPE
from .evaluation import evaluation_precision, evaluation_protocol

EXPECTED_VERSION = "qgrd_shared_query"
VARIANT = "shared_query"
OUTPUT_ROOTS = {
    "LSUI19": "experiments/qgrd_ablations/shared_query/lsui",
    "UIEB": "experiments/qgrd_ablations/shared_query/uieb",
}
PROTOCOLS = {
    "LSUI19": {
        "counts": {
            "train": 3466,
            "validation": 385,
            "test": 428
        },
        "manifests": {
            "train": "split/lsui19/train.tsv",
            "validation": "split/lsui19/validation.tsv",
            "test": "split/lsui19/test.tsv"
        },
        "hashes": {
            "train": "5cf9be63b7ed565ad3190936c61efe56c7b27c1e0cb7d8b0c9266ef62f87c6ab",
            "validation": "e81c35ae694ce9c0e2ba656ad5dece093ae7804d4ffd711eeb357103686f9c18",
            "test": "cee6a22aeb2903f1cd053f641eab3aa1733f55a394682e257cb4ab4b27b0373c"
        }
    },
    "UIEB": {
        "counts": {
            "train": 720,
            "validation": 80,
            "test": 90
        },
        "manifests": {
            "train": "split/uieb/train.tsv",
            "validation": "split/uieb/validation.tsv",
            "test": "split/uieb/test.tsv"
        },
        "hashes": {
            "train": "da477b9cb5588aaa113c0eb4a37caffcf5c37f13887e199543fee0b14c0ee7e9",
            "validation": "e263cfe365939550042bf68b89a7b1fca7940b9fb5980199f8c4372099de563f",
            "test": "3693f31fdc849684706ad55f95e0c33e2b362082ced8346da033620856ffe07a"
        }
    }
}


def verify_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    pinned = json.loads((package / "frozen_dependencies.json").read_text(encoding="utf-8"))
    for relative, expected in pinned.items():
        path = PROJECT_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen read-only source SHA256 mismatch: {relative}; restore the recorded source, do not silently repin")
    paths = sorted(package.rglob("*.py")) + [package / "frozen_dependencies.json"]
    return {**pinned, **{p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in paths}}


def validate_config(config: dict) -> None:
    if config["experiment"]["version"] != EXPECTED_VERSION:
        raise ValueError(f"{EXPECTED_VERSION} refuses experiment variant/version {config['experiment']['version']!r}")
    if config["model"].get("type") != EXPECTED_TYPE:
        raise ValueError(f"{VARIANT} refuses model.type {config['model'].get('type')!r}")
    evaluation_precision(config)
    dataset = config["data"].get("dataset")
    if dataset not in PROTOCOLS:
        raise ValueError(f"Unsupported dataset protocol: {dataset!r}")
    protocol = PROTOCOLS[dataset]
    if config["data"].get("expected_counts") != protocol["counts"]:
        raise ValueError("Dataset protocol expected_counts differ from the original V16 three-set protocol")
    for split in ("train", "validation", "test"):
        if config["data"][f"{split}_manifest"].replace("\\", "/") != protocol["manifests"][split]:
            raise ValueError(f"Dataset protocol {split} manifest differs from original V16")
    if config["experiment"]["output_root"] != OUTPUT_ROOTS[dataset]:
        raise ValueError("Output root must be the isolated variant/dataset directory")
    if config["test"].get("auto_run_after_training", False):
        raise ValueError("Held-out testing must remain explicitly invoked")
    if config["test"]["checkpoint"] != "best_psnr":
        raise ValueError("Controlled comparison requires default best_psnr checkpoint")


def protocol_identity(config: dict) -> dict:
    dataset = config["data"]["dataset"]
    return {"dataset": dataset, **deepcopy(PROTOCOLS[dataset])}


def validate_manifest_hashes(config: dict, manifests: dict[str, Path]) -> dict[str, str]:
    actual = {split: sha256_file(path) for split, path in manifests.items()}
    expected = PROTOCOLS[config["data"]["dataset"]]["hashes"]
    if actual != expected:
        raise ValueError("Dataset protocol manifest SHA256 mismatch; never re-split or substitute manifests")
    return actual


def _portable_config(config: dict) -> dict:
    result = deepcopy(config)
    result["data"].pop("root", None)  # Explicit --data-root relocation is the only config exception.
    return result


def assert_same_config(saved: dict, requested: dict, context: str) -> None:
    if _portable_config(saved) != _portable_config(requested):
        raise ValueError(f"{context}: resolved_config/variant/dataset/training protocol mismatch")


def verify_run_metadata(run_dir: Path, config: dict, sources: dict, manifests: dict) -> dict:
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    if info.get("version") != EXPECTED_VERSION or info.get("variant") != VARIANT:
        raise ValueError("Run variant/version mismatch")
    assert_same_config(info["resolved_config"], config, "Run")
    if info.get("source_sha256") != sources:
        raise ValueError("Run source SHA256 differs from current experiment/dependency sources")
    hashes = validate_manifest_hashes(config, manifests)
    if any(info.get(f"{split}_tsv_sha256") != digest for split, digest in hashes.items()):
        raise ValueError("Frozen split_snapshot differs from run_info hashes")
    if info.get("protocol_identity") != protocol_identity(config):
        raise ValueError("Run dataset protocol identity mismatch")
    if (info.get("evaluation_precision") != evaluation_precision(config)
            or info.get("evaluation_forward_protocol") != evaluation_protocol(config)):
        raise ValueError("Run evaluation precision protocol mismatch")
    return info


def verify_checkpoint(checkpoint: dict, config: dict, sources: dict) -> None:
    validate_config(config)
    if checkpoint.get("version") != EXPECTED_VERSION or checkpoint.get("variant") != VARIANT:
        raise ValueError("Checkpoint variant/version incompatible; cross-variant loading is forbidden")
    assert_same_config(checkpoint.get("resolved_config", {}), config, "Checkpoint")
    if checkpoint.get("model_type") != EXPECTED_TYPE:
        raise ValueError("Checkpoint model type incompatible")
    if checkpoint.get("source_sha256") != sources:
        raise ValueError("Checkpoint source SHA256 differs from current sources")
    if checkpoint.get("protocol_identity") != protocol_identity(config):
        raise ValueError("Checkpoint dataset protocol mismatch")
    if (checkpoint.get("evaluation_precision") != evaluation_precision(config)
            or checkpoint.get("evaluation_forward_protocol") != evaluation_protocol(config)):
        raise ValueError("Checkpoint evaluation precision protocol mismatch")
