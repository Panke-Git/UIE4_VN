"""Frozen PNG8 paper-evaluation primitives for the selected UIE versions.

The protocol is derived from the independently audited FX evaluator but is
vendored here so UIE4_VN has no runtime dependency on the FX tree.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor


SUPPORTED_PAPER_VERSIONS = frozenset({"v4", "v13", "v14", "v15", "v16", "v17", "v18"})
PROTOCOL_VERSION = "uie4-paper-png8-1.0.0"
LPIPS_PACKAGE_VERSION = "0.1.4"
LPIPS_ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LPIPS_LINEAR_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
CANONICAL_TEST_MANIFESTS = {
    "LSUI19": (428, "cee6a22aeb2903f1cd053f641eab3aa1733f55a394682e257cb4ab4b27b0373c"),
    "UIEB": (90, "3693f31fdc849684706ad55f95e0c33e2b362082ced8346da033620856ffe07a"),
}
METRIC_CONFIG = {
    "data_range": 1.0,
    "crop_border": 0,
    "ssim_window_size": 11,
    "ssim_sigma": 1.5,
}


class DependencyBlocked(RuntimeError):
    """A required fixed paper-evaluation dependency is unavailable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_name(data_config: dict[str, Any]) -> str:
    configured = str(data_config.get("dataset", "")).strip()
    if configured:
        if configured not in CANONICAL_TEST_MANIFESTS:
            raise ValueError(
                f"Paper evaluation supports {sorted(CANONICAL_TEST_MANIFESTS)}, "
                f"got {configured!r}"
            )
        return configured
    evidence: set[str] = set()
    for key in ("root", "train_manifest", "validation_manifest", "test_manifest"):
        value = str(data_config.get(key, "")).replace("\\", "/").lower()
        if "lsui" in value:
            evidence.add("LSUI19")
        if "uieb" in value:
            evidence.add("UIEB")
    if len(evidence) != 1:
        raise ValueError(
            "Legacy config has no data.dataset and root/manifest metadata does not "
            "identify exactly one of LSUI19 or UIEB"
        )
    return next(iter(evidence))


def official_enhanced_names(entries: Sequence[Any]) -> dict[str, str]:
    """Reproduce the ordered naming code in every selected version's test.py."""
    result: dict[str, str] = {}
    used: set[str] = set()
    for entry in entries:
        sample_id = str(entry.sample_id)
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id in frozen test manifest: {sample_id}")
        stem = Path(str(entry.input_relative)).stem
        name = f"{stem}_enhanced.png"
        if name in used:
            name = f"{stem}_{sample_id}_enhanced.png"
        if name in used:
            raise ValueError(f"Unresolvable official enhanced filename collision: {name}")
        used.add(name)
        result[sample_id] = name
    return result


def validate_exact_prediction_directory(
    prediction_dir: Path, expected_names: Sequence[str]
) -> dict[str, Path]:
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory does not exist: {prediction_dir}. "
            "Run the official version test first."
        )
    children = list(prediction_dir.iterdir())
    non_files = sorted(path.name for path in children if not path.is_file())
    actual = {path.name for path in children if path.is_file()}
    expected = set(expected_names)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if non_files or missing or unexpected:
        raise ValueError(
            "Prediction directory must contain exactly the frozen test outputs: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} "
            f"non_files={non_files[:5]}"
        )
    return {name: prediction_dir / name for name in expected_names}


def decode_rgb_png8(path: Path, *, expected_size: tuple[int, int]) -> np.ndarray:
    raw = path.read_bytes()
    if (
        len(raw) < 29
        or raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[12:16] != b"IHDR"
        or raw[24:26] != bytes((8, 2))
        or b"acTL" in raw
    ):
        raise ValueError(f"Prediction must be a non-animated true RGB PNG8: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Prediction must be a single RGB PNG8 image: {path}")
        if image.size != expected_size:
            raise ValueError(
                f"Prediction size {image.size} != required {expected_size}: {path}; "
                "paper evaluation never resizes predictions"
            )
        return np.asarray(image, dtype=np.float32) / 255.0


def rgb_array_to_tensor(array: np.ndarray) -> Tensor:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB array, got {value.shape}")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("Expected finite RGB values in [0,1]")
    return torch.from_numpy(value.copy()).permute(2, 0, 1)


def select_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    elif device.type != "cpu":
        raise ValueError("Paper evaluation supports only cpu or cuda[:index]")
    return device


class LPIPSMetric:
    """Fixed official LPIPS 0.1.4 AlexNet configuration."""

    def __init__(self, device: torch.device, *, allow_download: bool = False):
        try:
            version = importlib.metadata.version("lpips")
            if version != LPIPS_PACKAGE_VERSION:
                raise DependencyBlocked(
                    f"Required lpips=={LPIPS_PACKAGE_VERSION}, got {version}"
                )
            import lpips
        except (ImportError, OSError, RuntimeError) as error:
            raise DependencyBlocked(
                f"LPIPS dependency unavailable or incompatible: {error}"
            ) from error

        self.device = device
        alexnet_path = (
            Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
        )
        if not alexnet_path.is_file() and not allow_download:
            raise DependencyBlocked(
                f"Pretrained AlexNet weights are missing: {alexnet_path}. "
                "Run once with --allow-lpips-download or populate TORCH_HOME."
            )
        try:
            self.model = lpips.LPIPS(
                net="alex",
                version="0.1",
                lpips=True,
                spatial=False,
                pretrained=True,
                pnet_rand=False,
                pnet_tune=False,
                eval_mode=True,
                verbose=False,
            ).to(device).float().eval()
        except (OSError, RuntimeError, ValueError, EOFError) as error:
            raise DependencyBlocked(f"LPIPS initialization failed: {error}") from error
        self.model.requires_grad_(False)
        linear_path = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
        self.weights = {
            "alexnet": {"path": str(alexnet_path), "sha256": sha256_file(alexnet_path)},
            "linear": {"path": str(linear_path), "sha256": sha256_file(linear_path)},
        }
        expected = {
            "alexnet": LPIPS_ALEXNET_SHA256,
            "linear": LPIPS_LINEAR_SHA256,
        }
        for name, digest in expected.items():
            if self.weights[name]["sha256"] != digest:
                raise DependencyBlocked(f"LPIPS {name} weights differ from fixed SHA256")

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("LPIPS expects matching BCHW RGB tensors")
        for value in (prediction, target):
            if not torch.isfinite(value).all() or bool(((value < 0) | (value > 1)).any()):
                raise ValueError("LPIPS input must be finite RGB [0,1]")
        self.model.eval()
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            values = self.model(
                prediction.to(self.device, dtype=torch.float32),
                target.to(self.device, dtype=torch.float32),
                normalize=True,
            )
        if values.numel() != prediction.shape[0] or not torch.isfinite(values).all():
            raise RuntimeError("LPIPS must return one finite scalar per image")
        return values.reshape(-1).detach().cpu()


def paper_metric_protocol() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "representation": "portable_png8",
        "aggregation": "arithmetic mean of per-image scalars; every image has equal weight",
        "prediction": "single opaque RGB PNG, 8-bit, 256x256; evaluator never resizes",
        "gt_preprocessing": "Pillow RGB bilinear resize to 256x256, float32 / 255",
        "psnr": {**METRIC_CONFIG, "direction": "higher_is_better"},
        "ssim": {**METRIC_CONFIG, "padding": "zero", "direction": "higher_is_better"},
        "delta_e00": {
            "implementation": "src.shared.e00.delta_e00_from_lab",
            "illuminant": "D65",
            "observer": "2",
            "kL": 1.0,
            "kC": 1.0,
            "kH": 1.0,
            "direction": "lower_is_better",
        },
        "lpips": {
            "package": "lpips",
            "package_version": LPIPS_PACKAGE_VERSION,
            "network": "alex",
            "version": "0.1",
            "lpips": True,
            "spatial": False,
            "pretrained": True,
            "pnet_rand": False,
            "pnet_tune": False,
            "normalize": True,
            "amp": False,
            "direction": "lower_is_better",
            "alexnet_weights_sha256": LPIPS_ALEXNET_SHA256,
            "linear_weights_sha256": LPIPS_LINEAR_SHA256,
        },
    }


def json_score(value: float) -> float | str:
    if math.isinf(value) and value > 0:
        return "Infinity"
    if not math.isfinite(value):
        raise ValueError(f"Cannot serialize non-finite paper metric: {value}")
    return value
