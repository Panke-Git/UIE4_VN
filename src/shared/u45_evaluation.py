"""Self-contained official U45 identity and no-reference metric protocol.

This module vendors the audited FX_UIQM_v2 and FX_UCIQE_CIELAB_v1 numerical
definitions so UIE4_VN can be deployed without importing the sibling FX tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import color, filters
from skimage.color import rgb2lab

from .paper_evaluation import sha256_file


U45_PROTOCOL_VERSION = "UIE4_U45_NR_v1"
UIQM_VARIANT = "FX_UIQM_v2"
UCIQE_VARIANT = "FX_UCIQE_CIELAB_v1"
U45_SOURCE_REPOSITORY = "IPNUISTlegal/underwater-test-dataset-U45-"
U45_SOURCE_COMMIT = "58142a16017af9c40f36d25f5e8d383d7370d628"
U45_SOURCE_SUBTREE = "upload/U45/U45"
UIQM_COEFFICIENTS = (0.0282, 0.2953, 3.5753)
UCIQE_COEFFICIENTS = (0.4680, 0.2745, 0.2576)

# Git tree evidence copied from the fixed official source commit above.  The
# value is (Git blob SHA1, byte count); verification uses the actual Git blob
# object hash, not a filename/count-only guess.
OFFICIAL_U45_INDEX: dict[str, tuple[str, int]] = {
    "1.png": ("7ee9ba053a44b97ee9dd6e3fea8832bbffb05fa0", 79577),
    "10.png": ("9fd5f933b1432e99c51f2131e151a22411ab2d68", 114054),
    "11.png": ("b7776a86261ddc15c08183b664cd481fa0367aa4", 71157),
    "12.png": ("245e08ec3f31f7e467245d12903b0d77b50bc533", 114624),
    "13.png": ("288e9fe7e1ebac02382b81dd271aff297fb3326e", 135904),
    "14.png": ("8e20bbff01701bdfb800ccdd642325ee2a920291", 113958),
    "15.png": ("cf2653759d1d31304d8fd63984fb46e1229e2b93", 138872),
    "16.png": ("60a0a961231829f1c4ba7274bdc852a0ad642097", 105747),
    "17.png": ("f187453d8994da416b9e1f23e6e5e53d212d0f80", 99604),
    "18.png": ("b8e00d6be6b4d550aed322ae3d0595518a3ee86f", 117523),
    "19.png": ("4c1c72c12afb9c600dd0709b6b0728481221533e", 92319),
    "2.png": ("8036d54933fb720b6451b5bc9e731e0093ea54d5", 93839),
    "20.png": ("6b7cc04965d19a8f7fefd01112d9ad78b492ee18", 93043),
    "21.png": ("7f1dc6bdc36cc323b5b624a070773aba81c635a7", 121076),
    "22.png": ("be03a0860fadcc0668262231bbe446a37be1aa51", 81773),
    "23.png": ("04d26d37f77ad282d30bfd247c554f989dbaf415", 102246),
    "24.png": ("485a85435f7795c5b373de15f8f4f9928da15845", 101485),
    "25.png": ("8e156983006e298c9be0ad1aa7d3deb43db65ffd", 96862),
    "26.png": ("fdd14711a6ceb278d982b14a0c7684fc6c80d689", 80031),
    "27.png": ("6ed6a4c9909f6fc6090e4d17143c3c375d90e54b", 108943),
    "28.png": ("716f5edee65c895a01d82323011c3a766830a97d", 117214),
    "29.png": ("c8b9413c467be7f5e4d37fa09fb362a277931a05", 15115),
    "3.png": ("eaf6db85072ed6b41c4e1fb40f95e0300ba53421", 101035),
    "30.png": ("8b8294dacb2452d7746f152b13b59c44dbbb824c", 84222),
    "31.png": ("c9a5a61f456c74c31f77fa3bda1760fa5d363e88", 108351),
    "32.png": ("820cd50b61adb817fd5238124109359055b5329b", 102043),
    "33.png": ("5b402fbc225a405db47073727c4bba3b85702b75", 123354),
    "34.png": ("04ae5c8189f0bc9f26fe7268e7324a65903d25f9", 115529),
    "35.png": ("d99a4459fae3d25f21d56b7ea78e93ef38b46b74", 113216),
    "36.png": ("96ea44410773d8e69d111ee0441379a8fe5b42c1", 115213),
    "37.png": ("4a022d687f3a1726ca9ae6cd5c4f6591f91abb15", 113814),
    "38.png": ("525187064acb3fea230ae4228f998377c1c9982a", 112190),
    "39.png": ("afa5c6287297584a381f9ece7b91c23457095360", 89291),
    "4.png": ("df5923970fe68066525e1a5116d5f5114881c40a", 71662),
    "40.png": ("6b1a5a53ab50dbc7376a1ca17d56d8637a5ea0fa", 124235),
    "41.png": ("73d227e597a8cdb9df0bcf8366d13a95c96eca97", 110943),
    "42.png": ("aace57f1632b9e10761662c01dd983bd1184daa9", 90193),
    "43.png": ("cbc20c4d290dfd2e900c1e488b669549a15fdd41", 102784),
    "44.png": ("d4e5415ac02facd7e623fc63c9318a22462600c3", 114745),
    "45.png": ("a4eb11acb11b93ea14d0efb31b08b9c21f77acf3", 117111),
    "5.png": ("e7dcb0ad333767b25ce6dd170d6d8d3ff3df379b", 99770),
    "6.png": ("b9c3002594bf2518ca44aa239548a635eb8e234d", 88967),
    "7.png": ("7381854a008e0d9f3e2fcbcc3ba8bc1fca9be9dc", 106492),
    "8.png": ("f763ca6285e867407c4fdf71585f455417925d25", 75483),
    "9.png": ("115a5648a602e32205b5143c4906b19b3ca2d09d", 102907),
}


@dataclass(frozen=True)
class U45Entry:
    sample_id: str
    filename: str
    path: Path
    sha256: str
    git_blob_sha1: str
    width: int
    height: int
    original_mode: str


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def validate_rgb8(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError(f"Expected non-empty HWC RGB, got {rgb.shape}")


def decode_canonical_input(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as image:
        image.load()
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Animated U45 input is forbidden: {path}")
        if image.mode not in ("RGB", "L", "P") or "transparency" in image.info:
            raise ValueError(f"U45 input is not opaque RGB-compatible: {path} mode={image.mode}")
        mode = image.mode
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    validate_rgb8(array)
    return array, mode


def inspect_official_u45(data_root: Path) -> list[U45Entry]:
    root = data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"U45 clean input root does not exist: {root}")
    children = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("U45 clean root must be flat, contain files only, and contain no symlinks")
    actual_names = {path.name for path in children}
    expected_names = set(OFFICIAL_U45_INDEX)
    if actual_names != expected_names:
        raise ValueError(
            "U45 clean root differs from the fixed official 45-file index: "
            f"missing={sorted(expected_names - actual_names)[:5]} "
            f"unexpected={sorted(actual_names - expected_names)[:5]}"
        )
    entries: list[U45Entry] = []
    for filename in sorted(expected_names, key=lambda value: value.encode("utf-8")):
        path = root / filename
        raw = path.read_bytes()
        expected_blob, expected_size = OFFICIAL_U45_INDEX[filename]
        if len(raw) != expected_size or _git_blob_sha1(raw) != expected_blob:
            raise ValueError(
                f"U45 source identity mismatch for {filename}; use unchanged files from "
                f"{U45_SOURCE_REPOSITORY}@{U45_SOURCE_COMMIT}/{U45_SOURCE_SUBTREE}"
            )
        rgb, mode = decode_canonical_input(path)
        entries.append(
            U45Entry(
                sample_id=path.stem,
                filename=filename,
                path=path,
                sha256=hashlib.sha256(raw).hexdigest(),
                git_blob_sha1=expected_blob,
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
                original_mode=mode,
            )
        )
    return entries


def decode_prediction_png8(path: Path, *, expected_size: tuple[int, int]) -> np.ndarray:
    raw = path.read_bytes()
    if (
        len(raw) < 29
        or raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[12:16] != b"IHDR"
        or raw[24:26] != bytes((8, 2))
        or b"acTL" in raw
    ):
        raise ValueError(f"U45 prediction must be a non-animated true RGB PNG8: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"U45 prediction must be a single RGB PNG8: {path}")
        if image.size != expected_size:
            raise ValueError(
                f"U45 prediction size {image.size} != original input {expected_size}: {path}"
            )
        array = np.asarray(image, dtype=np.uint8)
    validate_rgb8(array)
    return array


def _eme(channel: np.ndarray, block_size: int = 8) -> float:
    num_x = math.ceil(channel.shape[0] / block_size)
    num_y = math.ceil(channel.shape[1] / block_size)
    result = 0.0
    weight = 2.0 / (num_x * num_y)
    for i in range(num_x):
        for j in range(num_y):
            block = channel[
                i * block_size : min((i + 1) * block_size, channel.shape[0]),
                j * block_size : min((j + 1) * block_size, channel.shape[1]),
            ]
            minimum = float(np.min(block))
            maximum = float(np.max(block))
            if minimum == 0.0:
                minimum += 1.0
            if maximum == 0.0:
                maximum += 1.0
            result += weight * math.log(maximum / minimum)
    return result


def _plip_sum(first: float, second: float, gamma: float = 1026.0) -> float:
    return first + second - first * second / gamma


def _plip_sub(first: float, second: float, k: float = 1026.0) -> float:
    return k * (first - second) / (k - second)


def _plip_mult(coefficient: float, value: float, gamma: float = 1026.0) -> float:
    return gamma - gamma * (1.0 - value / gamma) ** coefficient


def _logamee_v2(channel: np.ndarray, block_size: int = 8) -> float:
    num_x = math.ceil(channel.shape[0] / block_size)
    num_y = math.ceil(channel.shape[1] / block_size)
    total = 0.0
    weight = 1.0 / (num_x * num_y)
    for i in range(num_x):
        for j in range(num_y):
            block = channel[
                i * block_size : min((i + 1) * block_size, channel.shape[0]),
                j * block_size : min((j + 1) * block_size, channel.shape[1]),
            ]
            minimum = float(np.min(block))
            maximum = float(np.max(block))
            top = _plip_sub(maximum, minimum)
            bottom = _plip_sum(maximum, minimum)
            local = 0.0 if bottom == 0.0 else top / bottom
            if local != 0.0:
                total += local * np.log(local)
    return _plip_mult(weight, total)


def uiqm_components(rgb: np.ndarray) -> dict[str, float]:
    validate_rgb8(rgb)
    if rgb.shape[0] * rgb.shape[1] < 10:
        raise ValueError("UIQM is undefined for fewer than 10 pixels")
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            gray = color.rgb2gray(rgb)
            # Uint8 wraparound is intentionally preserved for FX_UIQM_v2 parity.
            rg = rgb[:, :, 0] - rgb[:, :, 1]
            yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
            rg_sorted = np.sort(rg, axis=None)
            yb_sorted = np.sort(yb, axis=None)
            trim = int(0.1 * len(rg_sorted))
            rg_trimmed = rg_sorted[trim:-trim]
            yb_trimmed = yb_sorted[trim:-trim]
            rg_mean = np.mean(rg_trimmed)
            yb_mean = np.mean(yb_trimmed)
            uicm = -0.0268 * np.sqrt(rg_mean**2 + yb_mean**2) + 0.1586 * np.sqrt(
                np.mean((rg_trimmed - rg_mean) ** 2)
                + np.mean((yb_trimmed - yb_mean) ** 2)
            )
            red_sobel = np.round(rgb[:, :, 0] * filters.sobel(rgb[:, :, 0])).astype(np.uint8)
            green_sobel = np.round(rgb[:, :, 1] * filters.sobel(rgb[:, :, 1])).astype(np.uint8)
            blue_sobel = np.round(rgb[:, :, 2] * filters.sobel(rgb[:, :, 2])).astype(np.uint8)
            uism = 0.299 * _eme(red_sobel) + 0.587 * _eme(green_sobel) + 0.114 * _eme(blue_sobel)
            uiconm = _logamee_v2(gray)
            score = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    except (ZeroDivisionError, FloatingPointError) as error:
        raise ValueError("UIQM produced an undefined non-finite operation") from error
    values = {"uicm": float(uicm), "uism": float(uism), "uiconm": float(uiconm), "uiqm": float(score)}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("UIQM produced a non-finite value")
    return values


def uiqm(rgb: np.ndarray) -> float:
    return uiqm_components(rgb)["uiqm"]


def uciqe_components(rgb: np.ndarray) -> dict[str, float | int]:
    validate_rgb8(rgb)
    lab = rgb2lab(
        rgb.astype(np.float64) / 255.0,
        illuminant="D65",
        observer="2",
        channel_axis=-1,
    )
    luminance = lab[:, :, 0]
    chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)
    sigma_c = float(np.std(chroma, ddof=0))
    tail_count = max(1, int(np.round(0.01 * luminance.size)))
    ordered = np.sort(luminance, axis=None)
    con_l = float(np.mean(ordered[-tail_count:]) - np.mean(ordered[:tail_count]))
    saturation = np.divide(
        chroma,
        luminance,
        out=np.zeros_like(chroma),
        where=luminance != 0,
    )
    mu_s = float(np.mean(saturation))
    score = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
    if not math.isfinite(score):
        raise ValueError("UCIQE produced a non-finite value")
    return {
        "sigma_c": sigma_c,
        "con_l": con_l,
        "mu_s": mu_s,
        "tail_count": tail_count,
        "uciqe": float(score),
    }


def uciqe(rgb: np.ndarray) -> float:
    return float(uciqe_components(rgb)["uciqe"])


def u45_metric_protocol() -> dict[str, Any]:
    return {
        "protocol_version": U45_PROTOCOL_VERSION,
        "dataset": "U45",
        "dataset_role": "test_only_no_reference",
        "expected_count": 45,
        "source_repository": U45_SOURCE_REPOSITORY,
        "source_commit": U45_SOURCE_COMMIT,
        "source_subtree": U45_SOURCE_SUBTREE,
        "prediction": "single opaque RGB PNG8 at original input width and height",
        "aggregation": "arithmetic mean of all 45 per-image scalars",
        "uiqm": {
            "variant": UIQM_VARIANT,
            "coefficients": UIQM_COEFFICIENTS,
            "block_size": 8,
            "all_black_block_policy": "zero local contrast",
            "direction": "higher_is_better",
        },
        "uciqe": {
            "variant": UCIQE_VARIANT,
            "coefficients": UCIQE_COEFFICIENTS,
            "lab": "skimage rgb2lab float64 RGB/255, D65, observer 2",
            "chroma_std": "population ddof=0",
            "tail_count": "max(1, round_ties_to_even(0.01*H*W))",
            "direction": "higher_is_better",
            "comparability_notice": (
                "Compare only values recomputed with this frozen implementation; "
                "literature UCIQE values from other implementations are not interchangeable."
            ),
        },
    }
