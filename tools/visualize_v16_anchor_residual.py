#!/usr/bin/env python3
"""Derive signed RGB intermediates from one verified V16 raw capture, without a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FORMULA = "I_c = I + R * (I - b[:, :, None, None]); no activation or clamp"
TITLES = ("I(x) − b", "R(x) ⊙ [I(x) − b]")
CAPTION = "Signed RGB visualization; shared scale; zero = mid-gray."


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(value):
    a = np.asarray(value, dtype=np.float64)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std(ddof=0))}


def describe(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "all": summary(value),
            "channels": {c: summary(value[0, i]) for i, c in enumerate("RGB")}}


def derive(I, b, R, I_c):
    if I.ndim != 4 or I.shape[:2] != (1, 3):
        raise ValueError(f"Expected one BCHW RGB sample, got {I.shape}")
    if b.shape != (1, 3) or R.shape != I.shape or I_c.shape != I.shape:
        raise ValueError("I, b, R and I_c have incompatible shapes")
    for name, value in (("I", I), ("b", b), ("R", R), ("I_c", I_c)):
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating-point values")
    # Retain the stored operand dtypes. In the real capture float32 I promotes
    # float16 b/R, exactly as the original PyTorch elementwise reconstruction.
    A = I - b[:, :, None, None]
    delta = R * A
    reconstructed = I + delta
    for value in (A, delta, reconstructed):
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite derived tensor")
    error = float(np.abs(reconstructed - I_c).max())
    if not np.allclose(reconstructed, I_c, atol=1e-6, rtol=1e-5):
        raise ValueError(f"Same-stage UICF reconstruction mismatch: max_abs_error={error}")
    return A, delta, reconstructed, error


def signed_display(A, delta, crop):
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= A.shape[-1] and 0 <= y0 < y1 <= A.shape[-2]):
        raise ValueError(f"Invalid valid-region crop: {crop}")
    valid = tuple(v[0, :, y0:y1, x0:x1] for v in (A, delta))
    # Float64 is only a display copy; raw A/delta keep their computation dtype.
    S = max(float(np.abs(v.astype(np.float64)).max()) for v in valid)
    displays = []
    for value in valid:
        v = (np.full(value.shape, 0.5, dtype=np.float64) if S == 0 else
             0.5 + 0.5 * value.astype(np.float64, copy=True) / S)
        if not np.isfinite(v).all() or v.min() < 0 or v.max() > 1:
            raise ValueError("Signed RGB display is outside [0,1]")
        # Zero maps to 0.5; quantized RGB represents this by round(127.5)=128.
        displays.append(np.rint(v.transpose(1, 2, 0) * 255).astype(np.uint8))
    return tuple(displays), S


def fresh_output(base):
    base = Path(base)
    if base.exists() and any(base.iterdir()):
        base = base.with_name(base.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def render(displays, output, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "none", "svg.hashsalt": "v16-anchor-residual",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    h, w = displays[0].shape[:2]
    stems = ("uicf_anchor_relative_signed", "uicf_correction_residual_signed")
    compacts = ("uicf_anchor_relative_compact", "uicf_correction_residual_compact")
    for image, title, stem, compact in zip(displays, TITLES, stems, compacts):
        Image.fromarray(image).save(output / f"{compact}.png", dpi=(dpi, dpi))
        fig, ax = plt.subplots(figsize=(3.6, 3.6 * h / w + 0.42), layout="constrained")
        ax.imshow(image, interpolation="nearest", resample=False, aspect="equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=13, pad=8)
        for extension in ("png", "pdf", "svg"):
            metadata = ({"Date": None} if extension == "svg" else
                        {"CreationDate": None, "ModDate": None} if extension == "pdf" else None)
            fig.savefig(output / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight",
                        pad_inches=0.06, metadata=metadata)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5 * h / w + 0.85), layout="constrained")
    for ax, image, title in zip(axes, displays, TITLES):
        ax.imshow(image, interpolation="nearest", resample=False, aspect="equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=13, pad=8)
    fig.supxlabel(CAPTION, fontsize=10)
    fig.savefig(output / "uicf_anchor_residual_comparison.png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return matplotlib.__version__


def load_verified(source):
    raw, meta = source / "uicf_raw.npz", source / "metadata.json"
    m = json.loads(meta.read_text())
    if sha256(raw) != m["sha256"]["raw_npz"]:
        raise ValueError("Previous raw NPZ hash differs from metadata")
    with np.load(raw, allow_pickle=False) as saved:
        required = {"I", "b", "R", "I_c", "field_mlp_output"}
        if not required.issubset(saved.files):
            raise ValueError(f"Incomplete previous raw capture: missing {required - set(saved.files)}")
        a = {key: saved[key].copy() for key in required}
    if m["reconstruction"]["formula"] != FORMULA or m["reconstruction"]["clamp"] is not False:
        raise ValueError("Unknown reconstruction formula/stage; refusing an assumed formula")
    if not m["field"]["is_actual_reconstruction_coefficient"] or m["geometry"]["channel_order"] != "RGB":
        raise ValueError("Previous capture is not the actual RGB reconstruction coefficient")
    if m["validation"]["forward_call_counts"] != {"uicf": 1, "backbone": 1, "field_mlp": 1}:
        raise ValueError("Previous capture does not identify one matched inference")
    if a["b"][0].tolist() != m["anchor"]["values_rgb"]:
        raise ValueError("Raw anchor differs from the recorded sample anchor")
    flat = a.pop("field_mlp_output")
    if not np.array_equal(flat.reshape(1, *a["I"].shape[-2:], 3).transpose(0, 3, 1, 2), a["R"]):
        raise ValueError("Stored R differs from the captured MLP reconstruction field")
    verified = {}
    for key in ("image", "checkpoint", "config"):
        path = Path(m["paths"][key])
        if sha256(path) != m["sha256"][key]:
            raise ValueError(f"Original {key} changed: {path}")
        verified[key] = str(path)
    # Check relevant current source against the source of the original inference.
    source_hashes = m.get("inference_source_sha256", m["source_sha256"])
    for name in ("src/shared/uicf_inr.py", "src/v16/dataset.py"):
        path = ROOT / name
        if sha256(path) != source_hashes[str(path)]:
            raise ValueError(f"Relevant implementation changed since original capture: {path}")
    geometry = m["geometry"]
    if any(geometry["external_padding_lrtb"]):
        raise ValueError("External padding needs explicit input provenance handling; this capture used none")
    with Image.open(verified["image"]) as im:
        if im.mode != "RGB" or list(im.size) != geometry["original_size_wh"]:
            raise ValueError("Original image mode/size differs")
        if geometry["resize"]["enabled"]:
            if geometry["resize"]["method"] != "PIL.Image.Resampling.BILINEAR":
                raise ValueError("Unknown prior resize method")
            im = im.resize(tuple(geometry["resize"]["to_wh"]), Image.Resampling.BILINEAR)
        reference = (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    if not np.array_equal(reference, a["I"]):
        raise ValueError("Stored formula input differs from original image with recorded preprocessing")
    return m, a, {"raw_sha256_matches": True, "original_image_checkpoint_config_hashes_match": True,
                  "relevant_source_hashes_match": True, "formula_input_matches_original_preprocessing_exactly": True,
                  "R_matches_original_field_mlp": True, "b_matches_original_metadata": True,
                  "same_original_forward_confirmed": True, "model_loads": 0, "model_forward_calls": 0}


def write_readme(output, m, command):
    text = f"""# UICF anchor-relative input and correction residual

本次读取先前同一次真实推理的原始浮点 NPZ，未加载模型、未重新推理。
来源：`{m['source_raw_npz']}`。
图片：`{m['image']}`。
checkpoint：`{m['checkpoint']}`。

A(x)=I(x)−b 是输入相对全局锚点的逐通道偏差。
delta(x)=R(x)⊙A(x) 是实际加回输入的校正残差。
重建为 I_c=I+delta，无额外系数、激活或 clamp。
与原始、未 clamp UICF 输出比较的最大绝对误差为 {m['validation']['reconstruction_max_abs_error']}。

两张图采用有符号 RGB 显示，保留 RGB 通道次序，不能按普通自然图像颜色直接解释。
共同尺度 S=max(max(abs(A)),max(abs(delta)))={m['display']['S']}，
仅在与上次相同的有效展示区域内计算，覆盖两个张量的全部三个通道。
V=0.5+0.5*X/S；原始零值是中灰，8-bit 量化为 128。
S=0 时直接输出中灰。没有分别归一化、截断负数、gamma、直方图均衡、
锐化、平滑或额外幅度放大。RGB 合成图不附加一维色条。
delta 接近中灰是此共享尺度下的真实幅度，不进行视觉增强。

`anchor_residual_raw.npz` 保存完整 I、b、R、A、delta、I_c_reconstructed 和原 I_c。
I、b、R 保留源 dtype，A/delta 使用原操作数的浮点提升规则，未转换为 uint8；
仅绘图副本转换到 RGB uint8（round(255*V)）。raw 数据不裁剪、不覆盖。
有效区域 crop（x0,y0,x1,y1）={m['display']['crop_xyxy']}；
展示尺寸（高、宽）={m['display']['shape_hw']}。
预处理与 padding/crop 完整记录见 metadata.json。本次没有新增 resize/padding。

`*_signed.png/pdf/svg` 为带标题版本，PNG {m['display']['dpi']} dpi，
PDF/SVG 标题保留矢量文字；`*_compact.png` 为有效区域原生像素、无外边距纯图片。
两类图片均由同一个映射后的 RGB 数组生成，带标题版本仅 nearest 显示。
`uicf_anchor_residual_comparison.png` 是共享尺度并排检查图。

WSL 复现（已有输出非空时自动另建 UTC 时间戳目录，保留旧文件）：

```bash
{command}
```

从 MacBook 运行：

```bash
ssh home-wsl {shlex.quote(command)}
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Previous directory with metadata.json and uicf_raw.npz")
    parser.add_argument("--output-dir", help="Default: <source-dir>/anchor_residual_vis; nonempty targets get a timestamp suffix")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    source = Path(args.source_dir).expanduser().resolve()
    previous_hashes = {name: sha256(source / name) for name in ("metadata.json", "uicf_raw.npz", "README.md")}
    previous, arrays, checks = load_verified(source)
    A, delta, reconstructed, error = derive(arrays["I"], arrays["b"], arrays["R"], arrays["I_c"])
    crop = previous["geometry"]["uicf_display_crop_xyxy"]
    displays, scale = signed_display(A, delta, crop)
    output = fresh_output(Path(args.output_dir).expanduser().resolve() if args.output_dir else source / "anchor_residual_vis")
    result = {**arrays, "A": A, "delta": delta, "I_c_reconstructed": reconstructed}
    np.savez_compressed(output / "anchor_residual_raw.npz", **result)
    mpl_version = render(displays, output, args.dpi)
    # Verify native resolution and byte identity of the compact display arrays.
    for image, name in zip(displays, ("uicf_anchor_relative_compact.png", "uicf_correction_residual_compact.png")):
        with Image.open(output / name) as saved:
            if not np.array_equal(np.asarray(saved), image):
                raise RuntimeError("Compact PNG differs from the shared signed display array")
    if previous_hashes != {name: sha256(source / name) for name in previous_hashes}:
        raise RuntimeError("Previous artifacts changed during processing")
    parts = [sys.executable, str(Path(__file__).resolve()), "--source-dir", str(source), "--dpi", str(args.dpi)]
    if args.output_dir:
        parts += ["--output-dir", str(Path(args.output_dir).expanduser().resolve())]
    command = shlex.join(parts)
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--short"], text=True).strip()
    m = {"created_utc": datetime.now(timezone.utc).isoformat(),
         "source_raw_npz": str(source / "uicf_raw.npz"), "source_metadata": str(source / "metadata.json"),
         "source_hashes_sha256": previous_hashes, "output_dir": str(output),
         "image": previous["paths"]["image"], "checkpoint": previous["paths"]["checkpoint"],
         "config": previous["paths"]["config"], "original_inference_environment": previous["environment"],
         "original_forward_call_counts": previous["validation"]["forward_call_counts"],
         "original_geometry": previous["geometry"], "actual_reconstruction_formula": FORMULA,
         "reconstruction_source": previous["reconstruction"]["source"],
         "b_rgb": arrays["b"][0].tolist(), "A": describe(A), "delta": describe(delta),
         "display": {"S": scale, "S_zero": scale == 0, "formula": "V = 0.5 + 0.5 * X / S; if S==0: V=0.5",
                     "scale_domain": "all valid pixels and RGB channels of both A and delta",
                     "zero_display_value": 0.5, "zero_uint8_value": 128,
                     "encoding": "round(255*V) RGB uint8, display copies only",
                     "shape_hw": list(displays[0].shape[:2]), "crop_xyxy": crop,
                     "channel_order": "RGB", "interpolation": "nearest", "resample": False,
                     "dpi": args.dpi, "colorbar": False, "extra_processing": None},
         "raw_tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in result.items()},
         "validation": {**checks, "reconstruction_max_abs_error": error, "comparison_stage": "unclamped UICF output",
                        "A_identity_max_abs_error": float(np.abs(A - (arrays['I'] - arrays['b'][:, :, None, None])).max()),
                        "delta_identity_max_abs_error": float(np.abs(delta - arrays['R'] * A).max()),
                        "all_derived_tensors_finite": True, "same_S_both_figures": True,
                        "compact_png_byte_identity": True, "previous_artifacts_unchanged": True},
         "environment": {"python_executable": sys.executable, "python": sys.version,
                         "platform": platform.platform(), "numpy": np.__version__,
                         "Pillow": PIL.__version__, "matplotlib": mpl_version},
         "git": {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                 "dirty": bool(status), "status_short": status},
         "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
         "reproduce_wsl_command": command}
    (output / "metadata.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(output, m, command)
    sums = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "transfer_sha256.json").write_text(json.dumps(sums, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "A_range": summary(A), "delta_range": summary(delta),
                      "shared_S": scale, "reconstruction_max_abs_error": error,
                      "model_loads": 0, "model_forward_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
