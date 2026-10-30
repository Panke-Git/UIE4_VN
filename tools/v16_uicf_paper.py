"""Single-forward, source-audited paper exports for the full canonical V16.

Invoked by tools/visualize_v16_uicf.py --paper. The legacy diagnostic CLI and
all model/state_dict code remain unchanged. No checkpoint or data is fetched.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import platform
import random
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import PIL
from PIL import Image, UnidentifiedImageError
import torch

from src.shared.uicf_inr import UnderwaterImplicitCorrectionField
from src.v16.dataset import LSUIDataset, _to_tensor, read_manifest
from src.v16.test import _checkpoint_path, _torch_load
from src.v16.utils import load_yaml, project_path

ROOT = Path(__file__).resolve().parents[1]
CMAP = "RdBu_r"
ZERO_EPSILON = float(np.finfo(np.float32).eps)
FORMULA = "I_c = I + R * (I - b[:, :, None, None]); no activation or clamp"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def location(obj):
    lines, start = inspect.getsourcelines(obj)
    return {"path": str(Path(inspect.getsourcefile(obj)).resolve()),
            "start_line": start, "end_line": start + len(lines) - 1}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def stats(array):
    values = np.asarray(array, dtype=np.float64)
    return {"min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std(ddof=0))}


def full_model_config(config):
    if config.get("experiment", {}).get("version") != "v16":
        raise ValueError("Paper export requires canonical v16, excluding v16_19 and other versions")
    model = config["model"]
    field = model["uicf"]
    if (model["type"] != "plain_unet_color_query_uicf_pre_backbone"
            or field.get("field_variant", "implicit") not in ("implicit", "implicit_inr")
            or any(field.get(key, True) is not True for key in (
                "use_spatial_conditioning", "use_global_field_conditioning", "use_learned_anchor"))):
        raise ValueError("Paper export requires full implicit UICF with every conditioning path enabled")
    if model.get("output_activation") != "sigmoid":
        raise ValueError("Expected full V16 sigmoid RGB output")


def choose_input(args, run_dir, config, data_root):
    skipped = []
    if args.image:
        path = Path(args.image).expanduser()
        path = path if path.is_absolute() else ROOT / path
        candidates = [path.resolve()]
        selection = {"method": "explicit --image", "manifest": None}
    else:
        manifest = run_dir / "split_snapshot/test.tsv"
        if not manifest.is_file():
            manifest = project_path(config["data"]["test_manifest"]).resolve()
        if manifest.is_file():
            entries = read_manifest(manifest)
            candidates = sorted((data_root / item.input_relative).resolve() for item in entries)
            selection = {"method": "first readable input after lexicographic absolute-path sort; no aesthetic selection",
                         "manifest": str(manifest), "manifest_sha256": sha256(manifest),
                         "manifest_count": len(entries), "column": "input_relative (not gt_relative)"}
        else:
            # Only the checkpoint's own documented LSUI/UIEB input tree is eligible.
            provenance = str(config["data"].get("root", "")).lower()
            if not any(name in provenance for name in ("lsui", "uieb")):
                raise FileNotFoundError("No test manifest; cannot establish an underwater input directory")
            folder = next((data_root / name for name in ("Val/input", "test/input", "input")
                           if (data_root / name).is_dir()), None)
            if folder is None:
                raise FileNotFoundError("No test manifest or documented LSUI/UIEB degraded input directory")
            candidates = sorted(path.resolve() for path in folder.rglob("*")
                                if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            selection = {"method": "manifest absent; first readable path in checkpoint dataset's degraded input tree",
                         "manifest": None, "directory": str(folder), "basis": provenance}
    for path in candidates:
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB":
                    raise ValueError(f"Expected RGB as in V16 dataset, got {image.mode}")
                rgb = image.copy()
        except (OSError, UnidentifiedImageError, ValueError) as error:
            if args.image:
                raise
            skipped.append({"path": str(path), "error": str(error)})
            continue
        selection.update(selected_path=str(path), skipped=skipped, sorted_candidate_count=len(candidates))
        return path, rgb, selection
    raise FileNotFoundError(f"No readable RGB input among {len(candidates)} candidates; skipped={skipped[:5]}")


def preprocess(image, config):
    """Same PIL bilinear and RGB /255 conversion as LSUIDataset evaluation.

    Input-only operation: no GT decoding, EXIF rotation, external padding, or crop.
    Backbone padding is downstream of UICF and must stay there.
    """
    evaluation = config["evaluation"]
    processed = image
    if evaluation.get("resize", True):
        size = int(evaluation["size"])
        processed = image.resize((size, size), Image.Resampling.BILINEAR)
    tensor = _to_tensor(processed).unsqueeze(0)
    # Validate the exact project's transform on the same decoded image, without I/O.
    transform = object.__new__(LSUIDataset)
    transform.evaluation = evaluation
    reference, _ = transform._evaluation_transform(image, image)
    if not torch.equal(tensor[0], reference):
        raise RuntimeError("Preprocessing differs from V16 evaluation transform")
    h, w = tensor.shape[-2:]
    return tensor, {"original_size_wh": list(image.size), "model_input_size_wh": [w, h],
                    "channel_order": "RGB", "normalization": "float32 / 255; no mean/std",
                    "exif_transpose": False, "resize": {
                        "enabled": bool(evaluation.get("resize", True)),
                        "from_wh": list(image.size), "to_wh": [w, h],
                        "method": "PIL.Image.Resampling.BILINEAR" if evaluation.get("resize", True) else None},
                    "external_padding_lrtb": [0, 0, 0, 0],
                    "uicf_display_crop_xyxy": [0, 0, w, h],
                    "preprocessing_matches_existing_dataset_exactly": True,
                    "evaluation_source": location(LSUIDataset._evaluation_transform)}


def capture_once(model, inputs, device, amp):
    captured, counts = {}, {"uicf": 0, "backbone": 0, "field_mlp": 0}
    def count(name):
        def hook(module, args, output):
            counts[name] += 1
            if name == "field_mlp":
                captured["field_mlp_output"] = output.detach()
        return hook
    def padded_hook(module, args):
        captured["backbone_padded_input"] = args[0].detach()
    def output_hook(module, args, output):
        captured["backbone_output_before_crop"] = output.detach()
    handles = [model.uicf.register_forward_hook(count("uicf")),
               model.backbone.register_forward_hook(count("backbone")),
               model.uicf.field_mlp.register_forward_hook(count("field_mlp")),
               model.backbone.encoder1.register_forward_pre_hook(padded_hook),
               model.backbone.output_activation.register_forward_hook(output_hook)]
    try:
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, enabled=amp):
                final, details = model.forward_with_uicf_details(inputs)
            from tools.visualize_v16_uicf import assert_uicf_consistency
            manual = assert_uicf_consistency(inputs, final, details)
            error = float((manual - details.enhanced).abs().max().cpu())
            if tuple(details.chromatic_anchor.shape) != (1, 3):
                raise ValueError("b is not one RGB vector; refusing to average or truncate")
            if (details.chromatic_anchor < 0).any() or (details.chromatic_anchor > 1).any():
                raise ValueError("Sigmoid RGB anchor outside [0,1]")
            field = details.correction_field
            flat_as_field = captured["field_mlp_output"].reshape(
                1, inputs.shape[-2], inputs.shape[-1], 3).permute(0, 3, 1, 2)
            if not torch.equal(flat_as_field, field):
                raise RuntimeError("Actual reconstruction field differs from field MLP output")
            if counts != {"uicf": 1, "backbone": 1, "field_mlp": 1}:
                raise RuntimeError(f"Expected one inference, got {counts}")
            tensors = {"b": details.chromatic_anchor, "R": field, "I": inputs,
                       "I_c": details.enhanced, "I_hat": final,
                       "global_feature": details.global_feature, **captured}
            arrays = {}
            for name, tensor in tensors.items():
                if not torch.isfinite(tensor).all():
                    raise FloatingPointError(f"Nonfinite tensor: {name}")
                arrays[name] = tensor.detach().cpu().numpy().copy()
            padded = captured["backbone_padded_input"]
            ph, pw = padded.shape[-2] - inputs.shape[-2], padded.shape[-1] - inputs.shape[-1]
            crop = captured["backbone_output_before_crop"][..., :inputs.shape[-2], :inputs.shape[-1]]
            if not torch.equal(final, crop):
                raise RuntimeError("Backbone output crop is not spatially aligned")
            return arrays, error, counts, [0, pw, 0, ph]
    finally:
        for handle in handles:
            handle.remove()


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "v16-uicf-paper", "axes.linewidth": 0.65,
        "savefig.facecolor": "white"})
    return matplotlib, plt


def export_figure(fig, directory, stem, dpi, formats=("png", "pdf", "svg")):
    for extension in formats:
        metadata = ({"Date": None} if extension == "svg" else
                    {"CreationDate": None, "ModDate": None} if extension == "pdf" else None)
        fig.savefig(directory / f"{stem}.{extension}", dpi=dpi,
                    bbox_inches="tight", pad_inches=0.07, metadata=metadata)


def anchor_figure(plt, b, compact):
    fig = plt.figure(figsize=(2.55, 2.40) if compact else (3.65, 3.45))
    ax = fig.add_subplot(111, projection="3d", proj_type="ortho")
    ax.view_init(elev=23, azim=-56)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), "RGB"):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color("0.40")
        axis.line.set_linewidth(0.65)
        axis.set_label_text(label, fontsize=10 if compact else 11)
        axis.labelpad = -1 if compact else 1
        axis.set_ticks([0, 1] if compact else [0, 0.5, 1])
        axis.set_ticklabels(["0", "1"] if compact else ["0", "0.5", "1"])
        axis.set_tick_params(labelsize=7 if compact else 8, pad=-1)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    for coordinate in range(3):
        foot = b.copy()
        foot[coordinate] = 0
        ax.plot(*np.stack((b, foot)).T, linestyle=(0, (2, 3)), color="0.72", linewidth=0.6)
    marker = ax.scatter(*b, s=90 if compact else 115, c=[b], edgecolors="0.18",
                        linewidths=0.65, depthshade=False)
    label_position = b + np.where(b > 0.85, -0.075, 0.055)
    ax.text(*label_position, r"$b$", fontsize=12 if compact else 13, color="0.12")
    if not compact:
        ax.set_title("Global RGB anchor", fontsize=12, pad=6)
    # Verify plotted coordinates and marker RGB before saving any figure.
    plotted = np.array([float(value[0]) for value in marker._offsets3d])
    if not np.array_equal(plotted, b) or not np.array_equal(marker.get_facecolors()[0, :3], b):
        raise RuntimeError("Plotted anchor coordinate or color changed")
    return fig


def render_figures(arrays, output, dpi):
    matplotlib, plt = plotting()
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import FuncFormatter
    b = arrays["b"][0].astype(np.float64)
    field = arrays["R"][0]  # Full valid UICF domain; no external padding in canonical test.
    observed_max = float(np.abs(field.astype(np.float64)).max())
    all_zero = observed_max == 0
    limit = ZERO_EPSILON if all_zero else observed_max
    norm = Normalize(vmin=-limit, vmax=limit, clip=False)
    cmap = matplotlib.colormaps[CMAP]
    for compact, stem in ((False, "uicf_rgb_anchor"), (True, "uicf_rgb_anchor_compact")):
        fig = anchor_figure(plt, b, compact)
        export_figure(fig, output, stem, dpi)
        plt.close(fig)
    height, width = field.shape[-2:]
    aspect = width / height
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.18 / aspect + 0.85), layout="constrained")
    for index, (ax, label) in enumerate(zip(axes, (r"$R_r(x)$", r"$R_g(x)$", r"$R_b(x)$"))):
        ax.imshow(field[index], cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        ax.set_axis_off()
        ax.set_title(label, fontsize=12, pad=5)
    fig.suptitle("Channel-wise correction fields", fontsize=13)
    fmt = FuncFormatter(lambda value, pos: f"{value:.3g}")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes),
                           fraction=0.025, pad=0.025, shrink=0.85,
                           ticks=[-limit, 0, limit], format=fmt)
    colorbar.set_label("Correction coefficient", fontsize=10, labelpad=7)
    colorbar.ax.tick_params(labelsize=8, width=0.5, length=2)
    colorbar.outline.set_linewidth(0.5)
    export_figure(fig, output, "uicf_channel_fields", dpi)
    plt.close(fig)
    for index, channel in enumerate("rgb"):
        # One pixel per coefficient, no titles, interpolation, border, or margins.
        rgba = cmap(norm(field[index]), bytes=True)
        Image.fromarray(rgba).convert("RGB").save(output / f"uicf_field_{channel}.png", dpi=(dpi, dpi))
    fig = plt.figure(figsize=(4.15, 0.85))
    cax = fig.add_axes((0.10, 0.63, 0.80, 0.17))
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=[-limit, 0, limit], format=fmt)
    bar.set_label("Correction coefficient", fontsize=10, labelpad=4)
    bar.ax.tick_params(labelsize=9, width=0.5, length=2)
    bar.outline.set_linewidth(0.5)
    export_figure(fig, output, "uicf_field_colorbar", dpi, ("png", "svg"))
    plt.close(fig)
    return {"colormap": CMAP, "vmin": -limit, "vmax": limit,
            "M_max_abs_valid_R": observed_max, "all_zero": all_zero,
            "zero_field_epsilon": ZERO_EPSILON if all_zero else None,
            "channels": ["R_r", "R_g", "R_b"], "interpolation": "nearest",
            "per_channel_normalization": False, "percentile_clipping": False,
            "smoothing": False, "display_shape_hw": [height, width],
            "display_crop_xyxy": [0, 0, width, height],
            "anchor_point_coordinates": b.tolist(), "anchor_marker_rgb": b.tolist(),
            "anchor_axis_limits": [[0, 1]] * 3, "png_dpi": dpi,
            "standalone_field_size_wh": [width, height],
            "matplotlib_version": matplotlib.__version__}


def previews(arrays, output):
    result = {}
    for key, filename in (("I", "model_input.png"), ("I_c", "uicf_corrected.png"),
                          ("I_hat", "final_enhanced.png")):
        raw = arrays[key][0]
        clipped_count = int(np.count_nonzero((raw < 0) | (raw > 1)))
        # Work on a new float32 array, leaving all saved computation arrays intact.
        display = np.clip(raw.astype(np.float32, copy=True), 0, 1)
        display = np.rint(display.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(display).save(output / filename)
        result[key] = {"file": filename, "method": "copy -> float32 -> clip [0,1] -> round(255*x) -> RGB uint8",
                       "out_of_display_range_elements": clipped_count, "raw_range": stats(raw)}
    return result


def read_json_if_exists(path):
    return json.loads(path.read_text()) if path.is_file() else None


def write_readme(output, metadata, command):
    p = metadata["paths"]
    b = metadata["anchor"]["values_rgb"]
    text = f"""# V16 UICF 真实论文图

本目录全部模型数据来自同一真实输入、同一 checkpoint、一次前向。
`uicf_rgb_anchor.*` 是 sigmoid 全局 RGB 锚点 b 在 [0,1]³ 中的单个点，点色即 b；
`uicf_rgb_anchor_compact.*` 去掉标题，适合框架图嵌入。
`uicf_channel_fields.*` 展示实际参与重建的三个系数通道，RGB 顺序固定。
`uicf_field_r/g/b.png` 是无边框、无文字、原像素比例的嵌图素材，
`uicf_field_colorbar.png/svg` 是同一色标的独立版本。

实际公式：`{FORMULA}`。
b 来自 encoder 特征全局均值及 Linear–GELU–Linear–Sigmoid，形状 [1,3]。
R 是 MLP 最后 Linear 输出 reshape/permute 后的 [1,3,H,W] 系数；
没有 tanh、缩放、插值或 clamp；本实现所谓“原始 MLP 输出”与实际系数数值相同。
UICF 的 I_c 不 clamp；后续 Color-Query U-Net 才使用 sigmoid 得到 I_hat。
实际改变量是 `ΔI = R * (I-b)`；R 的正负号本身不能解释为增亮或变暗。

三通道统一使用 {CMAP}，共用以零为中心的范围
[{metadata['plot']['vmin']}, {metadata['plot']['vmax']}]，M=max(abs(R)) 包含所有有效像素和三个通道。
这样相同颜色始终表示相同系数，零在色标中心；不做每通道 min–max、分位数裁剪、平滑或取绝对值。
全零时仅为绘图使用 float32 epsilon，并在 metadata 中明确记录。

`uicf_raw.npz` 保留完整计算张量及原 dtype：b、R、公式输入 I、UICF 输出 I_c、
最终 I_hat、global_feature、field_mlp_output，以及 backbone padding 后输入和裁剪前输出。
这些数值没有为了显示而 clamp 或归一化。预览 PNG 仅处理副本：clip [0,1] 后
round(255*x) 转 RGB uint8；`metadata.json` 记录超出显示范围的元素数。
`original_input{Path(p['image']).suffix}` 是输入原文件的逐字节副本。

本实验遵循配置的 evaluation resize，PIL bilinear 将
{metadata['geometry']['original_size_wh']} 调整为 {metadata['geometry']['model_input_size_wh']}（宽、高）。
这是原评估流程的 resize；所有场、I_c、I_hat 与实际模型输入对齐，不把它们拉伸回原图。
UICF 之前没有外部 padding。backbone 自行在右/下补到 16 的倍数并裁回输入大小；
不把 padding 前移到 UICF，以免改变全局锚点及场。
本次 backbone padding（左、右、上、下）={metadata['geometry']['backbone_padding_lrtb']}。

锚点 b={b}；重建最大绝对误差={metadata['validation']['reconstruction_max_abs_error']}。
checkpoint epoch={metadata['checkpoint']['epoch']}；strict=True，使用 model_state_dict，未使用 EMA。
图片选择方式及跳过记录、环境、git 状态、源码位置与 SHA256 见 metadata.json。

## 远端 WSL 复现命令

```bash
{command}
```

MacBook 可直接：

```bash
ssh home-wsl {shlex.quote(command)}
```

更换图片：在上述命令追加 `--image /绝对路径/输入图片.jpg`；
更换 checkpoint：将 `--checkpoint` 的值改为该完整 V16 权重的绝对路径或
best_psnr/best_ssim/best_loss/last，跨实验时同时更换 `--run-dir`。
`--data-root` 是配置所对应数据集根目录，不是单张图片目录。
`--output-dir` 指定精确输出目录；默认 `<run-dir>/result/uicf_paper_vis/<image-stem>/`。
CPU 可将 `--device cuda:0` 改成 `--device cpu`；CPU 自动关闭 CUDA AMP。
不同设备/精度可能产生浮点差异，均在 metadata 中记录。本次 CUDA AMP 与原测试一致。
使用 `--help` 查看全部参数；`--paper` 启用此单图论文导出，原有多样本 CLI 保持可用。
重复执行覆盖本目录同名产物，不删除其他文件。显存不足会直接报错，不自动缩图或分块。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def run(args):
    if args.indices is not None or args.sample_ids is not None or args.robust_percentile is not None:
        raise ValueError("--paper uses one sorted test input or --image, without robust clipping")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive")
    if args.device and args.gpu is not None:
        raise ValueError("Use --device or --gpu, not both")
    run_dir = project_path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config_resolved.yaml"
    config = load_yaml(config_path)
    full_model_config(config)
    selector = args.checkpoint or config["test"]["checkpoint"]
    checkpoint_path = _checkpoint_path(run_dir, selector).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No trained checkpoint; random initialization is forbidden: {checkpoint_path}")
    data_root = Path(args.data_root or config["data"]["root"]).expanduser().resolve()
    image_path, image, selection = choose_input(args, run_dir, config, data_root)
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
              run_dir / "result/uicf_paper_vis" / image_path.stem)
    inputs, geometry = preprocess(image, config)
    device = torch.device(args.device or (f"cuda:{args.gpu}" if args.gpu is not None else
                                         "cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
    if int(checkpoint.get("epoch", 0)) <= 0:
        raise ValueError("Checkpoint has no positive trained epoch; refusing a random/smoke checkpoint")
    from tools.visualize_v16_uicf import build_and_load_v16_model
    model = build_and_load_v16_model(config, checkpoint, device).requires_grad_(False)
    if type(model.uicf) is not UnderwaterImplicitCorrectionField:
        raise TypeError("Expected canonical implicit UICF")
    amp = bool(config["training"]["amp"]) and device.type == "cuda"
    print(f"Checkpoint: {checkpoint_path}\nInput: {image_path}\nDevice: {device}; AMP={amp}", flush=True)
    arrays, error, counts, padding = capture_once(model, inputs.to(device), device, amp)
    geometry.update(backbone_padding_lrtb=padding,
                    backbone_padding_mode=(None if not any(padding) else
                        "replicate" if min(inputs.shape[-2:]) < 2 or padding[1] >= inputs.shape[-1]
                        or padding[3] >= inputs.shape[-2] else "reflect"),
                    backbone_padding_stage="after UICF, before encoder1",
                    backbone_padding_source=location(model.backbone._pad),
                    backbone_forward_source=location(model.backbone._forward_impl),
                    backbone_padded_input_shape=list(arrays["backbone_padded_input"].shape),
                    backbone_output_crop_xyxy=geometry["uicf_display_crop_xyxy"],
                    field_resolution="same H,W as formula input; no interpolation")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "uicf_raw.npz", **arrays)
    shutil.copy2(image_path, output / f"original_input{image_path.suffix}")
    preview = previews(arrays, output)
    plot = render_figures(arrays, output, args.dpi)
    command_parts = [str(Path(sys.executable).resolve()), str(ROOT / "tools/visualize_v16_uicf.py"),
                     "--paper", "--run-dir", str(run_dir), "--checkpoint", str(checkpoint_path),
                     "--data-root", str(data_root), "--device", str(device), "--dpi", str(args.dpi)]
    if args.image:
        command_parts += ["--image", str(image_path)]
    if args.output_dir:
        command_parts += ["--output-dir", str(output)]
    command = shlex.join(command_parts)
    status = git("status", "--short", "--untracked-files=all")
    source_paths = {ROOT / name for name in (
        "tools/visualize_v16_uicf.py", "tools/v16_uicf_paper.py", "src/shared/uicf_inr.py",
        "src/shared/uicf_models.py", "src/shared/color_query_unet.py", "src/v4/models/unet.py",
        "src/v16/models/network.py", "src/v16/dataset.py", "src/v16/test.py")}
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {"image": str(image_path), "checkpoint": str(checkpoint_path),
                  "config": str(config_path), "run_dir": str(run_dir), "data_root": str(data_root),
                  "output_dir": str(output)},
        "sha256": {"image": sha256(image_path), "checkpoint": sha256(checkpoint_path),
                   "config": sha256(config_path), "raw_npz": sha256(output / "uicf_raw.npz")},
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(status), "status_short": status},
        "source_sha256": {str(path): sha256(path) for path in sorted(source_paths)},
        "environment": {"python_executable": sys.executable, "python_version": sys.version,
                        "conda_prefix": str(Path(sys.executable).parent.parent), "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__, "Pillow": PIL.__version__,
                        "matplotlib": plot["matplotlib_version"], "cuda_runtime": torch.version.cuda,
                        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "amp": amp, "model_dtype": str(next(model.parameters()).dtype)},
        "checkpoint": {"epoch": int(checkpoint["epoch"]), "version": checkpoint["version"],
                       "selector": selector, "keys": list(checkpoint), "load_key": "model_state_dict",
                       "strict": True, "missing_keys": [], "unexpected_keys": [], "ema_used": False,
                       "ema_note": "V16 train payload and test loader use model_state_dict; no EMA path",
                       "parameter_tensor_count": len(checkpoint["model_state_dict"]),
                       "model_config": config["model"],
                       "selection_evidence": {"best_record": read_json_if_exists(checkpoint_path.with_suffix(".json")),
                                              "test_summary": read_json_if_exists(run_dir / "result/test_summary.json")}},
        "image_selection": selection, "geometry": geometry,
        "anchor": {"shape": list(arrays["b"].shape), "values_rgb": arrays["b"][0].tolist(),
                   "range_definition": [0, 1], "source": "encoder -> global average pooling -> Linear/GELU/Linear/Sigmoid"},
        "field": {"shape": list(arrays["R"].shape), "channels": {
            name: stats(arrays["R"][0, i]) for i, name in enumerate(("R_r", "R_g", "R_b"))},
                  "range": stats(arrays["R"]),
                  "source": "field_mlp output reshaped BHWC then permuted BCHW; identical values",
                  "activation": None, "scaling": None, "interpolation": None,
                  "is_actual_reconstruction_coefficient": True},
        "reconstruction": {"formula": FORMULA, "source": location(UnderwaterImplicitCorrectionField.forward),
                           "clamp": False, "final_output": "Color-Query U-Net sigmoid then original-size crop"},
        "raw_tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for name, value in arrays.items()},
        "plot": plot, "preview": preview,
        "randomness": {"seed": seed, "random_sample_selection": False, "random_augmentation": False,
                       "cudnn_benchmark": False, "cudnn_deterministic": True, "matmul_tf32": False,
                       "strict_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "validation": {"reconstruction_max_abs_error": error, "same_stage": "unclamped UICF output",
                       "forward_call_counts": counts, "all_tensors_finite": True,
                       "field_mlp_equals_actual_R": True, "backbone_crop_matches_final": True,
                       "anchor_plot_coordinates_and_color_match_exactly": True},
        "reproduce_wsl_command": command,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(output, metadata, command)
    print(f"Saved: {output}\nb={arrays['b'][0].tolist()}\nR range={metadata['field']['range']}"
          f"\nReconstruction max abs error={error}; calls={counts}", flush=True)
