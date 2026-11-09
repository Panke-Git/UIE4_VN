#!/usr/bin/env python3
"""Model-free reference-derived correction demand, under the fixed V16 protocol.

No model, checkpoint, CUDA initialization, inference, or training is used.
The project dataset's manifest/integrity helpers are reused without changes.
See README_motivation_correction_demand.md for the predeclared protocol.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import PIL
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v16.dataset import (  # imports torch, but never its model/test modules
    ManifestEntry, read_manifest, validate_split_protocol, verify_image_entries,
)

EPSILON = 1e-12
DEGENERATE_ENERGY = 1e-10
SEED = 3520
SIGNED_LIMIT = 0.5
MAGNITUDE_LIMIT = 1.0
COUNTS = {"lsui": {"train": 3466, "validation": 385, "test": 428},
          "uieb": {"train": 720, "validation": 80, "test": 90}}
CHANNELS = "RGB"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()


def new_run_dir(parent: Path, dataset: str, stamp: str | None = None) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    for index in range(10000):
        suffix = "" if index == 0 else f"_{index:03d}"
        path = parent / f"run_{stamp}_{dataset}{suffix}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError("Cannot reserve a unique output directory")


def safe_pair_paths(entry: ManifestEntry, root: Path) -> tuple[Path, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"data.root is unavailable: {root}")
    paths = tuple((root / rel).resolve() for rel in (entry.input_relative, entry.gt_relative))
    for p in paths:
        if not p.is_relative_to(root.resolve()):
            raise ValueError(f"Image path escapes data.root: sample_id={entry.sample_id} path={p}")
    return paths  # type: ignore[return-value]


def load_pair(entry: ManifestEntry, root: Path, evaluation: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    input_path, gt_path = safe_pair_paths(entry, root)
    verify_image_entries([entry], root, "test")
    # Preserve the project's strict RGB validation, with no implicit conversion.
    with Image.open(input_path) as a, Image.open(gt_path) as b:
        a.load(); b.load()
        original = list(a.size)
        if evaluation.get("resize", True):
            size = int(evaluation["size"])
            a = a.resize((size, size), Image.Resampling.BILINEAR)
            b = b.resize((size, size), Image.Resampling.BILINEAR)
        # Exactly _to_tensor's float32 /255 arithmetic, in HWC layout.
        inputs = np.asarray(a, dtype=np.float32) / 255.0
        targets = np.asarray(b, dtype=np.float32) / 255.0
    meta = {"original_dimensions_wh": original,
            "analysis_dimensions_wh": [inputs.shape[1], inputs.shape[0]],
            "input_mode": "RGB", "gt_mode": "RGB"}
    return inputs, targets, meta


def residual(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    a, b = np.asarray(inputs, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 1:
        raise ValueError(f"Expected matching nonempty HWC RGB arrays: {a.shape}, {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Images contain NaN/Inf")
    if a.min() < 0 or b.min() < 0 or a.max() > 1 or b.max() > 1:
        raise ValueError("Images must be RGB values in [0,1]")
    return b - a


def patch_means(delta: np.ndarray, size: int) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 3 or delta.shape[-1] != 3 or size <= 0:
        raise ValueError("Patch averaging requires HWC RGB and a positive patch size")
    h, w, c = delta.shape
    if h % size or w % size:
        raise ValueError(f"Image {w}x{h} not divisible by patch size {size}; no crop/padding allowed")
    return delta.reshape(h // size, size, w // size, size, c).mean(axis=(1, 3))


def display_clipping(delta: np.ndarray, remaining: np.ndarray,
                     signed_limit: float = SIGNED_LIMIT,
                     magnitude_limit: float = MAGNITUDE_LIMIT) -> dict[str, float]:
    # Boolean comparisons allocate new arrays; no clipping/mutation of statistics.
    result = {"display_signed_clipped_fraction": float(np.mean(np.abs(delta) > signed_limit)),
              "display_magnitude_clipped_fraction": float(np.mean(np.linalg.norm(delta, axis=-1) > magnitude_limit)),
              "display_remaining_clipped_fraction": float(np.mean(np.linalg.norm(remaining, axis=-1) > magnitude_limit))}
    for c, label in enumerate(CHANNELS):
        result[f"display_signed_{label}_clipped_fraction"] = float(np.mean(np.abs(delta[..., c]) > signed_limit))
    return result


def compute_metrics(inputs: np.ndarray, targets: np.ndarray, primary_patch: int = 16,
                    patch_sizes: Sequence[int] = (8, 16, 32)) -> dict[str, Any]:
    delta = residual(inputs, targets)
    mu = delta.mean(axis=(0, 1))
    remaining = delta - mu
    raw = float(np.mean(delta ** 2))
    glob = float(np.mean(remaining ** 2))
    degenerate = raw <= DEGENERATE_ENERGY
    if glob > raw + 1e-12:
        raise ArithmeticError("E_global > E_raw: least-squares identity violated")
    decomposition_error = abs(raw - glob - float(np.mean(mu ** 2)))
    if decomposition_error > 1e-12:
        raise ArithmeticError("E_raw = E_global + mean(mu^2) identity violated")
    oracle = np.asarray(inputs, dtype=np.float64) + mu
    outside = (oracle < 0) | (oracle > 1)
    result: dict[str, Any] = {
        "residual_magnitude_mean": float(np.linalg.norm(delta, axis=-1).mean()),
        "e_raw": raw, "e_global": glob,
        "eta_spatial": None if degenerate else glob / (raw + EPSILON),
        "degenerate": degenerate, "decomposition_abs_error": decomposition_error,
        "channel_discrepancy": float(np.ptp(mu)),
        "channel_variance": float(mu.var(ddof=0)),
        "oracle_scalar_clipped_fraction": float(outside.mean()),
        "oracle_pixel_clipped_fraction": float(outside.any(axis=-1).mean()),
    }
    for c, label in enumerate(CHANNELS):
        result[f"mu_{label}"] = float(mu[c])
        result[f"mean_abs_{label}"] = float(np.abs(delta[..., c]).mean())
        result[f"spatial_std_{label}"] = float(delta[..., c].std(ddof=0))
    for first, second in (("R", "G"), ("R", "B"), ("G", "B")):
        result[f"mu_{first}_minus_{second}"] = result[f"mu_{first}"] - result[f"mu_{second}"]
    for p in sorted(set(patch_sizes) | {primary_patch}):
        block = patch_means(delta, p)
        var = float(block.var(axis=(0, 1), ddof=0).mean())
        if var > glob + 1e-12:
            raise ArithmeticError("Block-mean variance exceeds spatial pixel variance")
        result[f"patch_variance_{p}"] = var
        result[f"patch_spatial_retention_{p}"] = None if glob <= DEGENERATE_ENERGY else var / glob
    result["patch_spatial_variance"] = result[f"patch_variance_{primary_patch}"]
    result.update(display_clipping(delta, remaining))
    return result


def summarize(values: Sequence[Any]) -> dict:
    valid = np.asarray([float(x) for x in values if x is not None], dtype=np.float64)
    if not np.isfinite(valid).all():
        raise ArithmeticError("Non-finite summary input")
    if not len(valid):
        return {"mean": None, "median": None, "std": None, "p05": None, "p95": None,
                "valid_count": 0, "null_count": len(values)}
    return {"mean": float(valid.mean()), "median": float(np.median(valid)),
            "std": float(valid.std(ddof=0)),
            "p05": float(np.quantile(valid, .05, method="linear")),
            "p95": float(np.quantile(valid, .95, method="linear")),
            "valid_count": len(valid), "null_count": len(values) - len(valid)}


def select_samples(rows: Sequence[dict]) -> list[dict]:
    eligible = [r for r in rows if not r["degenerate"]]
    used: set[str] = set()
    selected = []
    for metric, quantile, rule in (("patch_spatial_variance", .5, "median spatial block variance"),
                                   ("patch_spatial_variance", .8, "80th-percentile spatial block variance"),
                                   ("channel_discrepancy", .8, "80th-percentile channel mean discrepancy")):
        ordered = sorted(eligible, key=lambda r: (r[metric], str(r["sample_id"]), r["input_path"], r["gt_path"]))
        if len(used) >= len(ordered):
            break
        rank = int(math.floor(quantile * (len(ordered) - 1) + .5))
        chosen_rank = min((i for i in range(len(ordered)) if ordered[i]["sample_id"] not in used),
                          key=lambda i: (abs(i - rank), i))
        row = ordered[chosen_rank]
        used.add(row["sample_id"])
        selected.append({"dataset": row["dataset"], "sample_id": row["sample_id"],
                         "input_path": row["input_path"], "gt_path": row["gt_path"],
                         "input_relative": row["input_relative"], "gt_relative": row["gt_relative"],
                         "selection_rule": rule, "selection_metric": metric,
                         "metric_value": row[metric], "target_quantile": quantile,
                         "target_rank_zero_based": rank, "selected_rank_zero_based": chosen_rank,
                         "candidate_count": len(ordered),
                         "original_dimensions_wh": [row["original_width"], row["original_height"]],
                         "analysis_dimensions_wh": [row["analysis_width"], row["analysis_height"]],
                         "quality_policy": "no post-visual reselection; inspect references manually"})
    return selected


def visualization_norms(full_range: bool = False) -> tuple[Any, Any]:
    from matplotlib.colors import Normalize, TwoSlopeNorm
    limit = 1.0 if full_range else SIGNED_LIMIT
    maximum = math.sqrt(3) if full_range else MAGNITUDE_LIMIT
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), Normalize(0.0, maximum, clip=True)


def save_figure(fig: Any, path: Path, formats: Sequence[str] = ("png", "pdf", "svg")) -> list[str]:
    result = []
    for fmt in formats:
        target = path.with_suffix("." + fmt)
        with target.open("xb") as handle:
            fig.savefig(handle, format=fmt, dpi=350, facecolor="white")
        result.append(str(target))
    return result


def render_grid(samples: Sequence[dict], base: Path, kind: str = "main",
                full_range: bool = False, formats: Sequence[str] = ("png", "pdf", "svg")) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    signed, magnitude = visualization_norms(full_range)
    cols = 6 if kind == "main" else 4 if kind == "oracle" else 5
    n = len(samples)
    height_inches = 1.40 * n + 1.14
    fig = plt.figure(figsize=(7.8 if cols == 6 else 7.2, height_inches))
    # Physical margins keep two-line oracle titles and the footer clear even
    # for a single-row export. Fractions alone shrink these gaps for n=1.
    left, right, bottom, top = .083, .992, .72 / height_inches, 1 - .42 / height_inches
    gs = fig.add_gridspec(n, cols, left=left, right=right, bottom=bottom, top=top,
                          wspace=.045, hspace=.09)
    titles = (["Input", "Reference", r"$\|\Delta_{\rm ref}\|_2$", r"$\Delta_R$", r"$\Delta_G$", r"$\Delta_B$"]
              if kind == "main" else
              ["Input", "GT-dependent Global\nRGB Oracle", "Reference", "Remaining magnitude"] if kind == "oracle" else
              ["Input", r"$\|\bar\Delta\|_2$", r"$\bar\Delta_R$", r"$\bar\Delta_G$", r"$\bar\Delta_B$"])
    axes = []
    for i, sample in enumerate(samples):
        a, y, d = sample["input"], sample["target"], sample["delta"]
        mu = d.mean(axis=(0, 1))
        if kind == "main":
            panels = [a, y, np.linalg.norm(d, axis=-1), d[..., 0], d[..., 1], d[..., 2]]
            styles = ["rgb", "rgb", "magnitude", "signed", "signed", "signed"]
        elif kind == "oracle":
            panels = [a, np.clip(a.astype(np.float64) + mu, 0, 1), y, np.linalg.norm(d - mu, axis=-1)]
            styles = ["rgb", "rgb", "rgb", "magnitude"]
        else:
            p = patch_means(d, sample["patch_size"])
            panels = [a, np.linalg.norm(p, axis=-1), p[..., 0], p[..., 1], p[..., 2]]
            styles = ["rgb", "magnitude", "signed", "signed", "signed"]
        row_axes = []
        for j, (panel, style) in enumerate(zip(panels, styles)):
            ax = fig.add_subplot(gs[i, j])
            if style == "rgb":
                ax.imshow(panel, interpolation="nearest", vmin=0, vmax=1)
            else:
                ax.imshow(panel, interpolation="nearest", cmap="coolwarm" if style == "signed" else "magma",
                          norm=signed if style == "signed" else magnitude)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(titles[j], fontsize=8, pad=5)
            if j == 0:
                ax.text(-.07, .5, sample["label"], transform=ax.transAxes, ha="right", va="center", fontsize=7)
            row_axes.append(ax)
        axes.append(row_axes)
    fig.canvas.draw()
    mag_column = 2 if kind == "main" else 3 if kind == "oracle" else 1
    mag_pos = axes[-1][mag_column].get_position()
    # Reserve horizontal space for endpoint tick labels on adjacent colorbars.
    mag_ax = fig.add_axes([mag_pos.x0, .34 / height_inches, mag_pos.width * .90, .04 / height_inches])
    cb = fig.colorbar(ScalarMappable(norm=magnitude, cmap="magma"), cax=mag_ax, orientation="horizontal",
                      ticks=[0, magnitude.vmax / 2, magnitude.vmax])
    cb.ax.tick_params(labelsize=6, length=2, pad=2)
    cb.ax.set_title("L2 magnitude (RGB [0,1])", fontsize=6, pad=4)
    if kind != "oracle":
        first = 3 if kind == "main" else 2
        p0, p1 = axes[-1][first].get_position(), axes[-1][-1].get_position()
        sax = fig.add_axes([p0.x0 + .012, .34 / height_inches,
                           p1.x1 - p0.x0 - .012, .04 / height_inches])
        cb = fig.colorbar(ScalarMappable(norm=signed, cmap="coolwarm"), cax=sax, orientation="horizontal",
                          ticks=[signed.vmin, 0, signed.vmax], extend="neither" if full_range else "both")
        cb.ax.tick_params(labelsize=6, length=2, pad=2)
        cb.ax.set_title("Signed reference residual (RGB [0,1])", fontsize=6, pad=4)
    note = ("Oracle display clipped to [0,1]; remaining residual computed before clipping."
            if kind == "oracle" else "256 x 256 paired bilinear resize; shared scales; display saturation recorded in metadata.")
    fig.text(.5, .045 / height_inches, note, ha="center", va="bottom", fontsize=6)
    paths = save_figure(fig, base, formats)
    plt.close(fig)
    return paths


def render_figures(selected: list[dict], rows: list[dict], root: Path,
                   evaluation: dict, out: Path, patch_size: int) -> tuple[list[str], list[dict]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                                "pdf.fonttype": 42, "ps.fonttype": 42,
                                "svg.fonttype": "none", "axes.linewidth": .5})
    samples = []
    clipping = []
    files = []
    for i, item in enumerate(selected):
        e = ManifestEntry(item["sample_id"], item["input_relative"], item["gt_relative"])
        a, y, _ = load_pair(e, root, evaluation)
        d = residual(a, y)
        mu = d.mean(axis=(0, 1))
        s = {"input": a, "target": y, "delta": d, "patch_size": patch_size,
             "label": f"{item['dataset'].upper()}\n{item['sample_id']}"}
        samples.append(s)
        record = {"sample_id": item["sample_id"], **display_clipping(d, d - mu),
                  "theoretical_range": display_clipping(d, d - mu, 1.0, math.sqrt(3))}
        clipping.append(record)
        individual = out / "individual_samples" / f"{i+1:02d}_{item['sample_id']}"
        individual.mkdir(parents=True, exist_ok=False)
        with (individual / "numeric_maps.npz").open("xb") as handle:
            np.savez_compressed(handle, input_rgb=a, reference_rgb=y, delta_ref=d,
                                mu_rgb=mu, oracle_unclipped=a.astype(np.float64) + mu,
                                remaining=d - mu, patch_mean_delta=patch_means(d, patch_size))
        files += render_grid([s], individual / "main", formats=("png",))
        files += render_grid([s], individual / "oracle", kind="oracle", formats=("png",))
    if samples:
        files += render_grid(samples, out / "motivation_main")
        files += render_grid(samples, out / "motivation_full_range", full_range=True)
        files += render_grid(samples, out / "global_oracle_diagnostic", kind="oracle")
        files += render_grid(samples, out / "patch_level_diagnostic", kind="patch")
    if rows:
        fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.9), layout="constrained")
        eta = [r["eta_spatial"] for r in rows if r["eta_spatial"] is not None]
        axes[0, 0].hist(eta, bins=np.linspace(0, 1, 21), color="#456b8c", edgecolor="white", linewidth=.4)
        axes[0, 0].set(xlabel=r"$E_{\rm global}/(E_{\rm raw}+\epsilon)$", ylabel="Images", xlim=(0, 1), title="Unexplained by a global additive offset")
        box_kw = dict(whis=(5, 95), showfliers=True,
                      flierprops={"markersize": 2, "markeredgewidth": .3})
        sizes = sorted({8, 16, 32, patch_size})
        axes[0, 1].boxplot([[r[f"patch_variance_{p}"] for r in rows] for p in sizes],
                           tick_labels=[str(p) for p in sizes], **box_kw)
        axes[0, 1].set(xlabel="Patch side length (pixels)", ylabel="Population variance (RGB units squared)", title="Coarse-scale spatial residual")
        for ax, key, title in ((axes[1, 0], "mu_", "Signed channel means"),
                               (axes[1, 1], "mean_abs_", "Absolute channel corrections")):
            ax.boxplot([[r[key + c] for r in rows] for c in CHANNELS], tick_labels=list(CHANNELS), **box_kw)
            ax.set(ylabel="RGB [0,1] units", title=title)
            ax.axhline(0, color="gray", linewidth=.5)
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{rows[0]['dataset'].upper()} test: n={len(rows)}; each image has equal weight\nBoxes: median / IQR; whiskers: 5th / 95th percentiles", fontsize=9)
        files += save_figure(fig, out / "dataset_distributions")
        plt.close(fig)
    return files, clipping


def protected_snapshot() -> dict[str, str]:
    rels = git("ls-files", "src/v16", "src/shared/uicf_inr.py", "src/shared/uicf_models.py", "configs", "split", "tools").splitlines()
    rels += ["tools/audit_uicf_experiments.py"]
    return {p: sha256(PROJECT_ROOT / p) for p in rels if (PROJECT_ROOT / p).is_file()}


def protocol_record(args: argparse.Namespace) -> dict:
    return {
        "protocol_version": "reference_correction_demand_v1_20261106",
        "dataset": args.dataset, "split": args.split, "seed": args.seed,
        "unit_of_aggregation": "equal-weight image", "variance_ddof": 0,
        "percentile_method": "numpy.quantile method=linear",
        "epsilon": EPSILON, "degenerate_e_raw_threshold": DEGENERATE_ENERGY,
        "patch_size": args.patch_size, "sensitivity_patch_sizes": sorted({8, 16, 32, args.patch_size}),
        "preprocessing": {"mode": "strict RGB", "load_dtype": "float32", "range": [0, 1],
                          "resize": [256, 256], "interpolation": "PIL.Image.Resampling.BILINEAR",
                          "augmentation": False, "residual_and_reductions_dtype": "float64",
                          "domain": "encoded RGB; not linear-light/physical attenuation"},
        "formulas": {"delta_ref": "Y-I (not UICF R)", "D": "L2_norm_RGB(Y-I)",
                     "mu": "mean_xy(delta_ref)", "oracle": "I+mu, GT-dependent analytical least-squares additive oracle",
                     "e_raw": "mean_xyc(delta_ref**2)", "e_global": "mean_xyc((delta_ref-mu)**2)",
                     "eta_spatial": "e_global/(e_raw+epsilon); null for degenerate e_raw",
                     "patch_variance": "mean_c Var_population_p(mean_pixels_in_nonoverlapping_patch(delta_ref))",
                     "patch_spatial_retention": "patch_variance/e_global; null if e_global<=degenerate threshold",
                     "channel_mu": "mean_xy(delta_c)", "channel_mean_abs": "mean_xy(abs(delta_c))",
                     "channel_spatial_std": "population_std_xy(delta_c)",
                     "channel_discrepancy": "max(mu)-min(mu)", "channel_variance": "population_var_RGB(mu)"},
        "selection": {"rules": ["patch variance q=.5", "patch variance q=.8", "channel discrepancy q=.8"],
                      "rank": "floor(q*(n-1)+0.5)", "ties": "metric, lexicographic sample_id, input, GT",
                      "duplicate_choice": "nearest unused rank; lower rank wins tie", "post_view_reselection": False,
                      "eligibility": "all successfully read nondegenerate test pairs; no metric outlier filtering"},
        "visualization": {"signed": {"cmap": "coolwarm", "vmin": -SIGNED_LIMIT, "center": 0, "vmax": SIGNED_LIMIT},
                          "magnitude": {"cmap": "magma", "vmin": 0, "vmax": MAGNITUDE_LIMIT},
                          "scale_scope": "fixed across all channels, images, datasets; linear",
                          "full_range_supplement": {"signed": [-1, 1], "magnitude": [0, math.sqrt(3)]},
                          "oracle": "clip to [0,1] for display ONLY; errors remain unclipped",
                          "png_dpi": 350, "pdf_fonttype": 42, "svg_fonttype": "none"},
        "claim_boundary": "Reference-derived spatial/channel heterogeneity; no physical-field or INR superiority claim",
        "sample_limit": args.sample_limit, "is_smoke_test": args.sample_limit is not None,
    }


def metric_table(stats: dict, metrics: Sequence[str]) -> str:
    lines = ["| 指标 | Mean | Median | Std (ddof=0) | P05 | P95 | 有效数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in metrics:
        s = stats[name]
        def f(v: Any) -> str:
            return "null" if v is None else f"{v:.6g}"
        lines.append(f"| {name} | {f(s['mean'])} | {f(s['median'])} | {f(s['std'])} | {f(s['p05'])} | {f(s['p95'])} | {s['valid_count']} |")
    return "\n".join(lines)


def write_report(out: Path, summary: dict, selected: list[dict], rows: list[dict], source: dict, patch_size: int) -> None:
    stats = summary["metrics"]
    name = summary["dataset"].upper()
    main_metrics = ["residual_magnitude_mean", "e_raw", "e_global", "eta_spatial", "patch_spatial_variance",
                    f"patch_spatial_retention_{patch_size}", "channel_discrepancy", "channel_variance"]
    channel_metrics = [prefix+c for prefix in ("mu_", "mean_abs_", "spatial_std_") for c in CHANNELS]
    channel_metrics += ["mu_R_minus_G", "mu_R_minus_B", "mu_G_minus_B"]
    selected_lines = []
    for item in selected:
        r = next(r for r in rows if r["sample_id"] == item["sample_id"])
        selected_lines.append(f"- **{item['sample_id']}**：{item['selection_rule']}，{item['selection_metric']}={item['metric_value']:.6g}；"
                              f"eta={r['eta_spatial']:.4f}，Vpatch={r['patch_spatial_variance']:.6g}；"
                              f"mu(R/G/B)=({r['mu_R']:.4f}, {r['mu_G']:.4f}, {r['mu_B']:.4f})；"
                              f"s(R/G/B)=({r['spatial_std_R']:.4f}, {r['spatial_std_G']:.4f}, {r['spatial_std_B']:.4f})；"
                              f"oracle 标量截断比例={r['oracle_scalar_clipped_fraction']:.3%}。")
    scope = "完整固定测试集" if summary["analysis_status"] == "PASS" else "非完整正式实验：不得把本结果称为完整测试集结论"
    if rows:
        eta = stats["eta_spatial"]["median"]
        retention = stats[f"patch_spatial_retention_{patch_size}"]["median"]
        lead = (f"逐图最优全局加性 RGB 偏移后，残差能量比中位数为 {eta:.2%}；"
                if eta is not None else "所有能量比例均为退化情形；")
        lead += (f"{patch_size}x{patch_size} 块平均保留空间残差能量的比例中位数为 {retention:.2%}。"
                 if retention is not None else "空间残差近零，块平均能量比例未定义。")
    else:
        lead = "没有有效图像，无法作出科学结论。"
    sensitivity = metric_table(stats, [f"patch_variance_{p}" for p in sorted({8, 16, 32, patch_size})]) if rows else "无有效结果。"
    text = f"""# {name}：参考残差校正需求分析

状态：**{summary['analysis_status']}**；{scope}。预期 {summary['expected_pairs']}，尝试 {summary['attempted_pairs']}，有效 {summary['valid_pairs']}，失败 {summary['failed_pairs']}，未尝试 {summary['unattempted_pairs']}。

{lead} 这是对参考图像和全局**加性**RGB 模型的描述性诊断，不是训练方法比较。数值显著性或视觉说服力不由文件生成成功来判定。

## A. 实验目标

判断成对输入到参考的恢复残差是否含有空间不均匀和通道相关模式，为自适应校正表示提供数据层面的动机。本分析不训练、不运行网络、不访问 checkpoint。delta_ref=Y-I 不是 UICF 的系数场 R(x)，也不是真实水下物理退化场。

## B. 数据来源与协议

- 数据根目录：`{source['data_root']}`。
- 测试清单：`{source['manifests']['test']['path']}`；SHA256 `{source['manifests']['test']['sha256']}`。
- 仓库 HEAD：`{source['repository_sha']}`。全部划分按 TSV 验证数量、ID/路径唯一性及跨划分互斥；不按物理 Train/Val 文件夹决定成员。
- Input/Reference 严格 RGB、原始尺寸匹配；同样的 PIL BILINEAR 256x256 Resize，float32/255；再提升到 float64 计算残差与归约。无测试裁剪、增强、gamma、CLAHE、色彩归一化或线性光转换。
- 原始和分析尺寸、每个原图 SHA256 见 source_manifest.json；失败见 failed_samples.csv。参考伪影与亚像素配准误差不能靠文件检查排除。
- seed={source['seed']}。每幅图像等权；总体方差/标准差 ddof=0，分位数采用线性插值。

## C. 数学定义

delta_ref(x)=Y(x)-I(x)，D(x)=||delta_ref(x)||2。mu_c=mean_x(delta_c)。GT-dependent Global RGB Oracle 为 I+mu，是 GT 决定的最小二乘最优全局加性偏移，不能部署或用于 SOTA 排名。

E_raw=mean_xc(delta^2)，E_global=mean_xc((delta-mu)^2)，eta=E_global/(E_raw+1e-12)。E_raw<=1e-10 的样本标记退化，eta=null。误差始终使用未截断 oracle，只有 RGB 显示截断到 [0,1]。

V_patch=mean_c Var_population_p(mean_patch(delta_c))；主尺度 {patch_size}，敏感性尺度 8/16/32，无重叠且不裁边。V_patch/E_global 衡量块平均保留的空间能量，不能排除所有参考风格或配准影响。

通道统计为 signed mean mu、mean absolute residual m、spatial population std s；D_channel=max(mu)-min(mu)，V_channel=Var_population_c(mu)。这些值没有衰减系数的物理含义。

## D. 定量结果

{metric_table(stats, main_metrics) if rows else '没有有效结果。'}

{metric_table(stats, channel_metrics) if rows else ''}

### 空间尺度敏感性

{sensitivity}

完整逐图结果及所有指标的有效/空值数量见 `statistics/`。能量分解 E_raw=E_global+mean(mu^2) 和 V_patch<=E_global 逐图检查；方差随嵌套块尺度增大不能增大。上述一致性是数学检查，不是物理机制检验。

## E. 图像分析

按完整已分析集合的固定排序选取，未观察图像后再选择。小样本 smoke 的选择只适用于 smoke 子集。

{chr(10).join(selected_lines) if selected_lines else '没有可选的非退化样本。'}

![固定规则选择的参考残差](figures/motivation_main.png)

![GT-dependent oracle 诊断](figures/global_oracle_diagnostic.png)

![块平均空间残差](figures/patch_level_diagnostic.png)

![逐图分布](figures/dataset_distributions.png)

主图所有 signed map 共享 [-0.5,0.5] 的零中心 coolwarm，所有 magnitude map 共享 [0,1] 的 magma；两个数据集也使用同一映射。主图显示饱和比例逐图保存在 CSV 和 figure_display_audit.json。额外 theoretical-range 图使用 [-1,1] 和 [0,sqrt(3)]，核查现象对显示截断的依赖。无逐图/逐通道 min-max。

## F. 论文可用性判断与边界

1. 清晰度：六列直接并列 Input/Reference/幅值/R/G/B，独立共享色条；最终须检查 PNG、PDF、SVG 的实际渲染记录。
2. 现象：结合全测试分布、未解释能量、块平均方差及成对通道差值判断。非零不自动等于视觉或统计显著，不从一幅热图得出总体结论。
3. 与 UICF 动机关系：若空间/通道模式均存在，支持研究自适应校正表示；不直接证明连续 INR 的必要性、优势或特定 UICF 模型表现。
4. 与现有实验区别：这是纯 Input/Reference 数据诊断；已有 R(x) 对齐或干预分析读取训练后模型，证据对象不同。
5. Introduction：仅在完整测试、跨数据集趋势和实际图像审计共同支持时采用。若近零、显示饱和或参考伪影主导，应降低结论或不用该图。
6. 审稿风险：GT-dependent oracle 只覆盖全局加性 RGB 偏移，不能推广到所有全局映射；参考来源/编辑风格/配准、256-square Resize、描述性选样与有限场景覆盖限制泛化。块平均减弱高频纹理影响，不消除全部混杂。

本数据集报告不混合两个数据集。跨数据集一致性及人工场景/参考质量判断应结合最终交付中的联合科学审计；任何未完成检查必须明确标记。

## G. Suggested caption

Reference-derived correction demand on paired underwater images. Inputs and references are processed identically. The correction magnitude and signed RGB residuals share fixed scales across examples. Representative images are selected deterministically by image-level spatial and channel statistics. These residuals characterize reference-dependent restoration requirements rather than a physical degradation field or a learned UICF coefficient field.

## 可复现资料

完整执行命令：

```bash
{source['command']}
```

环境、源码/清单/原图哈希、公式、阈值、选择规则、截断比例和完成计数在 `metadata/`。全零残差比例为空，不使用 epsilon 制造结论。ZIP 不含原始数据集或 checkpoint。
"""
    with (out / "MOTIVATION_ANALYSIS_CN.md").open("x", encoding="utf-8") as handle:
        handle.write(text)


def export_zip(out: Path) -> Path:
    target = out.with_suffix(".zip")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, Path(out.name) / p.relative_to(out))
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=tuple(COUNTS), required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", choices=("test",), default="test")
    p.add_argument("--output-dir", default="analysis/motivation_correction_demand")
    p.add_argument("--sample-limit", type=int, default=None, help="Smoke only; always PARTIAL")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--export-zip", action="store_true")
    p.add_argument("--audit-evidence", default=None, help="Frozen Phase-1 audit JSON to copy and reconcile")
    args = p.parse_args(argv)
    if args.sample_limit is not None and args.sample_limit <= 0:
        p.error("--sample-limit must be positive")
    if args.patch_size <= 0 or 256 % args.patch_size:
        p.error("--patch-size must be a positive divisor of 256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = new_run_dir(project_path(args.output_dir), args.dataset)
    for sub in ("figures", "statistics", "metadata"):
        (out / sub).mkdir()
    logger = logging.getLogger(f"correction_demand.{out.name}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(out / "analysis.log", mode="x", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); logger.addHandler(h)
    protocol = protocol_record(args)
    write_json(out / "metadata/protocol.json", protocol)
    expected = COUNTS[args.dataset]["test"]
    source: dict[str, Any] = {}
    rows: list[dict] = []
    failures: list[dict] = []
    status = "FAILED"
    zip_file: Path | None = None
    manifest_path: Path | None = None
    repo_sha = git("rev-parse", "HEAD")
    command_args = list(sys.argv[1:] if argv is None else argv)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *command_args])
    try:
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was already initialized; refusing model-free analysis")
        baseline = protected_snapshot()
        config_path = project_path(args.config or ("configs/config_v16.yaml" if args.dataset == "lsui" else "configs/uieb/config_v16_uieb.yaml"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config["experiment"]["version"] != "v16":
            raise ValueError("This audited protocol requires a V16 config")
        label = str(config["data"].get("dataset", "")).lower()
        if label != ("lsui19" if args.dataset == "lsui" else "uieb"):
            raise ValueError(f"Dataset/config identity mismatch: {args.dataset} vs {label}")
        if config["data"]["expected_counts"] != COUNTS[args.dataset]:
            raise ValueError("Configured sample counts differ from canonical fixed protocol")
        evaluation = config["evaluation"]
        if evaluation.get("resize") is not True or int(evaluation.get("size", 0)) != 256:
            raise ValueError("Audited protocol requires paired 256x256 bilinear Resize")
        root = Path(args.data_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"data.root is unavailable: {root}")
        manifests = {s: project_path(config["data"][f"{s}_manifest"]) for s in ("train", "validation", "test")}
        parsed = validate_split_protocol(manifests, COUNTS[args.dataset])
        manifest_path = manifests["test"]
        selected_entries = parsed["test"] if args.sample_limit is None else parsed["test"][:args.sample_limit]
        source = {"repository_sha": repo_sha, "branch": git("branch", "--show-current"),
                  "origin": git("remote", "get-url", "origin"), "repository_root": str(PROJECT_ROOT),
                  "git_status_at_start": git("status", "--short"), "python": sys.version,
                  "executable": sys.executable, "platform": platform.platform(),
                  "versions": {"numpy": np.__version__, "Pillow": PIL.__version__, "PyYAML": yaml.__version__,
                               "torch_dataset_import_only": torch.__version__},
                  "config": {"path": str(config_path), "sha256": sha256(config_path), "content": config},
                  "data_root": str(root), "seed": args.seed, "patch_size": args.patch_size,
                  "preprocessing": protocol["preprocessing"], "formulas": protocol["formulas"],
                  "visualization": protocol["visualization"], "command": command,
                  "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  "manifests": {s: {"path": str(p), "sha256": sha256(p), "count": len(parsed[s])} for s, p in manifests.items()},
                  "source_sha256": {p: sha256(PROJECT_ROOT / p) for p in [
                      "src/v16/dataset.py", "src/v16/test.py", "tools/analyze_motivation_correction_demand.py",
                      "tests/test_motivation_correction_demand.py", "tools/README_motivation_correction_demand.md"]},
                  "images": [], "checkpoint_accessed": False, "model_created": False}
        if args.audit_evidence:
            evidence_path = project_path(args.audit_evidence)
            evidence = json.loads(evidence_path.read_text())
            if evidence["sha"] != repo_sha:
                raise ValueError("Phase-1 audit SHA differs from execution HEAD")
            for s in manifests:
                if evidence["datasets"][args.dataset]["manifest_sha256"][s] != sha256(manifests[s]):
                    raise ValueError(f"Phase-1 {s} manifest differs from execution")
            source["phase1_audit_sha256"] = sha256(evidence_path)
            shutil.copyfile(evidence_path, out / "metadata/phase1_repository_data_audit.json")
        shutil.copyfile(config_path, out / "metadata/config_used.yaml")
        shutil.copyfile(PROJECT_ROOT / "tools/README_motivation_correction_demand.md", out / "metadata/frozen_design.md")
        for s, p in manifests.items():
            shutil.copyfile(p, out / "metadata" / (s + ".tsv"))
        logger.info("START dataset=%s expected=%d selected=%d root=%s", args.dataset, expected, len(selected_entries), root)
        for index, e in enumerate(selected_entries):
            context = {"dataset": args.dataset, "sample_id": e.sample_id,
                       "input_relative": e.input_relative, "gt_relative": e.gt_relative,
                       "input_path": str(root / e.input_relative), "gt_path": str(root / e.gt_relative)}
            try:
                a_path, b_path = safe_pair_paths(e, root)
                hashes = {"input_sha256": sha256(a_path), "gt_sha256": sha256(b_path)}
                a, y, dimensions = load_pair(e, root, evaluation)
                metrics = compute_metrics(a, y, args.patch_size)
                row = {**context, "original_width": dimensions["original_dimensions_wh"][0],
                       "original_height": dimensions["original_dimensions_wh"][1],
                       "analysis_width": a.shape[1], "analysis_height": a.shape[0], **metrics}
                rows.append(row)
                source["images"].append({**context, **hashes, **dimensions, "status": "valid"})
            except Exception as error:
                failures.append({**context, "error_type": type(error).__name__, "reason": str(error)})
                logger.error("FAILED sample=%s %s: %s", e.sample_id, type(error).__name__, error)
            if (index+1) % 50 == 0 or index+1 == len(selected_entries):
                logger.info("progress attempted=%d valid=%d failed=%d", index+1, len(rows), len(failures))
        write_csv(out / "statistics/per_image_metrics.csv", rows)
        write_csv(out / "statistics/failed_samples.csv", failures,
                  ["dataset", "sample_id", "input_relative", "gt_relative", "input_path", "gt_path", "error_type", "reason"])
        keys = [k for k in rows[0] if k not in set(context) | {"degenerate", "original_width", "original_height", "analysis_width", "analysis_height"}] if rows else []
        stats = {key: summarize([r[key] for r in rows]) for key in keys}
        for v in stats.values():
            v["failed_sample_count"] = len(failures)
        sensitivity = []
        for p in sorted({8, 16, 32, args.patch_size}):
            sensitivity.append({"dataset": args.dataset, "patch_size": p, **summarize([r[f"patch_variance_{p}"] for r in rows]),
                                "retained_spatial_energy_median": summarize([r[f"patch_spatial_retention_{p}"] for r in rows])["median"]})
        write_csv(out / "statistics/sensitivity_analysis.csv", sensitivity)
        chosen = select_samples(rows)
        # Selection is persisted before any example plotting.
        write_json(out / "metadata/selected_samples.json", chosen)
        generated, clipping = ([], [])
        if not args.no_figures:
            generated, clipping = render_figures(chosen, rows, root, evaluation, out / "figures", args.patch_size)
        write_json(out / "metadata/figure_display_audit.json", {"limits": protocol["visualization"], "selected_samples": clipping,
                   "full_test_image_weighted_clipping": {k: stats[k] for k in stats if k.startswith("display_")}})
        # Stability checks read only task sources, protected code and the paired images.
        stable_images = all(sha256(Path(i["input_path"])) == i["input_sha256"] and
                            sha256(Path(i["gt_path"])) == i["gt_sha256"] for i in source["images"])
        stable_source = all(sha256(PROJECT_ROOT / p) == h for p, h in baseline.items())
        monotone_patches = all(r["patch_variance_8"] + 1e-12 >= r["patch_variance_16"] and
                              r["patch_variance_16"] + 1e-12 >= r["patch_variance_32"] for r in rows)
        if not stable_images or not stable_source or not monotone_patches or torch.cuda.is_initialized():
            raise RuntimeError("Source/image stability, patch monotonicity or CPU-only invariant failed")
        complete = args.sample_limit is None and len(rows) == expected and not failures
        status = "PASS" if complete else "PARTIAL" if rows else "FAILED"
        audit = {"manifest_counts_and_leakage": "PASS", "image_integrity": "PASS" if not failures else "FAILED",
                 "energy_decomposition_and_bounds": "PASS", "nested_patch_variance_monotone": monotone_patches,
                 "source_code_unchanged": stable_source, "input_reference_bytes_unchanged": stable_images,
                 "cuda_initialized": torch.cuda.is_initialized(), "model_modules_imported": [m for m in sys.modules if m.startswith("src.") and (m.endswith(".models") or m.endswith(".test"))],
                 "unittest_status": "EXTERNAL; see delivery verification/unit_tests.log",
                 "scientific_visual_review": "EXTERNAL; see delivery scientific audit",
                 "analysis_status_meaning": "computational/protocol completeness, not scientific hypothesis acceptance"}
        if audit["model_modules_imported"]:
            raise RuntimeError("Unexpected model module import")
        write_json(out / "metadata/audit_checks.json", audit)
        summary = {"dataset": args.dataset, "split": "test", "analysis_status": status,
                   "expected_pairs": expected, "manifest_pairs": len(parsed["test"]),
                   "attempted_pairs": len(selected_entries), "valid_pairs": len(rows), "failed_pairs": len(failures),
                   "unattempted_pairs": expected-len(selected_entries),
                   "degenerate_pairs": sum(r["degenerate"] for r in rows),
                   "is_smoke_test": args.sample_limit is not None, "aggregation": "per-image equal weight",
                   "variance_ddof": 0, "metrics": stats,
                   "channel_direction_counts": {f"mu_{a}_gt_{b}": sum(r[f"mu_{a}"] > r[f"mu_{b}"] for r in rows) for a, b in [("R", "G"), ("R", "B"), ("G", "B")]},
                   "figure_file_count": len(generated), "selected_sample_count": len(chosen)}
        source.update({"completed_counts": {k: summary[k] for k in ("expected_pairs", "attempted_pairs", "valid_pairs", "failed_pairs", "unattempted_pairs")},
                       "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                       "cuda_initialized": False, "protected_hashes": baseline})
        if not args.no_figures:
            import matplotlib
            source["versions"]["matplotlib"] = matplotlib.__version__
        write_json(out / "metadata/source_manifest.json", source)
        write_json(out / "statistics/dataset_summary.json", summary)
        write_report(out, summary, chosen, rows, source, args.patch_size)
        with (out / "README.md").open("x", encoding="utf-8") as handle:
            handle.write(f"# {args.dataset.upper()} reference-derived correction demand\n\nStatus: {status}; valid {len(rows)}/{expected}; failed {len(failures)}.\n\n"
                         f"[中文科研报告](MOTIVATION_ANALYSIS_CN.md) | [Protocol](metadata/protocol.json) | [Source manifest](metadata/source_manifest.json)\n\n"
                         f"Command:\n\n```bash\n{command}\n```\n\n"
                         "The GT-dependent Global RGB Oracle is analytical and not deployable. Statistics use the unclipped oracle.\n"
                         "Plots may saturate at fixed documented display limits; raw arrays and theoretical-range plots are retained.\n")
        logger.info("FINISHED status=%s valid=%d failed=%d figures=%d", status, len(rows), len(failures), len(generated))
    except Exception as error:
        status = "FAILED"
        logger.exception("Analysis blocked: %s", error)
        write_json(out / "metadata/fatal_error.json", {"error_type": type(error).__name__, "reason": str(error),
                   "analysis_status": "FAILED", "valid_pairs_before_error": len(rows), "command": command})
        if not (out / "README.md").exists():
            with (out / "README.md").open("x", encoding="utf-8") as handle:
                handle.write(f"# FAILED analysis\n\n{type(error).__name__}: {error}\n\nSee analysis.log and metadata/fatal_error.json. No complete experiment is claimed.\n")
    finally:
        for h in (fh, sh):
            h.flush(); h.close(); logger.removeHandler(h)
    if args.export_zip:
        zip_file = export_zip(out)
    for key, value in (("REPOSITORY_SHA", repo_sha), ("DATASET", args.dataset), ("DATA_ROOT", args.data_root),
                       ("MANIFEST", str(manifest_path) if manifest_path else "UNAVAILABLE"), ("EXPECTED_PAIRS", expected),
                       ("VALID_PAIRS", len(rows)), ("FAILED_PAIRS", len(failures)),
                       ("TEST_STATUS", "EXTERNAL_UNIT_TEST_REPORT"), ("ANALYSIS_STATUS", status),
                       ("OUTPUT_DIR", out), ("ZIP_FILE", zip_file or "NOT_REQUESTED")):
        print(f"{key}: {value}")
    return 0 if status in ("PASS", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
