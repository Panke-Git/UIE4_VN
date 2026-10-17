"""Strict paired comparison with clean-ID clustered bootstrap confidence intervals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .eval import image_key
from .prepare import COLUMNS, sha256, validate_bundle, write_json
from .utils import fairness, load_yaml, project_path, protocol, read_csv, write_csv


def bootstrap(deltas, clean_ids, seed=3520, iterations=5000):
    if iterations < 2000:
        raise ValueError("At least 2000 bootstrap replicates required")
    deltas = np.asarray(deltas, dtype=np.float64)
    if not np.isfinite(deltas).all() or len(deltas) != len(clean_ids) or not len(deltas):
        raise ValueError("Invalid bootstrap data")
    groups = defaultdict(list)
    for clean, value in zip(clean_ids, deltas):
        groups[clean].append(value)
    values = [groups[k] for k in sorted(groups)]
    sums = np.array([np.sum(v) for v in values])
    sizes = np.array([len(v) for v in values])
    rng = np.random.default_rng(seed)
    result = np.empty(iterations)
    # Each replicate resamples clean IDs and includes all their haze versions.
    for i in range(iterations):
        indices = rng.integers(0, len(values), size=len(values))
        result[i] = sums[indices].sum() / sizes[indices].sum()
    return {"mean": float(deltas.mean()), "ci95": np.quantile(result, [0.025, 0.975]).tolist(),
            "positive_fraction": float(np.mean(deltas > 0)), "seed": seed, "iterations": iterations,
            "unit": "clean_id", "clean_groups": len(values),
            "method": "paired clustered percentile bootstrap; image-weighted mean within resampled clusters"}


def load_result(run, resolution):
    directory = Path(run) / f"test_{resolution}"
    if (directory / "FAILED.json").exists():
        raise ValueError(f"Evaluation failed/incomplete: {directory}")
    summary = json.loads((directory / "summary.json").read_text())
    if sha256(directory / "per_image.csv") != summary["per_image_sha256"]:
        raise ValueError("Per-image result hash mismatch")
    meta, splits, _ = validate_bundle(Path(run) / "split")
    c = load_yaml(Path(run) / "config_resolved.yaml")
    if summary["training_protocol"] != fairness(c):
        raise ValueError("Evaluation training protocol differs from saved config")
    if summary["variant"] != c["variant"] or summary["split_hashes"] != meta["hashes"]:
        raise ValueError("Evaluation differs from saved run/split")
    if sha256(Path(run) / "checkpoint/best_psnr.pt") != summary["checkpoint_sha256"]:
        raise ValueError("Evaluated checkpoint has changed")
    rows = read_csv(directory / "per_image.csv")
    expected = {r["sample_id"]: r for r in splits["test"]}
    if len(rows) != len(expected) or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate test IDs")
    for r in rows:
        if r["sample_id"] not in expected or any(r[k] != expected[r["sample_id"]][k] for k in COLUMNS):
            raise ValueError(f"Test ID/input/GT pairing mismatch: {r}")
        for key in ("psnr", "ssim"):
            r[key] = float(r[key])
            if not np.isfinite(r[key]):
                raise ValueError(f"Nonfinite result: {r}")
    for key in ("psnr", "ssim"):
        if not np.isclose(np.mean([r[key] for r in rows]), summary[key], rtol=0, atol=1e-12):
            raise ValueError("Summary is not mean of per-image scores")
    if summary["count"] != len(rows):
        raise ValueError("Summary sample count mismatch")
    return directory, summary, {r["sample_id"]: r for r in rows}


def align(full, wo, frows, wrows, resolution):
    if full["variant"] != "full" or wo["variant"] != "wo_uicf":
        raise ValueError("Expected independent full and wo_uicf runs")
    if full["model_type"] != "UICFPreBackbone(V15 PlainUNetColorQuery)" or wo["model_type"] != "V15 PlainUNetColorQuery":
        raise ValueError("Unexpected model types")
    for key in ("split", "split_hashes", "protocol", "checkpoint", "seed", "synthetic", "smoke", "training_protocol", "source_hashes"):
        if full[key] != wo[key]:
            raise ValueError(f"Incompatible comparison: {key}")
    if full["split"] != "test" or full["checkpoint"] != "best_psnr" or full["protocol"] != protocol(resolution):
        raise ValueError("Comparison requires official test and best validation PSNR with exact metric/resolution protocol")
    if full["training_runtime"]["amp_enabled"] != wo["training_runtime"]["amp_enabled"]:
        raise ValueError("Actual AMP training settings differ")
    if set(frows) != set(wrows):
        raise ValueError("Test sample IDs differ")
    for sid in frows:
        if any(frows[sid][key] != wrows[sid][key] for key in COLUMNS):
            raise ValueError(f"Paired sample identity differs: {sid}")


def panels(directory, fd, wd, ids):
    out = directory / "panels"
    out.mkdir()
    for sid in ids:
        name = image_key(sid)
        paths = [fd / "images/hazy" / name, wd / "images/enhanced" / name,
                 fd / "images/enhanced" / name, fd / "images/gt" / name]
        if not all(p.is_file() for p in paths):
            raise ValueError(f"Missing preselected visualization images: {sid}")
        ims = []
        for p in paths:
            with Image.open(p) as im:
                ims.append(im.convert("RGB"))
        if len({im.size for im in ims}) != 1:
            raise ValueError("Panel sizes differ")
        w, h = ims[0].size
        canvas = Image.new("RGB", (4 * w, h + 24), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (im, label) in enumerate(zip(ims, ["Hazy", "w/o UICF", "Full", "GT"])):
            canvas.paste(im, (i * w, 24))
            draw.text((i * w + 3, 4), label, fill="black")
        canvas.save(out / name)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full-run", required=True)
    p.add_argument("--wo-run", required=True)
    p.add_argument("--resolution", choices=("256", "native"), default="256")
    p.add_argument("--bootstrap-seed", type=int, default=3520)
    p.add_argument("--bootstrap-iterations", type=int, default=5000)
    a = p.parse_args(argv)
    fr, wr = project_path(a.full_run), project_path(a.wo_run)
    fd, fs, frows = load_result(fr, a.resolution)
    wd, ws, wrows = load_result(wr, a.resolution)
    align(fs, ws, frows, wrows, a.resolution)
    paired = []
    for sid in sorted(frows):
        f, w = frows[sid], wrows[sid]
        paired.append({**{k: f[k] for k in COLUMNS}, "full_psnr": f["psnr"], "wo_psnr": w["psnr"],
                       "delta_psnr": f["psnr"] - w["psnr"], "full_ssim": f["ssim"], "wo_ssim": w["ssim"],
                       "delta_ssim": f["ssim"] - w["ssim"]})
    result = {"full_run": str(fr), "wo_run": str(wr), "protocol": fs["protocol"], "count": len(paired),
              "synthetic": fs["synthetic"], "smoke": fs["smoke"], "full": {k: fs[k] for k in ("psnr", "ssim")},
              "wo_uicf": {k: ws[k] for k in ("psnr", "ssim")},
              "gain": {k: bootstrap([r[f"delta_{k}"] for r in paired], [r["clean_id"] for r in paired],
                                     a.bootstrap_seed, a.bootstrap_iterations) for k in ("psnr", "ssim")}}
    directory = fr / "comparisons" / f"{wr.name}_test_{a.resolution}"
    directory.mkdir(parents=True, exist_ok=False)
    # Fixed before metrics: common lexicographic first eight official-test IDs.
    ids = sorted(frows)[:min(8, len(frows))]
    panels(directory, fd, wd, ids)
    write_csv(directory / "paired_deltas.csv", paired)
    write_json(directory / "summary.json", result)
    write_json(directory / "visualization_selection.json", {"rule": "lexicographic first eight manifest IDs", "sample_ids": ids,
                                                            "file_names": {s: image_key(s) for s in ids}})
    title = "SYNTHETIC / SMOKE SOFTWARE CHECK — NOT PAPER EVIDENCE\n\n" if fs["smoke"] or fs["synthetic"] else ""
    table = title + f"Protocol: test_{a.resolution}; {fs['protocol']['resize']}; best validation PSNR checkpoint.\n\n"
    table += "| Variant | Haze4K PSNR ↑ | Haze4K SSIM ↑ |\n|---|---:|---:|\n"
    table += f"| w/o UICF (Color-Query backbone) | {ws['psnr']:.4f} | {ws['ssim']:.6f} |\n"
    table += f"| Full ICAR-UIE (UICF + QGRD) | {fs['psnr']:.4f} | {fs['ssim']:.6f} |\n"
    table += f"| Δ (Full − w/o UICF) | {result['gain']['psnr']['mean']:+.4f} | {result['gain']['ssim']['mean']:+.6f} |\n\n"
    for key in ("psnr", "ssim"):
        g = result["gain"][key]
        table += f"Δ{key.upper()} paired clean-ID bootstrap 95% CI: [{g['ci95'][0]:.6f}, {g['ci95'][1]:.6f}]; positive-image fraction {g['positive_fraction']:.4%}.\n\n"
    table += f"Bootstrap: {a.bootstrap_iterations} replicates, seed {a.bootstrap_seed}. "
    table += "From-scratch cross-task applicability and incremental UICF contribution only; not zero-shot, all-in-one, or an isolated QGRD claim.\n"
    (directory / "paper_table.md").write_text(table)
    print(table)
    print(f"COMPARISON_DIR={directory}")
    return directory


if __name__ == "__main__":
    main()
