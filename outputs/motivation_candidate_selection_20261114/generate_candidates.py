#!/usr/bin/env python3
"""Verify the audited Input–GT pairs and lay out the recorded candidate selections."""

import argparse
import ast
import base64
from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


RELEASE = Path("analysis/motivation_correction_demand/run_20261106_005616_delivery")
DATASETS = {"lsui": ("LSUI", 428), "uieb": ("UIEB T90", 90)}
SIZE = 256
PATCH = 16
EPS = 1e-12
ENERGY_MIN = 1e-10
TOLERANCE = 2e-10  # Frozen independent verifier uses a strict absolute bound.
CVPR_COLUMN_WIDTH_PT = 236.25
TITLES = ("Input", "Global RGB Shift Oracle", "Remaining Correction Magnitude", "Reference")
RGB_KEYS = tuple("mu_" + c for c in "RGB")
METRIC_KEYS = (*RGB_KEYS, "e_raw", "e_global", "eta_spatial", "patch_variance_16",
               "patch_spatial_retention_16", "oracle_pixel_clipped_fraction",
               "oracle_scalar_clipped_fraction", "display_remaining_clipped_fraction")
PERCENTILE_KEYS = {"shift": "RGB Shift L2", "eta": "Remaining Energy Ratio",
                   "v16": "V16", "retention": "Retained Spatial Energy Ratio",
                   "e_global": "Global Oracle Error"}
INK = HexColor("#17212b")
MUTED = HexColor("#3b4858")
RULE = HexColor("#dce1e6")
REGULAR = "CandidateSans"
BOLD = "CandidateSansBold"


@dataclass(frozen=True)
class Candidate:
    dataset: str
    band: str
    row: dict
    reason: str
    percentiles: dict
    sources: dict
    panels: tuple

    @property
    def sample_id(self):
        return self.row["sample_id"]

    @property
    def image_path(self):
        return Path("individual_samples") / self.dataset / (self.band + "_" + self.sample_id + ".png")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def validate_inventory(release, project):
    inventory_path = release / "artifact_inventory.json"
    inventory = read_json(inventory_path)
    for relative, record in inventory.items():
        path = release / relative
        require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                f"Release artifact no longer matches inventory: {path}")
    for name in ("analyze", "verify"):
        relative = f"tools/{name}_motivation_correction_demand.py"
        require(digest(project / relative) == inventory["source/" + relative]["sha256"],
                f"Current and archived scripts differ: {relative}")
    return inventory_path, len(inventory)


def read_audited_dataset(root, dataset, expected_count):
    metadata = root / "metadata"
    proto = read_json(metadata / "protocol.json")
    manifest = read_json(metadata / "source_manifest.json")
    summary = read_json(root / "statistics/dataset_summary.json")
    with (root / "statistics/per_image_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (root / "statistics/failed_samples.csv").open(newline="", encoding="utf-8") as handle:
        require(not list(csv.DictReader(handle)), f"{dataset}: release has failed sample rows")
    with (metadata / "test.tsv").open(newline="", encoding="utf-8") as handle:
        test_pairs = list(csv.reader(handle, delimiter="	"))
    fields = ("sample_id", "input_relative", "gt_relative")
    require(test_pairs == [[r[k] for k in fields] for r in rows]
            == [[r[k] for k in fields] for r in manifest["images"]],
            f"{dataset}: published test TSV, CSV and per-file checksums disagree")
    require(len(rows) == len({r["sample_id"] for r in rows}) == expected_count
            == summary["valid_pairs"] == summary["expected_pairs"]
            and summary["failed_pairs"] == 0 and summary["analysis_status"] == "PASS",
            f"{dataset}: coverage or test run status is invalid")
    for split, rec in manifest["manifests"].items():
        require(digest(metadata / f"{split}.tsv") == rec["sha256"], f"{dataset}: {split} hash mismatch")
    preprocessing = proto["preprocessing"]
    require(proto["protocol_version"] == "reference_correction_demand_v1_20261106"
            and proto["dataset"] == dataset and proto["split"] == "test"
            and proto["sample_limit"] is None and proto["is_smoke_test"] is False
            and proto["patch_size"] == PATCH and proto["variance_ddof"] == 0
            and proto["epsilon"] == EPS and proto["degenerate_e_raw_threshold"] == ENERGY_MIN
            and preprocessing["resize"] == [SIZE, SIZE] and preprocessing["range"] == [0, 1]
            and preprocessing["mode"] == "strict RGB" and preprocessing["load_dtype"] == "float32"
            and preprocessing["residual_and_reductions_dtype"] == "float64"
            and preprocessing["interpolation"] == "PIL.Image.Resampling.BILINEAR"
            and preprocessing["augmentation"] is False
            and proto["visualization"]["magnitude"] == {"cmap": "magma", "vmin": 0, "vmax": 1.0},
            f"{dataset}: release protocol differs from the implemented display protocol")
    return rows, manifest, read_json(metadata / "selected_samples.json")


def paths_for(info, original_root, publicdata, dataset):
    paths = {}
    for role in ("input", "gt"):
        relative = Path(info[f"{role}_relative"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"{dataset}: unsafe source manifest path {relative}")
        alternatives = [Path(original_root) / relative]
        if dataset == "lsui":
            alternatives.extend((publicdata / "LSUI19" / relative,
                                 publicdata / "LSUI" / relative.parent.name / relative.name))
        else:
            alternatives.append(publicdata / "UIEB19" / relative)
        for candidate_path in dict.fromkeys(alternatives):
            if candidate_path.is_file() and digest(candidate_path) == info[f"{role}_sha256"]:
                paths[role] = candidate_path.resolve()
                break
        else:
            attempted = ", ".join(str(p) for p in alternatives)
            raise FileNotFoundError(
                f"{dataset} {info['sample_id']} {role}: no exact published SHA-256 match at {attempted}"
            )
    return paths


def load_rgb(path, row):
    with Image.open(path) as image:
        native_size = int(row["original_width"]), int(row["original_height"])
        require(image.mode == "RGB" and image.size == native_size, f"Mode or size mismatch: {path}")
        encoded = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        return np.asarray(encoded, dtype=np.float32) / np.float32(255)


def calculate_arrays(input_rgb, reference_rgb):
    image = input_rgb.astype(np.float64)
    delta = reference_rgb.astype(np.float64) - image
    mu = delta.mean(axis=(0, 1))
    return {
        "input_rgb": input_rgb, "reference_rgb": reference_rgb, "delta_ref": delta,
        "mu_rgb": mu, "oracle_unclipped": image + mu, "remaining": delta - mu,
    }


def verify_metrics(arrays, row):
    mu = arrays["mu_rgb"]
    delta = arrays["delta_ref"]
    remaining = arrays["remaining"]
    oracle = arrays["oracle_unclipped"]
    outside = (oracle < 0) | (oracle > 1)
    e_raw = float(np.square(delta).mean())
    e_global = float(np.square(remaining).mean())
    require(row["degenerate"] == "False" and e_raw > ENERGY_MIN and e_global > ENERGY_MIN,
            f"Degenerate values for {row['sample_id']}")
    require(abs(e_raw - e_global - float(np.square(mu).mean())) < EPS,
            f"Energy decomposition violated for {row['sample_id']}")
    # Independently express V16 as the energy of block means of the centered residual.
    block_means = remaining.reshape(SIZE // PATCH, PATCH, SIZE // PATCH, PATCH, 3).mean(axis=(1, 3))
    v16 = float(np.square(block_means).mean())
    require(v16 <= e_global + EPS, f"Block mean energy exceeds remaining energy: {row['sample_id']}")
    magnitude = np.linalg.norm(remaining, axis=2)
    values = dict(zip(RGB_KEYS, map(float, mu), strict=True))
    values.update({
        "e_raw": e_raw, "e_global": e_global, "eta_spatial": e_global / (e_raw + EPS),
        "patch_variance_16": v16, "patch_spatial_retention_16": v16 / e_global,
        "oracle_pixel_clipped_fraction": float(outside.any(axis=2).mean()),
        "oracle_scalar_clipped_fraction": float(outside.mean()),
        "display_remaining_clipped_fraction": float((magnitude > 1).mean()),
    })
    for key, value in values.items():
        require(math.isfinite(float(row[key])) and abs(float(row[key]) - value) < TOLERANCE,
                f"{row['dataset']} {row['sample_id']} {key}: {value!r} vs release {row[key]}")
    return values


def published_magma(mpl_path):
    # Match the original Matplotlib 3.11.2 plot bytes without importing the
    # renderer or its dependencies. The 256 RGB triplets are data literals.
    tree = ast.parse(mpl_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_magma_data" for target in node.targets
        ):
            palette = np.asarray(ast.literal_eval(node.value), dtype=np.float64)
            require(palette.shape == (256, 3) and palette.min() >= 0 and palette.max() <= 1,
                    f"Malformed Matplotlib magma palette: {mpl_path}")
            return (palette * 255).astype(np.uint8)
    raise ValueError(f"Matplotlib magma palette not found: {mpl_path}")


def display_arrays(arrays, palette):
    input_rgb = arrays["input_rgb"]
    reference_rgb = arrays["reference_rgb"]
    oracle = np.clip(arrays["oracle_unclipped"], 0, 1)
    magnitude = np.linalg.norm(arrays["remaining"], axis=2)
    lut_indices = np.minimum((np.clip(magnitude, 0, 1) * len(palette)).astype(int), len(palette) - 1)
    return ((input_rgb * 255).astype(np.uint8), (oracle * 255).astype(np.uint8),
            palette[lut_indices], (reference_rgb * 255).astype(np.uint8))


def check_published_panels(svg, sample_index, panels):
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    # The historical diagnostic placed Reference third and Magnitude fourth.
    for panel, original in zip(panels, (1, 2, 4, 3), strict=True):
        axis = sample_index * 4 + original
        node = svg.find(f".//svg:g[@id='axes_{axis}']//svg:image", namespace)
        source = node.attrib["{http://www.w3.org/1999/xlink}href"].split(",", 1)[1]
        with Image.open(BytesIO(base64.b64decode(source))) as image:
            raster = np.asarray(image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB"))
        yy = ((np.arange(SIZE) + 0.5) * raster.shape[0] / SIZE).astype(int)
        xx = ((np.arange(SIZE) + 0.5) * raster.shape[1] / SIZE).astype(int)
        require(np.array_equal(panel, raster[yy[:, None], xx[None, :]]),
                f"Published SVG disagrees with the rebuilt pixels in axis {axis}")


def percentile_lookup(rows):
    features = {
        "shift": np.array([math.hypot(*(float(row[k]) for k in RGB_KEYS)) for row in rows]),
        "eta": np.array([float(row["eta_spatial"]) for row in rows]),
        "v16": np.array([float(row["patch_variance_16"]) for row in rows]),
        "retention": np.array([float(row["patch_spatial_retention_16"]) for row in rows]),
        "e_global": np.array([float(row["e_global"]) for row in rows]),
    }
    result = {r["sample_id"]: {} for r in rows}
    for name, values in features.items():
        require(np.isfinite(values).all(), f"Undefined percentile population: {name}")
        order = np.sort(values)
        left = np.searchsorted(order, values, side="left")
        right = np.searchsorted(order, values, side="right")
        for row, rank in zip(rows, 100 * (left + right) / (2 * len(values)), strict=True):
            result[row["sample_id"]][name] = float(rank)
    return result


def eligible(row, p, band, selection):
    g = selection["common_gates"]
    if (float(row["oracle_pixel_clipped_fraction"]) > g["max_oracle_pixel_clipping"]
        or float(row["oracle_scalar_clipped_fraction"]) > g["max_oracle_scalar_clipping"]
        or float(row["display_remaining_clipped_fraction"]) > g["max_magnitude_saturation"]
        or p["e_global"] < g["min_global_error_percentile"]):
        return False
    if not all(lo <= p[key] < hi for key, (lo, hi) in selection["bands"][band].items()):
        return False
    group = band[0]
    g = selection["group_gates"][group]
    eta = float(row["eta_spatial"])
    if group == "A":
        return g["eta_min"] <= eta <= g["eta_max"] and p["v16"] >= g["min_v16_percentile"]
    if group == "B":
        return (g["eta_min"] <= eta <= g["eta_max"] and p["retention"] >= g["min_retention_percentile"]
                and g["min_shift_percentile"] <= p["shift"] <= g["max_shift_percentile"])
    return (g["min_eta_percentile_inclusive"] <= p["eta"] < g["max_eta_percentile_exclusive"]
            and p["e_global"] <= g["max_global_error_percentile"]
            and p["retention"] >= g["min_retention_percentile"])


def verify_dataset(root, dataset, publicdata, palette, selection):
    _, expected_count = DATASETS[dataset]
    rows, manifest, historical = read_audited_dataset(root, dataset, expected_count)
    require({h["sample_id"] for h in historical} == set(selection["historical_review"][dataset]),
            f"{dataset}: historical review is incomplete")
    chosen = selection["selections"][dataset]
    order = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2")
    require(tuple(item["band"] for item in chosen) == order and len({item["id"] for item in chosen}) == 10,
            f"{dataset}: expected exactly 4 A, 4 B, 2 C unique selections in band order")
    percentages = percentile_lookup(rows)
    hist_by_id = {rec["sample_id"]: i for i, rec in enumerate(historical)}
    chosen_by_id = {rec["id"]: rec for rec in chosen}
    hist_metrics = {}
    candidates = {}
    stats = dict.fromkeys(METRIC_KEYS, 0.0)
    chosen_max = {}
    source_counts = Counter()
    eligible_counts = {}
    for band in order:
        band_pool = {r["sample_id"] for r in rows if eligible(r, percentages[r["sample_id"]], band, selection)}
        reviewed = selection["reviewed_ids_by_band"][dataset][band]
        require(len(set(reviewed)) == len(reviewed) and set(reviewed) <= band_pool,
                f"{dataset} {band}: reviewed IDs violate the recorded gates")
        eligible_counts[band] = len(band_pool)
    svg = ET.parse(root / "figures/global_oracle_diagnostic.svg").getroot()
    for row, info in zip(rows, manifest["images"], strict=True):
        ident = row["sample_id"]
        require(re.fullmatch(r"[A-Za-z0-9_-]+", ident) is not None and row["dataset"] == dataset,
                f"Invalid or unexpected sample identity: {ident}")
        paths = paths_for(info, manifest["data_root"], publicdata, dataset)
        for path in paths.values():
            location = path.relative_to(publicdata).parts[0] if path.is_relative_to(publicdata) else str(path.parent)
            source_counts[location] += 1
        input_rgb, reference_rgb = (load_rgb(paths[role], row) for role in ("input", "gt"))
        arrays = calculate_arrays(input_rgb, reference_rgb)
        values = verify_metrics(arrays, row)
        differences = {key: abs(value - float(row[key])) for key, value in values.items()}
        for key, diff in differences.items():
            stats[key] = max(stats[key], diff)
        if ident in hist_by_id:
            i = hist_by_id[ident]
            record = historical[i]
            require(float(row[record["selection_metric"]]) == record["metric_value"],
                    f"{dataset} {ident}: historic selection record and CSV disagree")
            asset = root / "figures/individual_samples" / f"{i + 1:02d}_{ident}" / "numeric_maps.npz"
            with np.load(asset, allow_pickle=False) as npz:
                arrays["patch_mean_delta"] = arrays["delta_ref"].reshape(
                    SIZE // PATCH, PATCH, SIZE // PATCH, PATCH, 3,
                ).mean(axis=(1, 3))
                for key, arr in arrays.items():
                    require(arr.dtype == npz[key].dtype and np.array_equal(arr, npz[key]),
                            f"{dataset} {ident}: stored array {key} differs")
                arrays = {key: npz[key] for key in arrays}
            check_published_panels(svg, i, display_arrays(arrays, palette))
            hist_metrics[ident] = {"selection_rule": record["selection_rule"], "percentiles": percentages[ident],
                                   "oracle_pixel_clipped_fraction": float(row["oracle_pixel_clipped_fraction"])}
            map_source = str(asset)
        else:
            map_source = "Reconstructed from the SHA-256 verified Input–GT pair"
        if ident in chosen_by_id:
            rec = chosen_by_id[ident]
            require(ident in selection["reviewed_ids_by_band"][dataset][rec["band"]],
                    f"{dataset} {ident}: no recorded visual review for {rec['band']}")
            sources = {role: str(path) for role, path in paths.items()}
            sources.update({f"{role}_sha256": info[f"{role}_sha256"] for role in paths})
            sources["array_source"] = map_source
            chosen_max[ident] = max(differences.values())
            candidates[ident] = Candidate(dataset, rec["band"], row, rec["reason"],
                                          percentages[ident], sources, display_arrays(arrays, palette))
    validation = {
        "verified_pairs": len(rows), "source_location_counts_files": dict(source_counts),
        "release_data_root": manifest["data_root"], "release_versions": manifest["versions"],
        "absolute_tolerance_strict": TOLERANCE, "max_absolute_difference_by_field": stats,
        "max_absolute_difference": max(stats.values()),
        "selected_max_absolute_difference_by_image": chosen_max,
        "historic_assets_exact_array_and_svg_pixel_matches": list(hist_metrics),
        "historical_metrics": hist_metrics, "eligible_count_by_band": eligible_counts,
    }
    return [candidates[rec["id"]] for rec in chosen], validation


def metric_values(candidate):
    row = candidate.row
    shift = math.hypot(*(float(row[k]) for k in RGB_KEYS))
    return (shift, float(row["eta_spatial"]), float(row["patch_variance_16"]),
            float(row["patch_spatial_retention_16"]), float(row["oracle_pixel_clipped_fraction"]))


def draw_metrics(candidate, pdf, left, y, *, inline=False):
    s, eta, v16, retained, clipped = metric_values(candidate)
    metrics = (f"||μ||₂ {s:.4f}", f"η {eta:.1%}", f"V16 {v16:.3e}",
               f"V16 / E_global {retained:.1%}", f"Oracle clip {clipped:.2%}")
    pdf.setFillColor(MUTED)
    pdf.setFont(REGULAR, 8)
    if inline:
        pdf.drawString(left, y, "     |     ".join(metrics))
    else:
        for i, metric in enumerate(metrics):
            pdf.drawString(left, y - i * 12, metric)


def draw_scale(pdf, palette, x, y, w, h, *, title_left=None):
    bar = Image.fromarray(palette.reshape(1, 256, 3))
    pdf.drawImage(ImageReader(bar), x, y, w, h)
    pdf.setStrokeColor(MUTED)
    pdf.setLineWidth(0.35)
    pdf.rect(x, y, w, h, stroke=1, fill=0)
    pdf.setFont(REGULAR, 8)
    pdf.setFillColor(INK)
    if title_left is not None:
        pdf.drawString(title_left, y + 0.7, "RGB L2 magnitude")
    for fraction, label in ((0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")):
        tick = x + w * fraction
        pdf.line(tick, y, tick, y - 2)
        pdf.drawCentredString(tick, y - 12, label)


def make_canvas(buffer, width, height, title):
    pdf = canvas.Canvas(buffer, pagesize=(width, height), pageCompression=1,
                        invariant=True, initialFontName=REGULAR)
    pdf.setTitle(title)
    pdf.setAuthor("")
    return pdf


def draw_panels(pdf, panels, x, y, size, gap):
    for i, panel in enumerate(panels):
        pdf.drawImage(ImageReader(Image.fromarray(panel)), x + i * (size + gap), y, size, size)


def contact_pdf(candidates, palette):
    label = DATASETS[candidates[0].dataset][0]
    margin, info_width, side, gap, stride = 20, 130, 122, 8, 137
    top, footer = 103, 69
    left = margin + info_width
    width = left + 4 * side + 3 * gap + margin
    height = top + 10 * stride + footer
    buf = BytesIO()
    pdf = make_canvas(buf, width, height, f"{label} | Global RGB shift oracle candidates")
    pdf.setFillColor(INK)
    pdf.setFont(BOLD, 22)
    pdf.drawString(margin, height - 39, f"{label} | Global RGB shift candidates")
    pdf.setFillColor(MUTED)
    pdf.setFont(REGULAR, 9.5)
    pdf.drawString(margin, height - 60, "A  Global shift    /    B  Spatial residual    /    C  Balanced")
    pdf.setFillColor(INK)
    pdf.setFont(BOLD, 9)
    pdf.drawString(margin, height - top + 15, "Sample / band")
    for i, title in enumerate(TITLES):
        center = left + i * (side + gap) + side / 2
        if i in (1, 2):
            lines = ("Global RGB", "Shift Oracle") if i == 1 else ("Remaining Correction", "Magnitude")
            for j, line in enumerate(lines):
                pdf.drawCentredString(center, height - top + 20 - j * 11, line)
        else:
            pdf.drawCentredString(center, height - top + 15, title)
    for index, candidate in enumerate(candidates):
        y = height - top - side - index * stride
        if candidate.band in ("B1", "C1"):
            pdf.setStrokeColor(RULE)
            pdf.setLineWidth(0.7)
            pdf.line(margin, y + side + 7.5, width - margin, y + side + 7.5)
        pdf.setFillColor(INK)
        pdf.setFont(BOLD, 12)
        pdf.drawString(margin, y + side - 16, candidate.sample_id)
        pdf.setFillColor(MUTED)
        pdf.setFont(BOLD, 8.5)
        pdf.drawString(margin, y + side - 32, f"Group {candidate.band[0]}  /  {candidate.band}")
        draw_metrics(candidate, pdf, margin, y + side - 54)
        draw_panels(pdf, candidate.panels, left, y, side, gap)
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(0.7)
    pdf.line(margin, footer + 5, width - margin, footer + 5)
    draw_scale(pdf, palette, margin + 111, 50, 180, 6, title_left=margin)
    pdf.setFont(REGULAR, 8.5)
    pdf.setFillColor(MUTED)
    pdf.drawString(margin, 19, "Oracle clip counts pixels with any out-of-range channel. Residual magnitude is computed before clipping.")
    pdf.showPage()
    pdf.save()
    return buf.getvalue(), (width, height)


def individual_pdf(candidate, palette):
    margin, side, gap = 14, 121, 6
    width, height = 2 * margin + 4 * side + 3 * gap, 229
    buf = BytesIO()
    title = f"{DATASETS[candidate.dataset][0]} / {candidate.sample_id}"
    pdf = make_canvas(buf, width, height, title)
    pdf.setFillColor(INK)
    pdf.setFont(BOLD, 16)
    pdf.drawString(margin, 204, title)
    pdf.setFont(REGULAR, 9)
    pdf.drawRightString(width - margin, 205, f"Group {candidate.band[0]} / {candidate.band}")
    for i, title in enumerate(TITLES):
        label = "Remaining magnitude" if i == 2 else title
        pdf.setFont(BOLD, 8.5)
        pdf.drawCentredString(margin + i * (side + gap) + side / 2, 178, label)
    draw_panels(pdf, candidate.panels, margin, 49, side, gap)
    draw_metrics(candidate, pdf, margin, 31, inline=True)
    draw_scale(pdf, palette, width - margin - 122, 17, 122, 4, title_left=width - margin - 211)
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


def column_preview(candidates, palette, pdftoppm, dpi):
    """Render every proposed three-panel crop at actual paper width in one review image."""
    title_height, stride, margin = 37, 91, 9
    width = 2 * (CVPR_COLUMN_WIDTH_PT + margin) + margin
    height = title_height + len(candidates["lsui"]) * stride + margin
    buf = BytesIO()
    pdf = make_canvas(buf, width, height, "Three-panel column-width proof")
    side, gap, x_pad = 67, 4, 1.5
    for index, (dataset, records) in enumerate(candidates.items()):
        gutter = margin + index * (CVPR_COLUMN_WIDTH_PT + margin)
        pdf.setFillColor(INK)
        pdf.setFont(BOLD, 11)
        pdf.drawString(gutter + x_pad, height - 19, DATASETS[dataset][0])
        for row, c in enumerate(records):
            y = height - title_height - (row + 1) * stride + 13
            pdf.setFillColor(INK)
            pdf.setFont(REGULAR, 8)
            pdf.drawString(gutter + x_pad, y + side + 3, f"{c.band} / {c.sample_id}")
            draw_panels(pdf, c.panels[:3], gutter + x_pad, y, side, gap)
            x = gutter + x_pad + 3 * side + 2 * gap + 4
            bar_img = Image.fromarray(palette[::-1].reshape(256, 1, 3))
            pdf.drawImage(ImageReader(bar_img), x, y + 5, 3.5, side - 10)
            pdf.setFont(REGULAR, 7)
            for text, step in (("0.0", 0), ("0.5", 0.5), ("1.0", 1)):
                pdf.drawString(x + 6, y + 3 + step * (side - 10), text)
    pdf.showPage()
    pdf.save()
    return render_png(buf.getvalue(), pdftoppm, dpi)


def render_png(contents, pdftoppm, dpi):
    process = subprocess.run([str(pdftoppm), "-r", str(dpi), "-png", "-singlefile", "-"],
                             input=contents, capture_output=True, check=True)
    require(process.stdout.startswith(b"\x89PNG\r\n\x1a\n"), "Poppler did not output a PNG")
    return process.stdout


def verify_sheet(contents, candidates, palette, page_size):
    reader = PdfReader(BytesIO(contents))
    require(len(reader.pages) == 1, "A contact sheet must contain one PDF page")
    page = reader.pages[0]
    require((float(page.mediabox.width), float(page.mediabox.height)) == page_size,
            "PDF page size differs from the layout")
    resources = page["/Resources"]["/XObject"]
    image_streams = [resources[operands[0]].get_object() for operands, operator in page.get_contents().operations
                     if operator == b"Do"]
    expected = [panel for candidate in candidates for panel in candidate.panels]
    expected.append(palette.reshape(1, 256, 3))
    require(len(image_streams) == 41, "Expected 40 panels and one shared colorbar per sheet")
    for obj, panel in zip(image_streams, expected, strict=True):
        require(obj["/Subtype"] == "/Image" and obj["/ColorSpace"] == "/DeviceRGB"
                and int(obj["/Width"]) == panel.shape[1] and int(obj["/Height"]) == panel.shape[0]
                and obj.get_data() == panel.tobytes(), "PDF embedded a different raster from its verified source")
    text = page.extract_text().splitlines()
    require(all(c.sample_id in text and f"Group {c.band[0]}  /  {c.band}" in text for c in candidates)
            and "RGB L2 magnitude" in text and "Reference" in text, "Vector labels are missing from the sheet")


def write_ranking(records, output):
    fieldnames = (
        "Dataset", "Image ID", "Selection Group", "Selection Band", "Selection Reason",
        "RGB Shift R", "RGB Shift G", "RGB Shift B", "RGB Shift L2",
        "Raw Error", "Global Oracle Error", "Remaining Energy Ratio", "V16",
        "Retained Spatial Energy Ratio", "Oracle Clipping Fraction", "Oracle Scalar Clipping Fraction",
        "Magnitude Above Display Max Fraction",
        *(f"{label} Percentile" for label in PERCENTILE_KEYS.values()),
        "Input Path", "GT Path", "Release Input Path", "Release GT Path",
        "Input SHA-256", "GT SHA-256", "Visualization Array Source",
        "Candidate PNG Path", "Contact Sheet PNG Path", "Contact Sheet PDF Path",
    )
    with (output / "motivation_candidate_ranking.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, candidates in records.items():
            stem = f"{dataset.upper()}_10_candidates"
            for c in candidates:
                r = c.row
                writer.writerow({
                    "Dataset": DATASETS[dataset][0], "Image ID": c.sample_id, "Selection Group": c.band[0],
                    "Selection Band": c.band, "Selection Reason": c.reason,
                    **{f"RGB Shift {ch}": r["mu_" + ch] for ch in "RGB"},
                    "RGB Shift L2": repr(metric_values(c)[0]),
                    "Raw Error": r["e_raw"], "Global Oracle Error": r["e_global"],
                    "Remaining Energy Ratio": r["eta_spatial"], "V16": r["patch_variance_16"],
                    "Retained Spatial Energy Ratio": r["patch_spatial_retention_16"],
                    "Oracle Clipping Fraction": r["oracle_pixel_clipped_fraction"],
                    "Oracle Scalar Clipping Fraction": r["oracle_scalar_clipped_fraction"],
                    "Magnitude Above Display Max Fraction": r["display_remaining_clipped_fraction"],
                    **{f"{label} Percentile": repr(c.percentiles[key]) for key, label in PERCENTILE_KEYS.items()},
                    "Input Path": c.sources["input"], "GT Path": c.sources["gt"],
                    "Release Input Path": r["input_path"], "Release GT Path": r["gt_path"],
                    "Input SHA-256": c.sources["input_sha256"], "GT SHA-256": c.sources["gt_sha256"],
                    "Visualization Array Source": c.sources["array_source"],
                    "Candidate PNG Path": str(output / c.image_path),
                    "Contact Sheet PNG Path": str(output / f"{stem}.png"),
                    "Contact Sheet PDF Path": str(output / f"{stem}.pdf"),
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "publicdata", "matplotlib-colormap", "font-dir", "pdftoppm", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path(__file__).with_name("candidate_selection.json"))
    args = parser.parse_args()
    project, publicdata, output = (path.resolve() for path in (args.project, args.publicdata, args.output))
    release = project / RELEASE
    require(not output.is_relative_to(release), "Use a new output directory outside the audited release")
    inventory, inventory_count = validate_inventory(release, project)
    inventory_hash_before = digest(inventory)
    selection = read_json(args.selection)
    palette = published_magma(args.matplotlib_colormap)
    regular = args.font_dir / "DejaVuSans.ttf"
    bold = args.font_dir / "DejaVuSans-Bold.ttf"
    pdfmetrics.registerFont(TTFont(REGULAR, str(regular)))
    pdfmetrics.registerFont(TTFont(BOLD, str(bold)))
    datasets, issues = {}, {}
    for key in DATASETS:
        datasets[key], issues[key] = verify_dataset(release / key, key, publicdata, palette, selection)
        print(key, issues[key]["verified_pairs"], "reproduced pairs,", len(datasets[key]), "selected",
              flush=True)

    output.mkdir(parents=True, exist_ok=True)
    individual = output / "individual_samples"
    individual.mkdir(exist_ok=False)
    for dataset, candidates in datasets.items():
        (individual / dataset).mkdir()
        stem = f"{dataset.upper()}_10_candidates"
        pdf, media = contact_pdf(candidates, palette)
        verify_sheet(pdf, candidates, palette, media)
        with (output / f"{stem}.pdf").open("xb") as handle:
            handle.write(pdf)
        png = render_png(pdf, args.pdftoppm, 300)
        with (output / f"{stem}.png").open("xb") as handle:
            handle.write(png)
        with Image.open(BytesIO(png)) as image:
            issues[dataset]["sheet_png_dimensions_px"] = list(image.size)
        issues[dataset]["sheet_pdf_dimensions_pt"] = list(media)
        for record in candidates:
            png = render_png(individual_pdf(record, palette), args.pdftoppm, 300)
            with (output / record.image_path).open("xb") as handle:
                handle.write(png)
    write_ranking(datasets, output)
    with (output / "single_column_proof.png").open("xb") as handle:
        handle.write(column_preview(datasets, palette, args.pdftoppm, 96))
    require(digest(inventory) == inventory_hash_before, "Release inventory changed during the run")
    validate_inventory(release, project)
    poppler = subprocess.run([str(args.pdftoppm), "-v"], capture_output=True, text=True, check=True)
    write_json(output / "reproduction_parameters.json", {
        "release": str(release), "release_inventory_sha256": inventory_hash_before,
        "release_artifacts_verified_before_and_after": inventory_count,
        "project": str(project), "publicdata": str(publicdata), "output": str(output),
        "selection": str(args.selection.resolve()), "selection_sha256": digest(args.selection),
        "script_sha256": digest(Path(__file__)),
        "command_argv": [sys.executable, "-B", *sys.argv],
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    **{name: version(name) for name in ("numpy", "Pillow", "reportlab", "pypdf")},
                    "pdftoppm": str(args.pdftoppm), "poppler": poppler.stderr.splitlines()[0]},
        "rendering": {"magma_table": str(args.matplotlib_colormap.resolve()),
                      "magma_table_sha256": digest(args.matplotlib_colormap),
                      "magma_rgb_lut_sha256": sha256(palette.tobytes()).hexdigest(),
                      "color_scale": [0, 1], "linear": True, "png_dpi": 300,
                      "font_dir": str(args.font_dir.resolve()),
                      "fonts_sha256": {path.name: digest(path) for path in (regular, bold)},
                      "proof_column_width_pt": CVPR_COLUMN_WIDTH_PT, "proof_display_dpi": 96},
        "datasets": issues,
    })


if __name__ == "__main__":
    main()
