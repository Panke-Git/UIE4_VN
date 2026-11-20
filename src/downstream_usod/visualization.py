"""Select a fixed set of readable, image-paired saliency examples."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .dataset import read_input
from .storage import atomic_file, write_csv


def select_examples(raw, icar, per_group):
    if raw.ids != icar.ids:
        raise ValueError("example selection needs paired test IDs")
    gains = raw.values[:, -1] - icar.values[:, -1]
    median = np.median(gains)
    order = (
        ("largest_mae_gain", sorted(range(len(gains)), key=lambda i: (-gains[i], raw.ids[i]))),
        ("near_median_mae_gain", sorted(range(len(gains)),
                                      key=lambda i: (abs(gains[i] - median), raw.ids[i]))),
        ("smallest_mae_gain", sorted(range(len(gains)), key=lambda i: (gains[i], raw.ids[i]))),
    )
    chosen, rows = set(), []
    for group, indexes in order:
        distinct = [i for i in indexes if i not in chosen][:per_group]
        chosen.update(distinct)
        rows.extend({"sample_id": raw.ids[i], "group": group, "mae_raw_minus_icar": float(gains[i])}
                    for i in distinct)
    return rows


def _panel(image, size, is_mask):
    image = image.convert("RGB")
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return ImageOps.pad(image, size, method=resample, color="#edf0f2")


def make_figure(samples, root, methods, predictions, destination, per_group):
    selection = select_examples(methods["raw"], methods["icar_uie"], per_group)
    by_id = {sample.sample_id: sample for sample in samples}
    selected_methods = list(methods)
    labels = {
        "raw": "Raw prediction", "icar_uie": "ICAR-UIE", "icar_uie_uieb": "ICAR-UIE (UIEB)",
        "waternet": "WaterNet", "ucolor": "Ucolor", "fdce_net": "FDCE-Net",
        "marine_net": "MARINE-Net", "swlgm": "SWLGM",
        "pugan": "PUGAN", "uvz": "UVZ", "imif": "IMIF", "hdams_net": "HDAMS-Net",
    }
    group_names = {
        "largest_mae_gain": "Largest MAE gains",
        "near_median_mae_gain": "Near median MAE gain",
        "smallest_mae_gain": "Smallest MAE gains",
    }
    panel = (224, 154)
    gap, margin, header, row_stride = 8, 18, 55, 187
    columns = ["Input", "GT mask", *[labels[name] for name in selected_methods]]
    width = 2 * margin + len(columns) * panel[0] + (len(columns) - 1) * gap
    canvas = Image.new("RGB", (width, header + row_stride * len(selection) + margin), "white")
    draw = ImageDraw.Draw(canvas)
    bold = ImageFont.load_default(size=19)
    detail = ImageFont.load_default(size=15)
    for i, label in enumerate(columns):
        draw.text((margin + i * (panel[0] + gap) + panel[0] // 2, 18),
                  label, font=bold, fill="#15232b", anchor="mt")
    for i, row in enumerate(selection):
        y = header + i * row_stride
        sample = by_id[row["sample_id"]]
        images = [read_input(root, sample, "rgb"), read_input(root, sample, "gt")]
        images.extend(predictions[name].load(sample) for name in selected_methods)
        draw.text((margin, y), f"{group_names[row['group']]} · ID {sample.sample_id} · "
                  f"MAE gain {row['mae_raw_minus_icar']:+.4f}", font=detail, fill="#213a48")
        for j, image in enumerate(images):
            canvas.paste(_panel(image, panel, j == 1), (margin + j * (panel[0] + gap), y + 23))
    write_csv(destination / "selected_samples.csv", selection)
    with atomic_file(destination / "saliency_examples.png") as stream:
        canvas.save(stream, format="PNG")
    return selection
