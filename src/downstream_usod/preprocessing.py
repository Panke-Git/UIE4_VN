"""Image conversions shared by Raw and every enhanced condition."""

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as tv


def rgb_tensor(image):
    return tv.to_tensor(image.convert("RGB"))


def tc_input(image, side):
    resized = image.convert("RGB").resize((side, side), Image.Resampling.BILINEAR)
    return tv.normalize(tv.to_tensor(resized), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def tc_output(probability, size):
    """Match Testing.py: d1 is already sigmoid, convert to PNG8 before resizing."""
    if (probability.ndim != 3 or probability.shape[0] != 1
            or not torch.isfinite(probability).all()
            or probability.min() < 0 or probability.max() > 1):
        raise ValueError("TC-USOD d1 must be a finite single-channel probability map")
    encoded = tv.to_pil_image(probability.detach().float().cpu())
    return encoded.resize(size, Image.Resampling.BILINEAR)


def rgb_png(tensor):
    if tensor.ndim != 3 or tensor.shape[0] != 3 or not torch.isfinite(tensor).all():
        raise ValueError("expected a finite CHW RGB output")
    rgb = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
