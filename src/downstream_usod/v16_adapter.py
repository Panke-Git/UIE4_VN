"""Load and run the frozen formal V16 model without modifying the backbone."""

import torch

from src.downstream_duo.model_inference_v16 import pad_tensor, starts
from src.v16.models.network import build_model

from .config import V16_CHECKPOINTS, read_yaml
from .preprocessing import rgb_png, rgb_tensor
from .storage import sha256_file, verify_hash


def v16_identity(config, checkpoint, source):
    expected, epoch = V16_CHECKPOINTS[source]
    return {"source": source, "epoch": epoch, "config_sha256": sha256_file(config),
            "checkpoint_sha256": verify_hash(checkpoint, expected)}


def load_v16(config, checkpoint, source, device, identity=None):
    identity = identity or v16_identity(config, checkpoint, source)
    if identity["source"] != source or (
        identity["checkpoint_sha256"], identity["epoch"]
    ) != V16_CHECKPOINTS[source]:
        raise ValueError("V16 identity does not match the formal source checkpoint")
    verify_hash(config, identity["config_sha256"])
    verify_hash(checkpoint, identity["checkpoint_sha256"])
    settings = read_yaml(config)
    if settings.get("experiment", {}).get("version") != "v16":
        raise ValueError("expected a resolved V16 run config")
    model_settings = settings["model"]
    field = model_settings.get("uicf", {})
    if (
        model_settings.get("type") != "plain_unet_color_query_uicf_pre_backbone"
        or model_settings.get("output_activation") != "sigmoid"
        or field.get("field_variant", "implicit") not in ("implicit", "implicit_inr")
        or any(field.get(name, True) is not True for name in (
            "use_spatial_conditioning", "use_global_field_conditioning", "use_learned_anchor"))
    ):
        raise ValueError("expected the canonical full V16 model")
    # These two fixed, project-owned training checkpoint hashes are checked above.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resolved = state.get("resolved_config", {})
    if (state.get("version") != "v16" or state.get("epoch") != identity["epoch"]
            or resolved.get("model") != model_settings):
        raise ValueError("checkpoint and resolved V16 config disagree on architecture or epoch")
    model = build_model(model_settings)
    model.load_state_dict(state["model_state_dict"], strict=True)
    if model.backbone.padding_factor != 16:
        raise ValueError("V16 padding factor changed")
    return model.to(device).eval().requires_grad_(False)


class V16Enhancer:
    def __init__(self, model, device, tile_size, overlap):
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("enhancer must have frozen weights and be in eval mode")

    def _forward(self, tensor):
        height, width = tensor.shape[-2:]
        prediction = self.model(pad_tensor(tensor.to(self.device)))[..., :height, :width].float().cpu()
        if (prediction.shape != tensor.shape or not torch.isfinite(prediction).all()
                or prediction.min() < -1e-6 or prediction.max() > 1 + 1e-6):
            raise ValueError("V16 produced invalid sigmoid RGB")
        return prediction.clamp_(0, 1)

    @torch.inference_mode()
    def __call__(self, images):
        if len({image.size for image in images}) != 1:
            raise ValueError("same-sized native inputs are required within an enhancement batch")
        tensor = torch.stack([rgb_tensor(image) for image in images])
        height, width = tensor.shape[-2:]
        size = self.tile_size
        if not size or max(height, width) <= size:
            return [rgb_png(item) for item in self._forward(tensor)]

        result = torch.zeros_like(tensor)
        denominator = torch.zeros((height, width))
        for y in starts(height, size, self.overlap):
            for x in starts(width, size, self.overlap):
                crop = tensor[..., y:y + size, x:x + size]
                rows, columns = crop.shape[-2:]
                wy = torch.hann_window(rows, periodic=False).clamp_min(0.01)
                wx = torch.hann_window(columns, periodic=False).clamp_min(0.01)
                window = wy[:, None] * wx[None, :]
                result[..., y:y + rows, x:x + columns] += self._forward(crop) * window
                denominator[y:y + rows, x:x + columns] += window
        if (denominator <= 0).any():
            raise RuntimeError("V16 tile stitching left uncovered pixels")
        return [rgb_png(item) for item in result.div_(denominator)]
