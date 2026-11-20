"""Inference from the existing PUGAN, UVZ, IMIF and HDAMS-Net FX adapters."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from PIL import Image

from .inference import enhance
from .preprocessing import rgb_png


def _load_pugan(checkpoint, source, device, auxiliary):
    import torch
    from adapters import data, runtime

    state = runtime.checkpoint_details(checkpoint, source, torch.device("cpu"))
    if state.get("synthetic") or state.get("validation_psnr") is None:
        raise ValueError("PUGAN requires a real validation-selected checkpoint")
    par_paths = {name: auxiliary[f"par_{name}"] for name in ("d", "t", "a")}
    if state.get("par_sha256") != {name: runtime.sha256(path) for name, path in par_paths.items()}:
        raise ValueError("PUGAN Par weights differ from the validated training run")
    generator, discriminator, depth_discriminator, par = runtime.build_parts(device, par_paths)
    del discriminator, depth_discriminator
    generator.load_state_dict(state["generator"], strict=True)
    generator.eval().requires_grad_(False)

    def predict(image):
        resized = image.resize((256, 256), Image.Resampling.BICUBIC)
        value = data.tensor(resized).unsqueeze(0).to(device)
        result = runtime.enhance(value, generator, par)[0]
        if tuple(result.shape) != (3, 256, 256):
            raise ValueError("PUGAN did not produce the released internal RGB dimensions")
        return Image.fromarray(data.png8(result)).resize(image.size, Image.Resampling.BICUBIC)

    return predict


def _load_uvz(checkpoint, source, device, auxiliary):
    import torch
    from adapters import infer_u45 as uvz

    state = uvz.load_checkpoint(checkpoint, torch.device("cpu"))
    details = uvz.checkpoint_provenance(state)
    domain = "lsui19" if source == "lsui" else "uieb"
    if (details["checkpoint_declared_training_dataset"] != domain
            or details["checkpoint_role"] != "validation_selected_best_psnr"
            or details["best_validation_psnr"] is None
            or state.get("synthetic") or details["settings"].get("synthetic")):
        raise ValueError("UVZ requires a real validation-selected checkpoint from the configured domain")
    depth_path = auxiliary["depth"]
    if uvz.assert_official_depth_checkpoint(depth_path) != details["depth_checkpoint_sha256"]:
        raise ValueError("UVZ depth checkpoint differs from the one used during training")
    depth = uvz.load_official_depth_model(depth_path, device)
    model, _ = uvz.load_enhancement_model(state, device)
    model.eval().requires_grad_(False)

    def predict(image):
        raw = np.asarray(image, dtype=np.uint8)
        value = uvz._to_internal_256(raw, device)
        uvz.ensure_256(value)
        result = model(value, uvz.estimate_depth_per_sample(depth, value))
        if tuple(result.shape) != (1, 3, 256, 256):
            raise ValueError("UVZ did not produce the released internal RGB dimensions")
        restored = uvz._restore_native(result, image.height, image.width)
        return Image.fromarray(uvz.png8(restored[0]))

    return predict


def _load_hdams(checkpoint, source, device):
    from adapters import data, runtime

    device = runtime.choose_device(str(device))
    model = runtime.build_model(device)
    runtime.load_checkpoint(checkpoint, model, dataset=source)
    model.eval().requires_grad_(False)

    def predict(image):
        value = data.tensor(image).unsqueeze(0).to(device)
        return rgb_png(runtime.native_forward(model, value)[0].cpu())

    return predict


def load_fx_model(method, checkpoint, source, device, auxiliary):
    if method == "pugan":
        return _load_pugan(checkpoint, source, device, auxiliary)
    if method == "uvz":
        return _load_uvz(checkpoint, source, device, auxiliary)
    if method == "hdams_net":
        return _load_hdams(checkpoint, source, device)
    raise ValueError(f"{method} does not have a learned FX model")


def _predict_imif(record, config):
    from adapters.imif import enhance as imif_enhance
    from adapters.infer import png8
    from adapters.protocol import sample_seed

    sample_id, image = record
    values, _ = imif_enhance(
        np.asarray(image, dtype=np.uint8), seed=sample_seed(3520, sample_id), config=config,
    )
    return Image.fromarray(png8(values))


def run_imif(stage, data_root, source, auxiliary, workers):
    from adapters.runtime import load_method_config, load_selected_state

    load_selected_state(auxiliary["selection"], source)
    config, _ = load_method_config()
    predict = partial(_predict_imif, config=config)
    workers = min(workers, len(stage.samples))
    if workers == 1:
        enhance(stage, lambda records: list(map(predict, records)), data_root, 1, with_sample_ids=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def predict_batch(records):
            return list(pool.map(predict, records))

        enhance(stage, predict_batch, data_root, workers, with_sample_ids=True)
