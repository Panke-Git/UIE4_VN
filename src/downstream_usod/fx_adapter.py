"""Input-only FX adapters using their existing model and preprocessing code."""

from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType

from .config import ADDITIONAL_FX_METHODS, FX_METHODS, PROJECT, mapping, sha_value
from .dataset import Sample
from .inference import ImageStage, enhance
from .storage import assert_separate_output, digest, read_json, run_lock, sha256_file, verify_hash

PAPER_DIRECTORIES = {
    "waternet": "An underwater image  enhancement benchmark dataset and beyond",
    "ucolor": "Underwater image enhancement via medium transmission-guided multi-color space embedding",
    "fdce_net": "FDCE-Net underwater image enhancement with embedding frequency and dual color encoder",
    "marine_net": "MARINE-Net Physics-guided lightweight CNN-INR architecture for underwater image restoration and color enhancement",
    "swlgm": "Shift-window local and global state space model for underwater image enhancement",
    "pugan": "Physical Model-Guided Underwater Image Enhancement Using GAN with Dual-Discriminators",
    "uvz": "Underwater variable zoom Depthguided perception network for underwater image enhancement",
    "imif": "Underwater Image Enhancement via Intelligent  Optimized Multi-Exposure Image Fusion",
    "hdams_net": "Hybrid-Domain Attention Learning for Lightweight Underwater Image Enhancement and Real-Time Vision Deployment",
}

FX_AUXILIARY_FILES = {
    "pugan": {"par_d": "Dnet.pth", "par_t": "Tnet.pth", "par_a": "coe.pth"},
    "uvz": {"depth": "model_best.pth"},
    "imif": {"selection": "validation_selected_state.json", "algorithm_state": "algorithm_state.json"},
}

SPATIAL_PROTOCOLS = {
    "pugan": "FX 256x256 bicubic input and bicubic restore after RGB PNG8 quantization",
    "uvz": "FX 256x256 bilinear RGB input, internal frozen depth model and native float bilinear restore",
    "imif": "Native RGB dimensions with fixed non-learning configuration and per-image seed",
    "hdams_net": "Native RGB dimensions with right/bottom factor-four padding and exact crop",
}


@dataclass(frozen=True)
class FXAsset:
    method: str
    root: Path
    paper: Path
    python: Path
    checkpoint: Path | None
    sha256: str | None
    source: str
    auxiliary: dict = field(default_factory=dict)
    workers: int = 1

    @classmethod
    def load(cls, paths, method, override=None):
        if method not in FX_METHODS:
            raise ValueError(f"unknown FX method {method}")
        configured = paths.section("fx")
        if method not in configured:
            raise FileNotFoundError(f"fx.{method} has no configured inference assets")
        item = mapping(configured[method], f"fx.{method}")
        root = Path(override).expanduser().resolve() if override else paths.path(None, "fx_root")
        paper = root / PAPER_DIRECTORIES[method]
        if not (paper / "adapters").is_dir() or not (root / "_shared").is_dir():
            raise FileNotFoundError(f"need the paper directory and FX/_shared under {root}")
        python = item.get("python")
        if not isinstance(python, str) or not python:
            raise ValueError(f"configure fx.{method}.python")
        # Preserve venv/bin/python as the invocation path, including its final symlink.
        python = (paths.file.parent / Path(python).expanduser()).absolute()
        if not os.access(python, os.X_OK) or not python.is_file():
            raise FileNotFoundError(f"FX model interpreter is unavailable: {python}")
        source = item.get("source")
        if source not in ("lsui", "uieb"):
            raise ValueError(f"fx.{method}.source must be lsui or uieb")
        assets = mapping(item.get("auxiliary", {}), f"fx.{method}.auxiliary")
        auxiliary = {}
        required = FX_AUXILIARY_FILES.get(method, {})
        if set(assets) != set(required):
            raise ValueError(f"fx.{method}.auxiliary requires keys {sorted(required)}")
        for name, basename in required.items():
            key = f"fx.{method}.auxiliary.{name}"
            spec = mapping(assets[name], key)
            path = paths.resolve(spec.get("path"), f"{key}.path")
            checksum = sha_value(spec.get("sha256"), f"{key}.sha256")
            if path.name != basename:
                raise ValueError(f"{key} requires {basename}")
            verify_hash(path, checksum)
            auxiliary[name] = {"path": str(path), "sha256": checksum}
        if method == "imif":
            if item.get("checkpoint") is not None or item.get("sha256") is not None:
                raise ValueError("IMIF requires validated algorithm state files and no learned checkpoint")
            selection = Path(auxiliary["selection"]["path"])
            algorithm = Path(auxiliary["algorithm_state"]["path"])
            if algorithm != selection.parent.parent / "algorithm_state.json":
                raise ValueError(f"IMIF algorithm state must be adjacent to the selection folder: {selection}")
            workers = item.get("workers", 1)
            if type(workers) is not int or workers < 1:
                raise ValueError("fx.imif.workers must be a positive integer")
            return cls(method, root, paper, python, None, None, source, auxiliary, workers)
        checkpoint = paths.resolve(item.get("checkpoint"), f"fx.{method}.checkpoint")
        sha = sha_value(item.get("sha256"), f"fx.{method}.sha256")
        selection_names = {
            "waternet": "best_psnr.pt", "ucolor": "best_psnr.pt", "fdce_net": "best_ssim.pt",
            "marine_net": "best_model.pth", "swlgm": "best_psnr.pt", "pugan": "best_psnr.pt",
            "uvz": "best_psnr.pt", "hdams_net": "best_psnr.pt",
        }
        if checkpoint.name != selection_names[method]:
            raise ValueError(f"fx.{method} requires the validation-selected {selection_names[method]}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FX validation-selected checkpoint is missing: {checkpoint}")
        verify_hash(checkpoint, sha)
        return cls(method, root, paper, python, checkpoint, sha, source, auxiliary)

    def provenance(self):
        if self.method == "imif":
            record = {"method_type": "non_learning", "validation_source": self.source}
        else:
            record = {"checkpoint_sha256": self.sha256, "training_source": self.source}
        if self.auxiliary:
            record["auxiliary_sha256"] = {name: spec["sha256"] for name, spec in self.auxiliary.items()}
        if self.method in SPATIAL_PROTOCOLS:
            record["spatial_protocol"] = SPATIAL_PROTOCOLS[self.method]
        return record


def _code_digest(asset):
    skip = {".git", "__pycache__", "outputs", "output", "result", "results", "validation",
            "data", "datasets", "checkpoints", ".venv", ".venv_eval"}
    files = {}
    for tree in (asset.paper, asset.root / "_shared"):
        for current, dirs, names in os.walk(tree):
            dirs[:] = [name for name in dirs if not name.startswith(".") and name not in skip]
            for name in names:
                path = Path(current) / name
                if path.suffix in (".py", ".yaml", ".yml"):
                    files[path.relative_to(asset.root).as_posix()] = sha256_file(path)
    return digest(files)


def run_fx(asset, run_root, data_root, samples, signature, device):
    protected = [asset.root, *(spec["path"] for spec in asset.auxiliary.values())]
    if asset.checkpoint is not None:
        protected.append(asset.checkpoint)
    assert_separate_output(run_root, protected)
    log = Path(run_root) / "logs" / f"fx_{asset.method}_worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    package = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(asset).items()}
    job = {"asset": package, "run": str(run_root), "data": str(data_root),
           "samples": [asdict(sample) for sample in samples], "signature": signature,
           "code_sha256": _code_digest(asset), "device": device, "provenance": asset.provenance()}
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run([str(asset.python), "-B", "-m", "src.downstream_usod.fx_adapter"],
                                input=json.dumps(job), cwd=PROJECT, text=True,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{asset.method} adapter exited {result.returncode}; see {log}")
    identity_path = Path(run_root) / "audit/stages/enhanced" / asset.method / "identity.json"
    identity = read_json(identity_path)
    stage = ImageStage(run_root, "enhanced", asset.method, samples, identity, "RGB")
    stage.complete()
    return stage


def _frozen(model, state, device):
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def _source_checkpoint(details, source, score_key):
    if details["checkpoint_declared_training_dataset"] != source:
        raise ValueError(f"FX checkpoint does not document the configured training domain {source}")
    if score_key and details.get(score_key) is None:
        raise ValueError(f"FX checkpoint lacks a recorded source-validation score {score_key}")


def _load_method(method, checkpoint_path, source, device, auxiliary=None):
    import numpy as np
    from PIL import Image
    import torch

    from .preprocessing import rgb_tensor

    if method in ADDITIONAL_FX_METHODS:
        from .fx_additional import load_fx_model

        return load_fx_model(method, checkpoint_path, source, device, auxiliary)

    if method == "marine_net":
        from adapters import inference as marine
        dataset = "lsui19" if source == "lsui" else "uieb"
        split = "lsui19" if source == "lsui" else "uieb"
        _, splits = marine.audit(dataset, PROJECT / "split" / split)
        checkpoint = marine.selected_checkpoint(checkpoint_path, dataset, splits)
        model = _frozen(marine.create_model(), checkpoint["model_state_dict"], device)

        def predict(image):
            value = marine.predict(model, rgb_tensor(image), device)
            return Image.fromarray((value.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        return predict

    from adapters.runtime import load_checkpoint
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    if checkpoint.get("synthetic") or checkpoint.get("smoke_test"):
        raise ValueError("synthetic FX checkpoints are ineligible")

    if method == "waternet":
        from adapters import infer_u45 as water
        _source_checkpoint(water._checkpoint_details(checkpoint), source, "best_validation_psnr")
        model = _frozen(water.WaterNet(), checkpoint["model"], device)

        def predict(image):
            rgb = np.asarray(image, dtype=np.uint8).copy()
            views = (rgb, *water.make_waternet_inputs(rgb))
            tensors = [water.to_tensor(view).unsqueeze(0).to(device) for view in views]
            return Image.fromarray(water._png8(model(*tensors)[0]))

    elif method == "ucolor":
        from adapters import infer_u45 as ucolor
        details = ucolor._checkpoint_details(checkpoint)
        _source_checkpoint(details, source, "best_validation_psnr")
        if tuple(details["widths"]) != ucolor.PRODUCTION_WIDTHS or details["attention_ratio"] != 16:
            raise ValueError("Ucolor checkpoint must use the released widths and attention ratio")
        model = _frozen(ucolor.Ucolor(tuple(details["widths"]), 16), checkpoint["model"], device)

        def predict(image):
            raw = np.asarray(image, dtype=np.uint8).copy()
            transmission = ucolor.transmission_png8(raw, window=details["transmission_window"],
                                                    minimum_transmission=details["minimum_transmission"])
            rgb, trans = ucolor._tensor_inputs(raw, transmission)
            rgb, trans, _ = ucolor._pad_for_model(rgb, trans)
            width, height = image.size
            value = model(rgb.to(device), trans.to(device))[0, :, :height, :width]
            return Image.fromarray(ucolor._png8(value))

    elif method == "fdce_net":
        from adapters import infer_u45 as fdce
        details = fdce.checkpoint_details(checkpoint)
        _source_checkpoint(details, source, "best_validation_ssim")
        if details["checkpoint_role"] != "validation_selected_best_ssim":
            raise ValueError("FDCE-Net requires the source validation SSIM-selected checkpoint")
        model = _frozen(fdce.FDCENet(), checkpoint["model"], device)
        if fdce.parameter_count(model) != fdce.EXPECTED_PARAMETER_COUNT:
            raise ValueError("FDCE-Net release parameter count changed")

        def predict(image):
            _, _, value = fdce.forward_padded(model, rgb_tensor(image).unsqueeze(0).to(device))
            return Image.fromarray(fdce.png8(value[0]))

    elif method == "swlgm":
        from adapters import u45 as swlgm
        details = swlgm.checkpoint_details(checkpoint)
        _source_checkpoint(details, source, "best_validation_psnr")
        if (details["checkpoint_role"] != "validation_selected_best_psnr"
                or details["checkpoint_selector"] != "validation_psnr"):
            raise ValueError("SWLGM requires the source validation PSNR-selected checkpoint")
        model = _frozen(swlgm.build_model(details["model"]), swlgm.model_state(checkpoint), device)

        def predict(image):
            value = swlgm.predict(model, rgb_tensor(image).unsqueeze(0).to(device), native_size=True)
            return Image.fromarray(swlgm.png8(value[0]))
    else:
        raise ValueError(f"unrecognized FX method {method}")
    return predict


def _worker(job):
    import torch

    asset = job["asset"]
    paper = Path(asset["paper"])
    fx = ModuleType("FX")
    fx.__path__ = [asset["root"]]
    sys.modules["FX"] = fx
    os.chdir(paper)
    sys.path.insert(0, str(paper))
    checkpoint = Path(asset["checkpoint"]) if asset["checkpoint"] else None
    if checkpoint is not None:
        verify_hash(checkpoint, asset["sha256"])
    auxiliary = {name: Path(spec["path"]) for name, spec in asset["auxiliary"].items()}
    for name, path in auxiliary.items():
        verify_hash(path, asset["auxiliary"][name]["sha256"])
    method = asset["method"]
    device = torch.device("cpu" if method == "imif" else job["device"])
    packages = {}
    names = ("torchvision", "numpy", "Pillow", "opencv-python", "opencv-python-headless", "mamba-ssm")
    if method in ADDITIONAL_FX_METHODS:
        names += ("scipy", "scikit-image", "einops", "timm", "PyYAML")
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    identity = {"signature": job["signature"], **job["provenance"],
                "fx_source_sha256": job["code_sha256"],
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "cuda": torch.version.cuda, "device": str(device), "packages": packages}}
    samples = [Sample(**item) for item in job["samples"]]
    root = Path(job["run"])
    # Keep this stage locked even if its parent command exits before the worker.
    with run_lock(root / "audit/stages/enhanced" / method):
        stage = ImageStage(root, "enhanced", method, samples, identity, "RGB")
        if (stage.audit / "complete.json").is_file():
            stage.complete()
            return
        if method == "imif":
            from .fx_additional import run_imif

            run_imif(stage, Path(job["data"]), asset["source"], auxiliary, asset["workers"])
            return
        model = _load_method(method, checkpoint, asset["source"], device, auxiliary)

        @torch.inference_mode()
        def predict(images):
            return [model(image) for image in images]

        enhance(stage, predict, Path(job["data"]), batch_size=1)


if __name__ == "__main__":
    _worker(json.load(sys.stdin))
