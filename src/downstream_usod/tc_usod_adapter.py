"""Use the authors' model from an independently checked out Git repository."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from .config import TC_COMMIT, TC_INPUT_SIZE
from .dataset import git
from .preprocessing import tc_input, tc_output
from .storage import verify_hash


def verify_model_source(repo):
    repo = Path(repo).resolve()
    if git(repo, "rev-parse", "HEAD") != TC_COMMIT:
        raise ValueError(f"TC-USOD source must be pinned to {TC_COMMIT}: {repo}")
    model_dir = repo / "TC_USOD/Models"
    expected = {}
    tree = git(repo, "ls-tree", "-r", TC_COMMIT, "--", "TC_USOD/Models")
    for row in tree.splitlines():
        metadata, relative = row.split("\t", 1)
        if Path(relative).suffix == ".py":
            expected[relative] = metadata.split()[2]
    actual = {path.relative_to(repo).as_posix() for path in model_dir.rglob("*.py")}
    if not expected or actual != expected.keys():
        raise ValueError(f"TC-USOD model files differ from the pinned source at {model_dir}")
    for relative, blob in expected.items():
        path = repo / relative
        if path.is_symlink() or git(repo, "hash-object", "--no-filters", "--", str(path)) != blob:
            raise ValueError(f"TC-USOD source differs from the pinned Git blob: {relative}")
    return model_dir


def build_architecture(repo):
    directory = verify_model_source(repo)
    name = "_uie4_tc_usod_official"
    package = sys.modules.get(name)
    if package is None:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
    elif package.__path__ != [str(directory)]:
        raise ValueError("cannot mix two TC-USOD source checkouts in one interpreter")
    network = importlib.import_module(name + ".USOD_Net")
    decoder = importlib.import_module(name + ".Decoder_Dconv")
    backbone_factory = network.T2t_vit_t_14

    def uninitialized_backbone(*, pretrained, **kwargs):
        return backbone_factory(pretrained=False, **kwargs)

    def uninitialized_resnet(*, pretrained):
        return decoder_models.resnet34(weights=None)

    decoder_models = decoder.models
    # The constructor's ImageNet initializers are redundant with the full strict load.
    # Keep the official graph and avoid backbone downloads or auxiliary weight files.
    with patch.object(network, "T2t_vit_t_14", uninitialized_backbone), patch.object(
        decoder, "models", SimpleNamespace(resnet34=uninitialized_resnet)
    ):
        return network.ImageDepthNet(SimpleNamespace(img_size=TC_INPUT_SIZE))


def load_tc(repo, checkpoint, sha256, device):
    verify_hash(checkpoint, sha256)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("expected the complete TC-USOD state_dict from the authors' baseline")
    prefixes = {key.startswith("module.") for key in state}
    if len(prefixes) != 1:
        raise ValueError("TC-USOD state_dict mixes wrapped and unwrapped keys")
    if True in prefixes:
        state = {key.removeprefix("module."): tensor for key, tensor in state.items()}
    model = build_architecture(repo)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


class TCUSOD:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("TC-USOD must have frozen weights and be in eval mode")

    @torch.inference_mode()
    def __call__(self, pairs):
        if any(rgb.size != depth.size for rgb, depth in pairs):
            raise ValueError("RGB and fixed depth must be aligned at native size")
        rgb = torch.stack([tc_input(image, TC_INPUT_SIZE) for image, _ in pairs]).to(self.device)
        depth = torch.stack([tc_input(image, TC_INPUT_SIZE) for _, image in pairs]).to(self.device)
        outputs = self.model(rgb, depth)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 11:
            raise ValueError("expected the 11 outputs of official TC-USOD")
        d1 = outputs[0]
        if d1.shape != (len(pairs), 1, TC_INPUT_SIZE, TC_INPUT_SIZE):
            raise ValueError(f"unexpected TC-USOD d1 shape: {d1.shape}")
        return [tc_output(output, image.size) for output, (image, _) in zip(d1, pairs, strict=True)]
