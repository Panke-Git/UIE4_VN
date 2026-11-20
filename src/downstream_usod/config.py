"""Experiment settings and host-specific asset locations."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import yaml

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT / "configs/downstream_usod/tc_usod_usod10k.yaml"
TC_REPOSITORY = "https://github.com/LinHong-HIT/USOD10K"
TC_COMMIT = "fcdd53926054c79c20a64d1c69621316c5562929"
TC_CHECKPOINT_SHA256 = "a17524c8374b60c12c7c81e87b3ebfe657ec49c4f8b5cfeedbb82eb77e2b0c29"
TEST_COUNT = 1026
TEST_IDS_SHA256 = "e1cb352150706d388c4683abf9b0ae0279fe36d75dd9b964280abb5c4cd8aa92"
OFFICIAL_PREDICTIONS = "TC_USOD/preds/USOD10K"
TC_INPUT_SIZE = 224
INITIAL_FX_METHODS = ("waternet", "ucolor", "fdce_net", "marine_net", "swlgm")
ADDITIONAL_FX_METHODS = ("pugan", "uvz", "imif", "hdams_net")
FX_METHODS = ("waternet", "ucolor", "pugan", "fdce_net", "uvz", "imif", "hdams_net", "swlgm", "marine_net")
DATA_DIRS = {"rgb": "USOD10K-TE-RGB", "depth": "USOD10K-TE-depth", "gt": "USOD10K-TE-GT"}

V16_CHECKPOINTS = {
    "lsui": ("ae0136c6fbb539d44caec25e7e35579cda50c3cadc0e6532de861c82a19e0ab4", 195),
    "uieb": ("781ea7923ed328ad1491d353abbef1aeed932b52e6f0ff2927fdbcf4d175aa12", 49),
}
V16_METHODS = {"icar_uie": "lsui", "icar_uie_uieb": "uieb"}
P0_METHODS = ("raw", "icar_uie")


def mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return value


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return mapping(yaml.safe_load(stream), str(path))


def sha_value(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{label} must be a full SHA-256 digest")
    return value.lower()


@dataclass(frozen=True)
class Protocol:
    tile_size: int
    overlap: int
    enhancement_batch: int
    saliency_batch: int
    bootstrap_replicates: int
    bootstrap_seed: int
    examples_per_group: int
    smoke_images: int

    @classmethod
    def load(cls, path):
        obj = cls(**read_yaml(path))
        for key, value in asdict(obj).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        if obj.tile_size == 0:
            if obj.overlap:
                raise ValueError("native inference needs overlap=0")
        elif obj.tile_size % 16 or not 0 < obj.overlap < obj.tile_size:
            raise ValueError("tiled inference needs a tile multiple of 16 and 0 < overlap < tile_size")
        if min(obj.enhancement_batch, obj.saliency_batch, obj.examples_per_group) < 1:
            raise ValueError("batch sizes and examples_per_group must be positive")
        if obj.bootstrap_replicates < 2 or not 1 <= obj.smoke_images < TEST_COUNT:
            raise ValueError("need at least two bootstrap draws and a bounded smoke subset")
        return obj


@dataclass(frozen=True)
class Paths:
    file: Path
    values: dict

    @classmethod
    def load(cls, path):
        file = Path(path).expanduser().resolve()
        return cls(file, read_yaml(file))

    def section(self, key):
        return mapping(self.values.get(key, {}), key)

    def resolve(self, value, key):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"configure {key} in {self.file}")
        return (self.file.parent / Path(value).expanduser()).resolve()

    def path(self, section, key):
        record = self.section(section) if section else self.values
        return self.resolve(record.get(key), ".".join(filter(None, (section, key))))

    @property
    def test_root(self):
        return self.path("usod10k", "test_root")

    @property
    def output(self):
        value = self.values.get("output_root")
        if value is not None:
            return self.resolve(value, "output_root")
        return PROJECT / "experiments/downstream_usod/tc_usod_20261119_v1"

    def data_dirs(self):
        return {role: self.section("usod10k").get(role + "_dir", default)
                for role, default in DATA_DIRS.items()}

    def v16_assets(self, source):
        entry = mapping(self.section("v16").get(source, {}), f"v16.{source}")
        return (self.resolve(entry.get("config"), f"v16.{source}.config"),
                self.resolve(entry.get("checkpoint"), f"v16.{source}.checkpoint"))

    def tc_asset(self):
        section = self.section("tc_usod")
        sha = sha_value(section.get("sha256", TC_CHECKPOINT_SHA256), "tc_usod.sha256")
        if sha != TC_CHECKPOINT_SHA256:
            raise ValueError("TC-USOD checkpoint must be the complete author-published baseline")
        return self.path("tc_usod", "checkpoint"), sha

    @property
    def device(self):
        return str(self.values.get("device", "cuda:0"))
