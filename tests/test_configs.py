from pathlib import Path

import yaml

from src.v1.dataset import validate_split_protocol
from src.v1.models import build_model as build_v1
from src.v2.models import build_model as build_v2
from src.v3.models import build_model as build_v3


ROOT = Path(__file__).resolve().parents[1]


def load(version: str) -> dict:
    return yaml.safe_load((ROOT / f"configs/config_{version}.yaml").read_text())


def test_split_protocol_counts_duplicates_and_leakage() -> None:
    manifests = {name: ROOT / "split" / "lsui19" / f"{name}.tsv" for name in ("train", "validation", "test")}
    parsed = validate_split_protocol(manifests)
    assert {name: len(rows) for name, rows in parsed.items()} == {
        "train": 3466, "validation": 385, "test": 428
    }


def test_configs_are_fair_except_experiment_module() -> None:
    configs = {version: load(version) for version in ("v1", "v2", "v3")}
    for section in ("data", "loss", "optimizer", "scheduler", "training", "checkpoint", "evaluation", "metrics", "test", "logging"):
        assert configs["v1"][section] == configs["v2"][section] == configs["v3"][section]
    backbone_keys = ("type", "img_channel", "width", "enc_blk_nums", "middle_blk_num", "dec_blk_nums")
    for key in backbone_keys:
        assert configs["v1"]["model"][key] == configs["v2"]["model"][key] == configs["v3"]["model"][key]
    assert configs["v1"]["model"]["middle_blk_num"] == 0


def test_all_committed_configs_use_the_replacement_bottleneck() -> None:
    expected_names = {
        "v1": "NAFEncDec_Identity",
        "v2": "NAFEncDec_PointINR",
        "v3": "NAFEncDec_GLINR",
    }
    for path in sorted((ROOT / "configs").glob("config_v*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        version = config["experiment"]["version"]
        if version not in expected_names:
            continue
        assert config["experiment"]["name"] == expected_names[version]
        assert config["model"]["middle_blk_num"] == 0


def test_backbone_parameter_shapes_identical() -> None:
    models = {
        "v1": build_v1(load("v1")["model"]),
        "v2": build_v2(load("v2")["model"]),
        "v3": build_v3(load("v3")["model"]),
    }
    prefixes = ("intro.", "encoders.", "downs.", "ups.", "decoders.", "ending.")
    shapes = {}
    for version, model in models.items():
        shapes[version] = {
            name: tuple(value.shape)
            for name, value in model.state_dict().items()
            if name.startswith(prefixes)
        }
    assert shapes["v1"] == shapes["v2"] == shapes["v3"]
    assert all(len(model.middle_blks) == 0 for model in models.values())


def test_nafnet_sources_are_byte_identical() -> None:
    sources = [(ROOT / "src" / version / "models" / "nafnet.py").read_bytes() for version in ("v1", "v2", "v3")]
    assert sources[0] == sources[1] == sources[2]
