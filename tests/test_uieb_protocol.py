from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("v4", "v13", "v14", "v15", "v16", "v17")
SPLITS = ("train", "validation", "test")
LSUI_COUNTS = {"train": 3466, "validation": 385, "test": 428}
UIEB_COUNTS = {"train": 720, "validation": 80, "test": 90}
UIEB_NAMES = {
    "v4": "PlainUNet_UIEB_T90",
    "v13": "PlainUNet_UICF_PreBackbone_UIEB_T90",
    "v14": "PlainUNet_UICF_ParallelBranch_UIEB_T90",
    "v15": "PlainUNet_ColorQuery_UIEB_T90",
    "v16": "PlainUNet_ColorQuery_UICF_PreBackbone_UIEB_T90",
    "v17": "PlainUNet_ColorQuery_UICF_ParallelBranch_UIEB_T90",
}
IDENTITY_FIELDS = ("sample_id", "input_relative", "gt_relative")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _manifests(dataset: str) -> dict[str, Path]:
    directory = ROOT / "split" / dataset
    return {split: directory / f"{split}.tsv" for split in SPLITS}


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    ("dataset", "expected_counts"),
    (("lsui19", LSUI_COUNTS), ("uieb", UIEB_COUNTS)),
)
def test_each_target_version_validates_both_protocols(
    version: str, dataset: str, expected_counts: dict[str, int]
) -> None:
    validate = importlib.import_module(f"src.{version}.dataset").validate_split_protocol
    parsed = validate(_manifests(dataset), expected_counts)
    assert {split: len(parsed[split]) for split in SPLITS} == expected_counts


def test_uieb_manifest_schema_paths_and_cross_split_identity_are_strict() -> None:
    dataset = importlib.import_module("src.v4.dataset")
    parsed = dataset.validate_split_protocol(_manifests("uieb"), UIEB_COUNTS)
    assert sum(len(entries) for entries in parsed.values()) == 890
    for split in SPLITS:
        path = _manifests("uieb")[split]
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            columns = line.split("\t")
            assert len(columns) == 3, f"{path}:{line_number}"
            assert all(columns)
            assert not Path(columns[1]).is_absolute()
            assert not Path(columns[2]).is_absolute()
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field in IDENTITY_FIELDS:
            first_values = {getattr(entry, field) for entry in parsed[first]}
            second_values = {getattr(entry, field) for entry in parsed[second]}
            assert first_values.isdisjoint(second_values), f"{first}/{second}/{field}"


def test_expected_counts_are_optional_but_complete_and_strict_when_provided() -> None:
    validate = importlib.import_module("src.v4.dataset").validate_split_protocol
    parsed = validate(_manifests("uieb"), None)
    assert {split: len(parsed[split]) for split in SPLITS} == UIEB_COUNTS
    with pytest.raises(ValueError, match="Missing expected count for split: test"):
        validate(_manifests("uieb"), {"train": 720, "validation": 80})
    with pytest.raises(ValueError, match="train count is 720, expected 719"):
        validate(
            _manifests("uieb"),
            {"train": 719, "validation": 80, "test": 90},
        )


@pytest.mark.parametrize("version", VERSIONS)
def test_lsui_and_uieb_configs_differ_only_in_allowed_protocol_fields(version: str) -> None:
    lsui = _load(ROOT / "configs" / f"config_{version}.yaml")
    uieb = _load(ROOT / "configs" / "uieb" / f"config_{version}_uieb.yaml")

    assert uieb["experiment"]["version"] == lsui["experiment"]["version"] == version
    assert uieb["experiment"]["seed"] == lsui["experiment"]["seed"] == 3520
    assert uieb["experiment"]["output_root"] == lsui["experiment"]["output_root"]
    assert uieb["experiment"]["name"] == UIEB_NAMES[version]

    allowed_data_differences = {
        "dataset",
        "root",
        "train_manifest",
        "validation_manifest",
        "test_manifest",
        "expected_counts",
    }
    assert {
        key: value for key, value in uieb["data"].items() if key not in allowed_data_differences
    } == {
        key: value for key, value in lsui["data"].items() if key not in allowed_data_differences
    }
    assert lsui["data"]["dataset"] == "LSUI19"
    assert lsui["data"]["expected_counts"] == LSUI_COUNTS
    assert uieb["data"] | {} == {
        **lsui["data"],
        "dataset": "UIEB",
        "root": "/root/autodl-tmp/pro/publicdata/UIEB19",
        "train_manifest": "split/uieb/train.tsv",
        "validation_manifest": "split/uieb/validation.tsv",
        "test_manifest": "split/uieb/test.tsv",
        "expected_counts": UIEB_COUNTS,
    }
    for section in (
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "checkpoint",
        "evaluation",
        "metrics",
        "test",
        "logging",
    ):
        assert uieb[section] == lsui[section], f"{version}/{section}"


@pytest.mark.parametrize("version", VERSIONS)
def test_target_entry_points_pass_configured_counts_and_use_generic_root_errors(
    version: str,
) -> None:
    for filename in ("train.py", "test.py"):
        source = (ROOT / "src" / version / filename).read_text(encoding="utf-8")
        assert 'config["data"].get("expected_counts")' in source
        assert "LSUI data.root is unavailable" not in source


@pytest.mark.parametrize("version", VERSIONS)
def test_dataset_source_has_no_hard_coded_protocol_sizes(version: str) -> None:
    source = inspect.getsource(
        importlib.import_module(f"src.{version}.dataset").validate_split_protocol
    )
    for forbidden_count in ("3466", "3851", "385", "428", "720", "80", "90"):
        assert forbidden_count not in source
