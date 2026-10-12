from copy import deepcopy
import pytest
import torch
from .helpers import ROOT, VARIANTS, config, module


@pytest.mark.parametrize("dataset,baseline", [("lsui", "configs/config_v16.yaml"), ("uieb", "configs/uieb/config_v16_uieb.yaml")])
def test_eight_configs_only_have_permitted_changes(dataset, baseline):
    import yaml
    reference = yaml.safe_load((ROOT / baseline).read_text())
    for variant in VARIANTS:
        cfg = config(variant, dataset)
        normalized = deepcopy(cfg)
        for key in ("version", "name", "output_root"):
            normalized["experiment"][key] = reference["experiment"][key]
        normalized["model"]["type"] = reference["model"]["type"]
        assert normalized == reference
        guard = module(variant, "provenance")
        guard.validate_config(cfg)
        paths = {s: ROOT / cfg["data"][s + "_manifest"] for s in ("train", "validation", "test")}
        guard.validate_manifest_hashes(cfg, paths)
        parsed = module(variant, "dataset").validate_split_protocol(paths, cfg["data"]["expected_counts"])
        assert {s: len(v) for s, v in parsed.items()} == cfg["data"]["expected_counts"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_protocol_rejects_two_set_or_substituted_counts(variant):
    guard = module(variant, "provenance")
    cfg = config(variant)
    cfg["data"]["expected_counts"]["train"] = 3851
    with pytest.raises(ValueError, match="expected_counts"):
        guard.validate_config(cfg)
    cfg = config(variant)
    cfg["data"]["test_manifest"] = "split/v16_19/lsui19/validation.tsv"
    with pytest.raises(ValueError, match="manifest"):
        guard.validate_config(cfg)


def test_metrics_dataset_losses_are_exact_v16_copies():
    from src.v16.metrics import batch_metrics as reference
    from src.shared.e00 import batch_delta_e00
    x, y = torch.rand(3, 3, 17, 21), torch.rand(3, 3, 17, 21)
    for variant in VARIANTS:
        for name in ("dataset", "metrics", "losses"):
            assert (ROOT / f"src/qgrd_ablations/{variant}/{name}.py").read_bytes() == (ROOT / f"src/v16/{name}.py").read_bytes()
        cfg = config(variant)["metrics"]
        actual = module(variant, "metrics").batch_metrics(x, y, cfg)
        expected = reference(x, y, cfg)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert torch.equal(batch_delta_e00(x, y, cfg), batch_delta_e00(x, y, cfg))
