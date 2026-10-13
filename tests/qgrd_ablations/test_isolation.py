import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from .helpers import ROOT, VARIANTS, config, module


def test_static_imports_are_isolated_and_runner_imports_are_local():
    for variant in VARIANTS:
        package = ROOT / "src/qgrd_ablations" / variant
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    name = node.module or ""
                    if node.level:
                        package_parts = list(path.parent.relative_to(ROOT).parts)
                        name = ".".join(package_parts[:len(package_parts) - node.level + 1]
                                        + ([name] if name else []))
                    imported = [name]
                else:
                    continue
                for name in imported:
                    assert not any(name.startswith(f"src.qgrd_ablations.{other}")
                                   for other in VARIANTS if other != variant), (path, name)
                if isinstance(node, ast.Name):
                    assert node.id not in {"ablation_mode", "ablation_type"}
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    assert node.id not in {"ablation_mode", "ablation_type"}
                if isinstance(node, ast.Attribute):
                    assert node.attr not in {"register_forward_hook", "register_forward_pre_hook"}
                if path.parent.name == "models" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert not any(a.arg in {"mode", "ablation_mode", "ablation_type"} for a in node.args.args)
        for runner in ("train.py", "test.py"):
            tree = ast.parse((package / runner).read_text())
            assert any(isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "models"
                       and any(a.name == "build_model" for a in n.names) for n in ast.walk(tree))


def test_versions_output_roots_and_all_checkpoint_guards():
    versions, roots = set(), []
    for variant in VARIANTS:
        cfg = config(variant)
        versions.add(cfg["experiment"]["version"])
        for ds in ("lsui", "uieb"):
            roots.append(Path(config(variant, ds)["experiment"]["output_root"]))
        guard = module(variant, "provenance")
        sources = guard.verify_source_hashes()
        payload = {"version": cfg["experiment"]["version"], "variant": variant,
                   "model_type": cfg["model"]["type"], "resolved_config": cfg,
                   "source_sha256": sources, "protocol_identity": guard.protocol_identity(cfg)}
        if variant == "shared_query":
            payload["evaluation_precision"] = "fp32"
            payload["evaluation_forward_protocol"] = module(variant, "evaluation").evaluation_protocol(cfg)
        guard.verify_checkpoint(payload, cfg, sources)
        for other in VARIANTS:
            if other == variant:
                continue
            bad = deepcopy(payload)
            bad["version"] = config(other)["experiment"]["version"]
            bad["variant"] = other
            with pytest.raises(ValueError, match="variant/version"):
                guard.verify_checkpoint(bad, cfg, sources)
        bad = deepcopy(payload)
        bad["resolved_config"] = config(variant, "uieb")
        with pytest.raises(ValueError, match="resolved_config"):
            guard.verify_checkpoint(bad, cfg, sources)
    assert len(versions) == 4 and len(set(roots)) == 8
    assert not any(a.is_relative_to(b) for a in roots for b in roots if a != b)


def test_all_test_entries_refuse_wrong_variant_before_writes(tmp_path):
    import yaml
    for variant in VARIANTS:
        run = tmp_path / variant
        run.mkdir()
        other = next(v for v in VARIANTS if v != variant)
        (run / "config_resolved.yaml").write_text(yaml.safe_dump(config(other)))
        before = list(run.iterdir())
        with pytest.raises(ValueError, match="refuses"):
            module(variant, "test").main(["--run-dir", str(run)])
        assert list(run.iterdir()) == before


def test_original_tracked_files_are_unchanged():
    snapshot = json.loads((ROOT / "src/qgrd_ablations/protection_snapshot.json").read_text())
    for path, expected in snapshot["tracked_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path


def test_pinned_sources_reject_tampering(monkeypatch):
    for variant in VARIANTS:
        guard = module(variant, "provenance")
        monkeypatch.setattr(guard, "sha256_file", lambda _p: "modified")
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            guard.verify_source_hashes()
