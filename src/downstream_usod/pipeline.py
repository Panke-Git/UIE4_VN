"""Official run gates, P0/P1/P2 and shared evaluation/reporting."""

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import platform
import re

import torch

from .config import FX_METHODS, INITIAL_FX_METHODS, P0_METHODS, PROJECT, TC_COMMIT, TC_REPOSITORY, V16_METHODS
from .dataset import git, manifest, official_ids, read_manifest, scan_test
from .evaluation import METRIC_PROTOCOL, METRICS, evaluate_method
from .fx_adapter import FXAsset, run_fx
from .inference import ImageStage, enhance, saliency
from .statistics import compare_methods
from .storage import (assert_separate_output, digest, event, read_json, seal_json, verify_hash,
                      write_csv, write_json, write_text)
from .tc_usod_adapter import TCUSOD, load_tc, verify_model_source
from .v16_adapter import V16Enhancer, load_v16, v16_identity
from .visualization import make_figure

FEATURE_BRANCH = "feat/downstream-usod-tc-usod"
DIRECTORIES = ("audit", "manifests", "enhanced", "predictions", "metrics", "visualizations",
               "logs", "reports")


def verify_release(expected):
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("--commit must be the full published feature commit")
    if Path(git(PROJECT, "rev-parse", "--show-toplevel")).resolve() != PROJECT:
        raise ValueError(f"incorrect deployment repository root: {PROJECT}")
    head = git(PROJECT, "rev-parse", "HEAD")
    branch = git(PROJECT, "branch", "--show-current")
    if head != expected or branch != FEATURE_BRANCH:
        raise ValueError(f"expected {FEATURE_BRANCH} at {expected}; found {branch} at {head}")
    if git(PROJECT, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("feature checkout has uncommitted or untracked files")
    return head


def verify_output_root(root, paths, fx_root=None):
    protected = [paths.file]
    for section, keys in (("usod10k", ("test_root",)), ("tc_usod", ("repository", "checkpoint"))):
        for key in keys:
            if paths.section(section).get(key):
                protected.append(paths.path(section, key))
    for source, entry in paths.section("v16").items():
        for key in ("config", "checkpoint"):
            if isinstance(entry, dict) and entry.get(key):
                protected.append(paths.resolve(entry[key], f"v16.{source}.{key}"))
    if fx_root or paths.values.get("fx_root"):
        protected.append(Path(fx_root).expanduser().resolve() if fx_root else paths.path(None, "fx_root"))
    for method, item in paths.section("fx").items():
        if isinstance(item, dict):
            if item.get("checkpoint"):
                protected.append(paths.resolve(item["checkpoint"], f"fx.{method}.checkpoint"))
            auxiliary = item.get("auxiliary")
            if isinstance(auxiliary, dict):
                for name, spec in auxiliary.items():
                    if isinstance(spec, dict) and spec.get("path"):
                        protected.append(paths.resolve(spec["path"], f"fx.{method}.auxiliary.{name}.path"))
    assert_separate_output(root, protected)


def claim(root):
    marker = root / "audit/experiment.json"
    if not marker.exists() and any(path.name != ".usod.lock" for path in root.iterdir()):
        raise ValueError(f"output directory is already in use: {root}")
    for name in (*DIRECTORIES, "smoke"):
        if (root / name).is_symlink():
            raise ValueError(f"experiment subdirectory cannot be a symlink: {root / name}")
    seal_json(marker, {"schema": 1, "experiment": "ICAR-UIE × TC-USOD", "dataset": "USOD10K Test"})
    for name in DIRECTORIES:
        (root / name).mkdir(exist_ok=True)


def prepare(paths, root):
    source = paths.path("tc_usod", "repository")
    ids = official_ids(source)
    samples = scan_test(paths.test_root, paths.data_dirs(), ids)
    value = manifest(samples, official=True)
    manifest_path = root / "manifests/test.json"
    seal_json(manifest_path, value)
    return read_manifest(read_json(manifest_path), ids), value["sha256"]


def environment(device, *, require_cuda):
    try:
        target = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"invalid device {device}") from error
    if require_cuda and target.type != "cuda":
        raise ValueError("AutoDL smoke and formal runs require a CUDA device")
    details = {
        "device": str(target), "python": platform.python_version(), "platform": platform.system(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "packages": {name: version(name) for name in ("torchvision", "timm", "numpy", "Pillow", "PyYAML")},
    }
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the configured interpreter")
        gpu = torch.cuda.get_device_properties(target)
        details["gpu"] = {"name": gpu.name, "bytes": gpu.total_memory,
                          "capability": [gpu.major, gpu.minor]}
    elif target.type != "cpu":
        raise ValueError("the supported device types are cpu and cuda")
    return details


def preflight(paths, root, release, load_models, phase, fx_root):
    checks = {}

    def check(name, operation):
        try:
            result = operation()
        except Exception as error:
            checks[name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            return None
        checks[name] = {"status": "passed", "details": result}
        return result

    check("environment", lambda: environment(paths.device, require_cuda=release is not None))
    check("tc_source", lambda: str(verify_model_source(paths.path("tc_usod", "repository"))))

    def test_check():
        samples, sha = prepare(paths, root)
        return {"count": len(samples), "manifest_sha256": sha}

    check("test", test_check)

    def tc_check():
        checkpoint, checksum = paths.tc_asset()
        verify_hash(checkpoint, checksum)
        if load_models:
            model = load_tc(paths.path("tc_usod", "repository"), checkpoint, checksum, paths.device)
            del model
        return {"checkpoint_sha256": checksum, "strict_load": load_models}

    check("tc_checkpoint", tc_check)
    for source in ("lsui", "uieb") if phase == "p2" else ("lsui",):

        def v16_check():
            config, checkpoint = paths.v16_assets(source)
            info = v16_identity(config, checkpoint, source)
            if load_models:
                model = load_v16(config, checkpoint, source, paths.device, info)
                del model
            return {**info, "strict_load": load_models}

        check(f"v16_{source}", v16_check)
    baselines = {}
    if phase == "p1":
        for method in FX_METHODS:
            try:
                asset = FXAsset.load(paths, method, fx_root)
                baselines[method] = {"status": "assets_verified", "sha256": asset.sha256, **asset.provenance()}
            except (OSError, ValueError) as error:
                baselines[method] = {"status": "unavailable", "reason": str(error)}
    report = {"status": "passed" if all(item["status"] == "passed" for item in checks.values()) else "blocked",
              "release": release, "checks": checks, "optional_fx": baselines}
    write_json(root / "audit" / f"preflight_{phase}.json", report)
    return report


class Experiment:
    def __init__(self, paths, root, protocol, release, fx_root=None):
        self.paths, self.root, self.protocol, self.fx_root = paths, root, protocol, fx_root
        self.repo = paths.path("tc_usod", "repository")
        verify_model_source(self.repo)
        self.tc_checkpoint, checksum = paths.tc_asset()
        verify_hash(self.tc_checkpoint, checksum)
        self.device = paths.device
        hardware = environment(self.device, require_cuda=True)
        self.samples, manifest_sha = prepare(paths, root)
        cfg, weights = paths.v16_assets("lsui")
        lsui = v16_identity(cfg, weights, "lsui")
        self.tc_sha256 = checksum
        self.identity = {"release": release, "test_manifest_sha256": manifest_sha,
                         "tc_repository": TC_REPOSITORY, "tc_commit": TC_COMMIT,
                         "tc_checkpoint_sha256": checksum, "lsui_v16": lsui,
                         "protocol": asdict(protocol), "metrics": METRIC_PROTOCOL,
                         "runtime": hardware}
        seal_json(root / "audit/run.json", self.identity)
        self.signature = digest(self.identity)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False

    def _enhancement(self, destination, samples, method):
        if method in V16_METHODS:
            source = V16_METHODS[method]
            cfg, checkpoint = self.paths.v16_assets(source)
            identity = (self.identity["lsui_v16"] if source == "lsui"
                        else v16_identity(cfg, checkpoint, source))
            return ImageStage(destination, "enhanced", method, samples,
                              {"signature": self.signature, "v16": identity}, "RGB")
        path = destination / "audit/stages/enhanced" / method / "identity.json"
        identity = read_json(path)
        if identity["signature"] != self.signature:
            raise ValueError(f"FX result {method} belongs to a different run")
        return ImageStage(destination, "enhanced", method, samples, identity, "RGB")

    def run_enhancement(self, method, destination, samples, fx=None):
        if method in FX_METHODS:
            asset = fx or FXAsset.load(self.paths, method, self.fx_root)
            torch.cuda.empty_cache()
            return run_fx(asset, destination, self.paths.test_root, samples, self.signature, self.device)
        stage = self._enhancement(destination, samples, method)
        if (stage.audit / "complete.json").is_file():
            stage.complete()
            return stage
        source = V16_METHODS[method]
        config, checkpoint = self.paths.v16_assets(source)
        model = load_v16(config, checkpoint, source, self.device, stage.identity["v16"])
        runner = V16Enhancer(model, self.device, self.protocol.tile_size, self.protocol.overlap)
        enhance(stage, runner, self.paths.test_root, self.protocol.enhancement_batch)
        return stage

    def predictions(self, method, destination, samples, enhanced=None):
        if method != "raw" and enhanced is None:
            enhanced = self._enhancement(destination, samples, method)
        identity = {"signature": self.signature, "tc_checkpoint_sha256": self.tc_sha256,
                    "enhanced": enhanced.complete() if enhanced else None}
        return ImageStage(destination, "predictions", method, samples, identity, "L"), enhanced

    def run_saliency(self, method, destination, samples, enhanced=None):
        stage, enhanced = self.predictions(method, destination, samples, enhanced)
        if (stage.audit / "complete.json").is_file():
            stage.complete()
            return stage
        tc = TCUSOD(load_tc(self.repo, self.tc_checkpoint, self.tc_sha256, self.device), self.device)
        saliency(stage, tc, self.paths.test_root, self.protocol.saliency_batch, enhanced)
        return stage

    def measure(self, samples, predictors, destination):
        results = {}
        for name, stage in predictors.items():
            identity = stage.complete()
            rows = evaluate_method(samples, self.paths.test_root, stage.load)
            rows.save(destination / "metrics" / name, identity)
            results[name] = rows
        return results

    def _smoke_samples(self):
        # Largest image plus evenly spaced sorted IDs, independent of model outputs and GT.
        largest = max(self.samples, key=lambda sample: (sample.width * sample.height, sample.sample_id))
        count = self.protocol.smoke_images
        selected = [largest]
        for i in range(count):
            sample = self.samples[i * (len(self.samples) - 1) // max(1, count - 1)]
            if sample not in selected:
                selected.append(sample)
        return sorted(selected[:count], key=lambda sample: sample.sample_id)

    def smoke(self, phase="p0", fx=None):
        if phase not in ("p0", "p2") and fx is None:
            raise ValueError("an FX smoke requires an explicit model asset")
        samples = self._smoke_samples()
        destination = self.root / "smoke" / phase
        method = fx.method if fx else ("icar_uie_uieb" if phase == "p2" else "icar_uie")
        raw = self.run_saliency("raw", destination, samples)
        enhanced = self.run_enhancement(method, destination, samples, fx)
        candidate = self.run_saliency(method, destination, samples, enhanced)
        rows = self.measure(samples, {"raw": raw, method: candidate}, destination)
        report = {"status": "passed", "purpose": "official-asset smoke, subset only",
                  "signature": self.signature, "sample_ids": [sample.sample_id for sample in samples],
                  "enhanced": enhanced.complete(), "predictions": {
                      "raw": raw.complete(), method: candidate.complete()},
                  "metrics": {name: row.overall() for name, row in rows.items()}}
        seal_json(self.root / "audit/smoke" / (phase + ".json"), report)
        return report

    def require_smoke(self, phase):
        path = self.root / "audit/smoke" / (phase + ".json")
        if not path.is_file():
            raise FileNotFoundError(f"run the {phase} smoke successfully before full inference")
        receipt = read_json(path)
        if receipt.get("status") != "passed" or receipt.get("signature") != self.signature:
            raise ValueError(f"{phase} smoke does not match this release, assets and environment")
        method = "icar_uie_uieb" if phase == "p2" else "icar_uie"
        stage = self._enhancement(self.root / "smoke" / phase, self._smoke_samples(), method)
        if stage.complete() != receipt.get("enhanced"):
            raise ValueError(f"{phase} smoke does not match the verified enhancement outputs")
        return receipt

    def require_p0(self):
        path = self.root / "reports/p0_summary.json"
        if not path.is_file():
            raise FileNotFoundError("complete P0 before running P1 or P2")
        receipt = read_json(path)
        if receipt.get("status") != "complete" or receipt.get("signature") != self.signature:
            raise ValueError("P0 report does not belong to this completed run")

    def report(self, phase, methods, stages, fx_status=None, *, visualize=True):
        output = self.root / "metrics" / phase
        table = [{"method": name, "count": len(rows.ids), **rows.overall()}
                 for name, rows in methods.items()]
        write_csv(output / "overall.csv", table)
        paired = compare_methods(methods, output, self.protocol.bootstrap_replicates,
                                 self.protocol.bootstrap_seed)
        if visualize:
            make_figure(self.samples, self.paths.test_root, methods, stages,
                        self.root / "visualizations" / phase, self.protocol.examples_per_group)
        coverage = fx_status or {}
        statuses = {item["status"] for item in coverage.values()}
        result = {"phase": phase, "status": "partial" if statuses - {"complete"} else "complete",
                  "signature": self.signature, "test_count": len(self.samples),
                  "metrics": table, "paired_bootstrap": paired, "fx": coverage}
        destination = self.root / "reports"
        lines = [f"# {phase.upper()} · ICAR-UIE × TC-USOD", "",
                 f"Official USOD10K Test · {len(self.samples)} images · {result['status']}", "",
                 "| Method | " + " | ".join(METRICS) + " |",
                 "| --- |" + " --- |" * len(METRICS)]
        for row in table:
            lines.append("| " + row["method"] + " | " + " | ".join(f"{row[m]:.6f}" for m in METRICS) + " |")
        lines.extend(("", f"Paired 95% intervals and image-level improvement rates are in metrics/{phase}."))
        for name, item in coverage.items():
            lines.append(f"\n- {name} · {item['status']}" + (f" · {item['reason']}" if "reason" in item else ""))
        write_text(destination / f"{phase}_summary.md", "\n".join(lines) + "\n")
        if phase == "p1":
            write_json(destination / "p1_summary.json", result)
        else:
            seal_json(destination / f"{phase}_summary.json", result)
        return result

    def run_fx_methods(self, stages, fx_methods):
        statuses = {}
        for method in fx_methods:
            try:
                fx = FXAsset.load(self.paths, method, self.fx_root)
            except (OSError, ValueError) as error:
                statuses[method] = {"status": "unavailable", "reason": str(error)}
                continue
            try:
                self.smoke(method, fx=fx)
                enhanced = self.run_enhancement(method, self.root, self.samples, fx=fx)
                stages[method] = self.run_saliency(method, self.root, self.samples, enhanced)
            except (OSError, ValueError, RuntimeError) as error:
                statuses[method] = {"status": "failed", "reason": str(error)}
                continue
            statuses[method] = {"status": "complete", **fx.provenance()}
        return statuses

    def run(self, phase, fx_methods=INITIAL_FX_METHODS):
        self.require_smoke("p0")
        if phase != "p0":
            self.require_p0()
        if phase == "p2":
            self.require_smoke("p2")
        stages = {"raw": self.run_saliency("raw", self.root, self.samples)}
        primary = self.run_enhancement("icar_uie", self.root, self.samples)
        stages["icar_uie"] = self.run_saliency("icar_uie", self.root, self.samples, primary)
        statuses = {}
        if phase == "p2":
            method = "icar_uie_uieb"
            enhanced = self.run_enhancement(method, self.root, self.samples)
            stages[method] = self.run_saliency(method, self.root, self.samples, enhanced)
        if phase == "p1":
            statuses = self.run_fx_methods(stages, fx_methods)
        rows = self.measure(self.samples, stages, self.root)
        result = self.report(phase, rows, stages, statuses)
        event(self.root / "logs/pipeline.jsonl", phase=phase, status=result["status"],
              counts={name: len(row.ids) for name, row in rows.items()})
        return result


def phase_methods(phase, fx_methods=INITIAL_FX_METHODS):
    requested = tuple(fx_methods)
    if set(requested) - set(FX_METHODS) or len(set(requested)) != len(requested):
        raise ValueError("FX method list must contain only unique supported method names")
    if phase == "p1":
        return (*P0_METHODS, *requested)
    if phase == "p2":
        return (*P0_METHODS, "icar_uie_uieb")
    return P0_METHODS
