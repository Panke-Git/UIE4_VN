"""Extend P1 with four new FX methods using the original sealed predictions."""

from .config import ADDITIONAL_FX_METHODS, FX_METHODS, INITIAL_FX_METHODS, P0_METHODS
from .evaluation import METRICS
from .inference import ImageStage
from .storage import digest, event, read_json, seal_json, sha256_file


def _reference_prediction(experiment, reference, method, signature, coverage):
    samples = experiment.samples
    enhanced = None
    enhancement_proof = None
    if method != "raw":
        identity_path = reference / "audit/stages/enhanced" / method / "identity.json"
        identity = read_json(identity_path)
        if identity.get("signature") != signature:
            raise ValueError(f"reference enhancement does not belong to the sealed run: {method}")
        if method == "icar_uie":
            if identity.get("v16") != experiment.identity["lsui_v16"]:
                raise ValueError("reference ICAR-UIE identity differs from the fixed LSUI model")
        elif (identity.get("checkpoint_sha256") != coverage[method]["checkpoint_sha256"]
              or identity.get("training_source") != "lsui"):
            raise ValueError(f"reference FX checkpoint differs from the P1 report: {method}")
        enhanced = ImageStage(reference, "enhanced", method, samples, identity, "RGB", read_only=True)
        enhancement_proof = enhanced.complete()

    stage = ImageStage(reference, "predictions", method, samples, {
        "signature": signature, "tc_checkpoint_sha256": experiment.tc_sha256, "enhanced": enhancement_proof,
    }, "L", read_only=True)
    for sample in samples:
        rgb = sample.hashes["rgb"]
        if enhanced is not None:
            receipt = read_json(enhanced.receipt_path(sample))
            if receipt["inputs"] != {"rgb": rgb}:
                raise ValueError(f"reference enhancement RGB differs for {method}/{sample.sample_id}")
            rgb = receipt["sha256"]
        if stage.completed(sample, {"rgb": rgb, "depth": sample.hashes["depth"]}) is None:
            raise ValueError(f"missing reference prediction: {method}/{sample.sample_id}")
    stage.complete()
    return stage


def _read_reference(experiment, reference):
    identity = read_json(reference / "audit/run.json")
    signature = digest(identity)
    if {k: v for k, v in identity.items() if k != "release"} != {
        k: v for k, v in experiment.identity.items() if k != "release"
    }:
        raise ValueError("reference and extension differ on data, models, metric protocol or runtime")
    if read_json(reference / "manifests/test.json") != read_json(experiment.root / "manifests/test.json"):
        raise ValueError("reference and extension have different test manifests")
    p0 = read_json(reference / "reports/p0_summary.json")
    p1_path = reference / "reports/p1_summary.json"
    p1 = read_json(p1_path)
    for phase, result in (("p0", p0), ("p1", p1)):
        if (result.get("phase"), result.get("status"), result.get("signature"), result.get("test_count")) != (
            phase, "complete", signature, len(experiment.samples),
        ):
            raise ValueError(f"reference {phase} summary is incomplete or has a different run signature")
    expected = (*P0_METHODS, *INITIAL_FX_METHODS)
    old_rows = {row["method"]: row for row in p1["metrics"]}
    if (set(old_rows) != set(expected) or len(p1["metrics"]) != len(expected)
            or any(row.get("count") != len(experiment.samples) for row in old_rows.values())
            or {row["method"]: row for row in p0["metrics"]} != {m: old_rows[m] for m in P0_METHODS}):
        raise ValueError("reference metrics do not cover the required P0 and P1 conditions")
    coverage = p1["fx"]
    if set(coverage) != set(INITIAL_FX_METHODS) or any(
        item.get("status") != "complete" or item.get("training_source") != "lsui" for item in coverage.values()
    ):
        raise ValueError("reference P1 requires the five completed LSUI FX baselines")
    stages = {m: _reference_prediction(experiment, reference, m, signature, coverage) for m in expected}
    seal_json(experiment.root / "audit/p1_reference.json", {
        "root": str(reference), "release": identity["release"], "run_signature": signature,
        "p1_report_sha256": sha256_file(p1_path), "reused_methods": list(expected),
    })
    return stages, old_rows, coverage


def extend_p1(experiment, reference):
    stages, prior, original_fx = _read_reference(experiment, reference)
    statuses = {**original_fx, **experiment.run_fx_methods(stages, ADDITIONAL_FX_METHODS)}
    stages = {m: stages[m] for m in (*P0_METHODS, *FX_METHODS) if m in stages}
    rows = experiment.measure(experiment.samples, stages, experiment.root)
    for name, archived in prior.items():
        if {key: archived[key] for key in METRICS} != rows[name].overall():
            raise ValueError(f"re-evaluation differs from the sealed reference P1 metrics: {name}")
    result = experiment.report("p1", rows, stages, {m: statuses[m] for m in FX_METHODS})
    event(experiment.root / "logs/pipeline.jsonl", phase="p1", status=result["status"],
          reference=str(reference), added=list(ADDITIONAL_FX_METHODS),
          counts={name: len(row.ids) for name, row in rows.items()})
    return result
