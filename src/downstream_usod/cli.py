"""Command-line stages for the frozen USOD10K experiment."""

import argparse
import json
from pathlib import Path
import sys

from .config import DEFAULT_PROTOCOL, FX_METHODS, Paths, Protocol
from .storage import event, run_lock


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--paths", type=Path, required=True, help="host-specific paths YAML")
    common.add_argument("--config", type=Path, default=DEFAULT_PROTOCOL, help="experiment recipe YAML")
    common.add_argument("--output-root", type=Path, help="override the output directory")
    common.add_argument("--fx-root", type=Path, help="override the external FX directory")
    common.add_argument("--commit", help="full published feature commit, required for GPU stages")
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", parents=[common], help="validate and freeze the official Test set")
    audit = commands.add_parser("preflight", parents=[common], help="check release, input assets and runtime")
    audit.add_argument("--phase", default="p0", choices=("p0", "p1", "p2"))
    audit.add_argument("--load-models", action="store_true", help="perform strict checkpoint loads")
    audit.add_argument("--deployed", action="store_true", help="require the published commit and CUDA")
    smoke = commands.add_parser("smoke", parents=[common], help="test real inputs, weights and alignment on CUDA")
    smoke.add_argument("--phase", default="p0", choices=("p0", "p2"))
    for command, description in {
        "enhance": "generate or resume one enhanced condition",
        "saliency": "generate or resume one saliency condition",
        "evaluate": "write per-image and aggregate metrics from completed predictions",
        "analyze": "write paired bootstrap and overall reports",
        "visualize": "render fixed-rule qualitative examples",
        "pipeline": "complete the specified formal experiment",
    }.items():
        action = commands.add_parser(command, parents=[common], help=description)
        action.add_argument("--phase", default="p0", choices=("p0", "p1", "p2"))
        action.add_argument("--fx-methods", nargs="+", default=FX_METHODS, choices=FX_METHODS,
                            help="selected P1 baselines, in display order")
        if command in ("enhance", "saliency"):
            action.add_argument("--method", required=True)
    return cli


def execute(args, paths, protocol, root):
    from .pipeline import Experiment, phase_methods, preflight, prepare, verify_release
    from .visualization import make_figure

    action = args.command
    formal = action not in ("prepare", "preflight")
    if (formal or (action == "preflight" and args.deployed)) and not args.commit:
        raise ValueError("deployment checks and GPU stages require --commit with the published feature SHA")
    release = verify_release(args.commit) if args.commit else None
    if action == "prepare":
        samples, manifest_sha = prepare(paths, root)
        return {"status": "prepared", "count": len(samples), "manifest_sha256": manifest_sha}
    if action == "preflight":
        return preflight(paths, root, release, args.load_models, args.phase, args.fx_root)

    experiment = Experiment(paths, root, protocol, release, args.fx_root)
    if action == "smoke":
        if args.phase == "p2":
            experiment.require_p0()
        return experiment.smoke(args.phase)

    methods = phase_methods(args.phase, args.fx_methods)
    if action == "pipeline":
        return experiment.run(args.phase, args.fx_methods)

    experiment.require_smoke("p0")
    if args.phase != "p0":
        experiment.require_p0()
    if args.phase == "p2":
        experiment.require_smoke("p2")

    if action in ("enhance", "saliency"):
        if args.method not in methods or (action == "enhance" and args.method == "raw"):
            raise ValueError(f"{args.method} is not used by {action} in {args.phase}")
        if args.method in FX_METHODS:
            from .fx_adapter import FXAsset
            fx = FXAsset.load(paths, args.method, args.fx_root)
            experiment.smoke(args.method, fx=fx)
        if action == "enhance":
            stage = experiment.run_enhancement(args.method, root, experiment.samples)
        else:
            stage = experiment.run_saliency(args.method, root, experiment.samples)
        return {"status": "complete", "method": args.method, **stage.complete()}

    stages = {name: experiment.predictions(name, root, experiment.samples)[0] for name in methods}
    rows = experiment.measure(experiment.samples, stages, root)
    if action == "analyze":
        return experiment.report(args.phase, rows, stages, visualize=False)
    if action == "visualize":
        selected = make_figure(experiment.samples, paths.test_root, rows, stages,
                               root / "visualizations" / args.phase, protocol.examples_per_group)
        return {"status": "complete", "selected": selected}
    return {"status": "complete", "metrics": {name: item.overall() for name, item in rows.items()}}


def main(argv=None):
    args = parser().parse_args(argv)
    root = None
    claimed = False
    try:
        from .pipeline import claim, verify_output_root

        paths = Paths.load(args.paths)
        protocol = Protocol.load(args.config)
        root = args.output_root.expanduser().resolve() if args.output_root else paths.output
        verify_output_root(root, paths, args.fx_root)
        with run_lock(root):
            claim(root)
            claimed = True
            try:
                report = execute(args, paths, protocol, root)
            except Exception as error:
                event(root / "logs/command_errors.jsonl", command=args.command,
                      error_type=type(error).__name__, error=str(error))
                raise
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["status"] == "blocked" or any(
            value["status"] == "failed" for value in report.get("fx", {}).values()
        ):
            return 1
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        if claimed:
            print(f"See {root / 'logs'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
