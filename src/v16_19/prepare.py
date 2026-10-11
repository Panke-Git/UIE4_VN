"""Explicitly freeze/verify the current LSUI19 or UIEB19 physical folders."""
import argparse
from pathlib import Path

from .protocol import COUNTS, freeze_dataset, verify_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze V16-19 Train/Val membership without moving data")
    parser.add_argument("--dataset", choices=tuple(COUNTS), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir) if args.output_dir else project / "split/v16_19" / args.dataset
    if not output.is_absolute():
        output = project / output
    if not output.resolve().is_relative_to((project / "split/v16_19").resolve()):
        raise ValueError("Preparation output must be inside split/v16_19; old manifests are protected")
    root = Path(args.data_root).resolve()
    protocol = verify_dataset(root, output) if args.verify_only else freeze_dataset(root, args.dataset, output)
    if protocol["dataset"] != args.dataset or protocol["counts"] != COUNTS[args.dataset]:
        raise ValueError("Frozen dataset/counts do not match the selected full dataset")
    print(f"{protocol['label']}: {protocol['counts']}; fingerprint={protocol['fingerprint']}")
    print(f"Verified protocol: {output}")


if __name__ == "__main__":
    main()
