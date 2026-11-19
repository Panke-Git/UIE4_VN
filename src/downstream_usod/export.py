"""Package the compact result summaries and an internal checksum manifest."""

import argparse
import io
from pathlib import Path
import tarfile

from .storage import atomic_file, run_lock, sha256_file


def export_results(root, archive, phases, include_figures=False):
    root = Path(root).resolve(strict=True)
    archive = Path(archive)
    files = []
    for phase in dict.fromkeys(phases):
        if phase not in ("p0", "p1", "p2"):
            raise ValueError(f"unknown experiment phase: {phase}")
        relatives = [
            f"reports/{phase}_summary.json", f"reports/{phase}_summary.md",
            f"metrics/{phase}/overall.csv", f"metrics/{phase}/paired_bootstrap.csv",
            f"metrics/{phase}/paired_bootstrap.json",
        ]
        if include_figures:
            relatives += [f"visualizations/{phase}/selected_samples.csv",
                          f"visualizations/{phase}/saliency_examples.png"]
        for name in relatives:
            path = root / name
            if not path.is_file() or root not in path.resolve().parents:
                raise FileNotFoundError(f"requested result is missing: {path}")
            files.append(path)
    with atomic_file(archive, exclusive=True) as stream, tarfile.open(fileobj=stream, mode="w:gz") as tar:
        sums = []
        for path in files:
            name = path.relative_to(root).as_posix()
            tar.add(path, arcname=name, recursive=False)
            sums.append(f"{sha256_file(path)}  {name}")
        data = ("\n".join(sums) + "\n").encode()
        item = tarfile.TarInfo("SHA256SUMS")
        item.size = len(data)
        tar.addfile(item, io.BytesIO(data))
    return archive


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-root", type=Path, required=True)
    cli.add_argument("--archive", type=Path, required=True)
    cli.add_argument("--phase", action="append", choices=("p0", "p1", "p2"))
    cli.add_argument("--include-figures", action="store_true")
    args = cli.parse_args()
    if not (args.output_root / "audit" / "run.json").is_file():
        cli.error("expected an existing USOD run")
    with run_lock(args.output_root):
        print(export_results(args.output_root, args.archive, args.phase or ["p0"], args.include_figures))


if __name__ == "__main__":
    main()
