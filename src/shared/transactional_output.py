"""Transactional directory output for paper-evaluation artifacts.

Work is written to a hidden sibling staging directory.  The public result path
is installed only after the caller exits successfully.  With ``overwrite`` an
existing result remains intact until the replacement is complete.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator
import uuid


@contextmanager
def transactional_output_directory(
    output_dir: Path, *, overwrite: bool
) -> Iterator[Path]:
    final = output_dir.resolve()
    if final.exists():
        if not final.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {final}")
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {final}. "
                "Use --overwrite only when replacing this complete result."
            )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent)
    )
    committed = False
    try:
        yield staging
        if not any(staging.iterdir()):
            raise RuntimeError("Refusing to commit an empty evaluation result")

        backup: Path | None = None
        if final.exists():
            backup = final.parent / f".{final.name}.backup-{uuid.uuid4().hex}"
            os.replace(final, backup)
        try:
            os.replace(staging, final)
            committed = True
        except BaseException:
            if backup is not None and backup.exists() and not final.exists():
                os.replace(backup, final)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)

