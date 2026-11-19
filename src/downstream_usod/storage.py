"""Publication and resume primitives shared by the experiment stages."""

from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


@contextmanager
def atomic_file(path, mode="wb", exclusive=False):
    """Publish a complete same-filesystem artifact after its writer succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if "b" in mode else {"encoding": "utf-8", "newline": ""}
    with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, prefix=".usod-",
                                     delete=False, **options) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            if exclusive:
                os.link(temporary, path)
            else:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path) as stream:
        stream.write(json_bytes(value))


def seal_json(path, value):
    """Keep published identities and receipts immutable across resumes."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"existing artifact has a different identity: {path}; use a new run")
        return
    with atomic_file(path, exclusive=True) as stream:
        stream.write(json_bytes(value))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with atomic_file(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, content):
    with atomic_file(path, "w") as stream:
        stream.write(content)


def event(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        stream.write(json.dumps({"time": stamp, **fields}, ensure_ascii=False) + "\n")


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".usod.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another USOD command holds the run lock: {root}") from error
        yield


def assert_separate_output(root, inputs):
    """Keep outputs away from all model, data and external source trees."""
    root = Path(root).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if root == source or root in source.parents or source in root.parents:
            raise ValueError(f"output root overlaps an input asset: {root} and {source}")
