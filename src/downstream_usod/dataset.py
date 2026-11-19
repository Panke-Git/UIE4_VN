"""Freeze the author-published test membership and original image geometry."""

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import subprocess

from PIL import Image

from .config import OFFICIAL_PREDICTIONS, TC_COMMIT, TEST_COUNT, TEST_IDS_SHA256
from .storage import digest

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ROLES = ("rgb", "depth", "gt")
SAMPLE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    width: int
    height: int
    files: dict
    hashes: dict

    @property
    def size(self):
        return self.width, self.height


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def official_ids(repo):
    if git(repo, "rev-parse", "HEAD") != TC_COMMIT:
        raise ValueError(f"TC-USOD checkout must be at {TC_COMMIT}: {repo}")
    entries = git(repo, "ls-tree", "-r", "--name-only", TC_COMMIT, "--", OFFICIAL_PREDICTIONS)
    filenames = [PurePosixPath(name) for name in entries.splitlines()]
    ids = sorted(entry.stem for entry in filenames if
                 entry.parent.as_posix() == OFFICIAL_PREDICTIONS and entry.suffix == ".png")
    identity = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    if len(set(ids)) != TEST_COUNT or len(ids) != TEST_COUNT or identity != TEST_IDS_SHA256:
        raise ValueError("the authors' published prediction IDs do not match the frozen Test split")
    return ids


def under(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"expected a relative path inside the test root: {relative!r}")
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        raise ValueError(f"test input points outside its root: {relative}")
    return resolved


def _index(root, relative):
    directory = under(root, relative)
    if not directory.is_dir():
        raise FileNotFoundError(f"required test directory is missing: {directory}")
    result, folded = {}, set()
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or path.name.startswith("._"):
            continue
        if not path.is_file():
            raise ValueError(f"invalid test image: {path}")
        if not SAMPLE_ID.fullmatch(path.stem) or path.stem.casefold() in folded:
            raise ValueError(f"invalid or duplicate sample ID: {path}")
        folded.add(path.stem.casefold())
        result[path.stem] = path.relative_to(root).as_posix()
    return result


def _decode(content, mode=None):
    with Image.open(BytesIO(content)) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("expected a single-frame test image")
        opened.load()
        return opened.convert(mode) if mode else opened.copy()


def scan_test(root, directories, ids, *, official=True):
    root = Path(root).resolve()
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids) or any(not SAMPLE_ID.fullmatch(i) for i in ids):
        raise ValueError("expected a non-empty unique list of safe sample IDs")
    if set(directories) != set(ROLES) or len({under(root, d) for d in directories.values()}) != 3:
        raise ValueError("RGB, depth and GT must use three distinct test directories")
    if official:
        identity = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
        if len(ids) != TEST_COUNT or identity != TEST_IDS_SHA256:
            raise ValueError("official test requires the full author-published membership in order")
    indexes = {role: _index(root, directories[role]) for role in ROLES}
    for role, index in indexes.items():
        missing = sorted(set(ids) - index.keys())
        extra = sorted(index.keys() - set(ids))
        if missing or extra:
            raise ValueError(f"{role} Test membership differs: missing {len(missing)} {missing[:6]}, "
                             f"extra {len(extra)} {extra[:6]}")
    samples = []
    for identifier in ids:
        paths = {role: indexes[role][identifier] for role in ROLES}
        hashes, sizes = {}, {}
        for role, relative in paths.items():
            content = under(root, relative).read_bytes()
            hashes[role] = hashlib.sha256(content).hexdigest()
            sizes[role] = _decode(content).size
        if len(set(sizes.values())) != 1 or min(sizes["rgb"]) < 2:
            raise ValueError(f"{identifier}: RGB, depth and GT must align at native size: {sizes}")
        samples.append(Sample(identifier, *sizes["rgb"], paths, hashes))
    return samples


def manifest(samples, *, official):
    body = {"schema": 1, "dataset": "USOD10K", "split": "official_test" if official else "synthetic",
            "source_commit": TC_COMMIT if official else None,
            "test_ids_sha256": TEST_IDS_SHA256 if official else None,
            "count": len(samples), "samples": [asdict(sample) for sample in samples]}
    return {**body, "sha256": digest(body)}


def read_manifest(value, expected_ids):
    body = {key: item for key, item in value.items() if key != "sha256"}
    if digest(body) != value.get("sha256") or body.get("schema") != 1:
        raise ValueError("invalid test manifest digest or schema")
    if (body.get("dataset"), body.get("split"), body.get("source_commit"),
            body.get("test_ids_sha256")) != ("USOD10K", "official_test", TC_COMMIT, TEST_IDS_SHA256):
        raise ValueError("production commands require an official USOD10K test manifest")
    samples = [Sample(**sample) for sample in body["samples"]]
    if [sample.sample_id for sample in samples] != expected_ids or body["count"] != len(samples):
        raise ValueError("saved manifest membership differs from the official split")
    return samples


def read_input(root, sample, role):
    """Read and hash exactly the bytes that will be decoded for this modality."""
    path = under(Path(root).resolve(), sample.files[role])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != sample.hashes[role]:
        raise ValueError(f"{sample.sample_id}: {role} changed after manifest freeze")
    image = _decode(content, "L" if role == "gt" else "RGB")
    if image.size != sample.size:
        raise ValueError(f"{sample.sample_id}: decoded {role} size differs from manifest")
    return image
