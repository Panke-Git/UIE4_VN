"""Run frozen inference with per-sample receipts for safe restart."""

from itertools import islice
from pathlib import Path

from PIL import Image

from .dataset import read_input
from .storage import atomic_file, digest, event, read_json, seal_json, sha256_file, verify_hash


class ImageStage:
    def __init__(self, root, kind, method, samples, identity, mode, *, read_only=False):
        self.root = Path(root)
        self.kind, self.method, self.mode = kind, method, mode
        self.samples = samples
        self.images = self.root / kind / method
        self.audit = self.root / "audit/stages" / kind / method
        self.log = self.root / "logs" / f"{kind}_{method}.jsonl"
        self.read_only = read_only
        self.identity = {**identity, "schema": 1, "kind": kind, "method": method, "mode": mode,
                         "sample_ids": [sample.sample_id for sample in samples]}
        seal = self.audit / "identity.json"
        if read_only:
            if read_json(seal) != self.identity or not self.images.is_dir():
                raise ValueError(f"inconsistent saved stage: {self.audit}")
        else:
            if not seal.exists() and self.images.exists() and any(self.images.iterdir()):
                raise ValueError(f"cannot claim an existing image directory without a receipt: {self.images}")
            seal_json(seal, self.identity)
            self.images.mkdir(parents=True, exist_ok=True)
        self.identity_sha256 = digest(self.identity)
        expected = {f"{sample.sample_id}.png" for sample in samples}
        extra = sorted(p.name for p in self.images.glob("*.png") if p.name not in expected)
        if extra:
            raise ValueError(f"unexpected outputs in {self.images}: {extra[:6]}")

    def image_path(self, sample):
        return self.images / f"{sample.sample_id}.png"

    def receipt_path(self, sample):
        return self.audit / "samples" / f"{sample.sample_id}.json"

    def completed(self, sample, inputs=None):
        receipt = self.receipt_path(sample)
        if not receipt.is_file():
            return None
        value = read_json(receipt)
        if (
            value["stage_sha256"] != self.identity_sha256
            or value["sample_id"] != sample.sample_id
            or value["size"] != list(sample.size)
            or (inputs is not None and value["inputs"] != inputs)
        ):
            raise ValueError(f"saved output has different inputs or stage identity: {receipt}")
        verify_hash(self.image_path(sample), value["sha256"])
        return value["sha256"]

    def save(self, sample, image, inputs):
        if self.read_only:
            raise ValueError(f"cannot write a reference stage: {self.audit}")
        if image.mode != self.mode or image.size != sample.size:
            raise ValueError(f"{sample.sample_id}: expected {self.mode} PNG at {sample.size}, "
                             f"got {image.mode} at {image.size}")
        path = self.image_path(sample)
        with atomic_file(path) as stream:
            image.save(stream, format="PNG")
        seal_json(self.receipt_path(sample), {
            "stage_sha256": self.identity_sha256, "sample_id": sample.sample_id,
            "size": list(sample.size), "inputs": inputs, "sha256": sha256_file(path),
        })

    def complete(self):
        digests = []
        for sample in self.samples:
            checksum = self.completed(sample)
            if checksum is None:
                raise ValueError(f"missing {self.method} {self.kind} output for {sample.sample_id}")
            digests.append([sample.sample_id, checksum])
        result = {"stage_sha256": self.identity_sha256, "count": len(digests),
                  "image_set_sha256": digest(digests)}
        receipt = self.audit / "complete.json"
        if self.read_only:
            if read_json(receipt) != result:
                raise ValueError(f"saved completion receipt changed: {receipt}")
        else:
            seal_json(receipt, result)
        return result

    def load(self, sample):
        if self.completed(sample) is None:
            raise ValueError(f"{self.method}: missing output for {sample.sample_id}")
        with Image.open(self.image_path(sample)) as image:
            if image.format != "PNG" or image.mode != self.mode or image.size != sample.size:
                raise ValueError(f"{self.method}: bad PNG encoding or dimensions for {sample.sample_id}")
            return image.copy()


def _groups(samples, batch_size, match_geometry):
    iterator = iter(samples)
    while batch := list(islice(iterator, batch_size)):
        if not match_geometry:
            yield batch
            continue
        by_size = {}
        for sample in batch:
            by_size.setdefault(sample.size, []).append(sample)
        yield from by_size.values()


def _run(stage, runner, batch_size, input_hashes, load_input, match_geometry):
    failed, written, reused = 0, 0, 0
    for batch in _groups(stage.samples, batch_size, match_geometry):
        pending = {}
        for sample in batch:
            # A corrupt saved receipt is a run-integrity error, not a failed inference.
            identity = input_hashes(sample)
            if stage.completed(sample, identity) is not None:
                reused += 1
            else:
                pending[sample.sample_id] = identity
        if not pending:
            continue
        try:
            # Preserve the original batch shape if a crash happened during publication.
            inputs = [load_input(sample) for sample in batch]
            outputs = runner(inputs)
            if len(outputs) != len(batch):
                raise ValueError("inference returned a different number of images")
        except (OSError, ValueError, RuntimeError) as error:
            for identifier in pending:
                event(stage.log, sample_id=identifier, status="failed",
                      error_type=type(error).__name__, error=str(error), actual_batch=len(batch))
            failed += len(pending)
            continue
        for sample, output in zip(batch, outputs, strict=True):
            if sample.sample_id not in pending:
                continue
            try:
                stage.save(sample, output, pending[sample.sample_id])
            except (OSError, ValueError, RuntimeError) as error:
                event(stage.log, sample_id=sample.sample_id, status="failed", error=str(error))
                failed += 1
                continue
            written += 1
            event(stage.log, sample_id=sample.sample_id, status="ok")
    event(stage.log, status="stage_end", written=written, reused=reused, failed=failed)
    if failed:
        raise RuntimeError(f"{stage.kind}/{stage.method}: {failed} failed; see {stage.log}; "
                           "successful samples can be resumed with the same settings")
    return stage.complete()


def enhance(stage, runner, data_root, batch_size, *, with_sample_ids=False):
    def load_input(sample):
        image = read_input(data_root, sample, "rgb")
        return (sample.sample_id, image) if with_sample_ids else image

    return _run(stage, runner, batch_size, lambda sample: {"rgb": sample.hashes["rgb"]},
                load_input, match_geometry=True)


def saliency(stage, runner, data_root, batch_size, enhanced=None):
    def inputs(sample):
        # Raw and each enhanced condition read the same immutable depth file.
        rgb = (enhanced.load(sample) if enhanced is not None
               else read_input(data_root, sample, "rgb"))
        return rgb, read_input(data_root, sample, "depth")

    def input_hashes(sample):
        rgb_hash = enhanced.completed(sample) if enhanced is not None else sample.hashes["rgb"]
        if rgb_hash is None:
            raise ValueError(f"missing enhancement for {sample.sample_id}")
        return {"rgb": rgb_hash, "depth": sample.hashes["depth"]}

    return _run(stage, runner, batch_size, input_hashes, inputs, match_geometry=False)
