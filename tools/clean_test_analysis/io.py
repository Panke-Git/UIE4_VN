"""Strict CSV/TSV input parsing and canonical test-sample alignment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class AnalysisInputError(ValueError):
    """Raised when post-hoc inputs cannot be aligned without dropping data."""


@dataclass(frozen=True)
class TestSample:
    __test__ = False

    sample_id: str
    input_path: str
    gt_path: str
    order: int


@dataclass(frozen=True)
class ModelMetric:
    sample_id: str
    psnr: float
    ssim: float


@dataclass(frozen=True)
class DifficultyMetric:
    sample_id: str
    psnr_256: float
    ssim_256: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisInputError(f"Expected JSON object: {path}")
    return value


def read_csv_rows(path: Path, *, allow_empty: bool = False) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV is unavailable: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise AnalysisInputError(f"CSV has no header: {path}")
        fieldnames = [field.strip() for field in reader.fieldnames]
        rows = [
            {str(key).strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    if not rows and not allow_empty:
        raise AnalysisInputError(f"CSV has no data rows: {path}")
    return rows, fieldnames


def resolve_field(
    fieldnames: Sequence[str],
    candidates: Sequence[str],
    *,
    required: bool = True,
    label: str = "field",
) -> str | None:
    normalized: dict[str, list[str]] = {}
    for field in fieldnames:
        normalized.setdefault(field.strip().lower(), []).append(field)
    for candidate in candidates:
        matches = normalized.get(candidate.lower(), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AnalysisInputError(f"Ambiguous {label} columns: {matches}")
    if required:
        raise AnalysisInputError(
            f"Missing {label}; expected one of {list(candidates)}, found {list(fieldnames)}"
        )
    return None


def parse_float(value: str, *, context: str, allow_blank: bool = False) -> float | None:
    text = value.strip()
    if not text and allow_blank:
        return None
    try:
        number = float(text)
    except ValueError as error:
        raise AnalysisInputError(f"Invalid numeric value {value!r} at {context}") from error
    if math.isnan(number):
        if allow_blank:
            return None
        raise AnalysisInputError(f"NaN is not allowed at {context}")
    return number


def read_test_manifest(path: Path, expected_count: int | None = 428) -> dict[str, TestSample]:
    if not path.is_file():
        raise FileNotFoundError(f"Test manifest is unavailable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        raise AnalysisInputError(f"Test manifest is empty: {path}")
    header = tuple(cell.strip().lower() for cell in rows[0])
    if header in {("sample_id", "input_path", "gt_path"), ("id", "input", "gt")}:
        rows = rows[1:]
    samples: dict[str, TestSample] = {}
    for order, row in enumerate(rows):
        if len(row) != 3 or any(not cell.strip() for cell in row):
            raise AnalysisInputError(f"{path}:{order + 1}: expected three non-empty TSV fields")
        sample_id, input_path, gt_path = (cell.strip() for cell in row)
        if sample_id in samples:
            raise AnalysisInputError(f"Duplicate test sample_id {sample_id!r} in {path}")
        samples[sample_id] = TestSample(sample_id, input_path, gt_path, order)
    if expected_count is not None and len(samples) != expected_count:
        raise AnalysisInputError(
            f"Test manifest has {len(samples)} samples, expected {expected_count}: {path}"
        )
    return samples


def _unique_map(items: Iterable[tuple[str, str]], label: str) -> dict[str, str]:
    grouped: dict[str, set[str]] = {}
    for key, sample_id in items:
        grouped.setdefault(key, set()).add(sample_id)
    ambiguous = {key: values for key, values in grouped.items() if len(values) > 1}
    if ambiguous:
        examples = list(ambiguous.items())[:5]
        raise AnalysisInputError(f"Ambiguous manifest {label} mapping: {examples}")
    return {key: next(iter(values)) for key, values in grouped.items()}


def _filename_maps(samples: Mapping[str, TestSample]) -> dict[str, dict[str, str]]:
    return {
        "path": _unique_map(((sample.input_path, sample.sample_id) for sample in samples.values()), "path"),
        "basename": _unique_map(
            ((Path(sample.input_path).name, sample.sample_id) for sample in samples.values()),
            "basename",
        ),
        "stem": _unique_map(
            ((Path(sample.input_path).stem, sample.sample_id) for sample in samples.values()),
            "stem",
        ),
    }


def canonical_id_from_filename(filename: str, samples: Mapping[str, TestSample]) -> str:
    """Resolve only exact manifest path/basename/stem and explicit ``_enhanced`` suffix forms."""
    maps = _filename_maps(samples)
    normalized = filename.strip().replace("\\", "/")
    if normalized in maps["path"]:
        return maps["path"][normalized]
    basename = Path(normalized).name
    if basename in maps["basename"]:
        return maps["basename"][basename]
    stem = Path(basename).stem
    if stem.endswith("_enhanced"):
        stem = stem[: -len("_enhanced")]
    if stem in maps["stem"]:
        return maps["stem"][stem]
    raise AnalysisInputError(f"Filename {filename!r} has no exact canonical test-manifest mapping")


def read_model_metrics(
    path: Path,
    samples: Mapping[str, TestSample],
    *,
    model_label: str,
) -> tuple[dict[str, ModelMetric], list[str]]:
    rows, fields = read_csv_rows(path)
    sample_field = resolve_field(fields, ("sample_id", "id"), required=False, label="sample ID")
    filename_field = resolve_field(
        fields, ("filename", "file_name", "input_path", "image"), required=False, label="filename"
    )
    if sample_field is None and filename_field is None:
        raise AnalysisInputError(
            f"{model_label} metrics need sample_id or filename for canonical alignment: {path}"
        )
    psnr_field = resolve_field(fields, ("psnr",), label="per-image PSNR")
    ssim_field = resolve_field(fields, ("ssim",), label="per-image SSIM")
    metrics: dict[str, ModelMetric] = {}
    for line, row in enumerate(rows, start=2):
        raw_id = row[sample_field] if sample_field else ""
        if raw_id:
            sample_id = raw_id
            if sample_id not in samples:
                raise AnalysisInputError(
                    f"{model_label} row {line} has extra/unknown sample_id {sample_id!r}"
                )
        else:
            if filename_field is None or not row[filename_field]:
                raise AnalysisInputError(f"{model_label} row {line} lacks both sample_id and filename")
            sample_id = canonical_id_from_filename(row[filename_field], samples)
        if sample_id in metrics:
            raise AnalysisInputError(f"{model_label} has duplicate mapping for sample_id {sample_id!r}")
        metrics[sample_id] = ModelMetric(
            sample_id=sample_id,
            psnr=float(parse_float(row[psnr_field], context=f"{path}:{line}:{psnr_field}")),
            ssim=float(parse_float(row[ssim_field], context=f"{path}:{line}:{ssim_field}")),
        )
    enforce_exact_ids(metrics, samples, label=model_label)
    return metrics, fields


def read_difficulty_metrics(
    path: Path,
    samples: Mapping[str, TestSample],
) -> tuple[dict[str, DifficultyMetric], list[str]]:
    rows, fields = read_csv_rows(path)
    sample_field = resolve_field(fields, ("sample_id", "id"), required=False, label="sample ID")
    input_field = resolve_field(fields, ("input_path", "filename"), required=False, label="input path")
    if sample_field is None and input_field is None:
        raise AnalysisInputError("Diagnostic difficulty CSV needs sample_id or input_path")
    psnr_field = resolve_field(
        fields,
        ("psnr_256", "raw_input_gt_psnr_256", "input_gt_psnr_256"),
        label="raw Input-to-GT PSNR_256",
    )
    ssim_field = resolve_field(
        fields,
        ("ssim_256", "raw_input_gt_ssim_256", "input_gt_ssim_256"),
        label="raw Input-to-GT SSIM_256",
    )
    metrics: dict[str, DifficultyMetric] = {}
    for line, row in enumerate(rows, start=2):
        sample_id = row[sample_field] if sample_field else ""
        if not sample_id:
            if input_field is None or not row[input_field]:
                raise AnalysisInputError(f"Difficulty row {line} lacks canonical identity")
            sample_id = canonical_id_from_filename(row[input_field], samples)
        if sample_id not in samples:
            raise AnalysisInputError(f"Difficulty row {line} has unknown sample_id {sample_id!r}")
        if sample_id in metrics:
            raise AnalysisInputError(f"Difficulty CSV has duplicate sample_id {sample_id!r}")
        metrics[sample_id] = DifficultyMetric(
            sample_id,
            float(parse_float(row[psnr_field], context=f"{path}:{line}:{psnr_field}")),
            float(parse_float(row[ssim_field], context=f"{path}:{line}:{ssim_field}")),
        )
    enforce_exact_ids(metrics, samples, label="diagnostic difficulty")
    return metrics, fields


def enforce_exact_ids(
    observed: Mapping[str, Any], expected: Mapping[str, Any] | set[str], *, label: str
) -> None:
    expected_ids = set(expected)
    observed_ids = set(observed)
    missing = sorted(expected_ids - observed_ids)
    extra = sorted(observed_ids - expected_ids)
    if missing or extra or len(observed) != len(expected_ids):
        raise AnalysisInputError(
            f"{label} sample alignment failed: observed={len(observed_ids)} "
            f"expected={len(expected_ids)} missing={missing[:10]} extra={extra[:10]}"
        )


def validate_run_version(run_dir: Path, expected_version: str) -> None:
    info = read_json(run_dir / "run_info.json")
    version = info.get("version")
    if version is not None and version != expected_version:
        raise AnalysisInputError(
            f"Expected {expected_version} run, but {run_dir} records version={version!r}"
        )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
