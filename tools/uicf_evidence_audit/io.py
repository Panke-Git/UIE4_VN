"""Strict text readers, provenance receipts and non-overwriting audit outputs."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile

from . import REPORT_FILES, combined_status

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    # Preserve NaN/Inf tokens for explicit invalid-value findings, never serialize
    # them back as observed numeric results.
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda token: "__NONFINITE_JSON_"+token+"__")


class Audit:
    def __init__(self):
        self.checks = []
        self.sources = {}
        self.cache = {}
        self.extra_protected = []

    def check(self, section, name, status, detail, sources=(), severity="P1"):
        item = dict(section=section, check=name, status=status, detail=str(detail),
                    sources=[str(p) for p in sources], severity=severity)
        self.checks.append(item)
        return status

    def status(self, section):
        return combined_status(c["status"] for c in self.checks if c["section"] == section)

    def read(self, path, kind, *, section, required=True):
        path = Path(path).absolute()
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        source = self.sources.setdefault(key, dict(path=key, resolved_path=str(path.resolve()),
            sha256=None, fields=[], sample_count=None, definitions=[], status="UNVERIFIED"))
        if not path.is_file():
            if required:
                self.check(section, "missing_file", "UNVERIFIED", f"缺少 {path}", [path])
            self.cache[key] = None
            return None
        try:
            data = path.read_bytes()
            source.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), status="PASS")
            text = data.decode("utf-8-sig")
            if kind == "json":
                value = strict_json(text)
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                def contains_nonfinite(node):
                    if isinstance(node, dict):
                        return any(contains_nonfinite(v) for v in node.values())
                    if isinstance(node, list):
                        return any(contains_nonfinite(v) for v in node)
                    return isinstance(node, str) and node.startswith("__NONFINITE_JSON_")
                if contains_nonfinite(value):
                    self.check(section,"nonfinite_json","FAIL",f"{path} 含 NaN/Inf JSON 数值",[path],"P0")
                source["fields"] = sorted(value)
                source["sample_count"] = value.get("sample_count",value.get("successful_sample_count"))
            elif kind == "yaml":
                import yaml
                try:
                    value = yaml.safe_load(text)
                except yaml.YAMLError as error:
                    raise ValueError(str(error)) from error
                if not isinstance(value, dict):
                    raise ValueError("Expected a YAML mapping")
                source["fields"] = sorted(value)
            elif kind == "csv":
                reader = csv.DictReader(io.StringIO(text, newline=""))
                names = reader.fieldnames
                if not names or any(not name for name in names) or len(names) != len(set(names)):
                    raise ValueError("Missing/duplicate CSV header")
                value = list(reader)
                if any(None in row or any(v is None for v in row.values()) for row in value):
                    raise ValueError("CSV row/header width mismatch")
                source.update(fields=names, sample_count=len(value))
            elif kind == "tsv":
                value = list(csv.reader(io.StringIO(text), delimiter="\t"))
                if value and value[0] in (["sample_id", "input_path", "gt_path"], ["id", "input", "gt"]):
                    value = value[1:]
                if any(len(row) != 3 or any(not x.strip() for x in row) for row in value):
                    raise ValueError("TSV expects three nonempty columns")
                value = [[v.strip() for v in row] for row in value]
                source.update(fields=["sample_id", "input_relative", "gt_relative"], sample_count=len(value))
            else:
                value = text
            self.cache[key] = value
            return value
        except (OSError, UnicodeError, ValueError, ImportError) as error:
            source["status"] = "FAIL" if not isinstance(error, ImportError) else "UNVERIFIED"
            self.check(section, "unreadable_file", source["status"], f"{path}: {error}", [path], "P0")
            self.cache[key] = None
            return None

    def annotate(self, path, fields=(), definition=None):
        source = self.sources.get(str(Path(path).absolute()))
        if source:
            source["fields"] = sorted(set(source["fields"]) | set(fields))
            if definition and definition not in source["definitions"]:
                source["definitions"].append(definition)

    def unchanged(self):
        for source in self.sources.values():
            if source["sha256"] is not None:
                path = Path(source["path"])
                ok = path.is_file() and sha256(path) == source["sha256"]
                self.check("source_integrity", "read_source_unchanged", "PASS" if ok else "FAIL",
                           source["path"], [path], "P0")
                if not ok:
                    source["status"] = "FAIL"
        for source in self.sources.values():
            related = [c["status"] for c in self.checks if source["path"] in c["sources"]]
            source["verification_status"] = combined_status(related or [source["status"]])


def is_within(child, parent):
    return child == parent or parent in child.parents


def safe_output_parent(parent, protected=(), *, project_root=ROOT):
    parent = Path(parent).expanduser().absolute()
    if any(p.is_symlink() for p in (parent, *parent.parents)):
        raise ValueError("审计输出路径不得经过符号链接")
    resolved = parent.resolve()
    analysis = project_root.resolve() / "analysis"
    if not is_within(resolved, analysis):
        raise ValueError("审计输出必须在当前工程 analysis/ 内")
    for item in protected:
        item = Path(item).resolve()
        if is_within(resolved, item) or is_within(item, resolved):
            raise ValueError(f"审计输出不得位于输入目录内或覆盖输入目录：{item}")
    if parent.exists() and not parent.is_dir():
        raise ValueError("审计输出父路径不是目录")
    return resolved


def new_output(parent):
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output = parent / f"task2_{stamp}"
    output.mkdir(exist_ok=False)
    return output


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_zip(output):
    destination = output.with_suffix(".zip")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REPORT_FILES:
            path = output / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"审计交付文件缺失或为符号链接：{path}")
            archive.write(path, arcname=f"{output.name}/{name}")
    return destination
