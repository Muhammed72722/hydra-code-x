from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

EXT_TO_LANG = {
    ".py": "python", ".pyw": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".go": "go",
    ".rs": "rust", ".cpp": "c++", ".cc": "c++", ".cxx": "c++", ".c": "c",
    ".h": "c", ".hpp": "c++", ".cs": "c-sharp", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".sql": "sql",
    ".lua": "lua", ".r": "r", ".R": "r", ".dart": "dart", ".ex": "elixir",
    ".exs": "elixir", ".hs": "haskell", ".fs": "fsharp", ".fsx": "fsharp",
    ".vue": "vue", ".svelte": "svelte", ".m": "objective-c", ".mm": "objective-c++",
    ".pl": "perl", ".pm": "perl", ".jl": "julia", ".groovy": "groovy",
    ".vim": "vimscript", ".sol": "solidity", ".zig": "zig", ".asm": "assembly",
    ".make": "makefile", ".mk": "makefile", ".cmake": "cmake", ".dockerfile": "dockerfile",
}

DEFAULT_ALLOWED_SPDX = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib",
    "Unlicense", "CC0-1.0", "0BSD", "BSL-1.1", "MPL-2.0", "EPL-2.0",
}

@dataclass
class SourceRecord:
    text: str
    language: str
    path: str
    repository: str
    license_spdx: str
    source: str
    sha256: str
    bytes: int
    quality_score: float = 0.0


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def language_from_path(path: str) -> Optional[str]:
    p = Path(path)
    name = p.name.lower()
    if name in {"dockerfile"}:
        return "dockerfile"
    if name in {"makefile", "gnumakefile"}:
        return "makefile"
    return EXT_TO_LANG.get(p.suffix.lower())


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def iter_json(path: Path) -> Iterator[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        yield payload
    else:
        raise TypeError(f"Unsupported JSON root in {path}")


def iter_parquet(path: Path) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read parquet sources") from exc
    table = pq.read_table(path)
    cols = table.column_names
    for row in table.to_pylist():
        yield {k: row.get(k) for k in cols}


def iter_records(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl(path)
    elif suffix == ".json":
        yield from iter_json(path)
    elif suffix == ".parquet":
        yield from iter_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {path}")


def normalize_record(raw: dict, source_path: str, source_name: str) -> Optional[SourceRecord]:
    text = raw.get("content") or raw.get("text") or raw.get("code")
    if not isinstance(text, str) or not text.strip():
        return None
    path = str(raw.get("path") or raw.get("file_path") or "unknown")
    language = str(raw.get("language") or raw.get("lang") or language_from_path(path) or "unknown")
    license_spdx = str(raw.get("license_spdx") or raw.get("license") or "NOASSERTION")
    repository = str(raw.get("repository") or raw.get("repo_name") or raw.get("repo") or "unknown")
    return SourceRecord(
        text=text,
        language=language.lower(),
        path=path,
        repository=repository,
        license_spdx=license_spdx,
        source=source_name,
        sha256=sha256_text(text),
        bytes=len(text.encode("utf-8", errors="replace")),
    )


def load_source(path: str | Path, source_name: str = "local") -> Iterator[SourceRecord]:
    p = Path(path)
    if p.is_dir():
        for child in sorted(p.rglob("*")):
            if child.is_file() and child.suffix.lower() in {".jsonl", ".json", ".parquet"}:
                for raw in iter_records(child):
                    rec = normalize_record(raw, str(child), source_name)
                    if rec is not None:
                        yield rec
    else:
        for raw in iter_records(p):
            rec = normalize_record(raw, str(p), source_name)
            if rec is not None:
                yield rec


def write_jsonl(records: Iterable[SourceRecord], output: str | Path) -> int:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            count += 1
    return count
