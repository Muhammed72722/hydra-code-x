from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json

@dataclass
class CodeRecord:
    record_id: str
    text: str
    language: str
    license_spdx: str
    repo_id: str | None = None
    path: str | None = None
    commit_sha: str | None = None
    source_uri: str | None = None
    source_date: str | None = None
    quality_score: float = 0.0
    synthetic: bool = False
    testable: bool = False
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

def stable_record_id(source: str, path: str | None, content: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode('utf-8', errors='ignore')); h.update(b'\0')
    h.update((path or '').encode('utf-8', errors='ignore')); h.update(b'\0')
    h.update(content.encode('utf-8', errors='ignore'))
    return h.hexdigest()[:32]

def write_jsonl(records: list[CodeRecord], output: str | Path) -> None:
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for record in records: f.write(record.to_json() + '\n')
