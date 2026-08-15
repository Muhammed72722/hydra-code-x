import json
from pathlib import Path

from hydra.data.source import language_from_path, load_source


def test_language_detection():
    assert language_from_path("foo.py") == "python"
    assert language_from_path("Dockerfile") == "dockerfile"
    assert language_from_path("x.rs") == "rust"


def test_jsonl_source(tmp_path: Path):
    p = tmp_path / "src.jsonl"
    p.write_text(json.dumps({
        "content": "def hello():\n    return 1\n",
        "path": "hello.py",
        "repository": "demo/repo",
        "license": "MIT",
    }) + "\n", encoding="utf-8")
    rows = list(load_source(p, "test"))
    assert len(rows) == 1
    assert rows[0].language == "python"
    assert rows[0].license_spdx == "MIT"
    assert len(rows[0].sha256) == 64
