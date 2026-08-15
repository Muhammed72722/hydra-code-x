#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra.data.source import DEFAULT_ALLOWED_SPDX, SourceRecord, load_source

COMMENT_PATTERNS = [r"^\s*#", r"^\s*//", r"^\s*/\\*", r"^\s*<!--", r"^\s*--"]

def quality(rec: SourceRecord) -> float:
    text = rec.text
    n = len(text.encode("utf-8", errors="replace"))
    if n < 80 or n > 1_000_000:
        return 0.0
    lines = text.splitlines()
    nonempty = [x for x in lines if x.strip()]
    if not nonempty:
        return 0.0
    alnum = sum(c.isalnum() for c in text)
    printable = sum(c.isprintable() or c in "\n\t\r" for c in text)
    weird = 1.0 - (printable / max(1, len(text)))
    avg_line = len(text) / max(1, len(lines))
    score = 1.0
    score *= min(1.0, math.log2(n + 2) / 14.0)
    score *= max(0.0, 1.0 - min(0.8, weird * 4.0))
    score *= min(1.0, (alnum / max(1, len(text))) * 2.5)
    score *= min(1.0, avg_line / 240.0 + 0.25)
    return max(0.0, min(1.0, score))


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def near_signature(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?)\b", "<NUM>", text)
    text = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", "<ID>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:16384]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", default="artifacts/corpus/train.jsonl")
    ap.add_argument("--source-name", default="local")
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--min-quality", type=float, default=0.35)
    ap.add_argument("--min-bytes", type=int, default=80)
    ap.add_argument("--max-bytes", type=int, default=1_000_000)
    ap.add_argument("--allow-license", nargs="*", default=sorted(DEFAULT_ALLOWED_SPDX))
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()

    langs = {x.lower() for x in args.languages} if args.languages else None
    allowed = {x.upper() for x in args.allow_license}
    exact_seen: set[str] = set()
    near_seen: set[str] = set()
    kept = []
    stats = {"seen": 0, "kept": 0, "license_reject": 0, "quality_reject": 0, "exact_dup": 0, "near_dup": 0, "language_reject": 0}

    for inp in args.input:
        for rec in load_source(inp, args.source_name):
            stats["seen"] += 1
            if langs and rec.language.lower() not in langs:
                stats["language_reject"] += 1
                continue
            if rec.license_spdx.upper() not in allowed:
                stats["license_reject"] += 1
                continue
            if not (args.min_bytes <= rec.bytes <= args.max_bytes):
                stats["quality_reject"] += 1
                continue
            rec.quality_score = quality(rec)
            if rec.quality_score < args.min_quality:
                stats["quality_reject"] += 1
                continue
            if rec.sha256 in exact_seen:
                stats["exact_dup"] += 1
                continue
            exact_seen.add(rec.sha256)
            sig = near_signature(rec.text)
            if sig in near_seen:
                stats["near_dup"] += 1
                continue
            near_seen.add(sig)
            kept.append(rec)
            stats["kept"] += 1
            if args.max_records and len(kept) >= args.max_records:
                break
        if args.max_records and len(kept) >= args.max_records:
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    with out.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    report = out.with_suffix(".report.json")
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
