from __future__ import annotations
import json, hashlib, random
from collections import defaultdict
from pathlib import Path


def _load_manifest(path: str):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line: continue
        yield json.loads(line)


def build_tokenizer_corpus(manifest: str, output: str, max_records: int = 10000, seed: int = 1337, min_quality: float = 0.55):
    rng=random.Random(seed)
    groups=defaultdict(list)
    for r in _load_manifest(manifest):
        text=r.get("text", "")
        if not text or len(text)<8: continue
        if float(r.get("quality_score", 0.0)) < min_quality: continue
        lang=(r.get("language") or "unknown").lower()
        groups[lang].append(r)
    langs=sorted(groups)
    for lang in langs:
        rng.shuffle(groups[lang])
    selected=[]
    # Round-robin by language to avoid a Python/JS-heavy sample.
    while len(selected)<max_records and langs:
        progressed=False
        for lang in langs:
            if groups[lang]:
                selected.append(groups[lang].pop())
                progressed=True
                if len(selected)>=max_records: break
        if not progressed: break
    rng.shuffle(selected)
    out=Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    seen=set(); kept=0
    with out.open("w", encoding="utf-8") as f:
        for r in selected:
            text=r.get("text", "")
            h=hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen: continue
            seen.add(h)
            f.write(text.replace("\x00", " ").strip()+"\n")
            kept+=1
    return {"output": str(out), "records": kept, "languages": {k: sum(1 for r in selected if (r.get("language") or "unknown").lower()==k) for k in langs}}
