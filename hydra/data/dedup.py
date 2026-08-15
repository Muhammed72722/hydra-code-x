from __future__ import annotations
import hashlib, re
from collections import defaultdict

def normalize_code(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace('\r\n','\n').replace('\r','\n')).strip()

def exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()

def near_signature(text: str, ngrams: int = 64) -> tuple[int, ...]:
    norm=normalize_code(text)
    if len(norm)<8: return (hash(norm),)
    grams=[norm[i:i+8] for i in range(0,len(norm)-7,8)]
    if len(grams)>ngrams:
        stride=max(1,len(grams)//ngrams); grams=grams[::stride][:ngrams]
    return tuple(sorted({hash(g) for g in grams}))

def exact_dedup(records: list[dict]) -> list[dict]:
    seen=set(); kept=[]
    for r in records:
        h=exact_hash(r['text'])
        if h not in seen: seen.add(h); kept.append(r)
    return kept

def bucket_by_signature(records: list[dict]) -> dict[int, list[dict]]:
    buckets=defaultdict(list)
    for r in records:
        sig=near_signature(r['text']); buckets[hash(sig[:8])].append(r)
    return buckets
