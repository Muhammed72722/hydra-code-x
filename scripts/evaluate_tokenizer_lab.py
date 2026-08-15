#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from dataclasses import dataclass, asdict
import sentencepiece as spm

IDENT_RE = re.compile(r'(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*')
OP_RE = re.compile(r'==|!=|<=|>=|->|=>|\+\+|--|&&|\|\||\+=|-=|\*=|/=|\*\*|//|<<|>>|[+\-*/%=<>!&|^~?:.,;(){}\[\]]')

@dataclass
class Metrics:
    vocab_size: int
    docs: int
    total_bytes: int
    total_tokens: int
    tokens_per_byte: float
    avg_bytes_per_token: float
    identifier_fragmentation: float
    operator_fragmentation: float
    fim_tokens_per_byte: float
    p95_sequence_tokens: float
    score: float
    model: str


def pctile(vals, p):
    if not vals: return 0.0
    vals = sorted(vals)
    k = (len(vals)-1)*p/100
    f, c = math.floor(k), math.ceil(k)
    if f == c: return float(vals[int(k)])
    return vals[f] + (vals[c]-vals[f])*(k-f)


def encode(sp, text):
    return sp.encode(text, out_type=int)


def frag_for_tokens(sp, spans):
    if not spans: return 0.0
    pieces = [len(encode(sp, s)) for s in spans]
    return sum(pieces)/len(pieces)


def fim_transform(text: str):
    if len(text) < 32: return None
    a = len(text)//3
    b = len(text)*2//3
    prefix, middle, suffix = text[:a], text[a:b], text[b:]
    seq = '<|fim_prefix|>' + prefix + '<|fim_suffix|>' + suffix + '<|fim_middle|>' + middle
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--output', default='artifacts/tokenizers/metrics.json')
    args = ap.parse_args()

    texts=[]
    for fp in args.input:
        p=Path(fp)
        if p.is_file():
            texts.append(p.read_text(encoding='utf-8', errors='replace'))
        elif p.is_dir():
            for f in p.rglob('*'):
                if f.is_file() and f.stat().st_size <= 2_000_000:
                    try: texts.append(f.read_text(encoding='utf-8', errors='replace'))
                    except Exception: pass
    candidates=json.loads(Path(args.candidates).read_text())
    results=[]
    for cand in candidates:
        sp=spm.SentencePieceProcessor(model_file=cand['model'])
        total_bytes=total_tokens=0
        ident_frags=[]; op_frags=[]; fim_vals=[]; seq_lens=[]
        for text in texts:
            if not text.strip(): continue
            ids=encode(sp,text)
            total_bytes += len(text.encode('utf-8'))
            total_tokens += len(ids)
            seq_lens.append(len(ids))
            ident_frags.append(frag_for_tokens(sp, IDENT_RE.findall(text)))
            op_frags.append(frag_for_tokens(sp, OP_RE.findall(text)))
            fim=fim_transform(text)
            if fim:
                fim_vals.append(len(encode(sp, fim)) / max(1, len(fim.encode('utf-8'))))
        tpb=total_tokens/max(1,total_bytes)
        ident=sum(ident_frags)/max(1,len(ident_frags))
        op=sum(op_frags)/max(1,len(op_frags))
        fim=sum(fim_vals)/max(1,len(fim_vals))
        p95=pctile(seq_lens,95)
        # Lower is better for all four metrics; score normalizes relative to the candidate set later.
        results.append(Metrics(cand['vocab_size'],len(texts),total_bytes,total_tokens,tpb,1/max(tpb,1e-9),ident,op,fim,p95,0.0,cand['model']))
    # Relative geometric utility: favor low token density, low fragmentation and useful FIM packing.
    cols=['tokens_per_byte','identifier_fragmentation','operator_fragmentation','fim_tokens_per_byte','p95_sequence_tokens']
    mins={c:min(getattr(r,c) for r in results) for c in cols}
    maxs={c:max(getattr(r,c) for r in results) for c in cols}
    for r in results:
        vals=[]
        for c in cols:
            x=getattr(r,c); lo,hi=mins[c],maxs[c]
            norm=1.0 if hi==lo else (hi-x)/(hi-lo)
            vals.append(norm)
        r.score = 0.30*vals[0] + 0.20*vals[1] + 0.15*vals[2] + 0.20*vals[3] + 0.15*vals[4]
    results.sort(key=lambda r:r.score, reverse=True)
    payload={'ranking':[asdict(r) for r in results], 'selection_rule':'relative normalized score; final choice requires small-model ablation'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))

if __name__ == '__main__': main()
