#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydra.data.corpus_sample import build_tokenizer_corpus

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", default="artifacts/tokenizer_corpus.txt")
    ap.add_argument("--max-records", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min-quality", type=float, default=0.55)
    args=ap.parse_args()
    print(json.dumps(build_tokenizer_corpus(args.manifest,args.output,args.max_records,args.seed,args.min_quality), indent=2))
