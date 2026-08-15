#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse,json
from hydra.data.tokenizer import ByteFallbackTokenizer,measure_token_stats,train_bpe_tokenizer
def load_texts(manifest:Path,limit:int):
    out=[]
    with manifest.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line)['text'])
            if len(out)>=limit: break
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('manifest'); ap.add_argument('--limit',type=int,default=10000); ap.add_argument('--bpe-dir'); a=ap.parse_args()
    texts=load_texts(Path(a.manifest),a.limit); print(json.dumps({'byte_fallback':measure_token_stats(texts,ByteFallbackTokenizer()).__dict__},indent=2))
    if a.bpe_dir:
        d=Path(a.bpe_dir); d.mkdir(parents=True,exist_ok=True); corpus=d/'_tokenizer_corpus.txt'; corpus.write_text('\n\n'.join(texts),encoding='utf-8')
        for vocab in (32768,49152,65536): train_bpe_tokenizer([str(corpus)],vocab,str(d))
if __name__=='__main__': main()
