#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

SPECIAL = [
    '<|pad|>','<|bos|>','<|eos|>','<|unk|>',
    '<|fim_prefix|>','<|fim_suffix|>','<|fim_middle|>',
    '<|fim_hole|>','<|fim_answer|>'
]


def build_with_sentencepiece(files: list[str], vocab_size: int, out_dir: Path, seed: int) -> Path:
    import sentencepiece as spm
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f'hydra-code-spm-{vocab_size}'
    spm.SentencePieceTrainer.train(
        input=','.join(files),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule_name='identity',
        user_defined_symbols=SPECIAL,
        split_digits=True,
        max_sentencepiece_length=16,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        random_seed=seed,
        hard_vocab_limit=False,
    )
    return prefix.with_suffix('.model')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--vocab-sizes', nargs='+', type=int, default=[32000,48000,65536])
    ap.add_argument('--output-dir', default='artifacts/tokenizers')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not __import__('importlib.util').util.find_spec('sentencepiece'):
        raise SystemExit('sentencepiece is required for tokenizer training.')
    results = []
    for size in args.vocab_sizes:
        model = build_with_sentencepiece(args.input, size, out, args.seed)
        results.append({'vocab_size': size, 'model': str(model)})
    (out / 'candidates.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
