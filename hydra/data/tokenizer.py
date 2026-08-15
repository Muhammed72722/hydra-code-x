from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
SPECIAL_TOKENS=['<|pad|>','<|bos|>','<|eos|>','<|unk|>','<|fim_prefix|>','<|fim_suffix|>','<|fim_middle|>','<|fim_hole|>','<|fim_answer|>']
@dataclass
class TokenStats:
    total_bytes:int; total_tokens:int; tokens_per_byte:float; average_bytes_per_token:float
class ByteFallbackTokenizer:
    def __init__(self): self.special_to_id={t:i for i,t in enumerate(SPECIAL_TOKENS)}; self.byte_offset=len(SPECIAL_TOKENS); self.vocab_size=self.byte_offset+256
    def encode(self,text:str)->list[int]: return [self.byte_offset+b for b in text.encode('utf-8',errors='replace')]
    def decode(self,ids)->str:
        data=bytearray()
        for idx in ids:
            if idx<self.byte_offset: continue
            v=idx-self.byte_offset
            if 0<=v<256:data.append(v)
        return data.decode('utf-8',errors='replace')
    def save(self,path): Path(path).write_text(json.dumps({'type':'byte_fallback','special_tokens':SPECIAL_TOKENS,'byte_offset':self.byte_offset,'vocab_size':self.vocab_size},indent=2),encoding='utf-8')

def measure_token_stats(records, tokenizer)->TokenStats:
    total_bytes=total_tokens=0
    for text in records:
        total_bytes+=len(text.encode('utf-8')); total_tokens+=len(tokenizer.encode(text))
    if total_bytes==0:return TokenStats(0,0,0.0,0.0)
    return TokenStats(total_bytes,total_tokens,total_tokens/total_bytes,total_bytes/max(1,total_tokens))

def train_bpe_tokenizer(corpus_files:list[str], vocab_size:int, output_dir:str)->None:
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    except ImportError as exc:
        raise RuntimeError("Install hydra-code-x[tokenizer] for BPE training.") from exc
    tok=Tokenizer(models.BPE(unk_token='<|unk|>')); tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer=trainers.BpeTrainer(vocab_size=vocab_size,special_tokens=SPECIAL_TOKENS,show_progress=True,initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train(corpus_files,trainer=trainer); tok.decoder=decoders.ByteLevel()
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); tok.save(str(out/f'hydra-code-bpe-{vocab_size}.json'))
