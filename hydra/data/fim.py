from __future__ import annotations
import random
from dataclasses import dataclass
@dataclass(frozen=True)
class FIMResult:
    text: str; loss_mask: list[int]; prefix_len: int; suffix_len: int; hole_len: int

def make_fim(text: str, rng: random.Random, min_hole: int=16) -> FIMResult:
    if len(text)<min_hole*3: return FIMResult(text,[1]*len(text),0,0,len(text))
    a=rng.randint(max(1,len(text)//5), max(2,len(text)//2))
    lo=min(a+min_hole,len(text)-min_hole); hi=max(lo,a+min_hole)
    b=rng.randint(lo,hi) if lo<=hi else lo
    prefix,hole,suffix=text[:a],text[a:b],text[b:]
    transformed=prefix+'<|fim_hole|>'+suffix+'<|fim_answer|>'+hole
    prefix_region=len(prefix)+len('<|fim_hole|>')+len(suffix)+len('<|fim_answer|>')
    mask=[0]*prefix_region+[1]*len(hole)
    return FIMResult(transformed,mask,len(prefix),len(suffix),len(hole))
