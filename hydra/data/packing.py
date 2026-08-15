from __future__ import annotations
from dataclasses import dataclass
@dataclass
class PackedSequence:
    input_ids:list[int]; labels:list[int]; loss_mask:list[int]

def pack_token_stream(sequences, max_length:int, pad_id:int):
    packed=[]; cur_ids=[]; cur_labels=[]; cur_mask=[]
    def flush():
        nonlocal cur_ids,cur_labels,cur_mask
        if not cur_ids:return
        pad=max_length-len(cur_ids)
        packed.append(PackedSequence(cur_ids+[pad_id]*pad,cur_labels+[-100]*pad,cur_mask+[0]*pad))
        cur_ids,cur_labels,cur_mask=[],[],[]
    for ids,mask in sequences:
        if not ids: continue
        mask=[1]*len(ids) if mask is None else mask
        start=0
        while start<len(ids):
            take=min(max_length-len(cur_ids),len(ids)-start)
            chunk=ids[start:start+take]; cm=mask[start:start+take]
            cur_ids.extend(chunk); cur_labels.extend(chunk); cur_mask.extend(cm); start+=take
            if len(cur_ids)==max_length: flush()
    flush(); return packed
