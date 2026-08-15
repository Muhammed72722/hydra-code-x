import random
from hydra.data.dedup import exact_dedup
from hydra.data.fim import make_fim
from hydra.data.packing import pack_token_stream
from hydra.data.quality import score_code
from hydra.data.tokenizer import ByteFallbackTokenizer,measure_token_stats
def test_quality_and_dedup():
    code='def f(x):\n    return x + 1\n'; assert score_code(code).quality_score>.5; assert len(exact_dedup([{'text':code},{'text':code}]))==1
def test_fim_deterministic():
    a=make_fim('def f(x):\n'+'x += 1\n'*20,random.Random(1)); b=make_fim('def f(x):\n'+'x += 1\n'*20,random.Random(1)); assert a.text==b.text and len(a.loss_mask)==len(a.text)
def test_packing():
    out=pack_token_stream([([1,2,3],None),([4,5],[1,0])],4,0); assert len(out)==2 and out[0].input_ids==[1,2,3,4]
def test_tokenizer():
    tok=ByteFallbackTokenizer(); text="print('hello')"; assert tok.decode(tok.encode(text))==text; assert measure_token_stats([text],tok).total_tokens>0
