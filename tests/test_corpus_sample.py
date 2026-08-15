from pathlib import Path
import json
from hydra.data.corpus_sample import build_tokenizer_corpus

def test_corpus_sample_is_deterministic(tmp_path):
    manifest=tmp_path/"m.jsonl"
    rows=[]
    for i, lang in enumerate(["python","rust","go","python"]):
        rows.append({"language":lang,"quality_score":0.9,"text":f"def f_{i}(x):\n    return x + {i}\n"})
    manifest.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    a=tmp_path/"a.txt"; b=tmp_path/"b.txt"
    ra=build_tokenizer_corpus(str(manifest), str(a), max_records=4, seed=99)
    rb=build_tokenizer_corpus(str(manifest), str(b), max_records=4, seed=99)
    assert a.read_text()==b.read_text()
    assert ra["records"]==rb["records"]
    assert ra["languages"]==rb["languages"]
    assert ra["records"]==4
