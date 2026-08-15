from pathlib import Path
import json

def test_scripts_exist():
    root=Path(__file__).parents[1]
    assert (root/'scripts/train_tokenizer_lab.py').exists()
    assert (root/'scripts/evaluate_tokenizer_lab.py').exists()

def test_fim_markers():
    text='def add(a, b):\n    return a + b\n'
    prefix=text[:10]; middle=text[10:20]; suffix=text[20:]
    seq='<|fim_prefix|>'+prefix+'<|fim_suffix|>'+suffix+'<|fim_middle|>'+middle
    assert '<|fim_prefix|>' in seq and '<|fim_middle|>' in seq
