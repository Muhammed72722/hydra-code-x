from pathlib import Path
import yaml

def test_experiment_specs_are_fixed_budget():
    tok = yaml.safe_load((Path(__file__).parents[1] / "experiments/tokenizer_ablation.yaml").read_text())
    a0 = yaml.safe_load((Path(__file__).parents[1] / "experiments/a0_baseline.yaml").read_text())
    assert tok["training"]["tokens"] == a0["fixed"]["training_tokens"]
    assert tok["training"]["precision"] == a0["fixed"]["precision"]
    assert tok["model"]["context"] == a0["fixed"]["context"]
    assert tok["candidates"] == [32768, 49152, 65536]
    assert a0["models"] == ["hydra_300m", "dense_transformer_300m"]
