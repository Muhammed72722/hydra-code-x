from hydra.model import HydraConfig, HydraModel
from hydra.utils.parameter_budget import count_parameters


def test_a0_parameter_budget_is_roughly_300m():
    cfg = HydraConfig()
    model = HydraModel(cfg)
    total, trainable = count_parameters(model)
    assert trainable == total
    # Clean A0 target: roughly 300M, with small implementation overhead.
    assert 295_000_000 <= total <= 320_000_000, total
