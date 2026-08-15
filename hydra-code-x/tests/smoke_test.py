import torch
from hydra.model import HydraConfig, HydraModel
from hydra.utils.parameter_budget import count_parameters


def test_forward_backward():
    cfg = HydraConfig(
        vocab_size=1024,
        d_model=256,
        n_layers=4,
        max_seq_len=128,
        ffn_dim=512,
        attention_layers=(1, 3),
        q_heads=4,
        kv_heads=2,
        attention_window=32,
        mamba_state=16,
        mamba_headdim=64,
        memory_slots=8,
        memory_topk=4,
    )
    model = HydraModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(ids, ids)
    assert out["logits"].shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    total, trainable = count_parameters(model)
    assert total == trainable
