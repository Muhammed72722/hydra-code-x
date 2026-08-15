import torch
from hydra.model import HydraConfig
from hydra.ssm import HydraMamba3Reference


def test_mamba_streaming_state_shape_and_finite():
    cfg = HydraConfig(d_model=128, mamba_state=16, mamba_headdim=32)
    m = HydraMamba3Reference(
        d_model=cfg.d_model,
        d_state=cfg.mamba_state,
        expand=cfg.mamba_expand,
        headdim=cfg.mamba_headdim,
    )
    x = torch.randn(2, 17, cfg.d_model)
    y, state = m(x, return_state=True)
    assert y.shape == x.shape
    ssm, prev_x, prev_b = state
    assert ssm.shape == (2, cfg.mamba_heads, cfg.mamba_headdim, cfg.mamba_state)
    assert prev_x.shape == (2, cfg.mamba_heads, cfg.mamba_headdim)
    assert prev_b.shape == (2, cfg.mamba_heads, cfg.mamba_state)
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(t).all() for t in state)


def test_incremental_matches_full_reference_path():
    torch.manual_seed(0)
    m = HydraMamba3Reference(d_model=64, d_state=16, expand=2, headdim=32)
    x = torch.randn(1, 8, 64)

    full, _ = m(x, return_state=True)

    state = None
    parts = []
    for i in range(x.size(1)):
        y_i, state = m(x[:, i:i+1], state=state, return_state=True)
        parts.append(y_i)
    step = torch.cat(parts, dim=1)

    assert torch.allclose(full, step, atol=1e-5, rtol=1e-4)
