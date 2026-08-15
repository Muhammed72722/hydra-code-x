from torch import nn
from .norm import RMSNorm
from .ffn import SwiGLU
from hydra.ssm import HydraMamba3Reference
from hydra.attention import LocalGQA


class HydraBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention = layer_idx in config.attention_layers
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)

        if self.is_attention:
            self.mixer = LocalGQA(
                config.d_model,
                config.q_heads,
                config.kv_heads,
                config.attention_window,
                config.rope_theta,
            )
        else:
            self.mixer = HydraMamba3Reference(
                d_model=config.d_model,
                d_state=config.mamba_state,
                expand=config.mamba_expand,
                headdim=config.mamba_headdim,
                rope_fraction=config.mamba_rope_fraction,
                dt_min=config.mamba_dt_min,
                dt_max=config.mamba_dt_max,
                dt_init_floor=config.mamba_dt_init_floor,
                A_floor=config.mamba_A_floor,
                is_outproj_norm=config.mamba_outproj_norm,
            )

        self.ffn = SwiGLU(config.d_model, config.ffn_dim)
        self.gate = nn.Linear(config.d_model, 1, bias=True)

    def forward(self, x, memory=None, state=None):
        h = self.norm1(x)
        mixer_out = self.mixer(h) if self.is_attention else self.mixer(h, state=state, return_state=True)
        if self.is_attention:
            y = mixer_out
            next_state = state
        else:
            y, next_state = mixer_out

        gate = 1.0 + 0.5 * self.gate(h).tanh()
        x = x + gate * y

        h = self.norm2(x)
        if memory is not None:
            h = h + memory(h)
        x = x + self.ffn(h)
        return x, next_state
