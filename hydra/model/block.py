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
            self.mixer = LocalGQA(config.d_model, config.q_heads, config.kv_heads, config.attention_window, config.rope_theta)
        else:
            self.mixer = HydraMamba3Reference(config.d_model, config.mamba_state, config.mamba_expand)
        self.ffn = SwiGLU(config.d_model, config.ffn_dim)
        self.gate = nn.Linear(config.d_model, 1, bias=True)

    def forward(self, x, memory):
        h = self.norm1(x)
        y = self.mixer(h)
        if isinstance(y, tuple):
            y = y[0]
        gate = 1.0 + 0.5 * self.gate(h).tanh()
        x = x + gate * y
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x
