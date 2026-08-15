from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class HydraConfig:
    vocab_size: int = 32_768
    d_model: int = 1_536
    n_layers: int = 12
    max_seq_len: int = 8_192
    ffn_dim: int = 4_096
    norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    attention_layers: Tuple[int, ...] = (2, 5, 8, 10)
    q_heads: int = 12
    kv_heads: int = 3
    attention_window: int = 2_048
    mamba_state: int = 64
    mamba_headdim: int = 64
    mamba_expand: int = 2
    memory_slots: int = 64
    memory_topk: int = 8
    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.q_heads

    def validate(self) -> None:
        assert self.d_model % self.q_heads == 0
        assert self.q_heads % self.kv_heads == 0
        assert self.d_model % self.mamba_headdim == 0
        assert all(0 <= i < self.n_layers for i in self.attention_layers)
        assert self.memory_topk <= self.memory_slots
