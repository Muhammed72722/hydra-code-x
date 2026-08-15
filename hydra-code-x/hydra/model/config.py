from dataclasses import dataclass
from typing import Tuple


@dataclass
class HydraConfig:
    # HYDRA-300M-A0 default: clean architecture baseline.
    vocab_size: int = 32_768
    d_model: int = 1_280
    n_layers: int = 12
    max_seq_len: int = 8_192
    ffn_dim: int = 3_584
    norm_eps: float = 1e-5
    rope_theta: float = 500_000.0

    # 4 attention + 8 SSM blocks.
    attention_layers: Tuple[int, ...] = (2, 5, 8, 10)
    q_heads: int = 10
    kv_heads: int = 2
    attention_window: int = 2_048

    # Mamba-3 reference-path parameters.
    mamba_state: int = 64
    mamba_headdim: int = 64
    mamba_expand: int = 2
    mamba_rope_fraction: float = 0.5
    mamba_dt_min: float = 1e-3
    mamba_dt_max: float = 0.1
    mamba_dt_init_floor: float = 1e-4
    mamba_A_floor: float = 1e-4
    mamba_mimo: bool = False  # A0 is a stable SISO reference baseline.
    mamba_mimo_rank: int = 4
    mamba_outproj_norm: bool = False

    # A0 keeps learned external memory disabled so the backbone is measurable.
    memory_slots: int = 0
    memory_topk: int = 0

    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.q_heads

    @property
    def mamba_inner_dim(self) -> int:
        return self.d_model * self.mamba_expand

    @property
    def mamba_heads(self) -> int:
        return self.mamba_inner_dim // self.mamba_headdim

    def validate(self) -> None:
        assert self.d_model % self.q_heads == 0, "d_model must divide q_heads"
        assert self.q_heads % self.kv_heads == 0, "q_heads must divide kv_heads"
        assert self.d_model % self.mamba_headdim == 0 or self.mamba_inner_dim % self.mamba_headdim == 0
        assert self.mamba_inner_dim % self.mamba_headdim == 0
        assert all(0 <= i < self.n_layers for i in self.attention_layers)
        assert len(set(self.attention_layers)) == len(self.attention_layers)
        assert 0 < self.attention_window <= self.max_seq_len
        assert self.mamba_state >= 2 and self.mamba_state % 2 == 0
        assert 0.0 < self.mamba_rope_fraction <= 1.0
        if self.memory_slots:
            assert self.memory_topk > 0
            assert self.memory_topk <= self.memory_slots
        else:
            assert self.memory_topk == 0
