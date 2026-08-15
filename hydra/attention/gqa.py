import math
import torch
from torch import nn
from .rope import apply_rope


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x.repeat_interleave(repeats, dim=1)


class LocalGQA(nn.Module):
    def __init__(self, d_model: int, q_heads: int, kv_heads: int, window: int, rope_theta: float):
        super().__init__()
        assert d_model % q_heads == 0
        assert q_heads % kv_heads == 0
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // q_heads
        self.window = window
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.rope_theta)
        k = repeat_kv(k, self.q_heads // self.kv_heads)
        v = repeat_kv(v, self.q_heads // self.kv_heads)
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        if self.window < t:
            idx = torch.arange(t, device=x.device)
            local = (idx[None, :] - idx[:, None]) < self.window
            causal = causal & local
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~causal[None, None], torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        y = torch.matmul(probs, v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(y)
