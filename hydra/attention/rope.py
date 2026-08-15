import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, theta: float = 500_000.0):
    # q/k: [B, H, T, D]
    d = q.size(-1)
    device = q.device
    dtype = q.dtype
    pos = torch.arange(q.size(-2), device=device, dtype=torch.float32)
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))
    angles = torch.outer(pos, inv)
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)
