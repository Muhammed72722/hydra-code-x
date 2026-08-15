"""Pure-PyTorch Mamba-3-style reference path for HYDRA.

The implementation mirrors the public Mamba-3 design choices (data-dependent
A/DT, heavy-tail A mapping, B/C normalization, rotary state mixing and a
trapezoid-controlled recurrence) while remaining an intentionally unfused,
portable reference. It is not claimed to be numerically identical to the
upstream fused SISO/MIMO kernels.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


MambaState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """Positive heavy-tail map used by the public Mamba-3 parameterization."""
    neg = x.clamp_max(0.0)
    pos = x.clamp_min(0.0)
    return pos + torch.reciprocal(1.0 - neg)


class HeadwiseRMSNorm(nn.Module):
    def __init__(self, n_heads: int, d_state: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_heads, d_state))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms * self.weight).to(x.dtype)


class HydraMamba3Reference(nn.Module):
    """Unfused Mamba-3-style SISO reference implementation.

    Input/output: [B, T, d_model]
    State tuple:
      ssm_state  [B, H, P, N]
      prev_x     [B, H, P]
      prev_B     [B, H, N]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        rope_fraction: float = 0.5,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        **_: object,
    ):
        super().__init__()
        if d_state < 2 or d_state % 2:
            raise ValueError("d_state must be an even integer >= 2")
        if not 0.0 < rope_fraction <= 1.0:
            raise ValueError("rope_fraction must be in (0, 1]")

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.d_inner = d_model * expand
        if self.d_inner % headdim:
            raise ValueError("d_model * expand must be divisible by headdim")
        self.nheads = self.d_inner // headdim
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm

        rope_pairs = max(1, int((d_state * rope_fraction) // 2))
        self.num_rope_pairs = min(rope_pairs, d_state // 2)
        self.rotary_dim = 2 * self.num_rope_pairs

        # Public Mamba-3 projection order:
        # [z, x, B, C, dd_dt, dd_A, trap, angle]
        proj_dim = (
            2 * self.d_inner
            + 2 * self.d_state
            + 3 * self.nheads
            + self.num_rope_pairs
        )
        self.in_proj = nn.Linear(d_model, proj_dim, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt = torch.exp(
            torch.rand(self.nheads, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(dt_init_floor)
        dt_bias = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(dt_bias)
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.B_norm = HeadwiseRMSNorm(self.nheads, d_state)
        self.C_norm = HeadwiseRMSNorm(self.nheads, d_state)

        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        self.post_norm = nn.LayerNorm(self.d_inner) if is_outproj_norm else None

    def initial_state(self, batch_size: int, device=None) -> MambaState:
        device = self.in_proj.weight.device if device is None else device
        ssm = torch.zeros(
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=torch.float32,
        )
        prev_x = torch.zeros(batch_size, self.nheads, self.headdim, device=device, dtype=torch.float32)
        prev_b = torch.zeros(batch_size, self.nheads, self.d_state, device=device, dtype=torch.float32)
        return ssm, prev_x, prev_b

    def _rotate_pairs(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Rotate the first state pairs of B/C with data-dependent angles."""
        # x: [B, T, H, N], angles: [B, T, R]
        k = self.num_rope_pairs
        first = x[..., : 2 * k].float()
        real = first[..., :k]
        imag = first[..., k:]
        phase = angles.unsqueeze(2).expand(-1, -1, self.nheads, -1).float()
        c, s = phase.cos(), phase.sin()
        r = real * c - imag * s
        i = real * s + imag * c
        rotated = torch.cat((r, i), dim=-1).to(x.dtype)
        if rotated.size(-1) == x.size(-1):
            return rotated
        return torch.cat((rotated, x[..., 2 * k:]), dim=-1)

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[Union[torch.Tensor, MambaState]] = None,
        return_state: bool = False,
    ):
        if u.ndim != 3 or u.size(-1) != self.d_model:
            raise ValueError(f"expected [B,T,{self.d_model}], got {tuple(u.shape)}")
        b, t, _ = u.shape
        p = self.headdim

        proj = self.in_proj(u)
        splits = [
            self.d_inner,
            self.d_inner,
            self.d_state,
            self.d_state,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_pairs,
        ]
        z, x, B, C, dd_dt, dd_A, trap, angles = proj.split(splits, dim=-1)
        xh = x.view(b, t, self.nheads, p)
        zh = z.view(b, t, self.nheads, p)

        # Data-dependent continuous-time parameters.
        A = -heavy_tail_activation(dd_A.float()).clamp(max=-self.A_floor)
        dt = F.softplus(dd_dt.float() + self.dt_bias.view(1, 1, -1))
        trap_gate = torch.sigmoid(trap.float())

        # A0 uses the SISO/shared-B/C path. The full HYDRA MIMO kernel is a
        # separate implementation milestone after the reference is stable.
        B = B.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        C = C.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        B = self.B_norm(B + self.B_bias.view(1, 1, self.nheads, -1))
        C = self.C_norm(C + self.C_bias.view(1, 1, self.nheads, -1))
        B = self._rotate_pairs(B, angles)
        C = self._rotate_pairs(C, angles)

        if state is None:
            state_t, prev_x, prev_B = self.initial_state(b, u.device)
        elif isinstance(state, tuple):
            state_t, prev_x, prev_B = state
            expected = (b, self.nheads, p, self.d_state)
            if tuple(state_t.shape) != expected:
                raise ValueError(f"bad Mamba state shape {tuple(state_t.shape)} != {expected}")
            state_t = state_t.float()
            prev_x = prev_x.float()
            prev_B = prev_B.float()
        else:
            expected = (b, self.nheads, p, self.d_state)
            if tuple(state.shape) != expected:
                raise ValueError(f"bad Mamba state shape {tuple(state.shape)} != {expected}")
            state_t = state.float()
            prev_x = torch.zeros_like(xh[:, 0], dtype=torch.float32)
            prev_B = torch.zeros_like(B[:, 0], dtype=torch.float32)

        ys = []
        for i in range(t):
            x_i = xh[:, i].float()          # [B,H,P]
            B_i = B[:, i].float()           # [B,H,N]
            C_i = C[:, i].float()           # [B,H,N]
            dt_i = dt[:, i]                 # [B,H]
            A_i = A[:, i]                   # [B,H]
            tr_i = trap_gate[:, i]          # [B,H]

            decay = torch.exp(A_i * dt_i).unsqueeze(-1).unsqueeze(-1)
            drive_now = x_i.unsqueeze(-1) * B_i.unsqueeze(2)
            drive_prev = prev_x.unsqueeze(-1) * prev_B.unsqueeze(2)

            # Reference quadrature blend: Euler/trapezoid interpolation.
            weight = tr_i.view(b, self.nheads, 1, 1)
            injected = dt_i.view(b, self.nheads, 1, 1) * (
                (1.0 - weight) * drive_now
                + weight * 0.5 * (drive_now + drive_prev)
            )
            state_t = decay * state_t + injected

            y = (state_t * C_i.unsqueeze(2)).sum(dim=-1)
            y = y + self.D.view(1, -1, 1) * x_i
            ys.append(y)

            prev_x = x_i
            prev_B = B_i

        y = torch.stack(ys, dim=1).to(u.dtype)
        y = y * F.silu(zh)
        y = y.reshape(b, t, self.d_inner)
        if self.post_norm is not None:
            y = self.post_norm(y)
        out = self.out_proj(y)

        next_state: MambaState = (state_t, prev_x, prev_B)
        return (out, next_state) if return_state else out
