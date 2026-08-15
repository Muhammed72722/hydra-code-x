import torch
from torch import nn
import torch.nn.functional as F


class HydraMamba3Reference(nn.Module):
    """Research reference selective SSM.

    This is intentionally a simple, pure-PyTorch state-space reference, not a
    claim of exact upstream Mamba-3 parity. It provides a stable interface for
    numerical tests and can later be swapped for a verified Mamba-3 kernel.
    """

    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.log_a = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.02)
        self.d_skip = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, state=None):
        b, t, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(u)
        params = self.x_proj(u)
        b_in, c_in, dt_raw = params.split([self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))

        a = -F.softplus(self.log_a).float()
        state_t = x.new_zeros((b, self.d_inner, self.d_state)) if state is None else state
        ys = []
        for i in range(t):
            x_t = u[:, i]
            dt_t = dt[:, i]
            b_t = b_in[:, i]
            c_t = c_in[:, i]
            decay = torch.exp(dt_t.float().unsqueeze(-1) * a)
            state_t = decay * state_t + dt_t.float().unsqueeze(-1) * x_t.float().unsqueeze(-1) * b_t.float().unsqueeze(1)
            y_t = (state_t * c_t.float().unsqueeze(1)).sum(-1) + self.d_skip.float() * x_t.float()
            ys.append(y_t.to(x.dtype))
        y = torch.stack(ys, dim=1)
        y = y * torch.sigmoid(gate)
        return self.out_proj(y), state_t
