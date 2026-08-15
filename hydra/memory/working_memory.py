import torch
from torch import nn


class WorkingMemory(nn.Module):
    def __init__(self, d_model: int, slots: int = 64, top_k: int = 8):
        super().__init__()
        self.slots = slots
        self.top_k = top_k
        self.keys = nn.Parameter(torch.randn(slots, d_model) * (d_model ** -0.5))
        self.values = nn.Parameter(torch.zeros(slots, d_model))
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.read = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        scores = torch.einsum("btd,sd->bts", q, self.keys) / (x.size(-1) ** 0.5)
        top_scores, top_idx = scores.topk(min(self.top_k, self.slots), dim=-1)
        values = self.values[top_idx]
        weights = torch.softmax(top_scores, dim=-1).unsqueeze(-1)
        retrieved = (values * weights).sum(dim=-2)
        return self.read(retrieved)
