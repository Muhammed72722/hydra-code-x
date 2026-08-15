from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically stable next-token loss with optional per-token mask."""
    # The data pipeline supplies labels aligned to input positions: label[t] is
    # the next token after input[t]. Keeping the objective aligned is important
    # for packed datasets and FIM, which explicitly build next-token targets.
    flat_logits = logits.float().reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    if loss_mask is None:
        return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

    flat_mask = loss_mask.bool().reshape(-1)
    if not torch.any(flat_mask):
        return flat_logits.sum() * 0.0
    return F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask], ignore_index=-100)
