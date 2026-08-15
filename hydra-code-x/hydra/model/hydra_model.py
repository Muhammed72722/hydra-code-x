import torch
from torch import nn
from .config import HydraConfig
from .embeddings import TokenEmbedding
from .norm import RMSNorm
from .block import HydraBlock
from hydra.memory import WorkingMemory


class HydraModel(nn.Module):
    def __init__(self, config: HydraConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embed = TokenEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([HydraBlock(config, i) for i in range(config.n_layers)])
        self.memory = (
            WorkingMemory(config.d_model, config.memory_slots, config.memory_topk)
            if config.memory_slots > 0
            else None
        )
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.embedding.weight

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        states = [None] * self.config.n_layers

        for i, block in enumerate(self.blocks):
            memory = self.memory if self.memory is not None else None
            x, states[i] = block(x, memory=memory, state=states[i])

        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits, "states": states}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            out["loss"] = loss
        return out
