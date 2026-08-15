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
        self.memory = WorkingMemory(config.d_model, config.memory_slots, config.memory_topk)
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.embedding.weight

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, self.memory)
            x = x + self.memory(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            out["loss"] = loss
        return out
