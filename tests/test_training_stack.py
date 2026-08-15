from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hydra.data import TokenShardDataset, collate_causal
from hydra.model import HydraConfig, HydraModel
from hydra.training import TrainingConfig, Trainer, causal_lm_loss


def test_token_dataset_and_loss(tmp_path: Path):
    shard = tmp_path / "tokens.pt"
    torch.save(torch.arange(0, 256, dtype=torch.long), shard)
    ds = TokenShardDataset(shard, seq_len=16, samples_per_epoch=4, seed=3)
    batch = collate_causal([ds[0], ds[1]])
    assert batch["input_ids"].shape == (2, 16)
    model_cfg = HydraConfig(
        vocab_size=128,
        d_model=128,
        n_layers=2,
        max_seq_len=32,
        ffn_dim=256,
        attention_layers=(1,),
        attention_window=32,
        q_heads=4,
        kv_heads=2,
        mamba_state=8,
        mamba_expand=2,
        mamba_headdim=32,
        memory_slots=0,
    )
    model = HydraModel(model_cfg)
    out = model(batch["input_ids"] % 128, batch["labels"] % 128)
    loss = causal_lm_loss(out["logits"], batch["labels"] % 128)
    assert torch.isfinite(loss)


def test_trainer_one_step(tmp_path: Path):
    cfg = HydraConfig(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        max_seq_len=32,
        ffn_dim=128,
        attention_layers=(1,),
        attention_window=32,
        q_heads=4,
        kv_heads=2,
        mamba_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        memory_slots=0,
    )
    model = HydraModel(cfg)
    ds = [{
        "input_ids": torch.randint(0, cfg.vocab_size, (16,)),
        "labels": torch.randint(0, cfg.vocab_size, (16,)),
    } for _ in range(4)]

    class Static:
        def __iter__(self):
            while True:
                for item in ds:
                    yield {k: v.unsqueeze(0) for k, v in item.items()}

    train_cfg = TrainingConfig(
        max_steps=1,
        grad_accum_steps=2,
        log_every=1,
        save_every=100,
        output_dir=str(tmp_path / "run"),
        bf16=False,
    )
    trainer = Trainer(model, train_cfg, torch.device("cpu"))
    trainer.train(Static())
    assert trainer.step == 1
