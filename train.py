from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hydra.data import TokenShardDataset, collate_causal
from hydra.model import HydraConfig, HydraModel
from hydra.training import Trainer, TrainingConfig


def resolve_device() -> torch.device:
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        return xm.xla_device()
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="HYDRA-CODE X training entrypoint")
    parser.add_argument("--data", nargs="+", required=True, help=".pt or .bin token shard(s)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--output", type=str, default="runs/hydra")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    cfg = HydraConfig()
    device = resolve_device()
    model = HydraModel(cfg).to(device)

    dataset = TokenShardDataset(
        args.data,
        seq_len=args.seq_len,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_causal,
    )

    train_cfg = TrainingConfig(
        max_steps=args.steps,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup,
        output_dir=args.output,
        seed=args.seed,
    )
    trainer = Trainer(model, train_cfg, device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at step {trainer.step}")
    print(f"device={device}")
    trainer.train(loader)


if __name__ == "__main__":
    main()
