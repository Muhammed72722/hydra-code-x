from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .objectives import causal_lm_loss


@dataclass
class TrainingConfig:
    max_steps: int = 10_000
    grad_accum_steps: int = 32
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 1_000
    output_dir: str = "runs/hydra"
    bf16: bool = True
    seed: int = 17


class Trainer:
    """Minimal backend-neutral trainer.

    It deliberately keeps optimizer/schedule/checkpoint logic independent from
    XLA so the same loop can be tested on T4/P100 before TPU execution.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.step = 0
        self.tokens_seen = 0
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)
        self.optimizer = torch.optim.AdamW(
            self._parameter_groups(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_multiplier)
        self.use_amp = config.bf16 and device.type in {"cuda", "xla"}
        self.amp_dtype = torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=False) if device.type == "cuda" else None

    def _parameter_groups(self) -> list[dict[str, Any]]:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith("bias") or getattr(p, "_no_weight_decay", False):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _lr_multiplier(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return max(1e-8, (step + 1) / max(1, self.config.warmup_steps))
        denom = max(1, self.config.max_steps - self.config.warmup_steps)
        progress = min(1.0, (step - self.config.warmup_steps) / denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.output_dir / f"step-{self.step:08d}.pt"
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "config": asdict(self.config),
        }
        torch.save(payload, target)
        return target

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.step = int(payload["step"])
        self.tokens_seen = int(payload.get("tokens_seen", 0))

    def train(self, loader: Iterable[dict[str, torch.Tensor]]) -> None:
        self.model.train()
        iterator = iter(loader)
        self.optimizer.zero_grad(set_to_none=True)

        while self.step < self.config.max_steps:
            running_loss = 0.0
            for _ in range(self.config.grad_accum_steps):
                batch = next(iterator)
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                loss_mask = batch.get("loss_mask")
                if loss_mask is not None:
                    loss_mask = loss_mask.to(self.device, non_blocking=True)

                autocast_enabled = self.use_amp
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=autocast_enabled,
                ):
                    out = self.model(input_ids)
                    loss = causal_lm_loss(out["logits"], labels, loss_mask=loss_mask)
                    loss = loss / self.config.grad_accum_steps

                loss.backward()
                running_loss += float(loss.detach())
                self.tokens_seen += int(input_ids.numel())

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            if self.step % self.config.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(json.dumps({
                    "step": self.step,
                    "loss": running_loss,
                    "ppl": math.exp(min(20.0, running_loss)),
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_seen": self.tokens_seen,
                }))

            if self.step % self.config.save_every == 0:
                self.save_checkpoint()
