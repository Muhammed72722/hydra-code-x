"""Small, dependency-light token dataset utilities for HYDRA training.

The production pipeline will later stream Arrow/WebDataset shards. This module
keeps A0 usable now with either `.pt` tensors or `.bin` integer token shards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


class TokenShardDataset(Dataset):
    """Random fixed-length windows sampled from token shards.

    Supported files:
      * `.pt`: 1-D integer tensor/list
      * `.bin`: raw little-endian uint16/uint32 selected by dtype
    """

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        seq_len: int,
        samples_per_epoch: int = 10_000,
        dtype: str = "uint16",
        seed: int = 17,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("no token shard paths supplied")
        self.seq_len = int(seq_len)
        self.samples_per_epoch = int(samples_per_epoch)
        self.dtype = dtype
        self.seed = int(seed)
        self._shards = [self._load(path) for path in self.paths]
        self._lengths = [int(x.numel()) for x in self._shards]
        if any(n < self.seq_len + 1 for n in self._lengths):
            raise ValueError("every shard needs at least seq_len + 1 tokens")
        self._weights = torch.tensor(self._lengths, dtype=torch.double)
        self._weights /= self._weights.sum()

    def _load(self, path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".pt":
            obj = torch.load(path, map_location="cpu", weights_only=True)
            tokens = obj if torch.is_tensor(obj) else torch.as_tensor(obj)
        elif path.suffix == ".bin":
            if self.dtype == "uint16":
                tokens = torch.from_file(str(path), dtype=torch.uint16)
            elif self.dtype == "uint32":
                tokens = torch.from_file(str(path), dtype=torch.uint32)
            else:
                raise ValueError("dtype must be uint16 or uint32 for .bin shards")
        else:
            raise ValueError(f"unsupported shard format: {path.suffix}")
        if tokens.ndim != 1:
            tokens = tokens.flatten()
        return tokens.to(torch.long).contiguous()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed + int(index))
        shard_idx = int(torch.multinomial(self._weights, 1, generator=g).item())
        tokens = self._shards[shard_idx]
        start = int(torch.randint(0, tokens.numel() - self.seq_len, (1,), generator=g).item())
        seq = tokens[start : start + self.seq_len + 1]
        return {
            "input_ids": seq[:-1].clone(),
            "labels": seq[1:].clone(),
        }


def collate_causal(samples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("empty batch")
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
    }


def collate_fim(
    samples: Sequence[dict[str, torch.Tensor]],
    *,
    fim_prefix_id: int,
    fim_middle_id: int,
    fim_suffix_id: int,
    fim_rate: float = 0.15,
    seed: int = 17,
) -> dict[str, torch.Tensor]:
    """Apply a deterministic FIM transform to some token sequences.

    The batch keeps a fixed shape by constructing `[prefix, suffix, middle]`
    in the common PSM order. Loss is masked for the control tokens.
    """
    if not 0.0 <= fim_rate <= 1.0:
        raise ValueError("fim_rate must be in [0, 1]")
    out_inputs: list[torch.Tensor] = []
    out_labels: list[torch.Tensor] = []
    out_masks: list[torch.Tensor] = []
    g = torch.Generator().manual_seed(seed)

    for sample_idx, sample in enumerate(samples):
        ids = sample["input_ids"]
        labels = sample["labels"]
        use_fim = bool(torch.rand((), generator=g).item() < fim_rate) and ids.numel() >= 8
        if not use_fim:
            out_inputs.append(ids)
            out_labels.append(labels)
            out_masks.append(torch.ones_like(labels, dtype=torch.bool))
            continue

        # Deterministic cut points, excluding the ends.
        local_g = torch.Generator().manual_seed(seed * 1_000_003 + sample_idx)
        n = ids.numel()
        a = int(torch.randint(2, n // 2, (1,), generator=local_g).item())
        b = int(torch.randint(a + 2, n - 1, (1,), generator=local_g).item())
        prefix, middle, suffix = ids[:a], ids[a:b], ids[b:]

        transformed = torch.cat([
            torch.tensor([fim_prefix_id], dtype=torch.long),
            prefix,
            torch.tensor([fim_suffix_id], dtype=torch.long),
            suffix,
            torch.tensor([fim_middle_id], dtype=torch.long),
            middle,
        ])
        # Next-token labels, padded/truncated back to the original sequence length.
        target = transformed[1:]
        if target.numel() < ids.numel():
            pad = torch.full((ids.numel() - target.numel(),), -100, dtype=torch.long)
            target = torch.cat([target, pad])
        else:
            target = target[: ids.numel()]
        inp = transformed[: ids.numel()]
        mask = target.ne(-100)
        out_inputs.append(inp)
        out_labels.append(target)
        out_masks.append(mask)

    return {
        "input_ids": torch.stack(out_inputs),
        "labels": torch.stack(out_labels),
        "loss_mask": torch.stack(out_masks),
    }
