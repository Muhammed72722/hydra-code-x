import argparse
import torch
from hydra.model import HydraConfig, HydraModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    cfg = HydraConfig()
    model = HydraModel(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    for step in range(args.steps):
        ids = torch.randint(0, cfg.vocab_size, (1, min(256, cfg.max_seq_len)), device=device)
        out = model(ids, ids)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(f"step={step:04d} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
