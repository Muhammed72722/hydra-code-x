# HYDRA-CODE X

Research implementation scaffold for HYDRA-CODE X.

## A0 target
- ~300M parameter hybrid decoder
- 12 layers: 8 Mamba-path layers + 4 GQA attention layers
- d_model=1536
- vocab=32768
- 8K context
- BF16-ready
- reference PyTorch backend first

## Important
`hydra/ssm/mamba3_reference.py` is a **research reference SSM**, not a claim of exact parity with upstream Mamba-3. The interface is intentionally isolated so an exact/optimized Mamba-3 kernel can replace it after numerical validation.
