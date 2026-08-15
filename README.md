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

## v0.1.4 Tokenizer Lab

The tokenizer lab trains and compares 32K, 48K, and 65,536-vocabulary SentencePiece BPE candidates.
It evaluates token density, identifier fragmentation, operator fragmentation, FIM token efficiency, and 95th-percentile sequence length. The ranking is only a screening heuristic; the final tokenizer is selected after a controlled small-model ablation.

## v0.1.6
- Added deterministic, quality-weighted tokenizer corpus sampling from JSONL manifests.
- Added language round-robin balancing for tokenizer experiments.
- Added deterministic SentencePiece seed control.
- Added regression test for reproducible corpus sampling.
