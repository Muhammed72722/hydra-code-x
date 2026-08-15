# HYDRA-CODE X

Research implementation of the HYDRA-CODE X coding foundation model.

## v0.1.1

The first milestone is **HYDRA-300M-A0**, a clean hybrid baseline:

- 12 decoder blocks
- 8 Mamba-3-style reference SSM blocks
- 4 GQA attention blocks
- 1280 hidden size
- 3584 SwiGLU intermediate size
- 32K vocabulary
- 8K training context
- FIM-ready causal language-model interface
- Working Memory disabled in A0 so it is a clean ablation baseline

### Important reference-model note

`hydra/ssm/mamba3_reference.py` is a pure-PyTorch research reference inspired by the public Mamba-3 design. It deliberately does **not** claim bitwise parity with the upstream fused SISO/MIMO kernels. The upstream implementation currently uses CUDA-oriented Triton/TileLang/CUTE kernels. HYDRA will validate the reference path first, then add dedicated CUDA and XLA backends.

## Development

```bash
pip install -e '.[test]'
pytest -q
```

For direct source-tree tests without installation:

```bash
PYTHONPATH=. pytest -q
```
