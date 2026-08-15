# HYDRA experiment protocol

These experiments are fixed-budget by design. Change one independent variable at a time.

## Tokenizer ablation
Train the same 300M HYDRA-A0 model with 32K, 48K and 65,536 vocabularies. Keep corpus, seed, token budget, optimizer, context and evaluation identical.

Tokenizer metrics are only a pre-filter. The final choice must use equal-compute small-model results.

## A0 backbone ablation
Compare HYDRA-300M against a parameter-matched dense Transformer baseline. The hybrid architecture only earns its complexity if it beats the simpler model under equal training conditions.

## Promotion rule
No feature is promoted to 1.3B unless it improves the primary coding score without a material regression in stability or efficiency.
