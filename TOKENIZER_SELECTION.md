# HYDRA-CODE X Tokenizer Selection Protocol

Candidates:
- 32K
- 48K
- 65,536

## Screening metrics

1. tokens / UTF-8 byte — lower is better.
2. identifier fragmentation — average pieces per identifier, lower is better.
3. operator fragmentation — average pieces per operator/symbol span, lower is better.
4. FIM tokens / byte — lower is better.
5. P95 sequence length on representative files — lower is better for packing efficiency.

The script computes a normalized screening score. This score is **not** the final decision.

## Final decision

Train matched 100M–300M small models with the top tokenizer candidates using:

- identical data shards
- identical token budget
- identical optimizer and schedule
- identical context length
- identical evaluation suite

Select based on a composite of:

- code completion
- FIM
- identifier-sensitive tasks
- repository packing efficiency
- validation loss at equal compute
- downstream coding benchmark

A tokenizer is rejected if it improves token efficiency but hurts identifier semantics or coding accuracy.
