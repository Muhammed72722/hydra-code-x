# HYDRA-CODE X data pipeline

This stage adds a dependency-light data pipeline scaffold for provenance, quality scoring, exact/near deduplication, tokenizer experiments, FIM transforms and sequence packing.

## Licensing and provenance

Source code must retain its original license metadata and provenance. BigCode states that The Stack contains code under the original repository licenses and that attribution requirements still apply; the dataset is also updated for validated removal requests. The Hugging Face dataset page is gated and its terms require users to acknowledge the dataset conditions. See the current BigCode/Hugging Face documentation before importing a live corpus. 

## Modules

- `hydra/data/provenance.py` — stable IDs and provenance records.
- `hydra/data/quality.py` — conservative heuristic quality signals.
- `hydra/data/dedup.py` — exact hashes and near-duplicate signature buckets.
- `hydra/data/fim.py` — tokenizer-agnostic FIM transform.
- `hydra/data/packing.py` — fixed-length packed sequences.
- `hydra/data/tokenizer.py` — byte fallback plus optional BPE training.
- `scripts/build_manifest.py` — local source-tree manifest builder.
- `scripts/benchmark_tokenizers.py` — 32K/48K/65K tokenizer candidate workflow.

## Policy

The generic local importer labels licenses as `UNKNOWN` until a source-specific manifest provides verified SPDX information. It must not be used as a license filter by itself.

The final HYDRA tokenizer is not assumed to be 65K. We will choose among 32K, 48K and 65K using token efficiency plus small-model coding validation.
