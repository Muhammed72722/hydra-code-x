# HYDRA-CODE X — Verified Data Sources

This document records what the project is currently willing to ingest. The default pipeline does **not** silently download or mix third-party corpora.

## The Stack v2

Current verified official sources:

- Hugging Face dataset: `bigcode/the-stack-v2`
- Official BigCode dataset documentation
- Training variants: `bigcode/the-stack-v2-train-full-ids` and `bigcode/the-stack-v2-train-smol-ids`

The Stack v2 is gated on Hugging Face. Bulk download requires agreement with Software Heritage and INRIA, and users are required to follow the original repository licenses and keep the dataset version current with validated removals. The training variants are repository-grouped and were built after exact and near-duplicate filtering. See the official dataset card before ingesting any content.

### HYDRA policy

1. Do not ingest a source unless its license/provenance fields are available.
2. Unknown or `NOASSERTION` license records are rejected by the default corpus builder.
3. Keep `repository`, `path`, `license_spdx`, source, and content hash in the corpus manifest.
4. Keep a frozen source-version manifest for every training run.
5. Never mix benchmark/test repositories into pretraining shards.

## The Stack v3

As of the current research pass, the verified BigCode/Hugging Face source surfaced here is The Stack v2. HYDRA does **not** assume a `bigcode/the-stack-v3` dataset exists or is usable until an official source and terms are independently verified.

## Other sources

Additional sources may be added later, but every source must supply:

- provenance
- license information
- removal/opt-out handling
- repository identity or source identity
- stable version/date
- reproducible download/index instructions
