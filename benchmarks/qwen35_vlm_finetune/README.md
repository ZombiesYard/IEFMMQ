# Qwen3.5 VLM Fine-Tuning Benchmarks

This directory contains tracked copies of the benchmark artifacts that were originally generated under `.tmp/benchmarks/`.

## Runs

- `base_vs_lora_current180_v1/`
  - Benchmark kind: `contaminated_dev_set`
  - Dataset: Run-001 reviewed data, same source pool as LoRA-v1 training.
  - Purpose: quick regression and in-distribution sanity check.

- `base_vs_lora_holdout_run002_v1/`
  - Benchmark kind: `heldout_new_session`
  - Dataset: Run-002 reviewed data, not used for LoRA-v1 training.
  - Purpose: independent-session generalization check.

Each run includes predictions, metrics, error lists, CSV fact scores, a Markdown report, and chart images.
