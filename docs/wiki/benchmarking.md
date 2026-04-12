# Benchmarking

The benchmark compares `Qwen/Qwen3.5-9B-Base` against the same base model with the SimTutor LoRA adapter loaded.

## Benchmark Script

```text
tools/benchmark_qwen35_vlm_facts.py
```

## Local Smoke Test

Linux/WSL example:

```bash
python tools/benchmark_qwen35_vlm_facts.py \
  --reviewed-jsonl datasets/vision_sft_holdout_run002/reviewed.jsonl \
  --base-model Qwen/Qwen3.5-9B-Base \
  --adapter models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/adapter \
  --output-dir benchmarks/qwen35_vlm_finetune/local_smoke \
  --lang en \
  --max-samples 2
```

## Remote Full Benchmark Example

`tcsh` example with placeholder paths:

```tcsh
cd <scratch-root>/iefmmq_vlm_ft_unsloth
source venv/bin/activate.csh

python work/benchmark_qwen35_vlm_facts.py \
  --reviewed-jsonl data/holdout_run002/reviewed.jsonl \
  --base-model Qwen/Qwen3.5-9B-Base \
  --adapter runs/full_qwen35_9b_base_bilingual_v1/adapter \
  --output-dir benchmarks/base_vs_lora_holdout_run002_v1 \
  --lang en \
  --benchmark-kind heldout_new_session \
  --max-new-tokens 768
```

## Metrics

- `json_valid_rate`: output parses as JSON.
- `schema_valid_rate`: output follows the expected fact schema.
- `fact_accuracy`: three-class accuracy over all facts.
- `macro_f1`: macro F1 over `seen`, `not_seen`, `uncertain`.
- `seen_f1`: F1 for the positive `seen` class.
- `sample_exact_match`: all facts for a sample must match.
- `critical_false_positive_count`: selected critical facts predicted as `seen` when gold is `not_seen` or `uncertain`.

## Current Results

Tracked benchmark artifacts:

- `benchmarks/qwen35_vlm_finetune/base_vs_lora_current180_v1/`
- `benchmarks/qwen35_vlm_finetune/base_vs_lora_holdout_run002_v1/`

Run-002 heldout summary:

| Metric | Base | LoRA-v1 |
|---|---:|---:|
| Fact accuracy | 0.7600 | 0.9150 |
| Seen F1 | 0.4714 | 0.8380 |
| Sample exact match | 0.0600 | 0.3600 |
| Critical false positives | 13 | 15 |

The main observed failure mode is `ins_go` false positives.
