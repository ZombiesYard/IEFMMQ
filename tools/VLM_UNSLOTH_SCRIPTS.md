# VLM Unsloth Training Scripts

Use separate entry points for Qwen and Gemma experiments.

## Qwen3.5

- Training: `tools/train_qwen35_vlm_unsloth.py`
- Benchmark: `tools/benchmark_qwen35_vlm_facts.py`
- Defaults remain tuned for `Qwen/Qwen3.5-9B-Base`.

## Gemma 4

- Training: `tools/train_gemma4_vlm_unsloth.py`
- Benchmark: `tools/benchmark_gemma4_vlm_facts.py`
- Defaults target `google/gemma-4-31B`.
- Gemma defaults set `--chat-template gemma-4`, `--lora-target-modules all-linear`,
  `--no-finetune-vision-layers`, and `--gpu-memory-utilization 0.95`.

The training data layout is unchanged. For the current comparison run, reuse the
existing Run-003 and Run-005 SFT exports and pass Run-005 twice to reproduce the
Run-003 + Run-005x2 recipe.
