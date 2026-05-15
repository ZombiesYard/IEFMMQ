# VLM Unsloth Training Scripts

Use separate entry points for Qwen and Gemma experiments.

## Qwen3.x

- Training: `tools/train_qwen35_vlm_unsloth.py`
- Benchmark: `tools/benchmark_qwen35_vlm_facts.py`
- Historical defaults remain tuned for `Qwen/Qwen3.5-9B-Base`.
- The current `Qwen/Qwen3.6-27B` run on cloud-247 reuses the same benchmark script
  with an explicit `--base-model Qwen/Qwen3.6-27B` override and a dedicated remote
  launcher under `/scratch/yz50/iefmmq_vlm_ft_unsloth/work/`.

## Gemma 4

- Training: `tools/train_gemma4_vlm_unsloth.py`
- Benchmark: `tools/benchmark_gemma4_vlm_facts.py`
- Defaults target `google/gemma-4-31B`.
- Gemma defaults set `--chat-template gemma-4`, `--lora-target-modules all-linear`,
  `--no-finetune-vision-layers`, and `--gpu-memory-utilization 0.95`.

The training data layout is unchanged. For the current comparison run, reuse the
existing Run-003 and Run-005 SFT exports and pass Run-005 twice to reproduce the
Run-003 + Run-005x2 recipe.
