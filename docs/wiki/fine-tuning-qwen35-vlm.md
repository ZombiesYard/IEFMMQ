# Fine-Tuning Qwen3.5 VLM

This page documents the Qwen3.5 VLM LoRA fine-tuning workflow.

## Model Boundary

This repository does not redistribute `Qwen/Qwen3.5-9B-Base`.

The fine-tuned artifact is a LoRA adapter:

```text
models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/adapter
```

To reconstruct the adapted model, obtain the base model separately and load the adapter on top of it.

## Training Stack

The training stack is:

```text
Unsloth + PEFT LoRA + TRL SFTTrainer
```

- Unsloth handles VLM loading, 4-bit preparation, LoRA injection, and vision batch collation.
- PEFT defines the LoRA adapter format.
- TRL `SFTTrainer` runs the supervised fine-tuning loop.

## Local Script

Training entrypoint:

```text
tools/train_qwen35_vlm_unsloth.py
```

Important defaults:

- `--model-name Qwen/Qwen3.5-9B-Base`
- `--max-seq-length 4096`
- `--num-train-epochs 4`
- `--learning-rate 2e-4`
- `--per-device-train-batch-size 1`
- `--gradient-accumulation-steps 4`
- `--lora-r 16`
- `--lora-alpha 16`
- `--lora-dropout 0.0`
- `--eval-ratio 0.1`
- `--seed 3407`

## Remote Training Example

The example below uses placeholders and assumes a remote shell with `tcsh`.

```tcsh
cd <scratch-root>/iefmmq_vlm_ft_unsloth
source venv/bin/activate.csh

setenv TMPDIR <scratch-root>/tmp
setenv HF_HOME <hf-cache-dir>
setenv HUGGINGFACE_HUB_CACHE <hf-cache-dir>/hub
setenv TRANSFORMERS_CACHE <hf-cache-dir>/transformers

python work/train_qwen35_vlm_unsloth.py \
  --train-jsonl data/sft_en.jsonl data/sft_zh.jsonl \
  --output-dir runs/full_qwen35_9b_base_bilingual_v1 \
  --model-name Qwen/Qwen3.5-9B-Base \
  --num-train-epochs 4 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 16 \
  --gradient-accumulation-steps 4
```

## Recorded LoRA-v1 Training Summary

| Field | Value |
|---|---:|
| Train rows | 324 |
| Eval rows | 36 |
| Epochs | 4 |
| Learning rate | 2e-4 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Train loss | 0.57036 |
| Final eval loss | 0.03269 |
| Runtime | 2201s |

See:

- `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/train_summary.json`
- `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/README.md`
