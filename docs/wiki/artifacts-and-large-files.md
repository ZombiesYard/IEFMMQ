# Artifacts and Large Files

This repository contains code, reports, reviewed datasets, benchmark artifacts, and a LoRA adapter.

## Git LFS

Large files should be tracked with Git LFS. The repository includes `.gitattributes` rules for:

- `*.safetensors`
- `*.bin`
- `*.pt`
- `datasets/**/*.jsonl`
- `models/**/tokenizer.json`

Small model metadata such as adapter configs, processor configs, training summaries, and chat templates should remain normal Git files so they stay human-reviewable in diffs.

Install Git LFS before committing large artifacts:

```bash
git lfs install
```

## Tracked Artifacts

| Path | Purpose |
|---|---|
| `models/qwen35_vlm_lora/` | LoRA adapter and model card |
| `datasets/vision_sft/` | Run-001 reviewed dataset and bilingual SFT JSONL |
| `datasets/vision_sft_holdout_run002/` | Run-002 heldout reviewed dataset |
| `benchmarks/qwen35_vlm_finetune/` | Base-vs-LoRA benchmark outputs |
| `Doc/Vision/Reports/` | Technical reports and figures |

## Ignored Artifacts

The raw screenshot capture cache is intentionally ignored:

```text
tools/.captures/
```

This directory can be several GB and contains raw intermediate capture material. Reviewed datasets and benchmark summaries are the intended shareable artifacts.

## Base Model

The repository does not redistribute `Qwen/Qwen3.5-9B-Base`.

To use the adapter, obtain the base model separately under its upstream terms and load:

```text
Qwen/Qwen3.5-9B-Base + models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/adapter
```

## Licensing Note

Repository code follows Apache-2.0. The Qwen base model has separate upstream terms. Reuse of the adapter should consider both the repository license and the upstream base-model license.
