# Base Model Reference

This directory stores the LoRA adapter trained for the SimTutor cockpit visual fact extraction experiment.

The base model weights are not vendored in this repository because `Qwen/Qwen3.5-9B-Base` is approximately tens of GB in the Hugging Face cache. Reconstruct the fine-tuned model by loading:

- Base model: `Qwen/Qwen3.5-9B-Base`
- LoRA adapter: `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/adapter`

The training configuration and metrics are recorded in `train_summary.json`.
