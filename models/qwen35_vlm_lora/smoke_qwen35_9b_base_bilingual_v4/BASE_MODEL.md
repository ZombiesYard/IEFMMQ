# Base Model Reference

This directory stores a smoke-test LoRA adapter for the SimTutor cockpit visual fact extraction experiment.

The base model weights are not vendored in this repository. Reconstruct the fine-tuned model by loading:

- Base model: `Qwen/Qwen3.5-9B-Base`
- LoRA adapter: `models/qwen35_vlm_lora/smoke_qwen35_9b_base_bilingual_v4/adapter`

This artifact is a small smoke run used for sanity checking and should not be treated as a main experimental model.

The training configuration and metrics are recorded in `train_summary.json`.
