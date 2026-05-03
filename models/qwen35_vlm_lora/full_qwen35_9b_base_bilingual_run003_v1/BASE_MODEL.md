# Base Model Reference

This directory stores the LoRA adapter trained for the SimTutor cockpit visual fact extraction experiment.

The base model weights are not vendored in this repository. Reconstruct the fine-tuned model by loading:

- Base model: `Qwen/Qwen3.5-9B-Base`
- LoRA adapter: `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_v1/adapter`

This artifact corresponds to the full Run-003-only training run based on the 13-fact ontology.

The training configuration and metrics are recorded in `train_summary.json`.
