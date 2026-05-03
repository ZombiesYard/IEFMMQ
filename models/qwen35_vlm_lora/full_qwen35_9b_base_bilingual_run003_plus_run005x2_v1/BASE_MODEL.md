# Base Model Reference

This directory stores the LoRA adapter trained for the SimTutor cockpit visual fact extraction experiment.

The base model weights are not vendored in this repository. Reconstruct the fine-tuned model by loading:

- Base model: `Qwen/Qwen3.5-9B-Base`
- LoRA adapter: `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/adapter`

This artifact corresponds to the current best Qwen training recipe:

- Run-003 bilingual once
- Run-005 composition rebalance bilingual twice

The training configuration and metrics are recorded in `train_summary.json`.
