# Qwen3.5 VLM LoRA Artifacts

This directory stores local copies of the Qwen3.5-9B VLM LoRA adapters used in the SimTutor cockpit visual fact extraction experiments.

The base model weights are not stored in this repository. Each artifact here should be reconstructed as:

```text
Qwen/Qwen3.5-9B-Base + local adapter
```

## Available Adapters

| Directory | Purpose | Notes |
|---|---|---|
| `full_qwen35_9b_base_bilingual_v1` | Run-001 bilingual LoRA-v1 | First full experiment using the original 8-fact ontology |
| `full_qwen35_9b_base_bilingual_run003_v1` | Run-003-only full LoRA | First full experiment using the 13-fact ontology |
| `full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1` | Current best Qwen adapter | Run-003 plus Run-005 composition rebalance, with Run-005 oversampled once |
| `smoke_qwen35_9b_base_bilingual_v4` | Smoke test adapter | Very small sanity-check run, not a main experimental artifact |

## Notes

- `adapter/` contains the PEFT LoRA weights and tokenizer/processor metadata.
- `train_summary.json` records the training configuration when available.
- `README.md` inside each run directory describes the experiment.
- `BASE_MODEL.md` inside each run directory records which base model should be used for reconstruction.
