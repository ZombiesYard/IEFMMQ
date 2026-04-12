---
base_model: Qwen/Qwen3.5-9B-Base
library_name: peft
pipeline_tag: image-text-to-text
tags:
- qwen3.5
- vision-language
- lora
- peft
- unsloth
- trl
- sft
- cockpit
- visual-fact-extraction
model_name: full_qwen35_9b_base_bilingual_v1
---

# SimTutor Qwen3.5-9B VLM LoRA Adapter

This directory contains the LoRA adapter for the SimTutor cockpit visual fact extraction experiment.

The fine-tuned model is not a standalone full-weight model. It should be reconstructed as:

```text
Qwen/Qwen3.5-9B-Base + this LoRA adapter
```

The base model weights are intentionally not stored in this repository. The adapter is the fine-tuned artifact produced by the experiment.

## Task

The adapter is trained to extract structured cockpit visual facts from a single F/A-18C composite-panel image. The output target is a JSON object with:

- `summary`
- `facts`

Each fact contains:

- `fact_id`
- `state`
- `evidence_note`

The first experiment uses eight facts:

- `fcs_page_visible`
- `bit_root_page_visible`
- `bit_page_failure_visible`
- `right_ddi_fcsmc_page_visible`
- `right_ddi_in_test_visible`
- `fcs_bit_result_visible`
- `ins_alignment_page_visible`
- `ins_go`

The model is not trained to output `frame_id`, `source_frame_id`, `session_id`, file paths, or self-reported `confidence`.

## Training Stack

This model was fine-tuned with an Unsloth-based VLM training stack:

- Base model loading and 4-bit VLM preparation: `unsloth.FastVisionModel`
- LoRA adapter injection: `FastVisionModel.get_peft_model`
- Vision data collation: `unsloth.trainer.UnslothVisionDataCollator`
- Supervised fine-tuning loop: `trl.SFTTrainer`
- Adapter format: PEFT LoRA

In short, the experiment used **Unsloth + PEFT LoRA + TRL SFTTrainer**. The TRL mention in generated metadata refers to the trainer used for the SFT loop, not to a text-only training setup.

## Training Data

The LoRA-v1 adapter was trained on Run-001 reviewed cockpit screenshots:

- Source session: `fa18c-coldstart-run-001`
- Reviewed images: 180
- SFT languages: English and Chinese
- Total SFT rows: 360
- Train rows: 324
- Eval rows: 36

The dataset format is OpenAI-compatible multimodal chat JSONL:

```json
{
  "messages": [
    {"role": "system", "content": "You are SimTutor visual fact extractor. Reply with JSON only."},
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "Inspect this composite cockpit panel image and output the 8 visual facts as JSON."}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"summary\":\"...\",\"facts\":[{\"fact_id\":\"fcs_page_visible\",\"state\":\"seen\",\"evidence_note\":\"...\"}]}"
    }
  ]
}
```

## Hyperparameters

Training summary:

| Parameter | Value |
|---|---:|
| Base model | `Qwen/Qwen3.5-9B-Base` |
| Max sequence length | 4096 |
| Epochs | 4 |
| Learning rate | 2e-4 |
| Per-device train batch size | 1 |
| Gradient accumulation steps | 4 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| Seed | 3407 |
| Train loss | 0.57036 |
| Final eval loss | 0.03269 |
| Runtime | 2201s |

See `train_summary.json` for the exact recorded training metadata.

## Evaluation Summary

Two benchmarks were run:

- Run-001 contaminated development set: same source pool as the training data.
- Run-002 heldout new session: independent reviewed session not used in LoRA-v1 training.

### Run-001: contaminated development set

| Metric | Base | LoRA-v1 |
|---|---:|---:|
| JSON valid rate | 1.0000 | 1.0000 |
| Schema valid rate | 1.0000 | 1.0000 |
| Fact accuracy | 0.7951 | 0.9694 |
| Seen F1 | 0.6080 | 0.9517 |
| Sample exact match | 0.1833 | 0.7778 |
| Critical false positives | 19 | 2 |

### Run-002: heldout new session

| Metric | Base | LoRA-v1 |
|---|---:|---:|
| JSON valid rate | 1.0000 | 1.0000 |
| Schema valid rate | 1.0000 | 1.0000 |
| Fact accuracy | 0.7600 | 0.9150 |
| Seen F1 | 0.4714 | 0.8380 |
| Sample exact match | 0.0600 | 0.3600 |
| Critical false positives | 13 | 15 |

The adapter improves most visual facts substantially. The main observed failure mode is `ins_go`, which produced 13 false positives on the Run-002 heldout set. This suggests that `ins_go` is too high-level for a single-frame VLM target and should be decomposed into lower-level visual evidence in future experiments.

## Loading

The adapter should be loaded on top of `Qwen/Qwen3.5-9B-Base`.

Pseudo-code:

```python
from transformers import AutoProcessor
from peft import PeftModel
from unsloth import FastVisionModel

base_model = "Qwen/Qwen3.5-9B-Base"
adapter_path = "models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/adapter"

model, processor = FastVisionModel.from_pretrained(
    model_name=base_model,
    max_seq_length=4096,
    load_in_4bit=True,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
```

The exact inference code depends on the local Qwen/Unsloth runtime and processor chat template.

## Limitations

- This is a small-data domain adaptation experiment.
- The LoRA-v1 adapter was trained only on Run-001.
- Run-002 shows useful cross-session generalization, but `ins_go` false positives remain a significant target-design issue.
- The current eight facts are a first-pass engineering ontology, not a final cockpit visual ontology.
- The adapter should be evaluated with additional independent sessions before being used as a general cockpit visual fact extractor.

## Related Artifacts

- Training summary: `train_summary.json`
- Base model note: `BASE_MODEL.md`
- Technical report: `Doc/Vision/Reports/qwen35_vlm_finetune_report_EN.md`
- Benchmark artifacts: `benchmarks/qwen35_vlm_finetune/`
