---
base_model: Qwen/Qwen3.5-9B-Base
library_name: peft
pipeline_tag: image-text-to-text
tags:
- base_model:adapter:Qwen/Qwen3.5-9B-Base
- qwen3.5
- vision-language
- lora
- peft
- unsloth
- trl
- sft
- cockpit
- visual-fact-extraction
---

# SimTutor Qwen3.5-9B VLM LoRA Adapter

This is a PEFT LoRA adapter for `Qwen/Qwen3.5-9B-Base`, trained for structured cockpit visual fact extraction.

The adapter was trained with an Unsloth VLM workflow:

- `unsloth.FastVisionModel` for base VLM loading and 4-bit preparation
- `FastVisionModel.get_peft_model` for LoRA injection
- `UnslothVisionDataCollator` for multimodal SFT batches
- `trl.SFTTrainer` for the supervised fine-tuning loop

So the training stack is best described as:

```text
Unsloth + PEFT LoRA + TRL SFTTrainer
```

## Intended Task

Input: one composite F/A-18C cockpit panel image.

Output: a JSON object containing a short summary and eight structured visual facts:

- `fcs_page_visible`
- `bit_root_page_visible`
- `bit_page_failure_visible`
- `right_ddi_fcsmc_page_visible`
- `right_ddi_in_test_visible`
- `fcs_bit_result_visible`
- `ins_alignment_page_visible`
- `ins_go`

Each fact uses:

- `state`: one of `seen`, `not_seen`, `uncertain`
- `evidence_note`: a short visual explanation

The adapter is not trained to emit `frame_id`, `source_frame_id`, `session_id`, file paths, or `confidence`.

## Training Summary

| Field | Value |
|---|---:|
| Base model | `Qwen/Qwen3.5-9B-Base` |
| Source data | Run-001 reviewed cockpit screenshots |
| Reviewed images | 180 |
| SFT rows | 360 bilingual rows |
| Train rows | 324 |
| Eval rows | 36 |
| Epochs | 4 |
| Learning rate | 2e-4 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| Train loss | 0.57036 |
| Final eval loss | 0.03269 |

## Evaluation Summary

On Run-002 heldout new session:

| Metric | Base | LoRA-v1 |
|---|---:|---:|
| Fact accuracy | 0.7600 | 0.9150 |
| Seen F1 | 0.4714 | 0.8380 |
| Sample exact match | 0.0600 | 0.3600 |
| Critical false positives | 13 | 15 |

Most facts improve substantially, but `ins_go` shows a concentrated false-positive failure mode. Future experiments should decompose `ins_go` into lower-level visual evidence.

## Loading

Load this adapter on top of `Qwen/Qwen3.5-9B-Base`.

```python
from peft import PeftModel
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B-Base",
    max_seq_length=4096,
    load_in_4bit=True,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "path/to/adapter")
model.eval()
```

## Limitations

This is a small-data research adapter trained for a narrow cockpit visual fact extraction task. It is not a general aviation VLM and should be evaluated on additional independent sessions before broader use.
