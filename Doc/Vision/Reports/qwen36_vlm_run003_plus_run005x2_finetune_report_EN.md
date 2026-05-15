# Qwen3.6-27B VLM Run-003 + Run-005x2 LoRA Fine-Tuning Report

## Abstract

This report documents a VLM LoRA fine-tuning and holdout benchmark round in which the visual backbone is switched from `Qwen/Qwen3.5-9B-Base` to the official `Qwen/Qwen3.6-27B`, while keeping the 13-fact ontology and the `Run-003 + Run-005x2` training recipe fixed.

The goal of this round is not to redefine facts or add new datasets. Instead, it asks two more practical questions:

1. Given the same training recipe, can a stronger official `Qwen/Qwen3.6-27B` backbone still benefit meaningfully from LoRA?
2. Can this line avoid the Gemma4-style deployment problem where training works but vLLM LoRA serving does not?

Training reuses the current best data recipe: `Run-003 bilingual once + Run-005 composition-rebalance bilingual twice`, for a total of 928 training rows over 4 epochs. Training was completed on cloud-247 with `1x H100 94GB`, using the official `Qwen/Qwen3.6-27B` plus an Unsloth + PEFT LoRA + TRL `SFTTrainer` stack with `load_in_4bit=True`. Final train loss is `0.1068`.

On `Run-002 newfacts holdout`, `Qwen3.6-27B + LoRA` improves substantially over the same-backbone base model: fact accuracy rises from `0.9108` to `0.9877`, macro F1 from `0.5218` to `0.6425`, seen F1 from `0.6712` to `0.9479`, and sample exact match from `0.26` to `0.86`. On `Run-004 random holdout`, the base model is already strong, but LoRA still brings consistent gains: fact accuracy rises from `0.9769` to `0.9892`, macro F1 from `0.5724` to `0.6075`, seen F1 from `0.8211` to `0.9147`, and sample exact match from `0.70` to `0.86`.

However, compared with the current best offline `Qwen3.5-9B Run-003 + Run-005x2` line, this `Qwen3.6-27B + LoRA` round does not fully surpass the old best across the two holdouts. Final fact accuracy, macro F1, seen F1, and sample exact match remain slightly below the qwen35 best line, although critical false positives are lower or comparable. Taken together, this is a **deployable, serviceable, and strong** new line, but under the current two offline holdouts it is better described as a **strong production candidate** than an unquestioned new SOTA replacement for the qwen35 best model.

## 1. Background and Motivation

### 1.1 Why Qwen3.6-27B was tested

In the previous round, `Qwen/Qwen3.5-9B-Base + Run-003 + Run-005x2 LoRA` already established the current strongest offline result and confirmed that both the 13-fact ontology and the composition-rebalance data strategy are effective.

From an engineering perspective, however, two questions remained open:

- whether a larger official Qwen backbone still offers room for improvement under the same training recipe;
- whether the resulting line can complete reliable `base + LoRA` online serving under `vLLM 0.19.0`.

The Gemma4 attempt already showed that “trainable” does not imply “servable.” This round therefore treats “trainable, benchmarkable, and vLLM-servable with LoRA” as a joint acceptance condition rather than evaluating training loss alone.

### 1.2 What stays fixed in this round

This round intentionally keeps the following variables unchanged:

- the 13-fact ontology;
- the training data source (`Run-003 + Run-005 composition rebalance`);
- the data weighting scheme (`Run-003 bilingual once + Run-005 bilingual twice`);
- the target task, which remains structured visual facts rather than free-form summary.

As a result, the round can be read relatively cleanly as:

> Under the same data recipe and the same task definition, what changes when the backbone is replaced by `Qwen/Qwen3.6-27B`?

## 2. Data and Training Recipe

### 2.1 Training data

This round uses the same data combination as the current best Qwen line:

| Dataset | reviewed images | SFT rows (EN+ZH) | Role |
|---|---:|---:|---|
| Run-003 | 220 | 440 | main 13-fact training set |
| Run-005 composition rebalance | 122 | 244 | multi-display co-occurrence and hard-negative supplement |

The actual input order during training is:

```text
Run-003/sft_en.jsonl
Run-003/sft_zh.jsonl
Run-005/sft_en.jsonl
Run-005/sft_zh.jsonl
Run-005/sft_en.jsonl
Run-005/sft_zh.jsonl
```

This yields a total of `928` training rows.

### 2.2 The fixed 13-fact ontology

This round continues to use the same 13 core facts:

`tac_page_visible`, `supt_page_visible`, `fcs_page_visible`, `fcs_page_x_marks_visible`, `bit_root_page_visible`, `fcsmc_page_visible`, `fcsmc_intermediate_result_visible`, `fcsmc_in_test_visible`, `fcsmc_final_go_result_visible`, `hsi_page_visible`, `hsi_map_layer_visible`, `ins_grnd_alignment_text_visible`, `ins_ok_text_visible`

Each fact still uses `seen / not_seen / uncertain`, and the downstream system continues to consume structured fact states only.

## 3. Fine-Tuning Setup

The training stack is:

```text
Qwen/Qwen3.6-27B
  + Unsloth VLM loading
  + load_in_4bit=True
  + PEFT LoRA
  + TRL SFTTrainer
```

Key settings are:

| Parameter | Value |
|---|---:|
| model_name | `Qwen/Qwen3.6-27B` |
| train_rows | 928 |
| eval_rows | 0 |
| epochs | 4 |
| learning_rate | 2e-4 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| effective batch size | 4 |
| max_seq_length | 4096 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| finetune_vision_layers | true |
| load_in_4bit | true |
| gpu_memory_utilization | 0.6 |
| seed | 3407 |
| train_runtime | 12619.29 s |
| train_steps_per_second | 0.074 |
| final train loss | 0.1068 |

Artifacts:

- LoRA adapter: `models/qwen36_vlm_lora/full_qwen36_27b_base_bilingual_run003_plus_run005x2_v1/adapter`
- train summary: `models/qwen36_vlm_lora/full_qwen36_27b_base_bilingual_run003_plus_run005x2_v1/train_summary.json`

## 4. Serving Compatibility Validation

This is the main practical difference between this round and the Gemma line.

After training, a separate serving smoke test was completed on cloud-247 and verified the following:

1. `Qwen/Qwen3.6-27B` base can be launched successfully under `vLLM 0.19.0`;
2. the LoRA adapter can be mounted via `--enable-lora --enable-tower-connector-lora`;
3. one vLLM service can expose both:
   - `simtutor-base`
   - `simtutor-vision`
4. both model names return valid OpenAI-compatible chat completions;
5. on the system side, `simtutor-*` model names automatically receive `enable_thinking=false`.

Therefore, the `Qwen3.6-27B + LoRA` line now satisfies all three:

- trainable
- benchmarkable
- online-serviceable

## 5. Evaluation Setup

### 5.1 Holdouts

This round continues to use the same two independent holdouts as the previous round:

| holdout | sample count | role |
|---|---:|---|
| `Run-002 newfacts` | 50 | stronger external generalization holdout |
| `Run-004 random` | 100 | random stress-test holdout |

Neither holdout has exact overlap with the `Run-003 + Run-005` training set. `Run-004` also has a skewed support distribution, so it is more suitable as a stress test than as the only main conclusion source.

### 5.2 Benchmark protocol

Both holdouts are evaluated under the same `vs_base` protocol as the previous report:

- first run `Qwen/Qwen3.6-27B` base on the holdout;
- then run the same backbone plus the `Run-003 + Run-005x2 LoRA`.

That means the total evaluation volume remains:

- `Run-004`: `100 base + 100 lora`
- `Run-002`: `50 base + 50 lora`

## 6. Results

### 6.1 Run-002 Newfacts Holdout

| Model | JSON valid | schema valid | fact accuracy | macro F1 | seen F1 | sample exact match | critical FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3.6-27B` base | 1.0000 | 1.0000 | 0.9108 | 0.5218 | 0.6712 | 0.2600 | 1 |
| `Qwen3.6-27B + LoRA` | 1.0000 | 1.0000 | 0.9877 | 0.6425 | 0.9479 | 0.8600 | 3 |

![Qwen3.6 / Run-002 newfacts overall accuracy](assets/qwen36_vlm_finetune/run002_newfacts/overall_accuracy.png)

![Qwen3.6 / Run-002 newfacts fact F1 by model](assets/qwen36_vlm_finetune/run002_newfacts/fact_f1_by_model.png)

![Qwen3.6 / Run-002 newfacts seen F1 by fact](assets/qwen36_vlm_finetune/run002_newfacts/seen_f1_by_fact.png)

![Qwen3.6 / Run-002 newfacts critical false positives](assets/qwen36_vlm_finetune/run002_newfacts/critical_false_positives.png)

Relative to the same-backbone base model, LoRA improves:

- fact accuracy by `+0.0769`
- macro F1 by `+0.1207`
- seen F1 by `+0.2767`
- sample exact match by `+0.60`
- critical false positives by `+2`

The main gains on this external holdout are very clear:

- `tac_page_visible`: `0.4706 -> 1.0000`
- `supt_page_visible`: `0.0000 -> 1.0000`
- `ins_grnd_alignment_text_visible`: `0.6875 -> 0.9512`
- `ins_ok_text_visible`: `0.0000 -> 0.5714`

In other words, LoRA does not merely make the model “answer better.” It substantially repairs some of the most important recall failures on `Run-002`, especially TAC/SUPT page recognition and INS alignment text detection.

At the same time, critical false positives increase from `1` to `3`, located in:

- `fcsmc_final_go_result_visible`
- `ins_grnd_alignment_text_visible`
- `ins_ok_text_visible`

So on `Run-002`, the main effect of this LoRA is “much better recall and exact match,” not “continued reduction of all high-risk false positives.”

### 6.2 Run-004 Random Holdout

| Model | JSON valid | schema valid | fact accuracy | macro F1 | seen F1 | sample exact match | critical FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3.6-27B` base | 1.0000 | 1.0000 | 0.9769 | 0.5724 | 0.8211 | 0.7000 | 7 |
| `Qwen3.6-27B + LoRA` | 1.0000 | 1.0000 | 0.9892 | 0.6075 | 0.9147 | 0.8600 | 7 |

![Qwen3.6 / Run-004 random overall accuracy](assets/qwen36_vlm_finetune/run004_random/overall_accuracy.png)

![Qwen3.6 / Run-004 random fact F1 by model](assets/qwen36_vlm_finetune/run004_random/fact_f1_by_model.png)

![Qwen3.6 / Run-004 random seen F1 by fact](assets/qwen36_vlm_finetune/run004_random/seen_f1_by_fact.png)

![Qwen3.6 / Run-004 random critical false positives](assets/qwen36_vlm_finetune/run004_random/critical_false_positives.png)

Relative to the same-backbone base model, LoRA improves:

- fact accuracy by `+0.0123`
- macro F1 by `+0.0352`
- seen F1 by `+0.0936`
- sample exact match by `+0.16`
- critical false positives by `+0`

This holdout tells a different story from `Run-002`: the base is already very strong, so the LoRA margin is clearly smaller, though still consistently positive. The clearest gains are concentrated in:

- `supt_page_visible`: `0.0000 -> 1.0000`
- `fcs_page_x_marks_visible`: `0.8889 -> 1.0000`
- `ins_grnd_alignment_text_visible`: `0.8403 -> 0.9466`

At the same time, it is important to state explicitly:

- `fcsmc_final_go_result_visible` is unchanged between base and LoRA, with `seen F1 = 0.9449` in both;
- all 7 critical false positives on this holdout remain concentrated in `fcsmc_final_go_result_visible`.

So on `Run-004`, LoRA behaves more like “boundary cleanup, recall repair, and exact-match improvement” than a full rewrite of the error pattern.

### 6.3 Comparison against the current Qwen3.5 best line

To decide whether this line should replace the current best Qwen setup, it must also be compared against the previous `Qwen3.5-9B Run-003 + Run-005x2` best line:

| holdout | model | fact accuracy | macro F1 | seen F1 | sample exact match | critical FP |
|---|---|---:|---:|---:|---:|---:|
| Run-002 | `Qwen3.5-9B + LoRA` | 0.9908 | 0.6515 | 0.9648 | 0.8800 | 4 |
| Run-002 | `Qwen3.6-27B + LoRA` | 0.9877 | 0.6425 | 0.9479 | 0.8600 | 3 |
| Run-004 | `Qwen3.5-9B + LoRA` | 0.9931 | 0.6100 | 0.9159 | 0.9200 | 8 |
| Run-004 | `Qwen3.6-27B + LoRA` | 0.9892 | 0.6075 | 0.9147 | 0.8600 | 7 |

This comparison shows:

1. `Qwen3.6-27B + LoRA` does not fully surpass the current qwen35 best line in final offline metrics.
2. Its critical false positives are lower or comparable.
3. Its base starting point is much stronger than the old qwen35 base, which suggests that the stronger official backbone already absorbs some capabilities that previously had to be learned through LoRA.

So the correct reading is not “Qwen3.6 is worse,” but rather:

> `Qwen3.6-27B` base is already very strong, so the marginal gain from the same LoRA data recipe is smaller than it was in the qwen35 era; the final result is strong, but not yet the new offline best.

## 7. Interpretation

### 7.1 Why Run-002 gains are large while Run-004 gains are small

`Run-002 newfacts` behaves like a stronger external generalization test and exposes more TAC/SUPT, INS text, and cross-page recall failures. By contrast, the base performance on `Run-004 random` is already very high, with many facts close to saturation even before LoRA.

The pattern is therefore clear:

- on the harder external holdout, LoRA still matters a lot;
- on the easier random holdout, LoRA mostly improves exact match and a smaller set of boundary facts.

### 7.2 Main remaining problems

The main remaining issues are concentrated in two areas:

1. `fcsmc_final_go_result_visible`
   - on `Run-004`, both base and LoRA retain 7 critical false positives;
   - this suggests a persistent optimistic bias on “completion-state” text.
2. INS completion-state refinement
   - on `Run-002`, LoRA raises `ins_ok_text_visible` from complete failure to `seen F1 = 0.5714`;
   - but there is still 1 FP and 2 FN, so this fact is not fully stable yet.

### 7.3 Practical engineering significance

Even though it does not yet beat the qwen35 best line on offline holdouts, this line still has strong engineering value:

- it uses the official `Qwen/Qwen3.6-27B`;
- it has been verified to support LoRA serving under `vLLM 0.19.0`;
- it supports the single-service dual-model-name deployment pattern;
- it can now proceed to system-level integration and runtime observation.

This makes it fundamentally different from the Gemma4 line: it is not a training artifact that looks promising but cannot be put online. It is a genuinely deployable candidate for system-level A/B and behavior validation.

## 8. Limitations

1. Both holdouts are still limited in scale, especially `Run-002` with only 50 images.
2. `Run-004` has skewed support and should not be treated as the only main conclusion source.
3. The comparison against the qwen35 best line crosses backbone families, so it is a practical comparison rather than a perfectly single-variable ablation.
4. This benchmark is still an offline visual fact extraction benchmark, not the full SimTutor online help chain.
5. The conclusion is tied to the current `Run-003 + Run-005x2` recipe; additional 27B-specific data or tuning could shift the result.

## 9. Conclusion

This `Qwen/Qwen3.6-27B + Run-003 + Run-005x2 LoRA` round establishes three things:

1. When the same best data recipe is moved to a stronger official Qwen backbone, LoRA still provides stable positive gains.
2. This line has now passed training, benchmarking, and vLLM online serving validation.
3. Under the current two offline holdouts, it still does not fully surpass the existing `Qwen3.5-9B` best line, so it is better viewed as a **strong production candidate** than an immediate full replacement for the old best model.

If the next step prioritizes system integration and online stability, this `Qwen3.6-27B` line is already worth continuing. If the next step still prioritizes purely offline benchmark improvement, the more promising direction is:

- collecting more high-risk completion-state samples for `final GO` and `INS OK`;
- retuning LoRA data weighting or prompt constraints specifically for the 27B base;
- running another small targeted iteration while preserving full vLLM serving compatibility.
