# Gemma-4-31B VLM LoRA Fine-Tuning Report (Run-003 + Run-005x2)

## Abstract

This report documents a backbone-transfer experiment in which the SimTutor F/A-18C cockpit visual fact extraction task is fine-tuned on `google/gemma-4-31B`. The experiment reuses the best data recipe established after Qwen Run-005. The 13-fact ontology is kept fixed, the training-set structure is not redesigned, and the same reviewed data are reused directly: the `Run-003` main training set plus the `Run-005 composition-rebalance` supplement, with the bilingual Run-005 subset repeated once more to increase the weight of multi-display co-occurrence and hard negatives.

Training again uses an Unsloth-accelerated PEFT LoRA VLM SFT setup with TRL `SFTTrainer`. Under that setup, Gemma LoRA clearly improves over the Gemma base model. On `Run-002 newfacts holdout`, fact accuracy improves from `0.8169` to `0.9492`, seen F1 from `0.5128` to `0.8123`, sample exact match from `0.04` to `0.44`, and critical false positives from `16` down to `0`. On `Run-004 random holdout`, fact accuracy improves from `0.8146` to `0.9715`, sample exact match from `0.12` to `0.74`, and critical false positives from `47` down to `13`.

However, Gemma-4-31B does not outperform the current best Qwen3.5-9B LoRA adapter. On `Run-002`, Gemma shows a more conservative error pattern and eliminates critical false positives entirely, but at the cost of lower recall and much lower sample exact match. On `Run-004`, Gemma still exhibits concentrated errors on `fcsmc_intermediate_result_visible`. Taken together, the two holdouts support a narrower conclusion: under the same ontology, the same data, and the same evaluation protocol, the Gemma backbone benefits substantially from LoRA fine-tuning, but its overall performance remains below the strongest Qwen adapter currently available in this project.

## 1. Experimental Objective

After Qwen Run-005, three points were already established:

1. the current 13-fact ontology is trainable and operationally interpretable;
2. the remaining performance bottleneck is driven more by data distribution than by further ontology revision;
3. the `Run-003 + Run-005x2` recipe already produces strong results on Qwen3.5-9B.

The present Gemma experiment therefore does not attempt to redesign the labeling scheme. Instead, it addresses a more controlled question:

> If ontology, training data, training targets, and holdouts are held constant, what changes when the backbone is replaced with `google/gemma-4-31B`?

This setup makes the experiment primarily about backbone behavior rather than about new data or new annotation variables.

## 2. What Is Kept the Same as Qwen Run-005

Gemma uses the same experimental setup as Qwen Run-005 in the following respects.

### 2.1 Ontology and output target

The same 13 core facts are used:

| fact_id | Meaning |
|---|---|
| `tac_page_visible` | TAC/TAC MENU page visible |
| `supt_page_visible` | SUPT/SUPT MENU page visible |
| `fcs_page_visible` | dedicated FCS page visible |
| `fcs_page_x_marks_visible` | X/fault fills visible in FCS channel boxes |
| `bit_root_page_visible` | BIT FAILURES/root page visible |
| `fcsmc_page_visible` | FCS-MC subpage visible |
| `fcsmc_intermediate_result_visible` | intermediate FCS-MC result visible |
| `fcsmc_in_test_visible` | FCS-MC IN TEST visible |
| `fcsmc_final_go_result_visible` | FCS-MC final GO visible |
| `hsi_page_visible` | HSI/POS page visible |
| `hsi_map_layer_visible` | colored HSI MAP layer visible |
| `ins_grnd_alignment_text_visible` | GRND/QUAL/TIME alignment text visible |
| `ins_ok_text_visible` | INS OK text visible |

The label space also remains the same:

- `seen`
- `not_seen`
- `uncertain`

Training targets continue to contain only structured fact outputs rather than non-visual fields such as `confidence` or `source_frame_id`. The export pipeline again uses `--drop-summary` because the downstream system consumes structured fact states rather than VLM summary text.

### 2.2 Training data

Gemma uses the same training corpus as Qwen:

| Dataset | reviewed images | SFT rows (EN+ZH) | Role |
|---|---:|---:|---|
| Run-003 | 220 | 440 | main training set |
| Run-005 composition rebalance | 122 | 244 | multi-display co-occurrence and hard-negative supplement |

The same weighted mixture is used:

```text
Run-003 bilingual once
+ Run-005 bilingual twice
```

As a result, the final training input is also unchanged:

- total training rows: `928`
- unique reviewed images: `342`

### 2.3 Evaluation sets

Gemma uses the same two holdouts:

1. `Run-002 newfacts holdout`
2. `Run-004 random holdout`

The same benchmark metrics are used:

- `json_valid_rate`
- `schema_valid_rate`
- `fact_accuracy`
- `macro_f1`
- `seen_f1`
- `sample_exact_match`
- `critical_false_positive_count`

The same overlap checks also still hold:

- exact overlap between `Run-002` and the training union = `0`
- exact overlap between `Run-004` and the training union = `0`

## 3. What Differs from Qwen Run-005

Although the data and evaluation protocol stay fixed, several Gemma-specific training choices differ from the Qwen run.

### 3.1 Backbone

Qwen Run-005 uses:

```text
Qwen/Qwen3.5-9B-Base
```

This round uses:

```text
google/gemma-4-31B
```

The comparison therefore reflects both model-family and model-scale differences, not merely prompt-template differences.

### 3.2 LoRA attachment strategy

The Qwen training script applies Unsloth LoRA with explicit language- and vision-side tuning flags. The Gemma training script instead uses:

- `lora_target_modules = "all-linear"`
- `finetune_vision_layers = false`

In practice, this means the Gemma run is closer to a uniform all-linear LoRA configuration than to a literal copy of the Qwen module-selection strategy.

### 3.3 Chat template and memory setting

Gemma training explicitly uses:

- `chat_template = "gemma-4"`
- `gpu_memory_utilization = 0.95`

By contrast, the Qwen run follows the Qwen-side defaults with a lower GPU-memory utilization hint. Both runs still use 4-bit loading and LoRA fine-tuning, but Gemma requires a more aggressive single-GPU memory configuration to run the `31B` backbone on one H100 NVL.

## 4. Gemma Training Configuration

According to the remote `train_summary.json`, the Gemma training settings are:

| Parameter | Value |
|---|---:|
| model | `google/gemma-4-31B` |
| train rows | `928` |
| eval rows | `0` |
| max sequence length | `4096` |
| epochs | `4` |
| learning rate | `2e-4` |
| per-device batch size | `1` |
| gradient accumulation | `4` |
| effective batch size | `4` |
| LoRA rank | `16` |
| LoRA alpha | `16` |
| LoRA dropout | `0.0` |
| LoRA target modules | `all-linear` |
| finetune vision layers | `false` |
| load in 4-bit | `true` |
| gpu memory utilization | `0.95` |
| chat template | `gemma-4` |
| seed | `3407` |
| train runtime | `8049.65 s` |
| train steps per second | `0.115` |
| final train loss | `0.0474` |

As in Qwen Run-005, `eval_ratio = 0`. The reason is also the same: Run-005 samples are already duplicated once within training, so internal same-pool evaluation is of limited value. External holdout results therefore remain the primary basis for judging generalization.

## 5. Results on Run-002 Newfacts Holdout

![Gemma Run-002 overall accuracy](assets/gemma4_vlm_finetune/gemma_run002_newfacts_overall_accuracy.png)

![Gemma Run-002 fact F1 by model](assets/gemma4_vlm_finetune/gemma_run002_newfacts_fact_f1_by_model.png)

![Gemma Run-002 seen F1 by fact](assets/gemma4_vlm_finetune/gemma_run002_newfacts_seen_f1_by_fact.png)

![Gemma Run-002 critical false positives](assets/gemma4_vlm_finetune/gemma_run002_newfacts_critical_false_positives.png)

### 5.1 Gemma base vs LoRA

| Metric | Base | LoRA |
|---|---:|---:|
| JSON valid rate | 1.0000 | 1.0000 |
| schema valid rate | 1.0000 | 1.0000 |
| fact accuracy | 0.8169 | 0.9492 |
| macro F1 | 0.4432 | 0.5857 |
| seen F1 | 0.5128 | 0.8123 |
| sample exact match | 0.04 | 0.44 |
| critical false positives | 16 | 0 |

Relative to the Gemma base model, LoRA yields a substantial improvement:

- `fact_accuracy`: `+0.1323`
- `macro_f1`: `+0.1425`
- `seen_f1`: `+0.2995`
- `sample_exact_match`: `+0.40`
- `critical_false_positive_count`: `16 -> 0`

### 5.2 Comparison with the best Qwen LoRA

On the same holdout, the best Qwen LoRA produces:

| Metric | Qwen LoRA | Gemma LoRA |
|---|---:|---:|
| fact accuracy | 0.9908 | 0.9492 |
| macro F1 | 0.6515 | 0.5857 |
| seen F1 | 0.9648 | 0.8123 |
| sample exact match | 0.88 | 0.44 |
| critical false positives | 4 | 0 |

This result reflects a clear trade-off:

1. Gemma reduces critical false positives to `0`, indicating a more conservative completion-state behavior.
2. That conservatism comes with lower recall and much lower sample exact match.
3. On this holdout, Gemma does not exceed Qwen in overall extraction quality.

### 5.3 Main strengths and weaknesses

Gemma LoRA is stable on:

- `bit_root_page_visible`
- `fcsmc_page_visible`
- `fcsmc_intermediate_result_visible`
- `fcsmc_in_test_visible`
- `fcsmc_final_go_result_visible`
- `hsi_map_layer_visible`

The main remaining weaknesses are:

- `supt_page_visible`, with seen F1 still at `0.0`
- `ins_grnd_alignment_text_visible`, with seen F1 at `0.7077`
- `ins_ok_text_visible`, with seen F1 at `0.0`

This suggests that Gemma handles FCS-MC subpage structure reasonably well, but still lags behind the best Qwen adapter on SUPT recognition and small INS alignment text.

## 6. Results on Run-004 Random Holdout

![Gemma Run-004 overall accuracy](assets/gemma4_vlm_finetune/gemma_run004_random_overall_accuracy.png)

![Gemma Run-004 fact F1 by model](assets/gemma4_vlm_finetune/gemma_run004_random_fact_f1_by_model.png)

![Gemma Run-004 seen F1 by fact](assets/gemma4_vlm_finetune/gemma_run004_random_seen_f1_by_fact.png)

![Gemma Run-004 critical false positives](assets/gemma4_vlm_finetune/gemma_run004_random_critical_false_positives.png)

### 6.1 Gemma base vs LoRA

| Metric | Base | LoRA |
|---|---:|---:|
| JSON valid rate | 0.99 | 1.00 |
| schema valid rate | 0.99 | 1.00 |
| fact accuracy | 0.8146 | 0.9715 |
| macro F1 | 0.4400 | 0.5630 |
| seen F1 | 0.6332 | 0.7959 |
| sample exact match | 0.12 | 0.74 |
| critical false positives | 47 | 13 |

Relative to the Gemma base model, LoRA again brings a strong improvement:

- `fact_accuracy`: `+0.1569`
- `macro_f1`: `+0.1229`
- `seen_f1`: `+0.1627`
- `sample_exact_match`: `+0.62`
- `critical_false_positive_count`: `47 -> 13`

### 6.2 Comparison with the best Qwen LoRA

| Metric | Qwen LoRA | Gemma LoRA |
|---|---:|---:|
| fact accuracy | 0.9931 | 0.9715 |
| macro F1 | 0.6100 | 0.5630 |
| seen F1 | 0.9159 | 0.7959 |
| sample exact match | 0.92 | 0.74 |
| critical false positives | 8 | 13 |

On this random holdout, the gap between Gemma and Qwen is even clearer. Gemma remains much stronger than the Gemma base model, but still below the best Qwen LoRA overall.

### 6.3 Main failure mode

The most prominent Gemma LoRA failure on Run-004 is concentrated in:

```text
fcsmc_intermediate_result_visible
```

Its behavior is:

- seen F1: `0.5185`
- critical false positives: `13`

This means that under random multi-display cockpit compositions, Gemma still tends to over-predict `PBIT GO / intermediate result` on some FCS-MC pages where that state is not actually present. Compared with Run-002, the problem is more visible under the more compositionally varied Run-004 distribution.

## 7. Discussion

### 7.1 Did Gemma learn the task?

Yes. On both `Run-002` and `Run-004`, Gemma LoRA improves over the Gemma base model by a wide margin. This shows that the current 13-fact task, the `Run-003 + Run-005x2` data recipe, and the current structured-output target are not uniquely effective for Qwen. They transfer to Gemma as well.

### 7.2 How does Gemma differ from Qwen?

The main difference is not whether the task is learnable, but how the backbone fails:

- Gemma is more conservative, especially on `Run-002`, where it reduces critical false positives to `0`
- that conservatism also lowers recall, seen F1, and sample exact match
- Qwen remains stronger on SUPT, small INS text, and simultaneous multi-fact correctness
- Gemma still does not solve the `fcsmc_intermediate_result_visible` false-positive issue on the random holdout

### 7.3 Implications for follow-up work

If the immediate goal is the strongest practical adapter, the current Qwen3.5-9B `Run-003 + Run-005x2` line remains the better choice.

If the goal is to study backbone-dependent risk preferences and error patterns, Gemma remains useful, especially for the following questions:

1. whether its conservative decision boundary can be improved with prompt or small-scale data adjustments;
2. whether `fcsmc_intermediate_result_visible` needs more targeted hard negatives;
3. whether INS small-text cues need dedicated visual supplementation or image-quality interventions.

## 8. Conclusion

This Gemma-4-31B VLM LoRA experiment completes a backbone-transfer replication of the Qwen Run-005 recipe under the same ontology, the same training data, and the same holdout protocol.

The conclusions can be summarized as follows:

1. the `Run-003 + Run-005x2` data recipe is also effective for Gemma, and Gemma LoRA shows stable and substantial gains over the Gemma base model;
2. Gemma exhibits a more conservative error pattern and can reduce critical false positives to `0` on `Run-002`;
3. however, Gemma LoRA still does not exceed the current best Qwen LoRA, especially in seen F1, sample exact match, SUPT recognition, and INS small-text detection.

The most accurate conclusion for this round is therefore not that Gemma is better than Qwen, but rather:

> On the current 13-fact cockpit visual fact extraction task, the Gemma backbone benefits clearly from the same LoRA fine-tuning recipe, but the best overall result still comes from Qwen3.5-9B.
