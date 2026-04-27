# VLM Backbone Comparison Summary

## Task Setup

This page summarizes the current backbone comparison under a matched task setup:

- Task: composite-panel visual fact extraction for F/A-18C cold start
- Ontology: fixed 13-fact ontology, no further fact redesign
- Training data: `Run-003 + Run-005 composition rebalance`
- Weighting strategy: `Run-003 bilingual once + Run-005 bilingual twice`
- Training input: `928` rows
- Evaluation sets:
  - `Run-002 newfacts holdout` (50 samples)
  - `Run-004 random holdout` (100 samples)
- Overlap checks:
  - exact overlap between `Run-002` and training = `0`
  - exact overlap between `Run-004` and training = `0`

## Compared Backbones

| backbone | adapter/run | note |
|---|---|---|
| `Qwen/Qwen3.5-9B-Base` | `Run-003 + Run-005x2 LoRA` | current best configuration |
| `google/gemma-4-31B` | `Run-003 + Run-005x2 LoRA` | replicated under the same data and evaluation |

## Training Configuration Comparison

### Shared setup

- same 13-fact ontology
- same training data and oversampling recipe
- same bilingual SFT format
- same LoRA rank / alpha / dropout:
  - `r=16`
  - `alpha=16`
  - `dropout=0.0`
- same:
  - `epochs=4`
  - `learning_rate=2e-4`
  - `batch_size=1`
  - `gradient_accumulation=4`
  - `effective batch size=4`
  - `max_seq_length=4096`
  - `seed=3407`
- same external holdout benchmarks

### Differences

| Item | Qwen | Gemma |
|---|---|---|
| backbone | `Qwen3.5-9B-Base` | `Gemma-4-31B` |
| LoRA attachment | Qwen-side explicit vision/language setup | `all-linear` |
| vision LoRA | enabled | `finetune_vision_layers=false` |
| chat template | Qwen-side default | `gemma-4` |
| gpu memory utilization | lower | `0.95` |
| train runtime | `6419.16 s` | `8049.65 s` |
| final train loss | `0.2156` | `0.0474` |

The final training loss should not be compared directly across backbones, because tokenizer behavior, template behavior, target distribution, and parameterization differ.

## Benchmark Results

### Run-002 Newfacts Holdout

| model | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_fp |
|---|---:|---:|---:|---:|---:|
| Qwen base | 0.8677 | 0.4513 | 0.5022 | 0.10 | 11 |
| Qwen LoRA | **0.9908** | **0.6515** | **0.9648** | **0.88** | 4 |
| Gemma base | 0.8169 | 0.4432 | 0.5128 | 0.04 | 16 |
| Gemma LoRA | 0.9492 | 0.5857 | 0.8123 | 0.44 | **0** |

### Run-004 Random Holdout

| model | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_fp |
|---|---:|---:|---:|---:|---:|
| Qwen base | 0.8038 | 0.4393 | 0.5659 | 0.03 | 70 |
| Qwen LoRA | **0.9931** | **0.6100** | **0.9159** | **0.92** | **8** |
| Gemma base | 0.8146 | 0.4400 | 0.6332 | 0.12 | 47 |
| Gemma LoRA | 0.9715 | 0.5630 | 0.7959 | 0.74 | 13 |

## Main Findings

### 1. The same data recipe works for both backbones

`Run-003 + Run-005x2` is not only effective for Qwen. Gemma also improves substantially over its base model on both holdouts, which supports the following:

- the current 13-fact ontology is transferable;
- the composition-rebalance + hard-negative + moderate-oversampling strategy remains effective across backbones.

### 2. The best overall result still comes from Qwen

On both holdouts, Qwen LoRA gives higher:

- `fact_accuracy`
- `seen_f1`
- `sample_exact_match`

The gap is especially clear on `sample_exact_match`, which requires all facts in a sample to be correct:

- Run-002: `0.88 vs 0.44`
- Run-004: `0.92 vs 0.74`

### 3. Gemma is more conservative, but not stronger overall

Gemma’s clearest advantage is that it reduces `critical false positives` to `0` on `Run-002`, indicating a more conservative behavior on high-risk completion-type facts.

That conservatism also comes with a cost:

- lower recall
- lower `seen_f1`
- much lower `sample_exact_match`

The more accurate summary is therefore:

> Gemma has a more conservative error style, but its overall extraction quality is still below the best Qwen LoRA.

## Error Pattern Comparison

### Qwen

Strengths:

- more stable TAC / SUPT / FCS / BIT root / FCS-MC page recognition
- stronger simultaneous multi-fact correctness
- stronger overall handling of small INS text

Main remaining issues:

- `fcsmc_final_go_result_visible`
- a small number of completion-type false positives

### Gemma

Strengths:

- clear base-to-LoRA improvement
- `critical_fp = 0` on `Run-002`
- stable structural recognition on several FCS-MC page states

Main weaknesses:

- weak `supt_page_visible`
- weak `ins_grnd_alignment_text_visible`
- weak `ins_ok_text_visible`
- concentrated false positives on `fcsmc_intermediate_result_visible` in `Run-004`

## Practical Conclusion

If the current goal is deployment or downstream system integration, **Qwen3.5-9B Run-003 + Run-005x2 LoRA remains the better practical choice**, because it is stronger in overall accuracy, seen F1, and exact match.

If the current goal is to study backbone-dependent behavior, Gemma should still be retained as a comparison backbone because it demonstrates:

- a more conservative critical-fact style;
- a different false-positive / false-negative trade-off from Qwen.
