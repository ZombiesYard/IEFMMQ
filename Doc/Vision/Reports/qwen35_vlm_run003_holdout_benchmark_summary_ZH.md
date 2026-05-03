# Qwen3.5 VLM Run-003 Holdout Benchmark Summary

本文汇总 Run-003 新 13-fact ontology LoRA 在两个 holdout set 上的 base vs LoRA 结果。Run-002 new-facts holdout 更适合作为主要独立评估；Run-004 random holdout 含较多重复与偏置样本，更适合作为随机操作压力参考。

## Dataset Scope

| split | samples | role |
|---|---:|---|
| Run-002 new-facts holdout | 50 | 主要独立 holdout，使用 Run-003 prompt 与新 facts 重新标注 |
| Run-004 random holdout | 100 | 随机操作/重复样本压力测试，不作为主要泛化结论 |

## Overall Metrics

| split | model | fact accuracy | macro F1 | seen F1 | exact match | critical FP |
|---|---|---:|---:|---:|---:|---:|
| Run-002 new-facts | base | 0.8677 | 0.4513 | 0.5022 | 0.1000 | 11 |
| Run-002 new-facts | LoRA | 0.9154 | 0.5306 | 0.6680 | 0.2000 | 20 |
| Run-004 random | base | 0.8038 | 0.4393 | 0.5659 | 0.0300 | 70 |
| Run-004 random | LoRA | 0.9185 | 0.4901 | 0.6390 | 0.5200 | 19 |

## Main Findings

Run-002 new-facts shows that the Run-003 LoRA improves overall structured visual fact extraction: fact accuracy rises from 0.8677 to 0.9154, seen F1 rises from 0.5022 to 0.6680, and sample exact match doubles from 0.10 to 0.20.

The main caveat is critical false positives. On Run-002 new-facts, total critical FP rises from 11 to 20. This is almost entirely caused by `ins_ok_text_visible`, where LoRA introduces 17 false positives. This means the new ontology fixed much of the old FCS-MC confusion, but the completion-style INS cue is still the key safety failure mode.

For FCS-MC facts on Run-002 new-facts, the LoRA is clearly better:

| fact | accuracy base -> LoRA | seen F1 base -> LoRA | seen FP base -> LoRA |
|---|---:|---:|---:|
| `fcsmc_page_visible` | 0.96 -> 0.98 | 0.89 -> 0.94 | 1 -> 0 |
| `fcsmc_intermediate_result_visible` | 0.84 -> 0.94 | 0.20 -> 0.00 | 8 -> 2 |
| `fcsmc_in_test_visible` | 0.90 -> 0.98 | 0.44 -> 0.91 | 1 -> 0 |
| `fcsmc_final_go_result_visible` | 0.94 -> 0.98 | 0.40 -> 0.80 | 2 -> 1 |

For HSI/INS facts on Run-002 new-facts, LoRA learns the alignment text much better, but still over-predicts OK:

| fact | accuracy base -> LoRA | seen F1 base -> LoRA | seen FP base -> LoRA | seen FN base -> LoRA |
|---|---:|---:|---:|---:|
| `ins_grnd_alignment_text_visible` | 0.16 -> 0.68 | 0.00 -> 0.76 | 0 -> 0 | 42 -> 16 |
| `ins_ok_text_visible` | 0.92 -> 0.58 | 0.00 -> 0.00 | 0 -> 17 | 4 -> 4 |

Run-004 random stress set shows a different pattern: LoRA reduces total critical FP from 70 to 19 and improves exact match from 0.03 to 0.52. Because this set is visually repetitive and skewed, it should support the qualitative claim that the adapter is more structured and less noisy under random cockpit states, but it should not replace Run-002 as the primary holdout.

## Interpretation

Run-003 validates the core hypothesis that a small, domain-specific LoRA can improve VLM extraction of F/A-18C cockpit display facts. The new ontology is also better than the old `ins_go` target because it separates page visibility, alignment text, map overlay, and OK text. However, the remaining high-risk problem is still a completion cue false positive: `ins_ok_text_visible`.

For the next iteration, the highest-value data addition is not more balanced generic pages. It is targeted hard negatives for `ins_ok_text_visible`: HSI alignment pages with GRND/QUAL/TIME visible but no OK, MAP overlay cases where OK-like artifacts appear, low-contrast alignment text, and transition frames around QUAL values before OK appears.

## Chart Assets

Charts are available under `Doc/Vision/Reports/assets/qwen35_vlm_finetune/` with prefixes:

- `run002_newfacts_*`
- `run004_random_*`

