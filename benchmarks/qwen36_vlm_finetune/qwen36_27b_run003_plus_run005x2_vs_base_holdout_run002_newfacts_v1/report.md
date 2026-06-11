# Qwen3.6-27B VLM Fact Benchmark

- benchmark_kind: `holdout_run002_newfacts`
- dataset: `/scratch/yz50/iefmmq_vlm_ft_unsloth/data/holdout_run002_newfacts/reviewed.jsonl`
- base_model: `Qwen/Qwen3.6-27B`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_qwen36_27b_base_bilingual_run003_plus_run005x2_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1.0 | 1.0 | 0.910769 | 0.521807 | 0.671228 | 0.26 | 1 |
| lora | 1.0 | 1.0 | 0.987692 | 0.642504 | 0.947896 | 0.86 | 3 |

## Fact Scores

### base

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 0.64 | 0.399287 | 0.727273 | 0.347826 | 0.470588 |
| supt_page_visible | 0.84 | 0.304348 | 0.0 | 0.0 | 0.0 |
| fcs_page_visible | 0.98 | 0.648889 | 1.0 | 0.923077 | 0.96 |
| fcs_page_x_marks_visible | 0.98 | 0.329966 | 0.0 | 0.0 | 0.0 |
| bit_root_page_visible | 0.92 | 0.588967 | 0.692308 | 1.0 | 0.818182 |
| fcsmc_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_intermediate_result_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_in_test_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_final_go_result_visible | 0.98 | 0.596491 | 0.666667 | 1.0 | 0.8 |
| hsi_page_visible | 0.98 | 0.552119 | 1.0 | 0.979592 | 0.989691 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 0.6 | 0.377315 | 1.0 | 0.52381 | 0.6875 |
| ins_ok_text_visible | 0.92 | 0.319444 | 0.0 | 0.0 | 0.0 |

### lora

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| supt_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_page_x_marks_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| bit_root_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_intermediate_result_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_in_test_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_final_go_result_visible | 0.98 | 0.596491 | 0.666667 | 1.0 | 0.8 |
| hsi_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 0.92 | 0.576333 | 0.975 | 0.928571 | 0.95122 |
| ins_ok_text_visible | 0.94 | 0.513057 | 0.666667 | 0.5 | 0.571429 |
