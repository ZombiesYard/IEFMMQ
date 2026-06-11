# Qwen3.6-27B VLM Fact Benchmark

- benchmark_kind: `holdout_run004_random`
- dataset: `/scratch/yz50/iefmmq_vlm_ft_unsloth/data/holdout_run004_random/reviewed.jsonl`
- base_model: `Qwen/Qwen3.6-27B`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_qwen36_27b_base_bilingual_run003_plus_run005x2_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1.0 | 1.0 | 0.976923 | 0.572389 | 0.821085 | 0.7 | 7 |
| lora | 1.0 | 1.0 | 0.989231 | 0.607541 | 0.914727 | 0.86 | 7 |

## Fact Scores

### base

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| supt_page_visible | 0.97 | 0.328257 | 0.0 | 0.0 | 0.0 |
| fcs_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_page_x_marks_visible | 0.99 | 0.627884 | 1.0 | 0.8 | 0.888889 |
| bit_root_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_intermediate_result_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_in_test_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_final_go_result_visible | 0.93 | 0.616331 | 0.895522 | 1.0 | 0.944882 |
| hsi_page_visible | 1.0 | 0.333333 | 1.0 | 1.0 | 1.0 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 0.81 | 0.535256 | 1.0 | 0.724638 | 0.840336 |
| ins_ok_text_visible | 1.0 | 0.333333 | 0.0 | 0.0 | 0.0 |

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
| fcsmc_final_go_result_visible | 0.93 | 0.616331 | 0.895522 | 1.0 | 0.944882 |
| hsi_page_visible | 1.0 | 0.333333 | 1.0 | 1.0 | 1.0 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 0.93 | 0.615039 | 1.0 | 0.898551 | 0.946565 |
| ins_ok_text_visible | 1.0 | 0.333333 | 0.0 | 0.0 | 0.0 |
