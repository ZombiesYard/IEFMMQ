# Gemma 4 VLM Fact Benchmark

- benchmark_kind: `holdout_run004_random`
- dataset: `datasets/vision_sft_holdout_run004_random/reviewed.jsonl`
- base_model: `google/gemma-4-31B`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.99 | 0.99 | 0.814615 | 0.440019 | 0.633198 | 0.12 | 47 |
| lora | 1.0 | 1.0 | 0.971538 | 0.562961 | 0.79593 | 0.74 | 13 |

## Fact Scores

### base

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 0.82 | 0.363117 | 0.111111 | 0.666667 | 0.190476 |
| supt_page_visible | 0.98 | 0.596564 | 1.0 | 0.666667 | 0.8 |
| fcs_page_visible | 0.68 | 0.349156 | 0.984375 | 0.670213 | 0.797468 |
| fcs_page_x_marks_visible | 0.97 | 0.578098 | 1.0 | 0.6 | 0.75 |
| bit_root_page_visible | 0.99 | 0.655914 | 1.0 | 0.9375 | 0.967742 |
| fcsmc_page_visible | 0.92 | 0.577426 | 0.963855 | 0.952381 | 0.958084 |
| fcsmc_intermediate_result_visible | 0.86 | 0.308244 | 0.0 | 0.0 | 0.0 |
| fcsmc_in_test_visible | 0.92 | 0.570521 | 0.916667 | 0.647059 | 0.758621 |
| fcsmc_final_go_result_visible | 0.81 | 0.536498 | 0.85 | 0.85 | 0.85 |
| hsi_page_visible | 0.93 | 0.321244 | 1.0 | 0.93 | 0.963731 |
| hsi_map_layer_visible | 0.43 | 0.200466 | 0.462366 | 0.86 | 0.601399 |
| ins_grnd_alignment_text_visible | 0.58 | 0.388496 | 0.9375 | 0.434783 | 0.594059 |
| ins_ok_text_visible | 0.7 | 0.27451 | 0.0 | 0.0 | 0.0 |

### lora

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| supt_page_visible | 0.97 | 0.328257 | 0.0 | 0.0 | 0.0 |
| fcs_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_page_x_marks_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| bit_root_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_page_visible | 0.98 | 0.640523 | 0.976744 | 1.0 | 0.988235 |
| fcsmc_intermediate_result_visible | 0.87 | 0.481125 | 0.35 | 1.0 | 0.518519 |
| fcsmc_in_test_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_final_go_result_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| hsi_page_visible | 1.0 | 0.333333 | 1.0 | 1.0 | 1.0 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 0.81 | 0.535256 | 1.0 | 0.724638 | 0.840336 |
| ins_ok_text_visible | 1.0 | 0.333333 | 0.0 | 0.0 | 0.0 |
