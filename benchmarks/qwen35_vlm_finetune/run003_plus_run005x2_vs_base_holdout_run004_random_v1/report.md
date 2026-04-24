# Qwen3.5 VLM Fact Benchmark

- benchmark_kind: `holdout_run004_random`
- dataset: `/scratch/yz50/iefmmq_vlm_ft_unsloth/data/holdout_run004_random/reviewed.jsonl`
- base_model: `Qwen/Qwen3.5-9B-Base`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.96 | 0.96 | 0.803846 | 0.439308 | 0.565909 | 0.03 | 70 |
| lora | 1.0 | 1.0 | 0.993077 | 0.610003 | 0.915938 | 0.92 | 8 |

## Fact Scores

### base

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 0.94 | 0.572695 | 0.6 | 1.0 | 0.75 |
| supt_page_visible | 0.94 | 0.489583 | 1.0 | 0.333333 | 0.5 |
| fcs_page_visible | 0.82 | 0.451885 | 1.0 | 0.808511 | 0.894118 |
| fcs_page_x_marks_visible | 0.91 | 0.317627 | 0.0 | 0.0 | 0.0 |
| bit_root_page_visible | 0.75 | 0.476316 | 0.432432 | 1.0 | 0.603774 |
| fcsmc_page_visible | 0.94 | 0.633638 | 0.9875 | 0.940476 | 0.963415 |
| fcsmc_intermediate_result_visible | 0.35 | 0.211976 | 0.090909 | 0.857143 | 0.164384 |
| fcsmc_in_test_visible | 0.88 | 0.540573 | 1.0 | 0.529412 | 0.692308 |
| fcsmc_final_go_result_visible | 0.85 | 0.575023 | 0.846154 | 0.916667 | 0.88 |
| hsi_page_visible | 0.85 | 0.306306 | 1.0 | 0.85 | 0.918919 |
| hsi_map_layer_visible | 0.95 | 0.646118 | 1.0 | 0.98 | 0.989899 |
| ins_grnd_alignment_text_visible | 0.31 | 0.16273 | 0.0 | 0.0 | 0.0 |
| ins_ok_text_visible | 0.96 | 0.326531 | 0.0 | 0.0 | 0.0 |

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
| fcsmc_in_test_visible | 0.99 | 0.65457 | 1.0 | 0.941176 | 0.969697 |
| fcsmc_final_go_result_visible | 0.92 | 0.608796 | 0.882353 | 1.0 | 0.9375 |
| hsi_page_visible | 1.0 | 0.333333 | 1.0 | 1.0 | 1.0 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_ok_text_visible | 1.0 | 0.333333 | 0.0 | 0.0 | 0.0 |
