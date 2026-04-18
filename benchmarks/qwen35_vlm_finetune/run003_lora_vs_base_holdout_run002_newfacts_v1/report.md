# Qwen3.5 VLM Fact Benchmark

- benchmark_kind: `holdout_run002_newfacts`
- dataset: `/scratch/yz50/iefmmq_vlm_ft_unsloth/data/holdout_run002_newfacts/reviewed.jsonl`
- base_model: `Qwen/Qwen3.5-9B-Base`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_qwen35_9b_base_bilingual_run003_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1.0 | 1.0 | 0.867692 | 0.451301 | 0.502213 | 0.1 | 11 |
| lora | 1.0 | 1.0 | 0.915385 | 0.530643 | 0.667952 | 0.2 | 20 |

## Fact Scores

### base

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 0.9 | 0.597768 | 1.0 | 0.782609 | 0.878049 |
| supt_page_visible | 0.9 | 0.315789 | 0.0 | 0.0 | 0.0 |
| fcs_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_page_x_marks_visible | 0.98 | 0.329966 | 0.0 | 0.0 | 0.0 |
| bit_root_page_visible | 0.9 | 0.572558 | 0.642857 | 1.0 | 0.782609 |
| fcsmc_page_visible | 0.96 | 0.6215 | 0.888889 | 0.888889 | 0.888889 |
| fcsmc_intermediate_result_visible | 0.84 | 0.37037 | 0.111111 | 1.0 | 0.2 |
| fcsmc_in_test_visible | 0.9 | 0.463166 | 0.666667 | 0.333333 | 0.444444 |
| fcsmc_final_go_result_visible | 0.94 | 0.45614 | 0.333333 | 0.5 | 0.4 |
| hsi_page_visible | 0.88 | 0.394928 | 1.0 | 0.877551 | 0.934783 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 0.16 | 0.091954 | 0.0 | 0.0 | 0.0 |
| ins_ok_text_visible | 0.92 | 0.319444 | 0.0 | 0.0 | 0.0 |

### lora

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 0.92 | 0.611932 | 1.0 | 0.826087 | 0.904762 |
| supt_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_page_x_marks_visible | 0.98 | 0.329966 | 0.0 | 0.0 | 0.0 |
| bit_root_page_visible | 0.86 | 0.428328 | 1.0 | 0.222222 | 0.363636 |
| fcsmc_page_visible | 0.98 | 0.643043 | 1.0 | 0.888889 | 0.941176 |
| fcsmc_intermediate_result_visible | 0.94 | 0.323024 | 0.0 | 0.0 | 0.0 |
| fcsmc_in_test_visible | 0.98 | 0.632618 | 1.0 | 0.833333 | 0.909091 |
| fcsmc_final_go_result_visible | 0.98 | 0.596491 | 0.666667 | 1.0 | 0.8 |
| hsi_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 0.68 | 0.421569 | 1.0 | 0.619048 | 0.764706 |
| ins_ok_text_visible | 0.58 | 0.244726 | 0.0 | 0.0 | 0.0 |
