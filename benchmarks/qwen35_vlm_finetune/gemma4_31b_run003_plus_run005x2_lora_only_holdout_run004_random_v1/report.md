# Gemma 4 VLM Fact Benchmark

- benchmark_kind: `holdout_run004_random`
- dataset: `/scratch/yz50/iefmmq_vlm_ft_unsloth/data/holdout_run004_random/reviewed.jsonl`
- base_model: `google/gemma-4-31B`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| lora | 1.0 | 1.0 | 0.973077 | 0.56434 | 0.799121 | 0.75 | 11 |

## Fact Scores

### lora

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| supt_page_visible | 0.97 | 0.328257 | 0.0 | 0.0 | 0.0 |
| fcs_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_page_x_marks_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| bit_root_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_page_visible | 0.98 | 0.640523 | 0.976744 | 1.0 | 0.988235 |
| fcsmc_intermediate_result_visible | 0.89 | 0.499048 | 0.388889 | 1.0 | 0.56 |
| fcsmc_in_test_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_final_go_result_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| hsi_page_visible | 1.0 | 0.333333 | 1.0 | 1.0 | 1.0 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 0.81 | 0.535256 | 1.0 | 0.724638 | 0.840336 |
| ins_ok_text_visible | 1.0 | 0.333333 | 0.0 | 0.0 | 0.0 |
