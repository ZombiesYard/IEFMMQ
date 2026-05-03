# Qwen3.5 VLM Fact Benchmark

- benchmark_kind: `holdout_run002_newfacts`
- dataset: `/scratch/yz50/iefmmq_vlm_ft_unsloth/data/holdout_run002_newfacts/reviewed.jsonl`
- base_model: `Qwen/Qwen3.5-9B-Base`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| lora | 1.0 | 1.0 | 0.990769 | 0.651478 | 0.964829 | 0.88 | 4 |

## Fact Scores

### lora

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 0.96 | 0.63961 | 1.0 | 0.913043 | 0.954545 |
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
| ins_grnd_alignment_text_visible | 0.98 | 0.640523 | 0.976744 | 1.0 | 0.988235 |
| ins_ok_text_visible | 0.96 | 0.592593 | 0.666667 | 1.0 | 0.8 |
