# Gemma 4 VLM Fact Benchmark

- benchmark_kind: `holdout_run002_newfacts`
- dataset: `/scratch/yz50/iefmmq_vlm_ft_unsloth/data/holdout_run002_newfacts/reviewed.jsonl`
- base_model: `google/gemma-4-31B`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| lora | 1.0 | 1.0 | 0.949231 | 0.585676 | 0.812294 | 0.44 | 0 |

## Fact Scores

### lora

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 0.92 | 0.611932 | 1.0 | 0.826087 | 0.904762 |
| supt_page_visible | 0.9 | 0.315789 | 0.0 | 0.0 | 0.0 |
| fcs_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_page_x_marks_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| bit_root_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_page_visible | 0.98 | 0.645007 | 0.9 | 1.0 | 0.947368 |
| fcsmc_intermediate_result_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_in_test_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcsmc_final_go_result_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| hsi_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| hsi_map_layer_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_grnd_alignment_text_visible | 0.62 | 0.388278 | 1.0 | 0.547619 | 0.707692 |
| ins_ok_text_visible | 0.92 | 0.319444 | 0.0 | 0.0 | 0.0 |
