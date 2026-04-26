# Gemma 4 VLM Fact Benchmark

- benchmark_kind: `holdout_run002_newfacts`
- dataset: `datasets/vision_sft_holdout_run002_newfacts/reviewed.jsonl`
- base_model: `google/gemma-4-31B`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1.0 | 1.0 | 0.816923 | 0.4432 | 0.512756 | 0.04 | 16 |
| lora | 1.0 | 1.0 | 0.949231 | 0.585676 | 0.812294 | 0.44 | 0 |

## Fact Scores

### base

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| tac_page_visible | 0.6 | 0.399573 | 0.551724 | 0.695652 | 0.615385 |
| supt_page_visible | 0.94 | 0.544567 | 0.75 | 0.6 | 0.666667 |
| fcs_page_visible | 0.82 | 0.475439 | 0.833333 | 0.384615 | 0.526316 |
| fcs_page_x_marks_visible | 0.98 | 0.329966 | 0.0 | 0.0 | 0.0 |
| bit_root_page_visible | 0.92 | 0.588967 | 0.692308 | 1.0 | 0.818182 |
| fcsmc_page_visible | 0.86 | 0.529915 | 0.583333 | 0.777778 | 0.666667 |
| fcsmc_intermediate_result_visible | 0.98 | 0.329966 | 0.0 | 0.0 | 0.0 |
| fcsmc_in_test_visible | 0.92 | 0.540404 | 0.666667 | 0.666667 | 0.666667 |
| fcsmc_final_go_result_visible | 0.94 | 0.513057 | 0.4 | 1.0 | 0.571429 |
| hsi_page_visible | 0.98 | 0.66323 | 1.0 | 0.979592 | 0.989691 |
| hsi_map_layer_visible | 0.6 | 0.320166 | 0.574468 | 1.0 | 0.72973 |
| ins_grnd_alignment_text_visible | 0.38 | 0.25184 | 1.0 | 0.261905 | 0.415094 |
| ins_ok_text_visible | 0.7 | 0.27451 | 0.0 | 0.0 | 0.0 |

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
