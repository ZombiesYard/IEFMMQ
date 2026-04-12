# Qwen3.5 VLM Fact Benchmark

- benchmark_kind: `heldout_new_session`
- dataset: `/scratch/yz50/iefmmq_vlm_ft_unsloth/data/holdout_run002/reviewed.jsonl`
- base_model: `Qwen/Qwen3.5-9B-Base`
- adapter: `/scratch/yz50/iefmmq_vlm_ft_unsloth/runs/full_qwen35_9b_base_bilingual_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1.0 | 1.0 | 0.76 | 0.424124 | 0.471372 | 0.06 | 13 |
| lora | 1.0 | 1.0 | 0.915 | 0.584653 | 0.837999 | 0.36 | 15 |

## Fact Scores

### base

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| fcs_page_visible | 0.46 | 0.305366 | 0.294118 | 0.769231 | 0.425532 |
| bit_root_page_visible | 0.8 | 0.440831 | 0.444444 | 0.444444 | 0.444444 |
| bit_page_failure_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| right_ddi_fcsmc_page_visible | 0.92 | 0.588967 | 0.692308 | 1.0 | 0.818182 |
| right_ddi_in_test_visible | 0.88 | 0.531165 | 0.5 | 1.0 | 0.666667 |
| fcs_bit_result_visible | 0.9 | 0.41065 | 0.25 | 0.333333 | 0.285714 |
| ins_alignment_page_visible | 0.2 | 0.129898 | 1.0 | 0.069767 | 0.130435 |
| ins_go | 0.92 | 0.319444 | 0.0 | 0.0 | 0.0 |

### lora

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| fcs_page_visible | 0.84 | 0.547831 | 0.619048 | 1.0 | 0.764706 |
| bit_root_page_visible | 0.94 | 0.606389 | 0.75 | 1.0 | 0.857143 |
| bit_page_failure_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| right_ddi_fcsmc_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| right_ddi_in_test_visible | 0.96 | 0.611296 | 0.75 | 1.0 | 0.857143 |
| fcs_bit_result_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| ins_alignment_page_visible | 0.88 | 0.541667 | 1.0 | 0.860465 | 0.925 |
| ins_go | 0.7 | 0.370042 | 0.1875 | 0.75 | 0.3 |
