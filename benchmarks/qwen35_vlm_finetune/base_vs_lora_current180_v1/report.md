# Qwen3.5 VLM Fact Benchmark

- benchmark_kind: `contaminated_dev_set`
- dataset: `<scratch-root>/iefmmq_vlm_ft_unsloth/data/reviewed.jsonl`
- base_model: `Qwen/Qwen3.5-9B-Base`
- adapter: `<scratch-root>/iefmmq_vlm_ft_unsloth/runs/full_qwen35_9b_base_bilingual_v1/adapter`

## Overall Metrics

| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1.0 | 1.0 | 0.795139 | 0.470475 | 0.60801 | 0.183333 | 19 |
| lora | 1.0 | 1.0 | 0.969444 | 0.637494 | 0.951653 | 0.777778 | 2 |

## Fact Scores

### base

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| fcs_page_visible | 0.677778 | 0.398974 | 0.885965 | 0.691781 | 0.776923 |
| bit_root_page_visible | 0.577778 | 0.375998 | 0.947368 | 0.327273 | 0.486486 |
| bit_page_failure_visible | 0.994444 | 0.664269 | 1.0 | 1.0 | 1.0 |
| right_ddi_fcsmc_page_visible | 0.916667 | 0.598668 | 0.745763 | 1.0 | 0.854369 |
| right_ddi_in_test_visible | 0.994444 | 0.659513 | 0.964286 | 1.0 | 0.981818 |
| fcs_bit_result_visible | 0.888889 | 0.371463 | 0.666667 | 0.1 | 0.173913 |
| ins_alignment_page_visible | 0.4 | 0.223802 | 0.727273 | 0.070796 | 0.129032 |
| ins_go | 0.911111 | 0.471115 | 0.75 | 0.333333 | 0.461538 |

### lora

| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |
|---|---:|---:|---:|---:|---:|
| fcs_page_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| bit_root_page_visible | 0.977778 | 0.651082 | 0.981818 | 0.981818 | 0.981818 |
| bit_page_failure_visible | 0.994444 | 0.664269 | 1.0 | 1.0 | 1.0 |
| right_ddi_fcsmc_page_visible | 0.988889 | 0.65679 | 0.956522 | 1.0 | 0.977778 |
| right_ddi_in_test_visible | 1.0 | 0.666667 | 1.0 | 1.0 | 1.0 |
| fcs_bit_result_visible | 0.977778 | 0.635499 | 1.0 | 0.85 | 0.918919 |
| ins_alignment_page_visible | 0.855556 | 0.553212 | 0.817518 | 0.99115 | 0.896 |
| ins_go | 0.961111 | 0.605768 | 1.0 | 0.722222 | 0.83871 |
