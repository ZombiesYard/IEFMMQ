# Experiment Inventory

> 目标：盘点当前仓库中**实际存在**的实验数据、训练产物、benchmark、图表与缺口。  
> 约束：不把脚本“理论上会产出的文件”误写成“仓库里现在确实有的文件”。

## 1. 数据集统计文件

当前 `datasets/` 目录下，仓库直接跟踪的是以下 `stats.json`：

| 数据集目录 | 证据文件 | 样本/统计摘要 | 角色 |
|---|---|---:|---|
| `datasets/vision_sft/` | `datasets/vision_sft/stats.json` | `total_tasks=180`, `reviewed_samples=180`, `exported_languages=[en, zh]` | Run-001 / 8-fact SFT 数据 |
| `datasets/vision_sft_holdout_run002/` | `datasets/vision_sft_holdout_run002/stats.json` | `total_tasks=50`, `reviewed_samples=50`, `exported_languages=[en]` | Run-002 / 8-fact holdout |
| `datasets/vision_sft_holdout_run002_newfacts/` | `datasets/vision_sft_holdout_run002_newfacts/stats.json` | `total_tasks=50`, `reviewed_samples=50`, `exported_languages=[en]` | Run-002 / 13-fact holdout |
| `datasets/vision_sft_holdout_run004_random/` | `datasets/vision_sft_holdout_run004_random/stats.json` | `total_tasks=100`, `reviewed_samples=100`, `exported_languages=[en]` | Run-004 random / 13-fact holdout |
| `datasets/vision_sft_run003/` | `datasets/vision_sft_run003/stats.json` | `total_tasks=220`, `reviewed_samples=220`, `exported_languages=[en, zh]`, `missing_summary_count=82` | Run-003 / 13-fact training source |
| `datasets/vision_sft_run005_composition_rebalance/` | `datasets/vision_sft_run005_composition_rebalance/stats.json` | `total_tasks=122`, `reviewed_samples=122`, `exported_languages=[en, zh]`, `missing_summary_count=25` | Run-005 composition rebalance |

### 1.1 8-fact v1 数据

8-fact 相关数据集：

- `datasets/vision_sft/stats.json`
- `datasets/vision_sft_holdout_run002/stats.json`

可直接确认的信息：

- Run-001 v1 训练源共有 180 个 reviewed samples
- Run-002 v1 holdout 共有 50 个 reviewed samples
- Run-001 使用双语导出；Run-002 v1 仅导出英文

### 1.2 13-fact v2 数据

13-fact 相关数据集：

- `datasets/vision_sft_holdout_run002_newfacts/stats.json`
- `datasets/vision_sft_holdout_run004_random/stats.json`
- `datasets/vision_sft_run003/stats.json`
- `datasets/vision_sft_run005_composition_rebalance/stats.json`

当前 v2 fact set 证据来自：

- `packs/fa18c_startup/vision_facts.yaml`
- `tools/export_vision_sft_dataset.py`
- `adapters/vision_fact_prompting.py`

13 个 facts 为：

- `tac_page_visible`
- `supt_page_visible`
- `fcs_page_visible`
- `fcs_page_x_marks_visible`
- `bit_root_page_visible`
- `fcsmc_page_visible`
- `fcsmc_intermediate_result_visible`
- `fcsmc_in_test_visible`
- `fcsmc_final_go_result_visible`
- `hsi_page_visible`
- `hsi_map_layer_visible`
- `ins_grnd_alignment_text_visible`
- `ins_ok_text_visible`

## 2. 训练产物

### 2.1 Qwen LoRA artifacts

| 目录 | 证据文件 | 可直接确认的信息 |
|---|---|---|
| `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/` | `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/train_summary.json` | `max_seq_length=4096`, `num_train_epochs=4.0`, `learning_rate=0.0002`, `train_rows=324`, `eval_rows=36` |
| `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_v1/` | `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_v1/train_summary.json` | `train_rows=396`, `eval_rows=44` |
| `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/` | `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/train_summary.json` | `train_rows=928`, `eval_rows=0` |
| `models/qwen35_vlm_lora/smoke_qwen35_9b_base_bilingual_v4/` | `models/qwen35_vlm_lora/smoke_qwen35_9b_base_bilingual_v4/train_summary.json` | smoke run, `train_rows=7`, `eval_rows=1` |

Qwen artifact 总览说明：

- `models/qwen35_vlm_lora/README.md`

### 2.2 Gemma LoRA artifacts

| 目录 | 证据文件 | 可直接确认的信息 |
|---|---|---|
| `models/gemma4_vlm_lora/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/` | `models/gemma4_vlm_lora/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/train_summary.json` | `max_seq_length=4096`, `num_train_epochs=4.0`, `learning_rate=0.0002`, `load_in_4bit=True`, `gpu_memory_utilization=0.95`, `chat_template=gemma-4`, `train_rows=928`, `eval_rows=0` |
| `models/gemma4_vlm_lora/smoke_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/` | `models/gemma4_vlm_lora/smoke_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/train_summary.json` | smoke run, `train_rows=4`, `eval_rows=0` |

Gemma README 目前更接近 trainer 自动生成 model card，而不是完整实验说明：

- `models/gemma4_vlm_lora/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/README.md`

## 3. Benchmark 目录总览

当前仓库存在以下 benchmark 目录：

- `benchmarks/qwen35_vlm_finetune/base_vs_lora_current180_v1/`
- `benchmarks/qwen35_vlm_finetune/base_vs_lora_holdout_run002_v1/`
- `benchmarks/qwen35_vlm_finetune/run003_lora_vs_base_holdout_run002_newfacts_v1/`
- `benchmarks/qwen35_vlm_finetune/run003_lora_vs_base_holdout_run004_random_v1/`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run004_random_v1/`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_lora_only_holdout_run002_newfacts_v1/`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_lora_only_holdout_run004_random_v1/`
- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run002_newfacts_v2/`
- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run004_random_v2/`
- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_lora_only_holdout_run002_newfacts_v1/`
- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_lora_only_holdout_run004_random_v1/`

每个完整 benchmark 目录已确认实际包含以下类型文件：

- `benchmark_summary.json`
- `metrics_base.json` / `metrics_lora.json` 或仅 `metrics_lora.json`
- `comparison.json`
- `predictions_base.jsonl` / `predictions_lora.jsonl` 或仅 `predictions_lora.jsonl`
- `errors_base.jsonl` / `errors_lora.jsonl` 或仅 `errors_lora.jsonl`
- `fact_scores.csv`
- `report.md`
- `charts/*.png`

### 3.1 训练/验证集 overlap 验证

来自 `Doc/Vision/Reports/vlm_backbone_comparison_summary_EN.md`：

- `Run-002` 与 training set 的 **exact overlap = 0**
- `Run-004` 与 training set 的 **exact overlap = 0**

这是 holdout 独立性的关键数据点，应在论文中明确说明。

证据目录样本：

- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/`
- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run004_random_v2/`

## 4. 当前最可引用的 benchmark 数字

### 4.1 8-fact Qwen v1：Run-002 heldout

证据文件：

- `benchmarks/qwen35_vlm_finetune/base_vs_lora_holdout_run002_v1/metrics_base.json`
- `benchmarks/qwen35_vlm_finetune/base_vs_lora_holdout_run002_v1/metrics_lora.json`
- `benchmarks/qwen35_vlm_finetune/base_vs_lora_holdout_run002_v1/benchmark_summary.json`

| 指标 | Base | LoRA-v1 |
|---|---:|---:|
| `json_valid_rate` | 1.0 | 1.0 |
| `schema_valid_rate` | 1.0 | 1.0 |
| `fact_accuracy` | 0.76 | 0.915 |
| `macro_f1` | 0.424124 | 0.584653 |
| `seen_f1` | 0.471372 | 0.837999 |
| `sample_exact_match` | 0.06 | 0.36 |
| `critical_false_positive_count` | 13 | 15 |

说明：

- 这是 8-fact 老实验线
- `critical_false_positive_count` 在此线里没有下降，反而从 13 到 15

### 4.2 13-fact Qwen：Run-003 + Run-005x2 vs Base on holdout Run-002 newfacts

证据文件：

- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/metrics_base.json`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/metrics_lora.json`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/benchmark_summary.json`

| 指标 | Base | LoRA |
|---|---:|---:|
| `json_valid_rate` | 1.0 | 1.0 |
| `schema_valid_rate` | 1.0 | 1.0 |
| `fact_accuracy` | 0.867692 | 0.990769 |
| `macro_f1` | 0.451301 | 0.651478 |
| `seen_f1` | 0.502213 | 0.964829 |
| `sample_exact_match` | 0.10 | 0.88 |
| `critical_false_positive_count` | 11 | 4 |

这是当前仓库里最适合作为 Qwen 主结果候选的 13-fact holdout 证据之一。

### 4.3 13-fact Qwen：Run-003 + Run-005x2 vs Base on holdout Run-004 random

证据文件：

- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run004_random_v1/metrics_base.json`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run004_random_v1/metrics_lora.json`

| 指标 | Base | LoRA |
|---|---:|---:|
| `json_valid_rate` | 0.96 | 1.0 |
| `schema_valid_rate` | 0.96 | 1.0 |
| `fact_accuracy` | 0.803846 | 0.993077 |
| `macro_f1` | 0.439308 | 0.610003 |
| `seen_f1` | 0.565909 | 0.915938 |
| `sample_exact_match` | 0.03 | 0.92 |
| `critical_false_positive_count` | 70 | 8 |

### 4.4 13-fact Gemma：Run-003 + Run-005x2 vs Base on holdout Run-002 newfacts

证据文件：

- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run002_newfacts_v2/metrics_base.json`
- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run002_newfacts_v2/metrics_lora.json`

| 指标 | Base | LoRA |
|---|---:|---:|
| `json_valid_rate` | 1.0 | 1.0 |
| `schema_valid_rate` | 1.0 | 1.0 |
| `fact_accuracy` | 0.816923 | 0.949231 |
| `macro_f1` | 0.4432 | 0.585676 |
| `seen_f1` | 0.512756 | 0.812294 |
| `sample_exact_match` | 0.04 | 0.44 |
| `critical_false_positive_count` | 16 | 0 |

### 4.5 13-fact Gemma：Run-003 + Run-005x2 vs Base on holdout Run-004 random

证据文件：

- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run004_random_v2/metrics_base.json`
- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run004_random_v2/metrics_lora.json`

| 指标 | Base | LoRA |
|---|---:|---:|
| `json_valid_rate` | 0.99 | 1.0 |
| `schema_valid_rate` | 0.99 | 1.0 |
| `fact_accuracy` | 0.814615 | 0.971538 |
| `macro_f1` | 0.440019 | 0.562961 |
| `seen_f1` | 0.633198 | 0.79593 |
| `sample_exact_match` | 0.12 | 0.74 |
| `critical_false_positive_count` | 47 | 13 |

## 5. 图表与报告文件

当前每个完整 benchmark 目录都已存有图表与报告，典型文件包括：

- `charts/overall_accuracy.png`
- `charts/fact_f1_by_model.png`
- `charts/seen_f1_by_fact.png`
- `charts/critical_false_positives.png`
- `charts/confusion_*.png`
- `report.md`

具体证据：

- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/charts/overall_accuracy.png`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/report.md`
- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run004_random_v2/charts/critical_false_positives.png`

另外，仓库中还有技术报告与报告图资产：

- `Doc/Vision/Reports/qwen35_vlm_finetune_report_EN.md`
- `Doc/Vision/Reports/qwen35_vlm_finetune_report_ZH.md`
- `Doc/Vision/Reports/assets/gemma4_vlm_finetune/*.png`

## 6. 日志与运行时产物

当前仓库可确认存在的运行 / replay 相关产物：

- `replay_eval/fa18c_startup_v04/suite.yaml`
- `replay_eval/fa18c_startup_v04/cases/*/dcs_bios_raw.jsonl`
- `replay_eval/fa18c_startup_v04/cases/*/Saved Games/DCS/SimTutor/frames/.../frames.jsonl`
- `logs/test_index/index.json`

可直接确认的含义：

- 有 replay evaluation 数据骨架
- 有 BIOS raw capture
- 有 frame sidecar 与 VLM-ready artifact 示例
- 5 个 replay evaluation cases：`batteryon_2min`、`Batteryon_enginGenoff_2min`、`fcs_reset_fcs_bit_2min`、`ins_2min`、`noop_2min`

但当前仓库**没有**一套已经完成、可直接在论文里当人因实验结果使用的 runtime 数据收集表。

## 7. 缺口与待核对项

以下内容在当前仓库中没有足够证据支持“已完成结果”写法：

| 项目 | 当前状态 | 处理方式 |
|---|---|---|
| VR/Quest 3 runtime logs | 仓库中未见可直接引用实验数据 | `TODO:VERIFY` |
| 用户研究 raw data / tables | 未见 participant-level 数据表 | `TODO:VERIFY` |
| NASA-TLX 实测结果 | 仅见表单与 proposal 计划，不见结果数据 | `TODO:COMPUTE` |
| retention / delayed post-test 数据 | 未见 | `TODO:COMPUTE` |
| runtime 数据收集系统最终字段设计 | 尚未冻结 | `TODO:VERIFY` |
| benchmark 主结果线最终选型 | Qwen / Gemma / 8-fact / 13-fact 尚需论文层面决定 | `TODO:VERIFY` |
| `run003_plus_run005x2_lora_only_holdout_run004_random_v1` 的 `benchmark_kind` | `benchmark_summary.json` 中写为 `contaminated_dev_set`，但目录名是 holdout_run004_random | `TODO:VERIFY` |

## 8. 当前可安全写入论文的结论边界

基于仓库现状，当前可以安全写入“Current Technical Results”的只有：

1. 数据集规模与 fact distribution，来源于 `datasets/*/stats.json`
2. LoRA training configuration，来源于 `models/*/train_summary.json`
3. Base vs LoRA benchmark 指标，来源于 `benchmarks/*/metrics_*.json` 与 `benchmark_summary.json`
4. 已存图表与 report 文件，来源于 `benchmarks/*/charts` 与 `report.md`

当前**不能**写成已完成结果的内容包括：

1. VR user study 结果
2. runtime data collection system 产出的 human-subject evidence
3. NASA-TLX、retention、learning gain 的真实数值

