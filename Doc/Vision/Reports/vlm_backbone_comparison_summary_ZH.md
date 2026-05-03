# VLM Backbone Comparison Summary

## 任务设置

本页总结当前两条 backbone 路线在相同任务定义下的对比结果：

- 任务：F/A-18C 冷启动 composite-panel 视觉事实抽取
- ontology：固定 13-fact ontology，不再修改 facts
- 训练数据：`Run-003 + Run-005 composition rebalance`
- 加权策略：`Run-003 bilingual once + Run-005 bilingual twice`
- 训练输入：`928` rows
- 评测集：
  - `Run-002 newfacts holdout`（50 samples）
  - `Run-004 random holdout`（100 samples）
- overlap 检查：
  - `Run-002` 与训练集 exact overlap = `0`
  - `Run-004` 与训练集 exact overlap = `0`

## Compared Backbones

| backbone | adapter/run | 说明 |
|---|---|---|
| `Qwen/Qwen3.5-9B-Base` | `Run-003 + Run-005x2 LoRA` | 当前最优方案 |
| `google/gemma-4-31B` | `Run-003 + Run-005x2 LoRA` | 在相同数据与评测下复现 |

## 训练配置对比

### 相同部分

- 相同 13-fact ontology
- 相同训练数据与过采样策略
- 相同 bilingual SFT 格式
- 相同 LoRA rank / alpha / dropout：
  - `r=16`
  - `alpha=16`
  - `dropout=0.0`
- 相同：
  - `epochs=4`
  - `learning_rate=2e-4`
  - `batch_size=1`
  - `gradient_accumulation=4`
  - `effective batch size=4`
  - `max_seq_length=4096`
  - `seed=3407`
- 相同外部 holdout benchmark

### 不同部分

| 项目 | Qwen | Gemma |
|---|---|---|
| backbone | `Qwen3.5-9B-Base` | `Gemma-4-31B` |
| LoRA 挂载 | Qwen 路线的 vision/language 显式配置 | `all-linear` |
| vision LoRA | 启用 | `finetune_vision_layers=false` |
| chat template | Qwen 路线默认 | `gemma-4` |
| gpu memory utilization | 较低 | `0.95` |
| train runtime | `6419.16 s` | `8049.65 s` |
| final train loss | `0.2156` | `0.0474` |

需要注意：final train loss 不宜直接跨 backbone 比较，因为 tokenizer、template、目标分布和模型参数化方式不同。

## Benchmark Results

### Run-002 Newfacts Holdout

| model | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_fp |
|---|---:|---:|---:|---:|---:|
| Qwen base | 0.8677 | 0.4513 | 0.5022 | 0.10 | 11 |
| Qwen LoRA | **0.9908** | **0.6515** | **0.9648** | **0.88** | 4 |
| Gemma base | 0.8169 | 0.4432 | 0.5128 | 0.04 | 16 |
| Gemma LoRA | 0.9492 | 0.5857 | 0.8123 | 0.44 | **0** |

### Run-004 Random Holdout

| model | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_fp |
|---|---:|---:|---:|---:|---:|
| Qwen base | 0.8038 | 0.4393 | 0.5659 | 0.03 | 70 |
| Qwen LoRA | **0.9931** | **0.6100** | **0.9159** | **0.92** | **8** |
| Gemma base | 0.8146 | 0.4400 | 0.6332 | 0.12 | 47 |
| Gemma LoRA | 0.9715 | 0.5630 | 0.7959 | 0.74 | 13 |

## Main Findings

### 1. 同一数据方案对两个 backbone 都有效

`Run-003 + Run-005x2` 不是只对 Qwen 有效。Gemma 在两套 holdout 上相对 base model 都获得了明显提升，说明：

- 当前 13-fact ontology 是可迁移的
- composition rebalance + hard negatives + moderate oversampling 的数据策略具有跨 backbone 有效性

### 2. 当前最佳整体结果仍来自 Qwen

在两个 holdout 上，Qwen LoRA 都给出更高的：

- `fact_accuracy`
- `seen_f1`
- `sample_exact_match`

特别是在需要多事实同时正确的 `sample_exact_match` 上，Qwen 优势更明显：

- Run-002：`0.88 vs 0.44`
- Run-004：`0.92 vs 0.74`

### 3. Gemma 更保守，但并不更强

Gemma 在 `Run-002` 上把 `critical false positives` 压到了 `0`，这是它最突出的优点，说明它在关键完成态 fact 上更保守。

但这种保守性也带来代价：

- recall 偏低
- `seen_f1` 低于 Qwen
- `sample_exact_match` 明显低于 Qwen

因此目前更准确的表述是：

> Gemma 的错误风格更保守，但总体抽取质量仍弱于最佳 Qwen LoRA。

## Error Pattern Comparison

### Qwen

优点：

- TAC / SUPT / FCS / BIT root / FCS-MC 页面辨认更稳
- 多事实同时命中更强
- INS 小字提示整体优于 Gemma

主要残余问题：

- `fcsmc_final_go_result_visible`
- 少量 completion-type false positives

### Gemma

优点：

- base -> LoRA 提升明显
- `Run-002` 上 `critical_fp = 0`
- 对部分 FCS-MC 页面结构判断稳定

主要问题：

- `supt_page_visible` 偏弱
- `ins_grnd_alignment_text_visible` 偏弱
- `ins_ok_text_visible` 偏弱
- `Run-004` 上 `fcsmc_intermediate_result_visible` 出现集中误报

## Practical Conclusion

如果当前目标是部署或后续系统集成，**优先选择 Qwen3.5-9B Run-003 + Run-005x2 LoRA** 更合理，因为它在总体准确率、seen F1 和 exact match 上都更强。

如果当前目标是研究 backbone 风格差异，Gemma 仍然值得保留，因为它展示了：

- 更保守的 critical-fact 判断风格
- 不同于 Qwen 的误报/漏报取舍

## Recommended Next Step

下一步更推荐继续沿 Qwen 路线做系统集成与运行链验证，同时把 Gemma 作为对照 backbone 保留在论文或报告中，用于说明：

1. 数据方案具有跨 backbone 可迁移性；
2. 当前最佳实用结果仍来自 Qwen；
3. backbone 差异主要体现在 error style，而不只是总体分数。
