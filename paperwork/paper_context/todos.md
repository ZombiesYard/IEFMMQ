# TODOs

> 目标：列出必须人工确认、手动核对或后续实现的数据/决策点。  
> 这些事项不应由当前整理脚本替你拍板。

## 1. 论文主结果线选择

- `TODO:VERIFY`：Qwen 还是 Gemma 作为论文主模型线？
- `TODO:VERIFY`：是否保留 8-fact v1 结果作为历史 baseline，还是只把它放进 appendix / pilot experiment？
- `TODO:VERIFY`：13-fact 结果里主表优先用 `holdout_run002_newfacts`、`holdout_run004_random`，还是两者并列？
- `TODO:VERIFY`：是否把 `run003_plus_run005x2` 作为“current best adapter”在全文统一命名？

## 2. benchmark 目录命名与语义

- `TODO:VERIFY`：`benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_lora_only_holdout_run004_random_v1/benchmark_summary.json` 中 `benchmark_kind=contaminated_dev_set` 是否为历史遗留标记错误？
- `TODO:VERIFY`：Gemma benchmark 的 `v2` 后缀是否表示第二轮脚本/报告格式，还是第二次实验版本？
- `TODO:VERIFY`：各目录名里的 `run003`, `run005x2`, `holdout_run002_newfacts`, `holdout_run004_random` 是否已在论文里有统一解释文本？

## 3. proposal 与当前系统的 framing

- `TODO:VERIFY`：你希望论文标题更偏向 `grounded tutoring system`，还是更偏向 `VLM adaptation for procedural tutoring`？
- `TODO:VERIFY`：VR 是否继续保留在标题或摘要级 framing 中，还是只放在 introduction / future evaluation context 中？
- `TODO:VERIFY`：论文是否明确写出“当前实现是 desktop/DCS runtime，VR runtime 仍在下一阶段实现”？

## 4. runtime 数据收集系统设计

- `TODO:VERIFY`：未来 runtime 数据收集要记录哪些 participant/session 字段？
- `TODO:VERIFY`：是否需要把 `help_cycle_id`、`overlay_targets`、`recent_actions`、`vision frame refs` 作为 user-study 数据的一部分长期保存？
- `TODO:VERIFY`：是否需要单独的 participant metadata schema？
- `TODO:VERIFY`：是否需要把 `NASA-TLX`、quiz、retention、transfer 数据和 runtime log 用统一 ID 关联？
- `TODO:VERIFY`：哪些数据必须匿名化，哪些仅本地保存？

## 5. 用户研究与实验设计

- `TODO:VERIFY`：最终对照组是否仍是 `video+notes`、`ungrounded text-only LLM`、grounded tutor 三组？
- `TODO:VERIFY`：受试者范围是否仍是 beginner / intermediate？
- `TODO:VERIFY`：目标样本量是否仍按 proposal 的 `n ≈ 5–10`，还是已经计划扩大？
- `TODO:VERIFY`：是否保留 retention test 和 transfer task？
- `TODO:VERIFY`：NASA-TLX 是否为最终 workload 指标，还是只做可选附加量表？

## 6. 文本与章节重写范围

- `TODO:VERIFY`：`paperwork/Introduction.tex` 是否需要几乎整章重写？
- `TODO:VERIFY`：`paperwork/Methodology.tex` 是否需要按当前 runtime + VLM pipeline 重组？
- `TODO:VERIFY`：`paperwork/Results.tex` 当前是否应完全改写为“current technical results only”？
- `TODO:VERIFY`：`paperwork/Discussion.tex` 是否需要显式增加一节“proposal-to-system shift”？
- `TODO:VERIFY`：`paperwork/Conclusion.tex` 是否要同时总结“已完成技术基线”和“下一阶段人因实验”两条线？

## 7. 图表与表格制作

- `TODO:COMPUTE`：最终论文主表用哪些 benchmark 目录的 `metrics_*.json` 生成？
- `TODO:COMPUTE`：是否需要统一重绘 benchmark figures，以保持论文视觉风格一致？
- `TODO:COMPUTE`：是否要从 `fact_scores.csv` 自动生成适合 LaTeX 的逐 fact 表格？
- `TODO:COMPUTE`：是否要补一张 ontology evolution 表，对比 8-fact 与 13-fact？
- `TODO:COMPUTE`：是否要补一张 dataset provenance 表，把 Run-001 / Run-002 / Run-003 / Run-005 的关系说清楚？

## 8. 训练与模型说明

- `TODO:VERIFY`：论文中对 Gemma 基座模型的正式写法，用 `google/gemma-4-31B` 还是 `unsloth/gemma-4-31b-unsloth-bnb-4bit`？
- `TODO:VERIFY`：Qwen 与 Gemma 的实验是否都进入主文，还是一条主文一条 appendix？
- `TODO:VERIFY`：`eval_rows=0` 的训练 run 在论文里是否需要额外解释训练/验证划分策略？
- `TODO:VERIFY`：论文是否要描述 why `run003` 内的 `missing_summary_count=82` 和 `run005` 内的 `missing_summary_count=25` 样本被保留/排除？
- `TODO:VERIFY`：Gemma adapter 的 `finetune_vision_layers=false` 设置是否需要在方法章中说明？
- `TODO:VERIFY`：`full_qwen35_9b_base_bilingual_v1` (8-fact LoRA) vs `full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1` (13-fact LoRA) 是否需要统一命名以区分？
- `TODO:VERIFY`：replay evaluation 的 5 个 cases 是否会被纳入论文（作为 runtime behavior evidence 而非 benchmark evidence）？

## 9. 证据边界提醒

以下内容在正式写英文正文前必须再次检查：

- `TODO:VERIFY`：所有 benchmark 数字是否都直接抄自 `metrics_*.json` 或 `benchmark_summary.json`
- `TODO:VERIFY`：所有 dataset 数字是否都直接抄自 `datasets/*/stats.json`
- `TODO:VERIFY`：所有“当前最佳”或“main line”表述是否已经经过人工确认，而不是默认选择
- `TODO:VERIFY`：所有 VR / user-study 相关句子是否都保持在 planned / future / pending 语气

