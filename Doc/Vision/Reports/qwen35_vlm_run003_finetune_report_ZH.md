# Run-003 Qwen3.5-9B VLM LoRA 微调实验报告

## 摘要

本报告记录 SimTutor F/A-18C 冷启动视觉感知链路中的 Run-003 VLM 微调实验。该实验的目标是验证：在重新设计更低层、可直接观察的视觉事实 ontology 后，少量人工复核数据是否仍能让 `Qwen/Qwen3.5-9B-Base` 学会稳定输出结构化 cockpit visual facts，并缓解旧版本 `ins_go` 带来的危险 false positives。

训练使用 Unsloth-accelerated PEFT LoRA VLM SFT with TRL `SFTTrainer`。Run-003 数据包含 220 张人工复核的组合屏图像，导出为英文和中文两套 multimodal SFT 数据，共 440 行。训练采用 4-bit LoRA，rank 16，alpha 16，dropout 0.0，训练 4 个 epoch，并使用 10% 训练内 eval split。

主要独立评测使用 Run-002 图像，并按新的 13-fact ontology 重新标注。在这个 50-sample holdout 上，LoRA 将 fact accuracy 从 `0.8677` 提升到 `0.9154`，seen F1 从 `0.5022` 提升到 `0.6680`，sample exact match 从 `0.10` 提升到 `0.20`。模型明显改善了 FCS-MC 状态和 HSI alignment text 的识别，但也出现新的主要失败模式：`ins_ok_text_visible` 的 false positives 从 0 增加到 17。另一个 Run-004 random stress set 中，LoRA 将 total critical false positives 从 70 降低到 19，但该集合存在重复和分布偏置，因此更适合作为随机操作压力参考，而不是主要泛化结论。

总体上，Run-003 支持“少量领域人工复核数据可以显著提升结构化 cockpit visual fact extraction”的核心假设。同时，它也说明干净的 single-fact boundary 数据还不够；如果要用于真实冷启动状态推理，还需要补充多屏幕 fact 共现样本和针对完成类文本的 hard negatives。

## 1. 实验动机

上一轮 Run-001 LoRA 使用 8 个视觉 facts。它在 Run-001 和 Run-002 上都带来了明显提升，但 Run-002 heldout 暴露出一个核心问题：`ins_go` false positives 很多。这个问题不只是模型能力不足，也与目标定义有关。`ins_go` 把较高层的流程完成状态压缩成单帧 VLM 标签，而实际图像里 MAP overlay、小字号 alignment text、countdown 中间态、显示切换瞬间都可能造成混淆。

因此，Run-003 将 ontology 重构为更低层的视觉证据。INS 不再直接要求模型判断“是否 GO”，而是拆成 HSI 页面、MAP 图层、GRND/QUAL/TIME alignment text、OK text。FCS-MC 也拆成页面可见性、PBIT/intermediate、IN TEST、final GO result。

本实验关注三个问题：

1. 新的 13-fact ontology 是否仍然可以被小规模 LoRA 学会？
2. 拆分高层完成信号后，是否能降低 FCS-MC 和 INS 相关 false positives？
3. 在独立 holdout 和随机 cockpit 截图中，还会暴露哪些新的 failure modes？

## 2. Visual Fact Ontology

输入是一张 VLM-ready composite-panel 图像，内部固定区域从上到下为：

| 区域 | 含义 |
|---|---|
| `left_ddi` | 左侧 Digital Display Indicator |
| `ampcd` | Advanced Multipurpose Color Display |
| `right_ddi` | 右侧 Digital Display Indicator |

每个视觉 fact 使用三分类标签：

| 状态 | 含义 |
|---|---|
| `seen` | 当前图像中能明确看到该事实 |
| `not_seen` | 当前图像中没有看到该事实 |
| `uncertain` | 图像模糊、局部遮挡、页面不完整，或单帧无法判断 |

Run-003 使用 13 个核心 facts：

| fact_id | 含义 |
|---|---|
| `tac_page_visible` | 能看到真正的 TAC/TAC MENU 页面，而不是单独的 TAC 选项标签 |
| `supt_page_visible` | 能看到真正的 SUPT/SUPT MENU 页面 |
| `fcs_page_visible` | 能看到专用 FCS 飞控页面 |
| `fcs_page_x_marks_visible` | FCS 页面格子里能看到 X/fault 填充 |
| `bit_root_page_visible` | 能看到 BIT FAILURES/root 页面 |
| `fcsmc_page_visible` | 能看到 FCS-MC 子页面及 MC1/MC2/FCSA/FCSB 行 |
| `fcsmc_intermediate_result_visible` | 能看到 PBIT GO 或其他非最终 FCS-MC 中间结果 |
| `fcsmc_in_test_visible` | 能看到 FCS-MC IN TEST 文本 |
| `fcsmc_final_go_result_visible` | 能看到最终 GO 结果，尤其 FCSA/FCSB GO |
| `hsi_page_visible` | 能看到 HSI navigation/POS 页面 |
| `hsi_map_layer_visible` | 能看到彩色 HSI MAP 图层 |
| `ins_grnd_alignment_text_visible` | 能看到 GRND/QUAL/TIME alignment text |
| `ins_ok_text_visible` | 能看到 INS alignment block 附近的 OK 文本 |

这一 ontology 比旧 8 facts 更细。它的设计目标是让 VLM 输出视觉证据，而不是直接输出流程状态。下游系统再结合 procedure context、telemetry 和时间稳定性进行状态推理。

## 3. 数据集构建

Run-003 使用手工引导的截图计划，目标是覆盖新 facts 的干净视觉边界和 hard negatives。最终 reviewed dataset 包含 220 张组合屏图像。初标由 Qwen 397B 完成，随后在 Label Studio 中人工复核。最终导出为 OpenAI-compatible multimodal chat JSONL。

训练目标刻意不包含 `frame_id`、`source_frame_id`、`confidence`、`expires_after_ms`、`sticky` 等字段。这些字段不是单张图像中可可靠推断的视觉事实，应由 runtime 系统在解析后补充，而不是让 VLM 生成。

### 3.1 Run-003 标签分布

| fact_id | seen | not_seen | uncertain |
|---|---:|---:|---:|
| `tac_page_visible` | 46 | 174 | 0 |
| `supt_page_visible` | 20 | 200 | 0 |
| `fcs_page_visible` | 30 | 190 | 0 |
| `fcs_page_x_marks_visible` | 16 | 204 | 0 |
| `bit_root_page_visible` | 18 | 202 | 0 |
| `fcsmc_page_visible` | 55 | 165 | 0 |
| `fcsmc_intermediate_result_visible` | 18 | 202 | 0 |
| `fcsmc_in_test_visible` | 18 | 202 | 0 |
| `fcsmc_final_go_result_visible` | 18 | 202 | 0 |
| `hsi_page_visible` | 106 | 114 | 0 |
| `hsi_map_layer_visible` | 79 | 141 | 0 |
| `ins_grnd_alignment_text_visible` | 58 | 162 | 0 |
| `ins_ok_text_visible` | 12 | 206 | 2 |

这个分布不是为了估计真实 cockpit state 频率，而是为了训练视觉边界和 hard negatives。它对定义单个 fact 很有效，但也可能低估真实冷启动中多个 facts 同时出现的情况。例如默认上电时常见 TAC、HSI/MAP、BIT root 同时可见。

### 3.2 SFT 数据格式

每行 SFT 数据包含：

```json
{
  "messages": [
    {"role": "system", "content": "You are SimTutor visual fact extractor. Reply with JSON only."},
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}, {"type": "text", "text": "...13-fact instruction..."}]},
    {"role": "assistant", "content": "{\"facts\":[{\"fact_id\":\"tac_page_visible\",\"state\":\"seen\",\"evidence_note\":\"\"}, ...]}"}
  ]
}
```

Run-003 导出时使用了 `--drop-summary`，并且 assistant target 中不包含人工 evidence notes。这样做是为了让监督目标集中在 13 个 fact states 本身。

## 4. 微调设置

训练栈如下：

```text
Qwen/Qwen3.5-9B-Base
  + Unsloth VLM loading and 4-bit preparation
  + PEFT LoRA adapters
  + TRL SFTTrainer
```

关键训练参数：

| 参数 | 数值 |
|---|---:|
| reviewed images | 220 |
| SFT rows | 440 |
| internal train rows | 396 |
| internal eval rows | 44 |
| epochs | 4 |
| learning rate | 2e-4 |
| batch size | 1 |
| gradient accumulation | 4 |
| effective batch size | 4 |
| max sequence length | 4096 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| seed | 3407 |
| runtime | 3375.8s |
| final train loss | 0.3479 |
| final internal eval loss | 0.0008 |

这里的 internal eval split 来自 Run-003 同一数据池，只能作为训练 sanity check，不能作为独立泛化指标。

## 5. Benchmark 方法

Benchmark 比较 base model 和 Run-003 LoRA adapter。两者接收相同 reviewed JSONL、相同图像和相同 13-fact 英文 benchmark prompt。输出被解析为 JSON，规范化后与人工标签比较。

指标如下：

| 指标 | 含义 |
|---|---|
| JSON valid rate | 输出是否可解析为 JSON object |
| schema valid rate | 输出字段是否符合预期 schema |
| fact accuracy | 所有 facts 上的三分类准确率 |
| macro F1 | `seen/not_seen/uncertain` 三类宏平均 F1 |
| seen F1 | `seen` 正类 F1 |
| sample exact match | 13 个 facts 全部正确才算该样本正确 |
| critical false positives | critical facts 中 gold 为 `not_seen/uncertain` 但预测为 `seen` 的错误 |

Run-003 critical facts 为：

```text
fcs_page_x_marks_visible
fcsmc_intermediate_result_visible
fcsmc_in_test_visible
fcsmc_final_go_result_visible
ins_grnd_alignment_text_visible
ins_ok_text_visible
```

## 6. Evaluation Sets

### 6.1 Run-002 New-Facts Holdout

Run-002 是之前独立采集的 session，本轮按新 13-fact ontology 重新标注。它包含 50 个样本，是 Run-003 的主要独立 holdout。

| fact_id | seen | not_seen |
|---|---:|---:|
| `tac_page_visible` | 23 | 27 |
| `supt_page_visible` | 5 | 45 |
| `fcs_page_visible` | 13 | 37 |
| `fcs_page_x_marks_visible` | 1 | 49 |
| `bit_root_page_visible` | 9 | 41 |
| `fcsmc_page_visible` | 9 | 41 |
| `fcsmc_intermediate_result_visible` | 1 | 49 |
| `fcsmc_in_test_visible` | 6 | 44 |
| `fcsmc_final_go_result_visible` | 2 | 48 |
| `hsi_page_visible` | 49 | 1 |
| `hsi_map_layer_visible` | 27 | 23 |
| `ins_grnd_alignment_text_visible` | 42 | 8 |
| `ins_ok_text_visible` | 4 | 46 |

### 6.2 Run-004 Random Stress Set

Run-004 是随机操作 holdout，共 100 个样本。它包含较多重复和分布偏置，不适合作为主要泛化结论，但适合作为随机 cockpit state 压力参考。

| fact_id | seen | not_seen |
|---|---:|---:|
| `bit_root_page_visible` | 16 | 84 |
| `fcsmc_page_visible` | 84 | 16 |
| `fcsmc_intermediate_result_visible` | 7 | 93 |
| `fcsmc_in_test_visible` | 17 | 83 |
| `fcsmc_final_go_result_visible` | 60 | 40 |
| `ins_ok_text_visible` | 0 | 100 |

## 7. 实验结果

### 7.1 Run-002 New-Facts Holdout

![Run-002 new-facts overall accuracy](assets/qwen35_vlm_finetune/run002_newfacts_overall_accuracy.png)

![Run-002 new-facts fact F1 by model](assets/qwen35_vlm_finetune/run002_newfacts_fact_f1_by_model.png)

![Run-002 new-facts seen F1 by fact](assets/qwen35_vlm_finetune/run002_newfacts_seen_f1_by_fact.png)

![Run-002 new-facts critical false positives](assets/qwen35_vlm_finetune/run002_newfacts_critical_false_positives.png)

| 指标 | Base | LoRA |
|---|---:|---:|
| JSON valid rate | 1.0000 | 1.0000 |
| schema valid rate | 1.0000 | 1.0000 |
| fact accuracy | 0.8677 | 0.9154 |
| macro F1 | 0.4513 | 0.5306 |
| seen F1 | 0.5022 | 0.6680 |
| sample exact match | 0.1000 | 0.2000 |
| critical false positives | 11 | 20 |

Run-002 显示 LoRA 在总体结构化抽取上有明显提升。fact accuracy 提升 4.77 个百分点，seen F1 提升 16.57 个百分点，sample exact match 提升 10 个百分点。这说明新的 13-fact target 是可学习的，LoRA 也获得了一定跨 session 泛化能力。

但 critical false positives 上升了。这个上升几乎集中在 `ins_ok_text_visible`：base 在该 fact 上为 0 个 false positives，而 LoRA 有 17 个。这说明旧 `ins_go` 的问题没有完全消失，而是转移到了更具体的 OK text 完成信号上。

关键 per-fact 变化如下：

| fact | accuracy base -> LoRA | seen F1 base -> LoRA | seen FP base -> LoRA | seen FN base -> LoRA |
|---|---:|---:|---:|---:|
| `supt_page_visible` | 0.90 -> 1.00 | 0.00 -> 1.00 | 0 -> 0 | 5 -> 0 |
| `fcsmc_page_visible` | 0.96 -> 0.98 | 0.89 -> 0.94 | 1 -> 0 | 1 -> 1 |
| `fcsmc_in_test_visible` | 0.90 -> 0.98 | 0.44 -> 0.91 | 1 -> 0 | 4 -> 1 |
| `fcsmc_final_go_result_visible` | 0.94 -> 0.98 | 0.40 -> 0.80 | 2 -> 1 | 1 -> 0 |
| `ins_grnd_alignment_text_visible` | 0.16 -> 0.68 | 0.00 -> 0.76 | 0 -> 0 | 42 -> 16 |
| `ins_ok_text_visible` | 0.92 -> 0.58 | 0.00 -> 0.00 | 0 -> 17 | 4 -> 4 |

![Run-002 new-facts INS OK base confusion matrix](assets/qwen35_vlm_finetune/run002_newfacts_confusion_ins_ok_text_visible_base.png)

![Run-002 new-facts INS OK LoRA confusion matrix](assets/qwen35_vlm_finetune/run002_newfacts_confusion_ins_ok_text_visible_lora.png)

### 7.2 Run-004 Random Stress Set

![Run-004 random overall accuracy](assets/qwen35_vlm_finetune/run004_random_overall_accuracy.png)

![Run-004 random fact F1 by model](assets/qwen35_vlm_finetune/run004_random_fact_f1_by_model.png)

![Run-004 random seen F1 by fact](assets/qwen35_vlm_finetune/run004_random_seen_f1_by_fact.png)

![Run-004 random critical false positives](assets/qwen35_vlm_finetune/run004_random_critical_false_positives.png)

| 指标 | Base | LoRA |
|---|---:|---:|
| JSON valid rate | 0.9600 | 1.0000 |
| schema valid rate | 0.9600 | 1.0000 |
| fact accuracy | 0.8038 | 0.9185 |
| macro F1 | 0.4393 | 0.4901 |
| seen F1 | 0.5659 | 0.6390 |
| sample exact match | 0.0300 | 0.5200 |
| critical false positives | 70 | 19 |

Run-004 显示 LoRA 大幅降低 total critical false positives，并显著提高 exact match。其中最明显的是 `fcsmc_intermediate_result_visible` false positives：base 有 60 个，而 LoRA 为 0。

同时，Run-004 也暴露了 recall 问题。例如 `bit_root_page_visible` 上，LoRA 没有 false positives，但有 11 个 false negatives，导致 seen F1 从 0.60 降到 0.48。`fcsmc_page_visible` 也没有 false positives，但漏掉 18 个正例。这说明 adapter 在更复杂或重复的 cockpit context 下变得偏保守。

因此，Run-004 的结果应解释为 mixed：LoRA 更结构化、更少乱报，但在部分页面和状态事实上 recall 下降。

## 8. 讨论

### 8.1 改善点

Run-003 在与新 ontology 匹配较好的低层视觉任务上表现明显改善。Run-002 上，FCS-MC page、IN TEST、final GO、HSI page、HSI MAP 和 alignment text 都有较好提升或稳定表现。模型也学会了 13-fact JSON contract，输出 schema 稳定。

更重要的是，LoRA 减少了部分 FCS-MC 相关危险混淆。在 Run-002 中，FCS-MC intermediate、IN TEST 和 final GO 三类 critical false positives 总数从 base 的 11 个降到 LoRA 的 3 个。这说明将 FCS-MC 阶段拆开是有效的。

### 8.2 失败点

新的主要失败点是 `ins_ok_text_visible`。在 Run-002 中，LoRA 将 17 个 human-labeled `not_seen` 样本误判为 `seen`。这与旧 `ins_go` 问题相似，但现在错误被定位到更具体的 OK text cue。换句话说，可解释性提高了，但安全问题尚未完全解决。

Run-004 还显示 adapter 对部分 facts 过于保守。它避免了很多 false positives，但漏掉了真实的 `bit_root_page_visible`、`fcsmc_page_visible`、`fcsmc_intermediate_result_visible` 和 `fcsmc_in_test_visible`。这很可能来自数据分布 mismatch：Run-003 主要采集干净的 single-fact boundary 样本，而真实冷启动画面经常在三个显示器上同时出现多个 facts。

### 8.3 为什么 Run-001 看起来更容易

Run-001 使用 8 个较粗的 facts，其中很多是页面级或较宽泛状态标签。Run-003 使用 13 个更细的 facts，需要区分 PBIT GO、IN TEST、final GO、GRND/QUAL/TIME 和 OK 等小字号状态文本。这些标签更难、更稀有，也更依赖 hard negatives。

因此，新任务更可解释，但也更难。某些细粒度 fact 指标下降，并不代表 ontology 更差；它可能只是把旧粗粒度标签隐藏的问题暴露出来了。

### 8.4 Prompt 与 Runtime 集成注意事项

当前 benchmark prompt 和 SFT prompt 都围绕新的 13 facts，二者是一致的。但 live runtime prompt 和 `vision_facts.yaml` 仍保留旧 facts，例如 `ins_go`、`fcs_bit_result_visible`、以及绑定 right DDI 的 FCS-MC facts。这不影响 Run-003 benchmark 的有效性，但意味着 Run-003 adapter 不能不加修改地直接接入 live 系统。

后续上线前，需要同步 runtime fact schema、输出 contract 和 decision boundaries。尤其不应要求 VLM 生成 `source_frame_id` 或 `confidence`；这些应由 runtime 系统在解析后补充或忽略。

## 9. 局限

本轮实验的主要局限为：

- Run-003 只有 220 张唯一图像；双语导出增加的是 instruction diversity，不是视觉多样性。
- 多个 critical facts 的 seen 样本较少，例如 `ins_ok_text_visible` 只有 12 个 seen。
- Run-003 采集计划强调干净 fact 边界，可能低估真实多屏 fact 共现。
- Run-004 可作为压力参考，但其图像分布重复且偏置明显。
- internal eval loss 不是独立泛化指标。
- live runtime prompt/schema 尚未迁移到新 13 facts。

## 10. 下一步

下一步建议先冻结当前 13 facts，以保持实验可比性。最有价值的是补充 Run-005 composition dataset：

| 类别 | 目的 |
|---|---|
| TAC + HSI MAP + BIT root | 学习默认冷启动多 fact 共现 |
| SUPT + HSI MAP + BIT root | 提高导航上下文下的页面 recall |
| FCS page + HSI + BIT root | 学习 FCS 与 BIT root 共现 |
| FCS-MC PBIT + HSI/TAC/SUPT | 区分 PBIT GO 与 final GO |
| FCS-MC IN TEST + HSI | 提高 IN TEST recall |
| FCS-MC final GO + INS no OK | 解耦 FCS 完成与 INS 完成 |
| HSI GRND/QUAL/TIME without OK | 降低 `ins_ok_text_visible` false positives |
| mixed non-target pages | 降低随机导航页面下的误报 |

训练应从 base model 重新开始，使用 Run-003 + Run-005，而不是在 Run-003 adapter 上继续训练。这样实验解释更干净，也更容易比较数据分布修正的效果。

## 11. 结论

Run-003 是一次成功的 ontology 与 fine-tuning 迭代，但还不是最终部署方案。它证明 9B VLM 可以通过小规模人工复核数据学习更精细的 13-fact cockpit visual schema，并在 Run-002 new-facts holdout 上提升总体结构化抽取表现，同时减少部分 FCS-MC 阶段混淆。

剩余问题不只是模型大小，而是稀有正例、完成类小字文本歧义，以及 clean training captures 与真实多屏 cockpit composition 之间的分布差异。下一轮应在保持 13 facts 不变的前提下，补充针对性 composition 与 hard-negative 数据。

