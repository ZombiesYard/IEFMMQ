# Run-005 Composition-Rebalance Qwen3.5-9B VLM LoRA 微调实验报告

## 摘要

本报告记录 SimTutor F/A-18C 冷启动视觉事实抽取链路中的 Run-005 组合补采与新一轮 LoRA 微调实验。与 Run-003 相比，本轮不再修改 ontology，而是保持当前 13 个 visual facts 不变，重点处理训练数据分布问题，包括多屏共现、hard negatives、小字状态正例，以及 Run-005 样本的适度过采样。

本轮训练仍使用 `Qwen/Qwen3.5-9B-Base`，训练栈为 Unsloth-accelerated PEFT LoRA VLM SFT with TRL `SFTTrainer`。训练数据由 Run-003 与 Run-005 组成：Run-003 为 220 张人工复核图像，Run-005 为 122 张 composition-rebalance 补采图像。导出为英文和中文两套 SFT 后，训练输入共 928 行，其中 Run-005 双语样本重复一遍参与训练，用于提高这部分补采样本在训练中的权重。

在更强的外部评测 `Run-002 newfacts holdout` 上，新 adapter 将 fact accuracy 从 `0.8677` 提升到 `0.9908`，seen F1 从 `0.5022` 提升到 `0.9648`，sample exact match 从 `0.10` 提升到 `0.88`，critical false positives 从 `11` 降到 `4`。在 `Run-004 random holdout` 100 样本 benchmark 上，它同样明显优于 base model，也显著优于旧的 Run-003 adapter：fact accuracy 从 `0.8038` 提升到 `0.9931`，seen F1 从 `0.5659` 提升到 `0.9159`，sample exact match 从 `0.03` 提升到 `0.92`，critical false positives 从 `70` 降到 `8`。

需要说明的是，`Run-004` 虽然与训练集不存在 exact image overlap，但其标签支持分布并不均匀，若干 facts 的正例数较少，因此 `seen F1 by fact` 图中大量接近 `1.0` 的结果需要结合 support count 一起解读。综合两组 holdout，可以认为：在固定 13-fact ontology 的前提下，补充 composition-balanced 数据并对其适度过采样，能够明显改善多屏共现场景下的视觉事实抽取质量；其中 `Run-002` 提供了更有力的外部泛化证据。

## 1. 背景与实验动机

### 1.1 从 Run-001 到 Run-003

本项目的视觉输入不是三张独立截图，而是一张固定布局的 composite-panel artifact，内部自上而下为：

| 区域 | 含义 |
|---|---|
| `left_ddi` | 左侧 DDI |
| `ampcd` | AMPCD |
| `right_ddi` | 右侧 DDI |

Run-001 使用 8 个较粗粒度 facts，证明了小规模领域数据能显著提升结构化视觉抽取，但也暴露出 `ins_go` 这类高层完成态 target 的问题。单帧图像很难稳定支持“流程是否已经完成”这类判断，尤其在 MAP overlay、小字号文本、切页瞬间和中间态存在时，false positive 风险很高。

为此，Run-003 将 ontology 拆成更低层、可直接观察的 13 个 facts，例如：

- 把 INS 相关判断拆成 `hsi_page_visible`、`hsi_map_layer_visible`、`ins_grnd_alignment_text_visible`、`ins_ok_text_visible`
- 把 FCS-MC 相关判断拆成 `fcsmc_page_visible`、`fcsmc_intermediate_result_visible`、`fcsmc_in_test_visible`、`fcsmc_final_go_result_visible`

Run-003 在 `Run-002 newfacts holdout` 上已经证明这套 13-fact ontology 是可学习的：fact accuracy 从 `0.8677` 提升到 `0.9154`，seen F1 从 `0.5022` 提升到 `0.6680`。但它也带来了两个新的信号：

1. `ins_ok_text_visible` 出现明显 false positives，说明高层 `ins_go` 被拆开以后，风险并没有完全消失，而是集中到了更具体的完成提示文本上。
2. 在 `Run-004 random holdout` 上，Run-003 adapter 虽然减少了误报，但在 `bit_root_page_visible`、`fcsmc_page_visible`、`fcsmc_intermediate_result_visible`、`fcsmc_in_test_visible` 上显得偏保守，说明训练集与真实多屏共现场景之间仍有分布差。

### 1.2 为什么本轮先不改 facts

本轮保持 13-fact ontology 不变，主要是为了控制实验变量。Run-003 之后已经可以确认两点：

- ontology 本身是可训练、可解释、可控的；
- 当前更大的瓶颈来自数据分布，而不是还要继续改 label 定义。

因此，Run-005 主要关注下面这个问题：

> 如果保持 13 facts 不变，只通过补采更合理的多屏组合样本和 hard negatives，能否显著改善模型表现？

## 2. 当前固定的 13-Fact Ontology

本轮继续使用 Run-003 已确定的 13 个核心 facts：

| fact_id | 含义 |
|---|---|
| `tac_page_visible` | 真正的 TAC/TAC MENU 页面可见，而不是单独的 TAC 按钮标签 |
| `supt_page_visible` | 真正的 SUPT/SUPT MENU 页面可见 |
| `fcs_page_visible` | 专用 FCS 页面可见 |
| `fcs_page_x_marks_visible` | FCS 页面通道格中有明确 X/fault 填充 |
| `bit_root_page_visible` | BIT FAILURES/root 页面可见 |
| `fcsmc_page_visible` | FCS-MC 子页面可见，且 MC1/MC2/FCSA/FCSB 行结构可辨认 |
| `fcsmc_intermediate_result_visible` | PBIT GO 或其他中间结果可见，但不是 final GO |
| `fcsmc_in_test_visible` | IN TEST 状态可见 |
| `fcsmc_final_go_result_visible` | FCS-MC final GO 结果可见 |
| `hsi_page_visible` | HSI/POS 页面可见 |
| `hsi_map_layer_visible` | 彩色 MAP 图层可见 |
| `ins_grnd_alignment_text_visible` | GRND/QUAL/TIME 对准文本块可见 |
| `ins_ok_text_visible` | 对准文本块附近的 OK 清晰可见 |

每个 fact 仍采用三分类：

| state | 含义 |
|---|---|
| `seen` | 当前图像中明确看到该事实 |
| `not_seen` | 当前图像中没有看到该事实 |
| `uncertain` | 单帧无法确定、文字模糊、局部遮挡或正在切换 |

下游系统仍只消费结构化 fact states，而不是模型 summary。因此本轮导出训练数据时继续使用 `--drop-summary`，并且 assistant target 中不包含 `confidence`、`source_frame_id` 等非视觉字段。

## 3. 为什么要做 Run-005 Composition Rebalance

Run-003 的问题不只是部分 facts 的 seen 样本偏少，更重要的是它更接近一套 clean boundary dataset：单个事实边界较清楚，但真实冷启动里常见的多屏组合覆盖不足。典型例子包括：

- 左侧 TAC / SUPT，AMPCD HSI，右侧 BIT root 同时存在
- HSI GRND/QUAL/TIME 与 FCS-MC `PBIT GO`、`IN TEST`、`final GO` 的真实共现
- `FCS-MC final GO` 与 `INS OK not seen` 的 hard negative 组合
- 纯非目标页、稳定黑屏、切页模糊等“不应乱报”的负例

因此，Run-005 不替代 Run-003，而是作为补采数据加入训练，主要目标是：

1. 补 `bit_root_page_visible`、`fcs_page_visible`、`fcsmc_page_visible` 的正例和真实共现。
2. 补 `PBIT GO`、`IN TEST`、`final GO` 这些小字状态文本的正例。
3. 补 `FCS-MC final GO + HSI no OK` 这类 hard negatives，避免把 FCS 完成误读成 INS 完成。
4. 保留一定的 non-target / blank / transition 样本，防止模型把所有模糊状态都报成 `seen`。

## 4. 数据集构建与分布设计

### 4.1 Run-003 与 Run-005 数据规模

| 数据集 | reviewed 图像数 | SFT 行数（EN+ZH） | 角色 |
|---|---:|---:|---|
| Run-003 | 220 | 440 | 原始 13-fact 主训练集 |
| Run-005 composition rebalance | 122 | 244 | 多屏共现与 hard-negative 补采集 |

Run-005 的导出统计如下：

| fact_id | seen | not_seen | uncertain |
|---|---:|---:|---:|
| `tac_page_visible` | 24 | 98 | 0 |
| `supt_page_visible` | 34 | 88 | 0 |
| `fcs_page_visible` | 40 | 79 | 3 |
| `fcs_page_x_marks_visible` | 18 | 104 | 0 |
| `bit_root_page_visible` | 43 | 78 | 1 |
| `fcsmc_page_visible` | 56 | 65 | 1 |
| `fcsmc_intermediate_result_visible` | 15 | 107 | 0 |
| `fcsmc_in_test_visible` | 19 | 103 | 0 |
| `fcsmc_final_go_result_visible` | 22 | 100 | 0 |
| `hsi_page_visible` | 109 | 13 | 0 |
| `hsi_map_layer_visible` | 37 | 85 | 0 |
| `ins_grnd_alignment_text_visible` | 72 | 48 | 2 |
| `ins_ok_text_visible` | 14 | 106 | 2 |

### 4.2 为什么要对 Run-005 过采样一次

训练时没有直接把 Run-003 和 Run-005 简单拼接，而是采用：

```text
Run-003 bilingual once
+ Run-005 bilingual twice
```

也就是说，Run-005 的 EN/ZH 样本整体再重复一遍参与训练。这样做的目的不是模拟“自然频率”，而是提高这批补采样本在训练中的影响，使模型更充分地学习：

- 多屏事实共现；
- 小字状态文本；
- final GO 与 INS OK 的解耦；
- non-target 与 unreadable 的边界。

加权后的训练 fact 分布如下：

| fact_id | seen | not_seen | uncertain |
|---|---:|---:|---:|
| `tac_page_visible` | 94 | 370 | 0 |
| `supt_page_visible` | 88 | 376 | 0 |
| `fcs_page_visible` | 110 | 348 | 6 |
| `fcs_page_x_marks_visible` | 52 | 412 | 0 |
| `bit_root_page_visible` | 104 | 358 | 2 |
| `fcsmc_page_visible` | 167 | 295 | 2 |
| `fcsmc_intermediate_result_visible` | 48 | 416 | 0 |
| `fcsmc_in_test_visible` | 56 | 408 | 0 |
| `fcsmc_final_go_result_visible` | 62 | 402 | 0 |
| `hsi_page_visible` | 324 | 140 | 0 |
| `hsi_map_layer_visible` | 153 | 311 | 0 |
| `ins_grnd_alignment_text_visible` | 202 | 258 | 4 |
| `ins_ok_text_visible` | 40 | 418 | 6 |

需要强调的是：`not_seen` 仍然在多数 facts 上占多数，这是多标签视觉事实任务的正常现象，因为一张图不可能同时包含全部事实。这里的目标不是追求每个 fact 严格 50/50，而是让关键 facts 有足够的正例、真实共现和 hard negatives。

## 5. 微调设置

### 5.1 为什么从 base model 重新训练

本轮没有在 Run-003 adapter 上继续训练，而是重新从 `Qwen/Qwen3.5-9B-Base` 开始。原因有两个：

1. 这样可以把改进更干净地归因到“数据分布修正”，而不是 adapter 继续累积训练带来的路径依赖。
2. 这样更方便把 `base -> Run-003 -> Run-003+Run-005x2` 视为同一实验链上的可比较节点。

### 5.2 训练栈与超参数

训练栈仍为：

```text
Qwen/Qwen3.5-9B-Base
  + Unsloth VLM loading and 4-bit preparation
  + PEFT LoRA adapters
  + TRL SFTTrainer
```

关键参数如下：

| 参数 | 数值 |
|---|---:|
| unique reviewed images | 342 |
| weighted bilingual SFT rows | 928 |
| eval rows | 0 |
| epochs | 4 |
| learning rate | 2e-4 |
| per-device batch size | 1 |
| gradient accumulation | 4 |
| effective batch size | 4 |
| max sequence length | 4096 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| seed | 3407 |
| train runtime | 6419.16 s |
| train steps per second | 0.145 |
| final train loss | 0.2156 |

本轮把 `eval_ratio` 设为 `0`。这是出于实验设计考虑：Run-005 样本在训练集中已被重复一遍，如果再从同一数据池切 internal eval，其参考价值会进一步降低。因此，本轮将泛化判断主要放在独立 holdout benchmark 上。

## 6. 评测设置

### 6.1 Run-002 Newfacts Holdout

`Run-002 newfacts` 是更强的外部 holdout。它来自独立采集的较早 session，并且按当前 13-fact ontology 重新标注。当前已确认：

| 项目 | 数值 |
|---|---:|
| sample count | 50 |
| unique images | 50 |
| exact overlap with Run-003 + Run-005 train union | 0 |

这一批数据的标签支持分布如下：

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

与 `Run-004` 相比，这一批虽然规模更小，但对 `ins_ok_text_visible`、`bit_root_page_visible`、`fcsmc_*` 等关键 facts 有更有意义的正例支持，因此更适合作为主要外部泛化依据。

### 6.2 Run-004 Random Holdout

`Run-004 random holdout` 当前也已完整跑通并拉回本地：

| 项目 | 数值 |
|---|---:|
| sample count | 100 |
| unique images | 100 |
| exact overlap with Run-003 + Run-005 train union | 0 |

### 6.3 Run-004 标签支持分布

Run-004 的支持分布并不均匀，这一点会直接影响 `seen F1 by fact` 的解读：

| fact_id | seen | not_seen |
|---|---:|---:|
| `tac_page_visible` | 3 | 97 |
| `supt_page_visible` | 3 | 97 |
| `fcs_page_visible` | 94 | 6 |
| `fcs_page_x_marks_visible` | 5 | 95 |
| `bit_root_page_visible` | 16 | 84 |
| `fcsmc_page_visible` | 84 | 16 |
| `fcsmc_intermediate_result_visible` | 7 | 93 |
| `fcsmc_in_test_visible` | 17 | 83 |
| `fcsmc_final_go_result_visible` | 60 | 40 |
| `hsi_page_visible` | 100 | 0 |
| `hsi_map_layer_visible` | 50 | 50 |
| `ins_grnd_alignment_text_visible` | 69 | 31 |
| `ins_ok_text_visible` | 0 | 100 |

这意味着：

- `tac_page_visible=1.0` 其实只是在 3 个正例上全对；
- `fcsmc_intermediate_result_visible=1.0` 对应的是 7 个正例；
- `ins_ok_text_visible` 在这个 holdout 上没有正例，因此它的 `seen F1` 无法反映“会不会识别 OK”，只能反映“会不会乱报 OK”。

## 7. 结果

### 7.1 Run-002 Newfacts Holdout

| 模型 | JSON valid | schema valid | fact accuracy | macro F1 | seen F1 | sample exact match | critical FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 1.0000 | 1.0000 | 0.8677 | 0.4513 | 0.5022 | 0.1000 | 11 |
| Run-003 LoRA | 1.0000 | 1.0000 | 0.9154 | 0.5306 | 0.6680 | 0.2000 | 20 |
| Run-003 + Run-005x2 LoRA | 1.0000 | 1.0000 | 0.9908 | 0.6515 | 0.9648 | 0.8800 | 4 |

![Run-005 / Run-002 newfacts overall accuracy](assets/qwen35_vlm_finetune/run005_run002_newfacts_overall_accuracy.png)

![Run-005 / Run-002 newfacts fact F1 by model](assets/qwen35_vlm_finetune/run005_run002_newfacts_fact_f1_by_model.png)

![Run-005 / Run-002 newfacts seen F1 by fact](assets/qwen35_vlm_finetune/run005_run002_newfacts_seen_f1_by_fact.png)

![Run-005 / Run-002 newfacts critical false positives](assets/qwen35_vlm_finetune/run005_run002_newfacts_critical_false_positives.png)

与 base model 相比，新 adapter 在这个外部 holdout 上的提升为：

- fact accuracy `+0.1231`
- macro F1 `+0.2002`
- seen F1 `+0.4626`
- sample exact match `+0.78`
- critical false positives `-7`

与旧的 Run-003 adapter 相比，新 adapter 的提升为：

- fact accuracy `+0.0754`
- macro F1 `+0.1208`
- seen F1 `+0.2969`
- sample exact match `+0.68`
- critical false positives `-16`

从关键 facts 看，新 adapter 并不是通过更保守的预测来减少误报，而是在提高 recall 的同时降低了误报：

| fact_id | Run-003 seen F1 | 新 adapter seen F1 | Run-003 FP/FN | 新 adapter FP/FN |
|---|---:|---:|---:|---:|
| `bit_root_page_visible` | 0.3636 | 1.0000 | 0 / 7 | 0 / 0 |
| `fcsmc_page_visible` | 0.9412 | 1.0000 | 0 / 1 | 0 / 0 |
| `fcsmc_intermediate_result_visible` | 0.0000 | 1.0000 | 2 / 1 | 0 / 0 |
| `fcsmc_in_test_visible` | 0.9091 | 1.0000 | 0 / 1 | 0 / 0 |
| `fcsmc_final_go_result_visible` | 0.8000 | 0.8000 | 1 / 0 | 1 / 0 |
| `ins_grnd_alignment_text_visible` | 0.7647 | 0.9882 | 0 / 16 | 1 / 0 |
| `ins_ok_text_visible` | 0.0000 | 0.8000 | 17 / 4 | 2 / 0 |

这说明 Run-005 补采不仅改善了 `bit_root` 和 `FCS-MC` 系列的漏检，也明显缓解了此前较突出的 `ins_ok_text_visible` 误报问题。

Run-002 上的关键混淆矩阵如下：

![Run-005 / Run-002 newfacts BIT root base](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_bit_root_page_visible_base.png)

![Run-005 / Run-002 newfacts BIT root lora](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_bit_root_page_visible_lora.png)

![Run-005 / Run-002 newfacts FCS-MC intermediate base](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_fcsmc_intermediate_result_visible_base.png)

![Run-005 / Run-002 newfacts FCS-MC intermediate lora](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_fcsmc_intermediate_result_visible_lora.png)

![Run-005 / Run-002 newfacts FCS-MC final GO base](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_fcsmc_final_go_result_visible_base.png)

![Run-005 / Run-002 newfacts FCS-MC final GO lora](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_fcsmc_final_go_result_visible_lora.png)

![Run-005 / Run-002 newfacts INS GRND base](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_ins_grnd_alignment_text_visible_base.png)

![Run-005 / Run-002 newfacts INS GRND lora](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_ins_grnd_alignment_text_visible_lora.png)

![Run-005 / Run-002 newfacts INS OK base](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_ins_ok_text_visible_base.png)

![Run-005 / Run-002 newfacts INS OK lora](assets/qwen35_vlm_finetune/run005_run002_newfacts_confusion_ins_ok_text_visible_lora.png)

### 7.2 Run-004 Random Holdout

| 模型 | JSON valid | schema valid | fact accuracy | macro F1 | seen F1 | sample exact match | critical FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.9600 | 0.9600 | 0.8038 | 0.4393 | 0.5659 | 0.0300 | 70 |
| Run-003 LoRA | 1.0000 | 1.0000 | 0.9185 | 0.4901 | 0.6390 | 0.5200 | 19 |
| Run-003 + Run-005x2 LoRA | 1.0000 | 1.0000 | 0.9931 | 0.6100 | 0.9159 | 0.9200 | 8 |

![Run-005 / Run-004 random overall accuracy](assets/qwen35_vlm_finetune/run005_run004_random_overall_accuracy.png)

![Run-005 / Run-004 random fact F1 by model](assets/qwen35_vlm_finetune/run005_run004_random_fact_f1_by_model.png)

![Run-005 / Run-004 random seen F1 by fact](assets/qwen35_vlm_finetune/run005_run004_random_seen_f1_by_fact.png)

![Run-005 / Run-004 random critical false positives](assets/qwen35_vlm_finetune/run005_run004_random_critical_false_positives.png)

与 base model 相比，新 adapter 的提升为：

- fact accuracy `+0.1892`
- macro F1 `+0.1707`
- seen F1 `+0.3500`
- sample exact match `+0.89`
- critical false positives `-62`

与旧 Run-003 adapter 相比，新 adapter 的提升为：

- fact accuracy `+0.0746`
- macro F1 `+0.1199`
- seen F1 `+0.2769`
- sample exact match `+0.40`
- critical false positives `-11`

### 7.3 Run-004 上关键 facts 的变化

在与 Run-005 补采目标最相关的 facts 上，改进较为明显：

| fact_id | Run-003 seen F1 | Run-003+Run-005x2 seen F1 | Run-003 FP/FN | 新 adapter FP/FN |
|---|---:|---:|---:|---:|
| `fcs_page_x_marks_visible` | 0.3333 | 1.0000 | 0 / 4 | 0 / 0 |
| `bit_root_page_visible` | 0.4762 | 1.0000 | 0 / 11 | 0 / 0 |
| `fcsmc_page_visible` | 0.8800 | 1.0000 | 0 / 18 | 0 / 0 |
| `fcsmc_intermediate_result_visible` | 0.0000 | 1.0000 | 0 / 7 | 0 / 0 |
| `fcsmc_in_test_visible` | 0.0000 | 0.9697 | 0 / 17 | 0 / 1 |
| `fcsmc_final_go_result_visible` | 0.7778 | 0.9375 | 17 / 11 | 8 / 0 |
| `ins_grnd_alignment_text_visible` | 0.8403 | 1.0000 | 0 / 19 | 0 / 0 |

从这些 facts 可以看到，Run-005 主要改善了两类问题：

1. **漏检问题**：例如 `bit_root_page_visible`、`fcsmc_intermediate_result_visible`、`fcsmc_in_test_visible`。
2. **阶段混淆问题**：例如把 `PBIT GO`、`IN TEST`、`final GO` 混在一起的错误。

### 7.4 Run-004 混淆矩阵示例

`bit_root_page_visible` 的 base 与新 adapter：

![Run-005 / Run-004 random BIT root base](assets/qwen35_vlm_finetune/run005_run004_random_confusion_bit_root_page_visible_base.png)

![Run-005 / Run-004 random BIT root lora](assets/qwen35_vlm_finetune/run005_run004_random_confusion_bit_root_page_visible_lora.png)

`fcsmc_intermediate_result_visible` 的 base 与新 adapter：

![Run-005 / Run-004 random FCS-MC intermediate base](assets/qwen35_vlm_finetune/run005_run004_random_confusion_fcsmc_intermediate_result_visible_base.png)

![Run-005 / Run-004 random FCS-MC intermediate lora](assets/qwen35_vlm_finetune/run005_run004_random_confusion_fcsmc_intermediate_result_visible_lora.png)

`fcsmc_final_go_result_visible` 的 base 与新 adapter：

![Run-005 / Run-004 random FCS-MC final GO base](assets/qwen35_vlm_finetune/run005_run004_random_confusion_fcsmc_final_go_result_visible_base.png)

![Run-005 / Run-004 random FCS-MC final GO lora](assets/qwen35_vlm_finetune/run005_run004_random_confusion_fcsmc_final_go_result_visible_lora.png)

`ins_grnd_alignment_text_visible` 的 base 与新 adapter：

![Run-005 / Run-004 random INS GRND base](assets/qwen35_vlm_finetune/run005_run004_random_confusion_ins_grnd_alignment_text_visible_base.png)

![Run-005 / Run-004 random INS GRND lora](assets/qwen35_vlm_finetune/run005_run004_random_confusion_ins_grnd_alignment_text_visible_lora.png)

## 8. 结果解读

### 8.1 为什么这次提升不太像偶然波动

由于本轮增幅较大，首先需要排除数据重复或评测偏差。当前已确认：

1. `Run-002` 与 `Run-004` 都与 `Run-003 + Run-005` 训练集不存在 exact overlap。
2. `Run-002` 50 张、`Run-004` 100 张都没有内部 exact 重图。
3. 改进并不只出现在低支持 fact 上；在 `Run-002` 这类更有意义的外部 holdout 上，`bit_root_page_visible`、`fcsmc_intermediate_result_visible`、`ins_grnd_alignment_text_visible`、`ins_ok_text_visible` 都明显提升。

因此，更合理的解释是：

- Run-005 补采的组合分布覆盖了 Run-003 的主要薄弱点；
- 过采样使这些补采样本在训练中获得了足够权重。

### 8.2 为什么 `seen F1 by fact` 看起来接近全 1

这张图容易被过度解读。主要原因不是模型已经“完美”，而是 `Run-004` 的支持分布偏斜：

- `tac_page_visible` 正例只有 3 个；
- `supt_page_visible` 正例只有 3 个；
- `fcsmc_intermediate_result_visible` 正例只有 7 个；
- `ins_ok_text_visible` 甚至没有正例。

因此，`Run-004` 的 `seen F1 by fact` 需要结合 support count 一起看，不能单独作为主要依据。相对而言，`Run-002 newfacts` 虽然样本更少，但它在 `ins_ok_text_visible`、`bit_root_page_visible`、`fcsmc_*` 上提供了更有信息量的正例，因此更适合作为本轮外部泛化的主要证据。

### 8.3 当前剩余的主要问题

当前剩余问题已经比 Run-003 收敛得多，但还没有完全消失：

- 在 `Run-002` 上，4 个 critical false positives 分别来自 `fcsmc_final_go_result_visible` 1 个、`ins_grnd_alignment_text_visible` 1 个、`ins_ok_text_visible` 2 个。
- 在 `Run-004` 上，剩余 8 个 critical false positives 全部集中在 `fcsmc_final_go_result_visible`。

整体来看，本轮已经较好地区分了 `bit_root`、`PBIT GO`、`IN TEST`、`GRND/QUAL/TIME` 等边界，但在 final GO 这类完成态文本上仍存在一定的积极偏置。

## 9. 局限

本报告需要明确以下限制：

1. `Run-002` 和 `Run-004` 都没有 exact overlap，但样本规模仍然不大，尤其 `Run-002` 只有 50 张。
2. `Run-004` 的标签支持明显偏斜，因此它更适合作为随机压力测试，而不是唯一主结论来源。
3. `Run-002` 虽然已经包含 `ins_ok_text_visible` 正例，但该 fact 的支持数仍然很少，后续仍需更多独立正例验证。
4. 本轮验证的是“固定 ontology，通过数据修正能否改善表现”，不是“ontology 已经最终定稿”。
5. 当前报告仍是离线 benchmark 结果，不等价于完整在线 SimTutor 决策链路效果。

## 10. 结论

本轮实验表明，在不继续修改 facts 的前提下，通过补充更有代表性的多屏组合样本、hard negatives，以及对补采数据进行适度过采样，可以明显提高 13-fact VLM 的稳定性。

与 base model 相比，`Run-003 + Run-005x2` 在 `Run-002 newfacts` 和 `Run-004 random` 两个 holdout 上都取得了显著提升；与旧 Run-003 adapter 相比，也表现出清晰的第二阶段改进。改进最明显的部分与 Run-005 的设计目标一致，包括 `bit_root`、`FCS-MC intermediate`、`IN TEST`、`GRND/QUAL/TIME` 和 `INS OK` 等此前较容易漏检或误报的 facts。

综合本轮结果，当前 13-fact ontology 可以先保持稳定，后续工作更适合继续补充有代表性的组合分布、增加独立正例支持，并在完整 SimTutor 下游推理链路中验证这一版 adapter，而不是立即再次改动 facts。下一步仍应继续收集新的独立 holdout，尤其补足 `INS OK` 和 `final GO` 这类稀有但风险较高的完成态样本。
