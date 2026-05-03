# Issue: 对齐 SimTutor 运行时 VLM 契约到 Run-003 / Run-005 的 13-fact 微调契约

## 背景

当前已经完成的 Qwen / Gemma 微调，训练数据使用的是 **Run-003 / Run-005 确定的 13-fact ontology**。  
但是 SimTutor 运行时用于视觉事实抽取的 prompt、schema 和下游绑定，仍然停留在更早的一套 **旧视觉 fact 契约** 上。

这意味着：

- 微调后的 adapter 学到的是 **新 13-fact 任务**
- SimTutor 运行时却仍在要求 **旧 fact 集合 + 旧输出字段**

因此，如果直接把当前微调后的 adapter 接到 SimTutor 运行时，多半会出现：

1. 输出格式偏移
2. fact_id 不匹配
3. schema 校验失败
4. 下游步骤推理无法正确消费视觉事实
5. 即使不崩，也会因为任务定义不一致而显著损失效果

---

## 问题不是“prompt 文字不同”，而是“契约已分叉”

### 训练侧（当前微调用）

训练导出 prompt 见：

- `tools/export_vision_sft_dataset.py`

训练目标的关键特征：

- 事实集合是新的 13 个 facts：
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
- top-level 输出允许：
  - `summary`
  - `facts`
- 每个 fact 只包含：
  - `fact_id`
  - `state`
  - `evidence_note`
- 明确禁止输出：
  - `source_frame_id`
  - `frame_id`
  - `confidence`

### 运行时（当前 SimTutor 实际使用）

运行时视觉 prompt 见：

- `adapters/vision_fact_prompting.py`

运行时 schema 与抽取器见：

- `adapters/vision_fact_extractor.py`

运行时视觉 facts config / binding 见：

- `packs/fa18c_startup/vision_facts.yaml`
- `core/vision_facts.py`

当前运行时仍使用旧 fact 集合，例如：

- `left_ddi_dark`
- `right_ddi_dark`
- `ampcd_dark`
- `left_ddi_menu_root_visible`
- `left_ddi_fcs_option_visible`
- `left_ddi_fcs_page_button_visible`
- `bit_page_visible`
- `bit_page_failure_visible`
- `right_ddi_fcsmc_page_visible`
- `right_ddi_fcs_option_visible`
- `fcs_reset_seen`
- `fcs_bit_interaction_seen`
- `takeoff_trim_seen`
- `ins_alignment_page_visible`
- `ins_go`

而且运行时 schema 还要求每个 fact 必须包含：

- `fact_id`
- `state`
- `source_frame_id`
- `evidence_note`

也就是说，当前运行时和训练侧至少存在下面这些硬冲突：

1. **fact_id 集合不同**
2. **top-level 输出契约不同**
3. **训练侧不输出 `source_frame_id`，运行时却强制要求**
4. **旧 runtime prompt 仍包含旧 ontology 的判别规则**
5. **下游 step_bindings / inference 仍依赖旧 fact 名称**

---

## 直接结论

**现在不能放心地把当前微调后的 adapter 直接接到 SimTutor 运行时。**

不是说一定完全不能跑，而是：

- 即使模型本身能力足够，运行时也会因为契约不一致而“吃亏”
- 风险不仅是精度下降，更可能是解析或绑定层面直接不匹配

---

## 推荐修复方向

推荐做 **正式契约对齐**，而不是临时 patch 一两句 prompt。

### 首选方案（推荐）

把 SimTutor 运行时视觉抽取链路迁移到 **Run-003 / Run-005 的 13-fact ontology**。

需要一起对齐的层面包括：

1. **运行时 prompt**
   - `adapters/vision_fact_prompting.py`

2. **运行时 response schema**
   - `adapters/vision_fact_extractor.py`

3. **运行时 vision facts 配置**
   - `packs/fa18c_startup/vision_facts.yaml`
   - `core/vision_facts.py`

4. **下游 step bindings / step inference / rule wording**
   - 所有仍引用旧 fact 名称的逻辑

5. **测试**
   - 更新旧 fact 断言
   - 新增对训练契约与运行契约一致性的测试

### 不推荐方案

做一个“新 13-fact -> 旧 runtime facts”的临时映射层。

原因：

- 有些旧 facts 和新 facts 并不是一一等价
- 会引入语义损失
- 会让后续维护更复杂
- 只是在掩盖训练契约与运行契约已经分叉的事实

如果必须短期兼容，也只能作为过渡方案，而且必须明确写成 temporary compatibility layer。

---

## 当前 Issue 的目标

实现“**运行时 VLM 契约对齐到 Run-003 / Run-005 微调契约**”，使当前微调后的 adapter 可以在 SimTutor 中以尽量原生的方式被使用。

---

## 建议范围

### 需要做

1. 梳理训练侧与运行侧的视觉事实契约差异
2. 确定运行时应采用的新 13-fact 集合
3. 更新 runtime prompt，使其与训练 prompt 的任务定义保持一致
4. 更新 runtime schema，至少处理：
   - 是否保留 `summary`
   - 是否继续要求 `source_frame_id`
5. 更新 `vision_facts.yaml` 与 `core/vision_facts.py`
6. 更新下游依赖旧 fact 的逻辑（尤其 step bindings / inference / prompting）
7. 更新相关测试并跑全绿

### 不需要做

1. 不重新采集数据
2. 不重新训练模型
3. 不修改多模态基础设施
4. 不扩展新的视觉 facts beyond 当前 13-fact ontology

---

## 特别注意

### 关于 `source_frame_id`

训练导出明确禁止模型输出 `source_frame_id`。  
因此这次对齐时需要明确决定：

- 是不是让运行时也不再要求模型输出 `source_frame_id`
- 改为由工具层根据当前候选帧、主帧或抽取上下文自动补 metadata

这通常是更合理的设计，因为：

- `source_frame_id` 属于可由系统补齐的 metadata
- 没必要把它变成模型训练负担

### 关于 `summary`

训练数据默认保留过 `summary`，但很多实际下游逻辑主要消费的是 `facts`。  
因此这次 issue 里要明确：

- 运行时是否真的需要 `summary`
- 若不需要，是否改为可选字段，或在运行时忽略它

---

## 建议优先查看的文件

### 训练侧

- `tools/export_vision_sft_dataset.py`
- `tests/test_export_vision_sft_dataset.py`

### 运行时视觉抽取侧

- `adapters/vision_fact_prompting.py`
- `adapters/vision_fact_extractor.py`
- `adapters/vision_prompting.py`
- `adapters/openai_compat_multimodal.py`

### 视觉 facts 配置与绑定

- `packs/fa18c_startup/vision_facts.yaml`
- `core/vision_facts.py`
- `adapters/step_inference.py`
- `adapters/prompting.py`

### 测试

- `tests/test_vision_fact_prompting.py`
- `tests/test_vision_fact_extractor.py`
- `tests/test_vision_fact_contracts.py`
- `tests/test_live_dcs.py`

---

## 验收标准

完成后至少满足：

1. 运行时 VLM prompt 的 fact 集合与微调目标一致
2. 运行时 schema 不再与训练输出格式冲突
3. 当前微调 adapter 能在 SimTutor runtime 中被合理消费
4. 下游 step inference / step binding 不再依赖已废弃旧 fact 名称
5. 相关单测 / 集成测试全绿

---

## 给新对话的直接指令

可以把下面这段直接贴给新对话：

> 请先阅读 `docs/dev_notes/issue_align_runtime_vlm_contract_with_run005_adapter.md`。  
> 当前问题不是单纯的 prompt wording 差异，而是训练侧与运行时 VLM 契约已经分叉。  
> 请先做仓库勘探，比较训练侧（`tools/export_vision_sft_dataset.py`）与运行时（`adapters/vision_fact_prompting.py`、`adapters/vision_fact_extractor.py`、`packs/fa18c_startup/vision_facts.yaml`、`core/vision_facts.py`）的差异，然后实施“将运行时 VLM 契约对齐到 Run-003 / Run-005 的 13-fact 微调契约”这一 issue。  
> 所有回复用中文，允许 TDD，但不要扩大范围，不要顺带重训模型。

