# Paper Outline

> 目标：给当前论文一个 IMRaD-like 的系统论文结构，同时显式保留你要继续推进的 VR / 用户研究主线。  
> 原则：把“当前可写结果”和“占位核心章节”明确拆开。

## 1. 推荐总结构

建议采用以下主章节：

1. Introduction
2. Related Work
3. System Context and Background
4. Methods
5. Current Technical Results
6. Planned Runtime Data Collection and Human-Subject Evaluation
7. Discussion
8. Conclusion

与当前 `paperwork/Thesis.tex` 的关系：

- `Introduction.tex`：继续使用
- `Related_Work.tex`：继续使用
- `Theoretical_Background.tex`：可改造成 `System Context and Background`
- `Methodology.tex`：继续使用，扩成 Methods 主章
- `Results.tex`：只写 current technical results
- `Discussion.tex`：继续使用
- `Conclusion.tex`：继续使用

## 2. 各章目标与内容边界

### 2.1 Introduction

目的：

- 定义 thesis-level research problem
- 交代 procedural tutoring 与 grounded assistance 的动机
- 诚实说明 proposal 与 current implementation 的偏移
- 宣布当前技术贡献与未来实验主线

应包含：

1. procedural learning 的挑战
2. VR / immersive training 的 thesis motivation
3. 为什么需要 grounded runtime
4. 当前已经完成的是哪一部分
5. 用户研究和 runtime 数据收集将作为下一阶段核心工作

不应包含：

- 任何尚未完成的人因实验结果句

主要证据：

- `paperwork/paper_context/proposal_summary.md`
- `paperwork/paper_context/system_delta_from_proposal.md`
- `paperwork/paper_context/claims_registry.md`

### 2.2 Related Work

目的：

- 把论文放到 procedural tutoring、RAG tutoring、VLM adaptation、simulator-based evaluation 的交叉位置

推荐小节：

1. Procedural tutoring and intelligent assistance
2. Grounded LLM systems and retrieval-augmented help
3. Vision-language model adaptation for structured outputs
4. Simulator-based training and instrumentation
5. Human-subject evaluation in procedural learning

不应做的事：

- 不要让 related work 看起来像“本文已经完成 VR study”

### 2.3 System Context and Background

目的：

- 帮读者理解当前实现系统所需的技术背景

推荐小节：

1. DCS and the F/A-18C cold-start task
2. Pack-driven procedure modeling
3. Telemetry processing and grounding
4. Visual fact ontology
5. LoRA-based VLM adaptation

关键证据：

- `packs/fa18c_startup/pack.yaml`
- `packs/fa18c_startup/vision_facts.yaml`
- `adapters/telemetry_pipeline.py`
- `tools/train_qwen35_vlm_unsloth.py`

### 2.4 Methods

目的：

- 完整描述当前仓库真正实现了什么

推荐小节：

#### 4.1 Overall system architecture

- `core/` / `ports/` / `adapters/`
- live runtime orchestration
- replay/logging path

证据：

- `live_dcs.py`
- `simtutor/__main__.py`
- `paperwork/paper_context/repo_inventory.md`

#### 4.2 Procedure pack and telemetry-grounded runtime

- step registry
- gating rules
- telemetry normalization / delta handling
- retrieval / source policy

证据：

- `packs/fa18c_startup/*.yaml`
- `adapters/telemetry_pipeline.py`
- `adapters/knowledge_local.py`

#### 4.3 Visual fact extraction pipeline

- candidate frames
- prompt contract
- 13-fact ontology
- runtime integration

证据：

- `adapters/vision_fact_extractor.py`
- `adapters/vision_fact_prompting.py`
- `packs/fa18c_startup/vision_facts.yaml`

#### 4.4 Dataset curation and model adaptation

- capture
- prelabel
- human review
- SFT export
- Qwen/Gemma LoRA training

证据：

- `tools/capture_vlm_dataset.py`
- `tools/generate_vlm_prelabels.py`
- `tools/export_vision_sft_dataset.py`
- `models/*/train_summary.json`

#### 4.5 Benchmark protocol

- base vs LoRA
- v1 8-fact line
- v2 13-fact line
- holdout definitions

证据：

- `tools/benchmark_qwen35_vlm_facts.py`
- `benchmarks/qwen35_vlm_finetune/*/benchmark_summary.json`

### 2.5 Current Technical Results

目的：

- 只报告当前仓库已有 artifact 支撑的技术结果

推荐小节：

#### 5.1 Dataset and training summary

- 训练数据规模
- v1/v2 ontology 差异
- Qwen/Gemma train rows 与 hyperparameters

#### 5.2 Qwen benchmark results

- 8-fact v1 holdout run002
- 13-fact holdout run002 newfacts
- 13-fact holdout run004 random

#### 5.3 Gemma benchmark results

- holdout run002 newfacts
- holdout run004 random

#### 5.4 Runtime / artifact readiness

- replay_eval data
- benchmark reports
- stored charts

可以写的结果句类型：

- “On holdout X, base vs LoRA differs by …”
- “The repository stores benchmark reports and confusion charts for …”

不能写的结果句类型：

- “Participants learned faster”
- “Workload decreased”
- “VR immersion improved retention”

### 2.6 Planned Runtime Data Collection and Human-Subject Evaluation

目的：

- 把 thesis 核心但尚未完成的 VR / 用户研究主线放在一个明确的占位章节里

推荐小节：

#### 6.1 Original evaluation vision

- 对照组：`video+notes`, `ungrounded text-only LLM`, grounded tutor
- 指标：completion time, error rate, NASA-TLX, retention, transfer

#### 6.2 Required runtime data collection upgrades

- 待补字段
- 待冻结日志 schema
- 待定义 participant/session linkage

#### 6.3 Planned experiment design

- participant profile
- procedure
- measurements
- analysis plan

#### 6.4 Threats to validity already known

- current system != final VR runtime
- current technical benchmark != learning outcome evidence

这一章只能写：

- 目标
- 设计
- 待收集数据
- 待填图表

这一章不能写：

- 任何已完成受试者结果

### 2.7 Discussion

目的：

- 解释 proposal 与 current system 的偏移
- 讨论 current technical line 的意义与局限

推荐小节：

1. Why the project shifted from VR-first MVP to telemetry/VLM-heavy implementation
2. What current benchmarks do and do not prove
3. 8-fact to 13-fact ontology evolution
4. Why human-subject evidence remains essential
5. Risks in interpreting benchmark success as learning success

### 2.8 Conclusion

目的：

- 收束为两件事：
  1. 当前已完成的 technical baseline
  2. 接下来要完成的 runtime data collection and human evaluation

## 3. 待填图表与表格槽位

### 当前可以准备的图表

1. System architecture diagram
2. Visual fact pipeline diagram
3. Dataset summary table
4. Qwen base-vs-LoRA benchmark table
5. Gemma base-vs-LoRA benchmark table
6. Representative confusion / per-fact chart figure

证据源：

- `benchmarks/qwen35_vlm_finetune/*/charts/*.png`
- `datasets/*/stats.json`
- `models/*/train_summary.json`

### 占位但不能填结果的图表

1. Participant flow chart
2. NASA-TLX summary figure
3. Completion time comparison
4. Retention / delayed post-test table
5. Transfer task comparison

这些只能放入未来 evaluation 章节作为 planned tables/figures。

## 4. 与 claims registry 的一致性约束

本 outline 与 `paperwork/paper_context/claims_registry.md` 的对应关系应固定如下：

- 第 5 章 `Current Technical Results` 只能使用 `verified` 或少量 `likely` claims
- 第 6 章 `Planned Runtime Data Collection and Human-Subject Evaluation` 只能使用 `TODO` claims
- 若后续某个 user-study claim 仍无数据，则不能从第 6 章提前挪进第 5 章

## 5. 一句话版本

推荐把整篇论文写成：

> 一篇以 VR/human-study thesis vision 为总目标、以当前 telemetry-grounded runtime 与 VLM adaptation results 为已完成技术基线的系统论文。

