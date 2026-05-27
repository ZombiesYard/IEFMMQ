# Proposal Summary

> 来源：`ThesisProposal1128.pdf`  
> 方法：基于 PDF 可恢复章节文本整理，不把 proposal 内容等同于当前实现系统。  
> 目标：为后续论文写作保留“原始研究意图”的可引用摘要。

## 1. 研究动机

proposal 的问题定义聚焦在 procedural learning，尤其是固定顺序、状态敏感、容易发生 context switching 的训练任务。

根据 proposal 中 `Motivation and Problem Definition` 一节，可恢复出的核心动机包括：

1. 学习者在复杂程序性任务中容易因为步骤顺序、状态确认和外部资料切换而出错
2. 传统的视频、笔记、手册式辅助会打断 VR 训练中的沉浸与流程连续性
3. 现有 embodied / egocentric / perception 相关工作更多关注开放式视觉理解，而不是固定 procedure 下的 grounded tutoring
4. proposal 明确希望避免“heavy-weight online visual perception at runtime”，而是优先依靠 read-only simulator state 与 retrieval-grounded explanation

可恢复章节锚点：

- `Motivation and Problem Definition`
- `Learner pain points`
- `Workflow breaks`
- `Technology limits`

## 2. 计划系统

proposal 的系统目标写在 `Goal of the Thesis` 与 `System Overview`。

### 2.1 平台与交互

proposal 的原始平台是：

- Meta Quest 3
- DCS-VR
- in-headset guidance

原始交互设想包括：

- gaze
- hand interactions
- voice
- read-only task-state signals from the simulator

### 2.2 推理与输出

proposal 的 planned architecture 是一条较轻量的 VR-first RAG+LLM 链路：

```text
Quest 3 interactions
-> PC / edge backend
-> retrieve manual/checklist chunks
-> LLM generates one short step
-> VR step card / overlay rendered in-headset
```

proposal 中对 LLM 输出的预期是结构化但轻量的 explanation triple：

- Action
- Check
- Why

也就是：

1. 一条可执行的 next step
2. 一条 quick self-check
3. 一条简短 rationale

### 2.3 数据与 grounding

proposal 假设系统主要依赖：

- read-only DCS state flags
- manual / checklist chunk retrieval
- grounded citations / source support

proposal 的 MVP 图示描述了一个最小系统：

- Quest 3 收集交互
- PC/edge 读取 DCS state
- 检索 manual snippets
- LLM 返回 concise step
- VR 内显示 compact cards

可恢复章节锚点：

- `Goal of the Thesis`
- `Objective 1: System`
- `System Overview`
- `Figure 1` / `Minimum Viable Product architecture (MVP)`

## 3. 计划验证

proposal 的验证路线写在 `Objective 2: Validation` 与 `Methodological Contribution`。

原始验证问题可以概括为：

> 在 DCS-VR 中，一个 retrieval-augmented grounded LLM tutor 能否可靠引导初学者完成 fixed, sequence-sensitive procedure，并优于 video+notes baseline？

### 3.1 对照组

proposal 中可恢复出的比较条件包括：

1. `video+notes`
2. `ungrounded text-only LLM`
3. `grounded RAG+LLM tutor`

### 3.2 指标

proposal 计划采集的评估指标包括：

- completion rate
- procedural error rate
- total time / task time
- dead-end count
- NASA-TLX
- pre/post quizzes
- 2-4 week retention
- transfer to a variant procedure

### 3.3 研究设计

proposal 预期做 controlled evaluation / A-B-C style human-subject study，参与者大致是：

- beginner to intermediate
- sample size 计划约 `n ≈ 5–10`

注意：这些都属于 proposal 的**计划验证方案**，不是当前仓库里已经完成的实验。

可恢复章节锚点：

- `Objective 2: Validation`
- `Methodological Contribution`
- `Work Plan`

## 4. 预期贡献

proposal 把贡献分成三类：

### 4.1 Technical contribution

proposal 预期交付一个：

- grounded
- citation-included
- in-headset tutoring system

系统将结合：

1. VR interaction inputs
2. minimal task/state data from simulators
3. retrieved manual/checklist chunks

### 4.2 Methodological contribution

proposal 预期设计一套用于 VR-based procedural learning 的 controlled evaluation framework，包括：

- multiple comparison conditions
- immediate performance metrics
- workload metrics
- longer-term learning outcomes

### 4.3 Practical contribution

proposal 还希望把 DCS World 验证为：

- evaluation testbed
- repeatable environment for procedural learning systems

proposal 的工作计划也围绕这三类贡献展开：

- 早期 weeks：task preparation / evaluation design / RAG+LLM backend
- 中期 weeks：VR display MVP / state grounding / V1 freeze
- 后期 weeks：rehearsal / main study / analysis / write-up

可恢复章节锚点：

- `Contributions`
- `Technical Contribution`
- `Methodological Contribution`
- `Practical Contribution`
- `Work Plan`

## 5. 对后续论文写作的直接含义

proposal 提供的是：

1. 原始 thesis vision
2. 原始 validation ambition
3. 原始人因实验路线

但它**不能**直接当作当前系统描述使用。后续论文需要明确区分：

- proposal-level research agenda
- current repository-level implemented system
- future runtime data collection and human-subject evaluation

