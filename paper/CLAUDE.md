# CLAUDE.md — Thesis Writing Workspace

> 论文写作专属上下文。代码库架构与开发规则参见上级目录的 `../CLAUDE.md`。

---

## 论文元信息

- **类型**：Bachelor Thesis，TU Clausthal，ISSE
- **导师**：Prof. Dr. Christian Bartelt（一导），Prof. Dr. Rüdiger Ehlers（二导）
- **主文件**：`Thesis.tex`（KOMA-Script `scrreprt`，英式英语，`natbib` + `plainnat`）
- **参考文献**：`Literature.bib`
- **图片资源**：`images/`
- **规划文件**：`paper_context/`（写作顺序第 1-4 步已完成）

## LaTeX 章节结构

| 文件 | 内容 | 状态 |
|------|------|------|
| `Preamble.tex` | 封面 + 原创性声明 | 模板已就位 |
| `Abstract.tex` | 摘要 | 待写（最后） |
| `Introduction.tex` | 引言 | 待写（倒数第二步） |
| `Related_Work.tex` | 相关工作 | 待写 |
| `Theoretical_Background.tex` | 理论背景 | 待写 |
| `Methodology.tex` | 方法论 | 待写（核心章节） |
| `Results.tex` | 实验结果 | 待写 |
| `Discussion.tex` | 讨论与局限 | 待写 |
| `Conclusion.tex` | 结论 | 待写（倒数第三步） |
| `Appendix.tex` | 附录 | 待写 |

## 证据与声明纪律

这是最重要的规则：

- **绝不编造**实验结果、已实现特性、用户研究结果或统计显著性。
- 每条技术声明必须指向具体证据：源代码路径、`artifacts/` 实验日志、`benchmarks/` 数据、`datasets/` 数据集、`docs/dev_notes/` 开发笔记。
- 证据缺失或未经核实时，写 `TODO:VERIFY`，绝不静默填补空白。
- 在 LaTeX 中用 `% Evidence: <file-path>` 注释标注证据来源。
- 严格区分四种状态：
  - **Implemented** — 代码存在，测试通过
  - **Experimental prototype** — 代码存在，非生产级
  - **Planned** — 设计完成但未实现
  - **Future work** — 超出范围，已确认

## 写作风格

- 学术英语，清晰优先于华丽。避免营销语言和夸大声明。
- 术语必须与 `../.github/copilot-instructions.md` 第 6 节的术语表保持一致。
- 每次改动保持小而可审查（git diff 友好）。
- 写长篇之前，先总结：(a) 可用证据，(b) 未解决的 TODO。

## 论文写作顺序

严格遵循此顺序，**不从摘要开始**：

1. ~~仓库盘点~~ → `paper_context/repo_inventory.md`
2. ~~实验盘点~~ → `paper_context/experiment_inventory.md`
3. ~~声明登记~~ → `paper_context/claims_registry.md`
4. ~~论文大纲~~ → `paper_context/paper_outline.md`
5. **系统设计** → `Theoretical_Background.tex` / `Methodology.tex`
6. **方法论** → `Methodology.tex`（VLM 微调设置、评估协议）
7. **实现** → `Related_Work.tex` 或 Methodology 内专节
8. **评估/结果** → `Results.tex`
9. **讨论与局限** → `Discussion.tex`
10. **引言** → `Introduction.tex`
11. **结论** → `Conclusion.tex`
12. **摘要** → `Abstract.tex`
13. **标题** — 最后确定

## 术语速查

全文统一使用以下术语（详见 `../CLAUDE.md` 和 `../.github/copilot-instructions.md`）：

| 术语 | 含义 |
|------|------|
| tutor orchestrator | 顶层循环：observe → evaluate gates → decide help → execute |
| procedure engine | 加载 procedure pack，追踪步骤进度 |
| gating rule engine | 评估每步的前置/完成条件 |
| observation | 模拟器状态快照（遥测变量 + 可选视觉事实） |
| tutor request | 触发帮助时发送给 LLM 的结构化查询 |
| tutor response | LLM 返回的结构化 JSON，经验证后映射为覆盖层动作 |
| event log | 所有 tutor 交互、观测、结果的 JSONL 记录 |
| VLM adaptation | Qwen3.5-VLM 通过 LoRA 微调用于座舱视觉事实提取 |
| visual fact | 关于座舱 UI 元素的类型化命题（`seen`/`not_seen`/`uncertain`） |
| grounded procedural guidance | 引用具体座舱状态的帮助，而非通用建议 |

## 审稿检查清单

修改任何章节后，逐项确认：

- [ ] 主要声明是否清晰、具体？
- [ ] 声明是否有可引用的证据支撑（代码路径/日志/benchmark）？
- [ ] 未经核实的声明是否标记了 `TODO:VERIFY`？
- [ ] 已实现特性是否与计划/未来工作明确区分？
- [ ] 该章节是否符合论文整体大纲（见 `paper_context/paper_outline.md`）？
- [ ] LaTeX 是否仍能编译通过？
- [ ] 术语是否与术语表一致？
