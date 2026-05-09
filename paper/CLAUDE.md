# CLAUDE.md — Thesis Writing Workspace (paper/)

> 论文写作专属上下文。此目录是从 `../paperwork/` 模板衍生出来的独立写作工作区。

---

## 为什么在 paper/ 下写作

- `../paperwork/` 是 TU Clausthal ISSE 的原始 LaTeX 论文模板（`scrreprt`，英式英语，`natbib`）。
- `paper/` 基于该模板但使用了**更新后的章节结构**（8 章，反映实际的 VLM 工作系统），而非模板中的占位章节。
- 模板的关键元素已被保留：文档类 `scrreprt`、标题页含 TUC logo 和导师信息、原创性声明、罗马/阿拉伯数字页码、德语 Zusammenfassung 以及 `plainnat` 引用格式。
- 此分离使得论文草稿可以独立于原始模板文件进行演进，同时仍遵循 TU Clausthal 的格式要求。

---

## 论文元信息

- **类型**：Master's Thesis，TU Clausthal，ISSE
- **导师**：Prof. Dr. Christian Bartelt（一导），Prof. Dr. Rüdiger Ehlers（二导）
- **主文件**：`main.tex`（KOMA-Script `scrreprt`，英式英语 + 德语，`natbib` + `plainnat`）
- **参考文献**：`references.bib`（参见 `../paperwork/Literature.bib` 获取原始模板引用）
- **图片资源**：`images/`（TUC logo 从 `../paperwork/images/` 复制）
- **规划文件**：`../paperwork/paper_context/`（第 1–4 步：仓库盘点、实验盘点、声明登记、论文大纲——均已完成）

---

## LaTeX 章节结构

| 文件 | 内容 | 状态 |
|------|------|------|
| `Preamble.tex` | 标题页 + 原创性声明 | **已就位**（改编自 `../paperwork/Preamble.tex`） |
| `sections/00_abstract.tex` | 摘要 + Zusammenfassung | 待写（最后） |
| `sections/00_acknowledgements.tex` | 致谢 | 待写 |
| `sections/01_introduction.tex` | 引言 | 待写 |
| `sections/02_background_related_work.tex` | 背景与相关工作 | 待写 |
| `sections/03_system_design.tex` | 系统设计 | **已写 prose**（~4000 词） |
| `sections/04_methodology.tex` | 方法论 | 骨架就位 |
| `sections/05_implementation.tex` | 实现 | 骨架就位 |
| `sections/06_evaluation.tex` | 评估（技术结果 + 已规划用户研究） | 骨架就位 |
| `sections/07_discussion.tex` | 讨论 | 骨架就位 |
| `sections/08_conclusion.tex` | 结论 | 骨架就位 |
| `sections/09_appendix.tex` | 附录 | 骨架就位 |

---

## 证据与声明纪律

这是最重要的规则：

- **绝不编造**实验结果、已实现特性、用户研究结果或统计显著性。
- 每条技术声明必须指向具体证据：源代码路径、benchmark JSON 文件、数据集统计、开发笔记或 `../paperwork/paper_context/` 文件。
- 证据缺失或未经核实时，写 `TODO:VERIFY`，绝不静默填补空白。
- 在 LaTeX 中用 `% Evidence: <file-path>` 注释标注证据来源。
- 严格区分四种状态：
  - **Implemented** — 代码存在，测试通过
  - **Experimental prototype** — 代码存在，非生产级
  - **Planned** — 设计完成但未实现
  - **Future work** — 超出范围，已确认
- 仅使用来自 `paper_context/claims_registry.md` 的 **verified** 或 **likely** 声明作为事实性陈述。**TODO** 声明只能在 Planned Evaluation 章节或 Future Work 中以未来工作的形式出现。

---

## 写作风格

- 学术英语，清晰优先于华丽。避免营销语言和夸大声明。
- 术语必须与 `../paperwork/paper_context/terminology.md` 保持一致。
- 每次改动保持小而可审查（git diff 友好）。
- 写长篇之前，先总结：(a) 可用证据，(b) 未解决的 TODO。

---

## 论文写作顺序

严格遵循此顺序，**不从摘要开始**：

1. ~~仓库盘点~~ → `../paperwork/paper_context/repo_inventory.md`
2. ~~实验盘点~~ → `../paperwork/paper_context/experiment_inventory.md`
3. ~~声明登记~~ → `../paperwork/paper_context/claims_registry.md`
4. ~~论文大纲~~ → `../paperwork/paper_context/paper_outline.md`
5. **系统设计** → `sections/03_system_design.tex` ✅ **已完成**
6. **方法论** → `sections/04_methodology.tex`
7. **实现** → `sections/05_implementation.tex`
8. **评估/结果** → `sections/06_evaluation.tex`（先技术结果，后已规划用户研究）
9. **讨论与局限** → `sections/07_discussion.tex`
10. **引言** → `sections/01_introduction.tex`
11. **背景与相关工作** → `sections/02_background_related_work.tex`
12. **结论** → `sections/08_conclusion.tex`
13. **摘要** → `sections/00_abstract.tex`
14. **标题** — 最后确定

---

## 编译

```bash
cd paper
make          # tectonic main.tex（单次运行，自动获取依赖）
make clean    # 移除构建产物
```

注意：`pdflatex` + `bibtex` 不可用；本仓库使用 `tectonic`（从 GitHub releases 下载的预构建二进制文件）。

---

## 审稿检查清单

修改任何章节后，逐项确认：

- [ ] 主要声明是否清晰、具体？
- [ ] 声明是否有可引用的证据支撑（代码路径/日志/benchmark）？
- [ ] 未经核实的声明是否标记了 `TODO:VERIFY`？
- [ ] 已实现特性是否与 planned/future work 明确区分？
- [ ] 该章节是否符合 `paper_context/paper_outline.md` 中的论文整体大纲？
- [ ] 该章节使用的声明是否与 `paper_context/claims_registry.md` 中的状态一致？
- [ ] LaTeX 是否仍能编译通过（`make`）？
- [ ] 术语是否与 `paper_context/terminology.md` 一致？
- [ ] 正文中是否没有将 TODO 声明当作已完成结果呈现？
