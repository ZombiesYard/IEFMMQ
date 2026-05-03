# New Chat Handoff: v0.4 Runtime Status and Pending Issues

这份文档用于开启一个**新对话**时，向新的 Codex 说明仓库背景、工程约束、当前系统状态，以及 issue 处理方式。  
尽管文件名里还保留了 `multitarget_overlay` 这个旧历史名字，**本文件现在应视为通用 handoff 文档**，不是某个特定 issue 的提示词。

---

## 1. Role

你是首席软件架构师 + 资深 Python 工程师（Python 3.10+），对 Clean Architecture / Hexagonal Architecture 非常熟悉，能交付生产级代码与测试。如果后续有和karpathy-guidelines skill冲突的地方以karpathy-guidelines skill为准。

---

## 2. Language

所有回复必须使用中文，并使用中文逻辑思考。

---

## 3. Repo / Environment

### 本地仓库

- Repo root:
  `/mnt/l/Documents/files/Yu Zhang TU Clausthal/Thesis/IEFMMQ`
- Git remote:
  `https://github.com/ZombiesYard/IEFMMQ.git`

### 远程服务器

- SSH:
  `ssh yz50@cloud-247.rz.tu-clausthal.de`
- 远程 shell 是 `tcsh`，不是 bash
- 只使用 `/scratch/yz50`
- 不要使用 home 目录存放模型 / cache / 训练输出
- 不要随便运行 `rm`；如需删除任何远程文件，必须先明确询问用户
- 远程很多环境变量要用 `setenv`，不是 `export`

### 当前已知远程缓存习惯

如果需要在远程运行模型、benchmark 或训练，优先确保这些目录与环境变量：

```tcsh
mkdir -p /scratch/yz50/tmp
mkdir -p /scratch/yz50/.cache/uv
mkdir -p /scratch/yz50/.cache/huggingface
mkdir -p /scratch/yz50/.cache/pip

setenv TMPDIR /scratch/yz50/tmp
setenv UV_CACHE_DIR /scratch/yz50/.cache/uv
setenv PIP_CACHE_DIR /scratch/yz50/.cache/pip
setenv HF_HOME /scratch/yz50/.cache/huggingface
setenv HUGGINGFACE_HUB_CACHE $HF_HOME/hub
setenv TRANSFORMERS_CACHE $HF_HOME/transformers
setenv PATH /scratch/yz50/.local/bin:$PATH
rehash
```

---

## 4. AUTONOMY & REPO ACCESS

你被明确授权在本地对整个仓库进行自主探索与检索，包括但不限于：

- 自主列目录、打开任意文件（`core/ports/adapters/packs/docs/tests/...`）
- 自主全库搜索（`grep` / `rg`）定位契约、调用链、测试入口、schema、pack 结构
- 自主运行与修复测试（`pytest` 等），并根据失败堆栈反向定位问题
- 自主生成/修改最小必要的文档与 schema 变更（遵守版本规则）
- 为完成当前 issue，允许进行小范围重构（提取函数、重命名变量、补注释、补类型标注等），但不得扩大功能范围

默认先做“仓库勘探 + grep 定位 + 依赖链确认”，再进入实现。  
不需要等待用户给文件路径。

---

## 5. OPTIONAL DEV ARTIFACTS

允许临时开发产物，例如：

- `scripts/dev_debug_*.py`
- `scripts/dev_repro_*.py`
- `docs/dev_notes.md`
- `docs/dev_notes/<issue_slug>.md`

但必须遵守：

- 默认不进入最终交付；如果只是一次性调试，提交前删除或不纳入最终 PR
- 若决定保留，必须放在明确的 dev/diagnostic 位置，并在总结里说明用途
- 不得破坏安全边界（尤其不得引入自动点击等）

---

## 6. TDD Permission

允许先写 failing tests，再实现代码使其通过。  
最终要求：

- 全量测试必须通过
- 不得留下故意失败的测试

---

## 7. Scope Rule

每次只实现 **ONE 当前 Issue**。  
当前对话里粘贴的 issue 就是唯一任务。

允许：

- 为完成该 issue 修改多个文件、模块、测试
- 若存在阻塞 bug，允许一并修复，但必须明确说明它为什么是 blocking fix

不允许：

- 顺带实现其他功能
- 扩大产品范围

---

## 8. Project Context at Start

### v0.3 状态

v0.3 已实现且可用：

- 系统能在触发 help 后，通过 telemetry enrich
- 使用去噪后的 `delta_summary`
- grounded LLM 输出结构化 JSON
- 然后安全映射执行 overlay 高亮

v0.3 的重要约束继续有效：

- DCS-BIOS 输入是高频增量写入，噪声大
- 进入 prompt 的必须是去噪 / 预算 / 聚合后的摘要
- 禁止把 raw delta / 全量 BIOS 直接塞进 prompt

### v0.4 总目标

在 v0.3 基础上引入 Qwen3.5 的 VLM 能力，用视觉补齐 BIOS 无法直接观测的步骤证据，例如：

- FCS RESET
- FCS BIT
- 页面状态

目标是避免模型在 live / replay 中被错误推进到后续步骤。

---

## 9. v0.4 Global Decisions（必须遵守）

### 部署拓扑

- `simtutor/live_dcs.py` 跑在 DCS 同机
- VLM 跑在远程ssh物理机`yz50@cloud-247.rz.tu-clausthal.de`，但还没有安装模型，只有vlm设置完毕，需要连接并确认
- 组合面板图帧优先走 DCS 同机磁盘路径
- overlay / ACK 如涉及网络，必须可配置，默认兼容 `127.0.0.1`

### 视觉输入形态：组合面板图

固定组合图区域（首版冻结）：

- 左 DDI
- AMPCD
- 右 DDI

布局参考：

- `Doc/Vision/assets/fa18c_composite_panel_fullscreen_v2.svg`
- `Doc/Vision/assets/fa18c_composite_panel_v2.svg`

特点：

- 从上到下垂直排列：左 DDI、AMPCD、右 DDI
- 右侧是模拟器画面
- 不包含外景 / 座舱大面积背景 / 无关控制台

### 采帧策略

- 平时持续低频采集组合图（约 1–2 fps）
- help 触发时立即补抓 1 张
- 每次 help cycle 默认使用两张候选：
  - `pre_trigger_frame`
  - `trigger_frame`

### replay

- replay 使用历史 sidecar frames
- 不做“重截图”

### 帧交换

- DCS 导出图片到固定目录
- Python 读取新帧并生成 `VisionObservation`
- 目录规范：
  `<Saved Games>/<DCS variant>/SimTutor/frames/<session_id>/<channel>/`
- 文件名规范：
  `<capture_wall_ms>_<frame_seq>.png`
- manifest：
  `frames.jsonl`
- DCS 端必须“临时文件 -> 原子 rename”
- Python 端只消费最终文件

### BIOS 与帧对齐

使用 `t_wall / capture_wall_ms` 对齐 `Observation` 与 `VisionObservation`

- live: `<= 250ms`
- replay: `<= 100ms`

必须输出可审计 metadata：

- `sync_status`
- `sync_delta_ms`
- `frame_id`
- `frame_stale`
- `sync_miss_reason`

### 多模态失败必须降级

- multimodal path 失败：必须退回 text-only（telemetry-only）
- vision sidecar 缺失：必须标记 `vision_unavailable` 并安全退化

### 安全边界

系统只允许 overlay：

- `highlight`
- `clear`
- `pulse`

严禁：

- 自动点击
- 自动执行
- `performClickableAction`

overlay targets 必须走 allowlist（`ui_map` / `pack.ui_targets`）  
LLM 输出必须可审计。  
日志不得包含 API key、完整 prompt。

---

## 10. Architecture Rules

保持 Clean / Hexagonal：

- Domain Core（纯）：
  procedure engine、tutor orchestrator、evaluation/scoring、policy/gating、可选 error diagnosis
- Ports（稳定接口）：
  `EnvironmentPort / OverlayPort / UserIOPort / ModelPort / KnowledgePort / EventStorePort`
- `VisionPort` 已接入，但 core 不得依赖 DCS / 截图实现细节
- contracts 通信，禁止跨层导入 adapter 内部对象

Schema 规则：

- v1 schema 不破坏
- 新增字段优先可选字段，或新增 v2 schema

---

## 11. Current System and Experiment State

最近两轮 VLM 相关实验已经完成：

- Qwen 最佳路线：
  `Run-003 + Run-005x2`
- Gemma 对照路线：
  `Run-003 + Run-005x2`

相关英文报告：

- `Doc/Vision/Reports/qwen35_vlm_run005_finetune_report_EN.md`
- `Doc/Vision/Reports/gemma4_vlm_run003_plus_run005x2_finetune_report_EN.md`
- `Doc/Vision/Reports/vlm_backbone_comparison_summary_EN.md`

当前主要训练产物本地路径：

- Qwen:
  `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1`
- Gemma:
  `models/gemma4_vlm_lora/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1`

当前 benchmark 结果表明：

- Qwen `Run-003 + Run-005x2` 是目前最强的实用路线
- Gemma `Run-003 + Run-005x2` 明显优于 Gemma base，但整体仍弱于最佳 Qwen adapter

当前 SimTutor runtime 的实际情况：

- 系统整体**已经能跑**
- 单帧 help 模式**已经能正确生成回复和高亮**
- 多目标高亮**看起来大概率已经能成功**，但还需要更系统化测试
- 线上 / 当前接入的 VLM 仍是**阿里云的原版未微调模型**
- 下一阶段目标是：在 `yz50@cloud-247.rz.tu-clausthal.de` 那台 H100 机器上部署 VLM，并接入**微调后的 Qwen 或 Gemma adapter**

关于模型兼容性，有一个必须牢记的约束：

- **从 `Qwen/Qwen3.5-9B-Base` 微调得到的 LoRA adapter，不能直接给 `Qwen3.5-27B` 使用。**

原因不是策略层面，而是模型结构层面：

- LoRA adapter 绑定的是具体 backbone 的层数、hidden size、module shape
- 9B 和 27B 不是同一个 base model 尺寸
- 因此 9B adapter 应视为只兼容 `Qwen/Qwen3.5-9B-Base`

除非另做专门转换或重新训练，否则不要把 9B LoRA 直接挂到 27B 上。

---

## 12. Issue Selection Rules

新对话开始时，**任务优先级**按下面顺序理解：

1. **用户在当前对话里明确指定的 issue / 任务**
2. **用户给出的 GitHub issue 编号或 URL**
3. **用户让读取的 issue 专用说明文档**
4. 本文档中的一般背景
5. backlog 作为候选参考

这意味着：

- **用户当前明确指定的任务永远优先**
- 不要把本文档当成“默认必须做多目标高亮”的指令
- backlog 只是候选池，不是当前任务本身

如果用户给了 issue 编号，并且本地 `gh` 已经登录，优先直接读取 GitHub issue：

```bash
gh issue view <ISSUE_NUMBER> --repo ZombiesYard/IEFMMQ
```

如果用户还给了本地 issue 文档，则：

- GitHub issue 用来确认标题 / 正文 / 最新状态
- 本地 issue 文档用来补仓库内部背景、设计意图和相关文件线索

---

## 13. Open Issue Backlog

以下 issue 都已经明确，适合拆成独立 GitHub issues：

1. **Align runtime VLM contract with the Run-003 / Run-005 fine-tuned 13-fact adapter**
   - 运行时 prompt / schema / step bindings 仍与训练侧分叉
   - 详见：
     `docs/dev_notes/issue_align_runtime_vlm_contract_with_run005_adapter.md`

2. **Add stable multi-target overlay tests for fake LLM output**
   - 用 fake LLM JSON 测试多个高亮目标同时通过系统链路
   - 核心目标：`fcs_bit_switch` + `right_mdi_pb5`

3. **Fix startup guidance when all displays are dark**
   - 当 DDI 和 AMPCD 都没亮时，系统目前会先高亮 AMPCD
   - 虽然最终操作并不完全错误，但会让用户困惑
   - 期望行为：先引导 DDI 上电，再处理 AMPCD

4. **Investigate why the bleed air knob step is skipped**
   - 手册中存在该步骤
   - 当前系统疑似把它跳过了

5. **Clean remote home usage and move stray caches fully under `/scratch/yz50`**
   - `yz50` 的 home 目录曾被误用，2GB 配额满了
   - 需要清理 home 中遗留的 cache / temp / HF 等文件，并确认后续不再写回 home

6. **Prepare runtime rollout for fine-tuned VLM on the H100 host**
   - 当前 runtime 还在用阿里云原版模型
   - 需要部署并切换到本地 H100 机器上的微调后 Qwen / Gemma

7. **Add explicit model-adapter compatibility guardrails**
   - 明确阻止把 Qwen3.5-9B 的 LoRA adapter 挂到 Qwen3.5-27B 上
   - 最好在配置 / 启动时显式报错，而不是隐式失败

8. **Record training/inference time and token usage more systematically**
   - 包括训练时长、token 消耗、运行时生成耗时
   - 这个优先级较低

---

## 14. Typical High-Value Entry Points

不同 issue 会命中不同文件，但新对话通常可以先从这些入口做勘探：

### 运行时 / 模型接入

- `adapters/openai_compat_model.py`
- `adapters/base_help_model.py`
- `adapters/prompting.py`
- `adapters/response_mapping.py`
- `core/llm_schema.py`

### overlay / action / UI target

- `adapters/action_executor.py`
- `core/overlay.py`
- `packs/fa18c_startup/ui_map.yaml`

### pack / procedure / inference

- `packs/fa18c_startup/pack.yaml`
- `adapters/step_inference.py`
- `core/*` 与 `packs/*`

### 视觉事实 / VLM 契约

- `adapters/vision_fact_prompting.py`
- `adapters/vision_fact_extractor.py`
- `packs/fa18c_startup/vision_facts.yaml`
- `core/vision_facts.py`
- `tools/export_vision_sft_dataset.py`

### 常见测试入口

- `tests/test_openai_compat_model.py`
- `tests/adapters/test_action_executor.py`
- `tests/test_overlay_planner.py`
- `tests/test_vision_fact_prompting.py`
- `tests/test_vision_fact_extractor.py`
- `tests/test_live_dcs.py`

---

## 15. Issue-Specific Hints Belong in Issue Notes

像下面这些内容：

- 某个具体 issue 的现有断言线索
- 某组特定 targets（例如 `fcs_bit_switch` / `right_mdi_pb5`）
- 某个 issue 的验收意图
- 某个 issue 的建议开场 prompt

不应该继续堆在通用 handoff 里。

原则上应放到：

- GitHub issue 本身
- 对应的 `docs/dev_notes/issue_*.md`

这样新对话才不会因为 handoff 文档里残留的旧案例而跑偏。

---

## 16. Expected Workflow for the New Chat

收到当前 issue 后，默认按这个顺序做：

1. 如果用户给了 GitHub issue 编号，先用 `gh issue view` 读取 issue
2. 读取用户指定的本地 issue 文档（如果有）
3. 仓库勘探
   - 列出关键目录与相关文件
   - `rg` 定位 contracts / tests / adapters / packs / schemas / prompts
4. 明确计划修改 / 新增的文件清单
5. 给出简短实现计划
6. 如有必要，先写 failing tests
7. 实现最小必要代码
8. 跑测试并修到全绿
9. 输出 PR 风格总结
   - 改了什么
   - 如何验证
   - 已知限制

如果用户没有给 issue 编号，只给了自然语言任务，则把用户当前消息当成唯一任务来源。

---

## 17. Suggested Generic Starting Pattern

如果用户要开启一个新 issue 对话，推荐的启动模式是：

1. 让新对话先读取本文件
2. 明确说明：**忽略 backlog 中的其他候选项，以当前指定 issue 为准**
3. 如果给了 GitHub issue 编号，要求新对话直接用 `gh issue view` 读取 issue
4. 如有必要，再补充对应的本地 `issue_*.md`

这样可以把：

- 通用背景
- 当前 issue
- 仓库勘探

三件事拆开，避免重复和误导
