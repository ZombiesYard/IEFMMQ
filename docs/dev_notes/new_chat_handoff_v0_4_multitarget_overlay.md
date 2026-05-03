# New Chat Handoff: v0.4 Runtime Status and Pending Issues

这份文档用于开启一个**新对话**时，向新的 Codex 说明仓库背景、工程约束、当前系统状态，以及**当前推荐优先 issue**。  
可直接把本文件内容贴给新对话，或让新对话先读取本文件。

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

## 12. Recommended Next Issue

如果新对话没有收到用户重新指定的 issue，**推荐优先处理下面这个 issue**：

**测试多个按钮高亮，并通过 fake LLM 的 JSON 返回多个 overlay targets，至少包含 `fcs_bit_switch` 和 `right_mdi_pb5`。**

这个 issue 的核心是：

1. 验证当前系统是否能接受**多目标 overlay/highlight**
2. 使用 fake LLM / fake model 返回结构化 JSON
3. JSON 中包含多个高亮目标
4. 目标至少包括：
   - `fcs_bit_switch`
   - `right_mdi_pb5`
5. 测试当前链路是否能正确：
   - 解析模型输出
   - 映射为多个 overlay actions
   - 通过 allowlist 校验
   - 执行到 overlay sender / action executor 层
6. 只能测试高亮，不允许引入任何自动点击或自动执行

但这不是唯一待办。当前还有一组已经识别出来的高优先级 / 中优先级问题，见下面的“Open Issue Backlog”。

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

## 14. Very Likely Relevant Files

新对话开始时，建议优先查看这些文件：

### 直接相关实现

- `adapters/openai_compat_model.py`
- `adapters/base_help_model.py`
- `adapters/action_executor.py`
- `adapters/response_mapping.py`
- `core/overlay.py`
- `packs/fa18c_startup/ui_map.yaml`

### 直接相关测试

- `tests/test_openai_compat_model.py`
- `tests/adapters/test_action_executor.py`
- `tests/test_overlay_planner.py`

### 额外相关线索

- `adapters/prompting.py`
- `core/llm_schema.py`
- `packs/fa18c_startup/pack.yaml`

---

## 15. Important Existing Clues Already Found

仓库里已经能找到和“多目标高亮”非常相关的现有线索，不要从零开始乱做：

### 在 `tests/test_openai_compat_model.py`

已存在与 `fcs_bit_switch` 和 `right_mdi_pb5` 相关的多目标断言线索，例如：

- `targets: ["fcs_bit_switch", "right_mdi_pb5"]`
- `assert [action["target"] for action in res.actions] == ["fcs_bit_switch", "right_mdi_pb5"]`

### 在 `tests/adapters/test_action_executor.py`

已存在多目标执行相关断言，例如：

- `assert [item["target"] for item in report.executed] == ["fcs_bit_switch", "right_mdi_pb5"]`

### 在 `adapters/prompting.py`

已经出现多目标语义约束：

- 当系统允许多目标时，`overlay.targets` 应同时返回 `fcs_bit_switch` 与 `right_mdi_pb5`

这说明：

- 当前仓库**并不是完全没有多目标概念**
- 很可能已经有部分实现或半实现
- 当前 issue 更像是：**把“多目标高亮”测试补齐、打通、或修正到稳定可用**

---

## 16. Expected Workflow for the New Chat

收到当前 issue 后，必须按这个顺序做：

1. 仓库勘探
   - 列出关键目录与相关文件
   - grep 定位 contracts / tests / overlay / model adapter / ui_map

2. 明确将修改 / 新增的文件清单

3. 给出简短实现计划
   - 当前系统是否已经部分支持多目标
   - 是补测试、补解析、还是补执行路径

4. 如有必要，先写 failing tests

5. 实现最小必要代码

6. 跑测试并修到全绿

7. 输出 PR 风格总结
   - 改了什么
   - 如何运行
- 如何测试
- 已知限制

---

## 17. Acceptance Intent for the Multi-Target Overlay Issue

如果当前处理的是“多目标高亮测试”这个 issue，至少应满足以下结果：

- fake LLM JSON 可以表达多个 overlay target
- `fcs_bit_switch` 与 `right_mdi_pb5` 能一起通过当前系统链路
- 测试清楚验证多目标高亮行为
- 不破坏单目标高亮现有逻辑
- 不引入自动点击
- 不扩大为其他 feature

---

## 18. Suggested Starting Prompt for the New Chat

可以把下面这段作为新对话的开头：

> 请先阅读 `docs/dev_notes/new_chat_handoff_v0_4_multitarget_overlay.md`。  
> 先根据文档里的当前系统状态与 open issue backlog 做仓库勘探。  
> 如果我没有另外指定 issue，请优先实现多目标高亮测试这一项：通过 fake LLM 的 JSON 返回多个 overlay targets，至少包含 `fcs_bit_switch` 和 `right_mdi_pb5`，验证当前系统从模型输出到 overlay/action executor 的多目标高亮链路。  
> 所有回复请用中文，遵守文档里的架构、安全边界、TDD 和 scope 约束；同时注意运行时 VLM 契约目前与微调 adapter 的 13-fact 契约存在分叉，不要忽略这一背景。
