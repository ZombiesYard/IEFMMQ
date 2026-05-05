# New Chat Handoff: v0.4 Runtime Status and Working Conventions

这份文档用于开启一个**新对话**时，向新的 Codex 说明仓库背景、工程约束、当前系统状态，以及通用工作方式。  
它是**长期有效的通用 handoff**，不是某一个固定 issue 的任务说明。  
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

### 本地 Python / 测试环境（这点非常重要）

仓库元数据里能看到：

- `pyproject.toml` 使用 Poetry 风格依赖声明
- 仓库根目录存在 `uv.lock`
- `README.md` 里的通用示例偏向：
  `python -m venv .venv` / `python -m pytest -q`

但**当前这台本地工作机上，最稳妥、已经实际验证通过的测试入口不是这些通用示例，而是下面这套命令**：

```bash
source ~/venvs/iefmmq-wsl/bin/activate
PYTHONPATH=/usr/lib/python3/dist-packages python -m pytest -q
```

原因与经验规则：

- 当前工作机会话里，系统默认 Python 不一定装全测试依赖
- 我们真实遇到过：
  `pytest` 收集阶段报
  `ModuleNotFoundError: No module named 'PIL'`
- 但切到
  `~/venvs/iefmmq-wsl`
  后，这些测试可以正常跑
- 同时，这个环境下显式带上：
  `PYTHONPATH=/usr/lib/python3/dist-packages`
  是当前已验证可用的稳妥做法

因此：

- **如果只是看仓库文件，看到 `uv.lock` 不代表当前本地测试一定要用 `uv run`**
- **如果只是看 README，看到 `.venv` 示例也不代表当前机器上 `.venv` 一定存在且可用**
- 对这个仓库在当前工作机上的本地测试，优先使用：

```bash
source ~/venvs/iefmmq-wsl/bin/activate && PYTHONPATH=/usr/lib/python3/dist-packages python -m pytest -q
```

如果需要跑单测子集，沿用同一入口，例如：

```bash
source ~/venvs/iefmmq-wsl/bin/activate && PYTHONPATH=/usr/lib/python3/dist-packages python -m pytest -q tests/test_vision_fact_extractor.py
```

补充理解：

- 另一个模型/对话如果用了 `uv`，通常不是“逻辑上必须用 uv”
- 更可能是因为：
  - 仓库里有 `uv.lock`
  - 远程 vLLM 相关环境历史上确实出现过 uv-managed Python
  - 有些 agent 会把“看见 `uv.lock`”当成优先信号
- 但对**当前本地仓库测试**来说，我们已经有更可靠的经验：  
  **优先用 `~/venvs/iefmmq-wsl`，不要默认切到 `uv run`**

### 远程服务器

- SSH:
  `ssh yz50@cloud-247.rz.tu-clausthal.de`
- 远程 shell 是 `tcsh`，不是 bash
- 只使用 `/scratch/yz50`
- 不要使用 home 目录存放模型 / cache / 训练输出
- 不要随便运行 `rm`；如需删除任何远程文件，必须先明确询问用户
- 远程很多环境变量要用 `setenv`，不是 `export`

### 远程 `tcsh` / SSH 实战注意事项

这是一个容易踩坑的地方，必须记住：

- 远程默认交互环境是 `tcsh`，很多 bash 写法会直接失效
- 优先使用：
  `ssh yz50@cloud-247.rz.tu-clausthal.de "tcsh -f -c '...'"`  
  也就是：
  - 外层本地 shell 用双引号
  - 远程 `tcsh -f -c` 里的命令块优先用单引号
- `tcsh -f` 很重要：它会尽量减少被远程用户初始化脚本干扰
- 如果命令本身需要复杂管道、变量展开或多步操作，优先把整段逻辑放进 `tcsh -f -c '...'`
- 在远程 `tcsh` 里：
  - 用 `setenv VAR value`，不要用 `export VAR=value`
  - 激活环境通常用：
    `source /scratch/yz50/iefmmq_vlm_ft_unsloth/venv/bin/activate.csh`
  - 修改 `PATH` 后最好 `rehash`
  - 但对 `vllm` 这类已经知道绝对路径的命令，不要过度依赖
    `activate.csh + rehash`
    在非交互式 `tcsh -f -c` 里，“已经激活但命令仍然找不到”是我们真实遇到过的坑

我们之前实际采用并验证过的稳妥模式有两种：

1. 简单单条命令：

```bash
ssh yz50@cloud-247.rz.tu-clausthal.de "tcsh -f -c 'nvidia-smi'"
```

2. 需要进入目录、激活环境、再运行命令时：

```bash
ssh yz50@cloud-247.rz.tu-clausthal.de "tcsh -f -c 'cd /scratch/yz50/iefmmq_vlm_ft_unsloth; source venv/bin/activate.csh; setenv TMPDIR /scratch/yz50/tmp; python --version'"
```

如果一定要在本地先用 bash 拼一层再 ssh，也可以用：

```bash
/bin/bash -lc "ssh -o BatchMode=yes yz50@cloud-247.rz.tu-clausthal.de \"tcsh -f -c 'nvidia-smi'\""
```

这个写法的经验规则是：

- 最外层是本地 bash 的双引号
- 内层 ssh 远程整段再包一层转义双引号
- 最里面真正给 `tcsh -c` 的命令块用单引号

常见错误来源：

- 把 `export`、`source venv/bin/activate`、`&&` 之类 bash 习惯直接搬进远程 `tcsh`
- 本地和远程两层引号混用，导致远程命令在本地就先被展开
- 在远程 home 目录写缓存，触发 2GB quota 问题
- 在非交互式 `tcsh` 里以为 `source activate.csh` 之后命令一定能被正确发现

这次实际又踩到过一个典型坑：

- 在 `tcsh` 里如果命令包含某些 `!`、方括号模式或会触发 history expansion 的片段，可能直接报：
  `Event not found.`

我们这次就遇到过类似情况：本来只是想在远程列出 home 目录占用，但因为命令片段和 `tcsh` 的历史展开规则撞上了，结果命令在真正执行前就失败了。

遇到这种情况时，优先采用下面的解决办法：

1. **不要硬跟 `tcsh` 斗引号和历史展开**
2. 对于纯文件系统检查、`du`、`ls`、`find`、`sort` 这类不依赖 `tcsh` 语法的命令，直接改成走远程 `/bin/sh -lc`

例如这条是稳定可用的：

```bash
ssh yz50@cloud-247.rz.tu-clausthal.de "/bin/sh -lc 'cd ~ && du -sh .[^.]* * 2>/dev/null | sort -h | tail -n 40'"
```

也就是说，远程命令实际上有两种推荐模式：

### 模式 A：需要远程环境、`setenv`、`activate.csh`、vLLM / Python 服务启动

用 `tcsh -f -c`：

```bash
ssh yz50@cloud-247.rz.tu-clausthal.de "tcsh -f -c 'cd /scratch/yz50/iefmmq_vlm_ft_unsloth; source venv/bin/activate.csh; setenv TMPDIR /scratch/yz50/tmp; python --version'"
```

### 模式 B：只做纯 shell 检查、磁盘排查、日志查看、目录统计

优先用 `/bin/sh -lc`，通常更省心：

```bash
ssh yz50@cloud-247.rz.tu-clausthal.de "/bin/sh -lc 'cd ~ && du -sh .cache .local 2>/dev/null'"
```

经验规则：

- **涉及 `setenv` / `source activate.csh` / `rehash` / venv / tcsh 环境变量时，用 `tcsh -f -c`**
- **涉及 `du` / `find` / `ls` / `sort` / `tail` / 纯文件系统检查时，优先用 `/bin/sh -lc`**
- **涉及远程 vLLM 启动时，如果你已经知道可执行文件路径，优先直接调用绝对路径，而不是假设 `activate.csh` 后 `vllm` 一定能被 shell 找到**

这样能明显减少：

- `Event not found`
- 引号嵌套错位
- 本地 shell 先展开远程命令
- `tcsh` 特有语法把普通检查命令搞坏
- `venv` 明明激活了，但 `vllm: Command not found`

### 当前已知远程缓存习惯

如果需要在远程运行模型、benchmark 或训练，优先确保这些目录与环境变量：

```tcsh
mkdir -p /scratch/yz50/tmp
mkdir -p /scratch/yz50/.cache/uv
mkdir -p /scratch/yz50/.cache/huggingface
mkdir -p /scratch/yz50/.cache/pip
mkdir -p /scratch/yz50/.cache/vllm
mkdir -p /scratch/yz50/.cache/triton
mkdir -p /scratch/yz50/.cache/torchinductor
mkdir -p /scratch/yz50/.config/vllm

setenv TMPDIR /scratch/yz50/tmp
setenv UV_CACHE_DIR /scratch/yz50/.cache/uv
setenv PIP_CACHE_DIR /scratch/yz50/.cache/pip
setenv HF_HOME /scratch/yz50/.cache/huggingface
setenv HUGGINGFACE_HUB_CACHE $HF_HOME/hub
setenv TRANSFORMERS_CACHE $HF_HOME/transformers
setenv VLLM_CACHE_ROOT /scratch/yz50/.cache/vllm
setenv VLLM_CONFIG_ROOT /scratch/yz50/.config/vllm
setenv VLLM_NO_USAGE_STATS 1
setenv TRITON_CACHE_DIR /scratch/yz50/.cache/triton
setenv TORCHINDUCTOR_CACHE_DIR /scratch/yz50/.cache/torchinductor
setenv PATH /scratch/yz50/.local/bin:$PATH
rehash
```

做 vLLM / Qwen-VL 部署时，必须额外记住下面几点：

- 只设置 `HF_HOME` / `TRANSFORMERS_CACHE` **不够**
- vLLM 还会写：
  - `VLLM_CACHE_ROOT`
  - `VLLM_CONFIG_ROOT`
  - Triton cache（`TRITON_CACHE_DIR`）
  - TorchInductor cache（`TORCHINDUCTOR_CACHE_DIR`）
- 如果这些变量没显式指到 `/scratch/yz50`，很容易再次偷偷写回 remote home，然后触发 `yz50` 的 2GB home quota 问题
- vLLM usage stats 也会默认写配置目录，因此建议固定：
  `setenv VLLM_NO_USAGE_STATS 1`
- 如果某些底层库仍然顽固地走 `~/.cache` 语义，一个稳妥兜底办法是让启动该服务的进程额外带上：
  `setenv HOME /scratch/yz50`
- 对 vLLM 命令本身，也建议优先固定成绝对路径，例如：
  `/scratch/yz50/vllm_qwen35/venv/bin/vllm`
  不要把“能否找到 `vllm` 命令”完全交给 `activate.csh` 和 `rehash`
- 如果远程 home 被清理过，还要额外检查：
  `/scratch/yz50/vllm_qwen35/venv/bin/python*`
  是否仍然指向一个真实存在的解释器
- 我们这次真实遇到过：
  `/scratch/yz50/vllm_qwen35/venv/bin/python -> /home/yz50/.local/share/uv/python/...`
  但 home 下那份 uv Python 已经因为 quota 清理而消失
- 这时会出现一个很迷惑的状态：
  - `venv/bin/vllm` 文件还在
  - `venv/lib/python3.12/site-packages/vllm` 也还在
  - 但 `venv/bin/python3` 已经是断掉的 symlink
  - 所以无论是 `python ...`、`python -m vllm...`，还是直接执行 `venv/bin/vllm`，都可能起不来
- 排查时必须显式看：
  - `ls -l /scratch/yz50/vllm_qwen35/venv/bin/python*`
  - `cat /scratch/yz50/vllm_qwen35/venv/pyvenv.cfg`
- 这类情况下，绝对路径 `vllm` 本身也不一定够，因为它的 shebang 通常仍会指向
  `venv/bin/python3`
  ，而这个解释器入口已经坏了
- 一个实用修复办法是：
  - 把 `venv/bin/python` 重新指到 `/usr/bin/python3.12`
  - 再把 `venv/bin/python3`、`venv/bin/python3.12` 重新链回 `python`
  - 同时把 `pyvenv.cfg` 里的 `home = ...` 改成 `/usr/bin`
  - 然后再验证：
    - `/scratch/yz50/vllm_qwen35/venv/bin/python3 -c 'import vllm'`
    - `/scratch/yz50/vllm_qwen35/venv/bin/vllm --version`
- 因此，远端启动脚本不要默认假设“vLLM venv 的 python 一定没坏”
- 如果要用 Python helper 启动 vLLM，优先让 helper 本身由系统
  `/usr/bin/python3`
  运行，而不是强依赖可能已经损坏的 venv python
- 另外一个这次实际踩到的命令拼接坑：
  - 不要把**本地**的 `$PATH` 直接插进远程 `ssh "... /bin/bash -lc '...'"` 命令
  - 本地 PATH 里如果带空格、括号或 Windows 路径，可能直接把远程 bash 命令炸掉
  - 远端启动 vLLM 时，优先手工给一个最小且可控的 PATH，例如：
    `/scratch/yz50/.local/bin:/scratch/yz50/vllm_qwen35/venv/bin:/usr/local/bin:/usr/bin:/bin`

这次在 H100 上还实际碰到过一个容易误导新对话的现象：

- `python3 -m vllm.entrypoints.openai.api_server` 可能报
  `ModuleNotFoundError: No module named 'vllm'`
- 但与此同时，真正可用的 vLLM 可执行文件其实就在：
  `/scratch/yz50/vllm_qwen35/venv/bin/vllm`

也就是说，排查顺序应该是：

1. 先确认你到底用的是哪个 Python / 哪个 venv
2. 再确认 `vllm` 的绝对路径是否存在
3. 如果绝对路径存在，优先直接调用它
4. 不要因为一次 `python -m vllm...` 失败，就误判“服务器上没装 vllm”

这次在 H100 上实际踩到过的报错包括：

- `OSError: [Errno 28] No space left on device: '/home/yz50/.cache/vllm'`
- `OSError: [Errno 28] No space left on device: '/home/yz50/.triton/cache/...'`

看到这类报错时，优先检查是不是还有 cache root 漏了，而不是先怀疑模型本身。

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
- 多目标高亮**已有一定基础**，但是否需要补测试、补契约或补执行路径，应以当前 issue 为准
- 线上 / 当前接入的 VLM 仍是**阿里云的原版未微调模型**
- 下一阶段目标是：在 `yz50@cloud-247.rz.tu-clausthal.de` 那台 H100 机器上部署 VLM，并接入**微调后的 Qwen 或 Gemma adapter**

当前实验与训练侧的**事实 ontology 基线已经稳定为 13 facts**。  
如果某个具体 issue 仍在讨论旧 runtime contract、旧 fact IDs 或迁移问题，应以对应的 issue 文档或 GitHub issue 为准，而不是把“仍未切到 13 facts”当作全局事实。

关于模型兼容性，有一个必须牢记的约束：

- **从 `Qwen/Qwen3.5-9B-Base` 微调得到的 LoRA adapter，不能直接给 `Qwen3.5-27B` 使用。**

原因不是策略层面，而是模型结构层面：

- LoRA adapter 绑定的是具体 backbone 的层数、hidden size、module shape
- 9B 和 27B 不是同一个 base model 尺寸
- 因此 9B adapter 应视为只兼容 `Qwen/Qwen3.5-9B-Base`

除非另做专门转换或重新训练，否则不要把 9B LoRA 直接挂到 27B 上。

---

## 12. Task Selection Rule

新对话开始时，**永远以用户当前明确指定的任务为准**。  
如果用户给了：

- GitHub issue 编号
- PR 编号
- 本地 `docs/dev_notes/issue_*.md`
- 明确的自然语言任务描述

就应优先处理它，而不是自行从 backlog 中挑选任务。

如果用户没有明确给出任务，才可以把下面的 backlog 当作候选池，并先向用户确认或在回复里说明你准备处理哪一项。

---

## 13. Open Issue Backlog

以下条目是**候选问题池 / 历史待办方向**。  
它们不自动构成“当前任务”，也不应覆盖用户在当前对话里显式指定的 issue。

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
   - 已验证过一个真实坑：
     - 对 Qwen3.5 VL + LoRA，如果只传 `--enable-lora` 而不传
       `--enable-tower-connector-lora`
       ，vLLM 可能只加载语言侧 LoRA，而把视觉塔 / connector 的 LoRA 忽略掉
   - 判断信号：
     - 如果日志里出现大量
       `visual.* will be ignored`
       ，说明 adapter 没有被完整应用到多模态部分
     - 正确做法是启动时显式加上：
       `--enable-tower-connector-lora`
   - 这次在 H100 上成功拉起的关键参数组合是：
     - base model:
       `Qwen/Qwen3.5-9B-Base`
     - LoRA adapter:
       `/scratch/yz50/iefmmq_vlm_models/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/adapter`
     - served model name:
       `simtutor-qwen35-9b-lora`
     - LoRA alias:
       `simtutor`
   - 另一个已知部署坑：
     - 远端若缺少 `ninja`，FlashInfer / GDN warmup 可能报
       `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`
     - 在我们这次部署里，这没有阻止服务最终启动，但会让 warmup 退化并产生 warning

7. **Add explicit model-adapter compatibility guardrails**
   - 明确阻止把 Qwen3.5-9B 的 LoRA adapter 挂到 Qwen3.5-27B 上
   - 最好在配置 / 启动时显式报错，而不是隐式失败

8. **Record training/inference time and token usage more systematically**
   - 包括训练时长、token 消耗、运行时生成耗时
   - 这个优先级较低

---

## 14. How to Start Any New Issue

收到当前 issue 后，建议按这个顺序做：

1. 读取任务来源
   - 如果用户给了 GitHub issue 编号，优先使用：
     `gh issue view <ISSUE_NUMBER> --repo ZombiesYard/IEFMMQ`
   - 如果用户同时给了本地 `docs/dev_notes/issue_*.md`，把它作为仓库内部补充背景
   - 如果两者冲突，以 GitHub issue 为主，并明确指出冲突点

2. 仓库勘探
   - 列出关键目录与相关文件
   - 使用 `rg` / `grep` 定位 contracts / tests / adapters / packs / schemas / 调用链

3. 明确将修改 / 新增的文件清单

4. 给出简短实现计划

5. 如有必要，先写 failing tests

6. 实现最小必要代码

7. 跑测试并修到全绿

8. 输出 PR 风格总结
   - 改了什么
   - 如何运行
   - 如何测试
   - 已知限制

---

## 15. Suggested Generic Prompt for a New Chat

可以把下面这段作为新对话的通用开头模板：

> 请使用 karpathy-guidelines skill。  
> 先阅读 `docs/dev_notes/new_chat_handoff_v0_4_multitarget_overlay.md`。  
> 当前唯一任务由我在本条消息中明确指定；请忽略 handoff 文档里的 backlog 候选项，不要自行挑选别的任务。  
> 如果我给了 GitHub issue 编号，请先用 `gh issue view <ISSUE_NUMBER> --repo ZombiesYard/IEFMMQ` 读取 issue；如果我同时给了本地 `docs/dev_notes/issue_*.md`，再把它作为补充背景。  
> 先做仓库勘探和 grep 定位，不要直接改代码。  
> 请先给我：GitHub issue 摘要、关键调用链、计划修改文件清单、实施计划，然后再开始改代码。  
> 所有回复请使用中文，并遵守本文档中的架构、安全边界、TDD 和 scope 约束。
