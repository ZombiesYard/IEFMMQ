# SimTutor v0.4 双阶段运行时部署实验报告

## 摘要

本报告记录 SimTutor 双阶段教学推理管线在本地 vLLM 服务器（NVIDIA A100 80GB）上的 v0.4 部署实验。系统使用一个 `Qwen/Qwen3.5-9B-Base` 模型，以两个模型名分别提供服务：`simtutor-vision`（带 LoRA）用于从组合面板截图中提取视觉事实；`simtutor-qwen35-9b-base`（纯 base）用于纯文本的步骤诊断与高亮目标推理。与此前的单模型名方案相比，这一分离消除了 LoRA adapter 对文本生成路径的干扰。

本次部署同时引入五项基础设施改进：传输层重试、JPEG 图像压缩、连接池调优、放宽 JSON Schema 校验，以及服务端全局关闭 thinking。在一轮 15 次 help cycle 的实机测试中，系统取得了 14/15 的 model-mode 响应率，`RemoteProtocolError` 与 `ValidationError` 均降至零。当前最主要的剩余问题是部分可观测条件下的 LLM 幻觉——当 VLM 返回全部 `not_seen`（屏幕未亮）时，9B base 模型有时无视视觉证据，生成错误的高亮引导。

## 1. 系统架构

### 1.1 双阶段流水线

每次 help cycle 向同一 vLLM 服务器发起两次独立调用：

| 阶段 | 模型名 | LoRA | 输入 | 输出 |
|---|---|---|---|---|
| 视觉事实抽取 | `simtutor-vision` | 是 | 组合面板图（~200 KB JPEG）+ 事实抽取 prompt | 13 个视觉事实（`seen`/`not_seen`/`uncertain`） |
| 教学响应生成 | `simtutor-qwen35-9b-base` | 否 | VLM 事实文本 + 遥测数据（VARS, GATES）+ RAG + 确定性步骤提示 + 规程规则 | `{"diagnosis","next","overlay","explanations"}` |

LoRA adapter 基于 Run-003 + Run-005x2 双语数据微调，训练任务为 `image → facts JSON`。该 adapter 不应用于教学响应阶段，因为 LoRA 权重会使模型偏向 facts 输出格式（`summary`、`facts` 等 key），与教学响应所需的完全不同输出 schema（`diagnosis`、`overlay`、`explanations`）产生冲突。

### 1.2 部署拓扑

| 组件 | 位置 | 端口 |
|---|---|---|
| vLLM 服务器 | `cloud-247.rz.tu-clausthal.de` | 6324 |
| SSH 隧道 | Windows PowerShell `localhost:16324 → cloud-247:6324` | 16324 |
| DCS + SimTutor | Windows 10, Python 3.12 | — |
| 截图 sidecar | 同机 Windows | 触发端口 7795 |

```
┌─────────────────────────────────────────────────────────────────┐
│ Windows 10 (DCS + SimTutor)                                     │
│                                                                 │
│  ┌──────────┐   UDP:7795   ┌────────────────┐                  │
│  │ DCS hook │──────────────→│ Vision sidecar │                  │
│  │ 截图触发  │              │ 截图 + 组合     │                  │
│  └──────────┘              └───────┬────────┘                  │
│                                    │ PNG (~5 MB)               │
│                                    ▼                           │
│                         ┌────────────────────┐                 │
│                         │ SimTutor live_dcs  │                 │
│                         │                    │                 │
│                         │  Stage 1: VLM 调用 │                 │
│                         │  model=simtutor-   │                 │
│                         │  vision (LoRA)     │                 │
│                         │  image→facts JSON  │                 │
│                         │         │          │                 │
│                         │         ▼          │                 │
│                         │  Stage 2: LLM 调用 │                 │
│                         │  model=simtutor-   │                 │
│                         │  qwen35-9b-base    │                 │
│                         │  facts+telemetry   │                 │
│                         │  →help response    │                 │
│                         └────────┬───────────┘                 │
│                                  │ http://localhost:16324      │
│                                  │ SSH 隧道                    │
└──────────────────────────────────┼──────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────┐
│ cloud-247.rz.tu-clausthal.de     │                              │
│                                  ▼                              │
│  ┌──────────────────────────────────────────┐                  │
│  │ vLLM 0.19.0 :6324                        │                  │
│  │                                          │                  │
│  │  served models:                          │                  │
│  │  - simtutor-qwen35-9b-base (base)        │                  │
│  │  - simtutor-vision (base + LoRA)         │                  │
│  │                                          │                  │
│  │  GPU: NVIDIA A100 80GB                   │                  │
│  │  enable_thinking: false (全局)            │                  │
│  └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 基础设施改进

与此前的单模型名方案相比，本次部署进行了五项改进：

### 2.1 传输层重试

`httpx.Client` 的连接池会复用 TCP 连接。在 SSH 隧道环境下，keep-alive 连接可能变 stale，导致后续请求出现 `RemoteProtocolError`。此前在多模态 session 中该错误占比约 79%。

**修复**: `OpenAICompatModel._post_with_transport_retry` 捕获 `httpx.RequestError` / `ConnectionError` / `TimeoutError`，重置 HTTP 客户端（关闭旧连接、创建新 socket），重试一次。本轮测试中 `RemoteProtocolError` 完全消除。

### 2.2 JPEG 图像压缩

DCS 组合面板截图为 PNG 格式，原始文件约 5 MB，base64 编码后约 6.7 MB。在存在丢包的 SSH 隧道上，大 payload 的失败概率显著更高。

**修复**: `openai_compat_multimodal.py` 中的 `_encode_image_for_vlm` 在 base64 编码前将 PNG 转换为 JPEG（85% 质量）。典型的 880×1440 座舱截图从 ~5 MB 压缩至 ~200 KB（约 25× 缩小）。LoRA adapter 训练时使用 PNG 图像，但经抽查验证，对 JPEG 85% 质量无明显精度损失。

### 2.3 连接池调优

默认的 `httpx.Client(timeout=60.0)` 使用单一超时值和无限制的连接复用。

**修复**: `_make_http_client` 使用以下配置：

| 设置 | 值 | 说明 |
|---|---|---|
| `connect` 超时 | 5 s | 无法连接时快速失败 |
| `read` 超时 | 60 s | 为 LLM 推理留足时间 |
| `write` 超时 | 30 s | 覆盖 ~200 KB 上传 |
| `pool` 超时 | 5 s | 快速获取连接 |
| `max_keepalive_connections` | 2 | 防止 stale 连接累积 |
| `max_connections` | 4 | 单用户系统适中上限 |
| `keepalive_expiry` | 30 s | 定期刷新连接 |

### 2.4 放宽 JSON Schema

LLM 教学响应 schema 此前要求 `evidence.quote.minLength=1`、`explanations.minItems=1` 和 `explanations[].minLength=1`。9B base 模型经常遗漏 `quote` 文本或生成空 `explanations`，导致 schema 校验失败，从而拒绝原本正确的 overlay target。

**修复**: `quote.minLength` 改为 `0`，`explanations.minItems` 改为 `0`，`explanations[].minLength` 改为 `0`。下游代码本已具备默认 explanation 填充逻辑，空 quote 也不影响 evidence 有效性。

### 2.5 服务端全局关闭 thinking

Qwen3.5-9B-Base 默认开启 `enable_thinking`，生成 `<think>...</think>` 标签，消耗 token 并破坏 JSON 解析。经测试确认，vLLM 0.19.0 忽略请求中的 `chat_template_kwargs: {"enable_thinking": false}`。

**修复**: 在模型 snapshot 目录中放置 `generation_config.json`（内容为 `{"enable_thinking": false}`）。这会全局禁用所有请求的 thinking，无论请求参数如何。

### 2.6 Few-shot 示例

在 prompt 中注入两个完整的请求-响应对，展示正确的 diagnosis、overlay evidence 和 explanations 格式。注入仅在生产模式（`_owns_client=True`）下生效，测试不受影响。此前尝试的 assistant prefill（`{"role": "assistant", "content": "{"}`）经测试确认会导致模型复读对话历史而非续写 JSON，已移除。

## 3. 测试结果（2026-05-15）

### 3.1 Session 概览

| 指标 | 值 |
|---|---|
| 总 help cycles | 15 |
| 总 observation 事件 | 6137 |
| 视觉帧 observation | 366 |
| Overlay 事件 | ~30 |
| Session 时长 | ~5 分钟 |

### 3.2 Generation mode 分布

| Mode | 次数 | 占比 | 说明 |
|---|---|---|---|
| `model` | 14 | 93% | LLM 生成有效教学响应 |
| `fallback` | 1 | 7% | S06: LLM 抛出 `ValueError`（空输出） |
| `repair` | 0 | 0% | — |
| `RemoteProtocolError` | **0** | 0% | 传输重试消除了所有网络失败 |
| `ValidationError` | **0** | 0% | Schema 放宽消除了所有校验拒绝 |

与此前修复前基准（35 次 cycle: 40% model, 49% fallback, 11% repair）相比，model-mode 率从 40% 提升至 93%。

### 3.3 逐步骤明细

| # | 步骤 | Mode | 延迟 (ms) | 消息（节选） |
|---|---:|---:|---:|---|
| 1 | S01 | model | 1143 | 操作 battery_switch 打开电瓶开关 |
| 2 | S02 | model | 1232 | 将 fire_test_switch 拨到 TEST A |
| 3 | S03 | model | 1316 | 打开 APU 开关 |
| 4 | S03 | model | 1293 | APU 未就绪，左键打开 APU 开关 |
| 5 | S04 | model | 1194 | 将 ENG CRANK 拨到 RIGHT |
| 6 | S05 | model | 1403 | 操作 left_mdi_pb18，满足 vars.rpm_r>=25 |
| 7 | S05 | model | 1333 | 右油门须从 OFF 推到 IDLE |
| 8 | S06 | fallback | 1393 | S06 未完成: vars.rpm_r_gte_60==true |
| 9 | S07 | model | 1211 | 按下 lights_test_button |
| 10–15 | S08 | model (×6) | 1376–1499 | 左 DDI 当前显示 TAC 页面，按左 MDI PB18 |

### 3.4 教学文本投送

全部 15 次 cycle 均成功将教学文本投送到 DCS（UDP → Lua hook → `trigger.action.outText`）。零 `sender_unavailable` 发生。

## 4. 剩余问题

### 4.1 部分可观测条件下的 LLM 幻觉

本轮测试中最突出的失败模式出现在 #10–15（步骤 S08）。关键观察：

- **#10**: 全部 13 个 VLM facts 均为 `not_seen`。屏幕未亮。`fused_missing` 正确报告 `vars.mpcd_on==true`。LLM 却输出"左 DDI 当前显示 TAC 页面，按 PB18"。
- **#11**: 同样——全部 `not_seen`，LLM 幻觉 TAC 页面引导。
- **#12–15**: VLM 返回部分 `seen` facts（不同阶段分别看到 `tac_page_visible`、`supt_page_visible`、`fcs_page_visible`）。LLM 持续输出"TAC 页面可见，按 PB18"，无视 `fused_missing` 条件（先为 `vars.mpcd_on==true`，后为 `vars.hud_on==true`）。

根因是 9B base 模型的指令遵从天花板。prompt 中明确要求信任 VLM 标注，但模型倾向于依赖自身参数先验。融断步骤推断正确识别了 S08 及缺失条件，但这些信息在长上下文中不够突出，LLM 未将其视为硬约束。

**建议修复**: 在 `LiveDcsTutorLoop.run_help_cycle` 中增加确定性守卫。当 VLM 返回零 `seen` facts 且步骤需要视觉确认时，跳过 LLM 直接用模板消息（如"屏幕可能未通电，请先完成前置步骤"）。将安全保证放在代码中而非依赖 LLM 读懂 prompt。

### 4.2 VLM 事实抽取走 LoRA 且为纯图像输入

视觉事实抽取 prompt 为纯图像文本（"你是 SimTutor 视觉事实抽取器..."）加组合面板图像。LoRA 恰恰是用此格式训练的（Run-003 + Run-005x2 双语数据），因此 VLM 生成高质量 facts。本轮全部 15 次 cycle 均为 `multimodal_path_success=True`。

### 4.3 延迟

每次 cycle 延迟主要由两次模型调用支配：

| 组件 | 典型延迟 |
|---|---|
| VLM 事实抽取 | ~500–1000 ms |
| 教学响应生成 | ~1000–1500 ms |
| 每次 help cycle 合计 | ~1500–2500 ms |
| 含连接预热 | 首次调用快 ~100 ms |

对通过热键手动触发的教学系统而言，该延迟可接受。若用于连续自动 help，建议冷却时间 ≥3 s。

## 5. 结果解读

93% 的 model-mode 率应在以下语境中理解：

1. **测试仅 15 次 cycle**。样本量较小。需要更多显示状态切换的长 session 才能获得可靠估计。
2. **`missing=[]` 异常**: 部分请求的确定性步骤提示显示 `missing_conditions` 为空，尽管门控条件实际未满足。这可能由 S08/S26 步骤重构（PR #237）引入的回归导致，LLM 通过直接读取 EVIDENCE_SOURCES 中的 VARS 数据来弥补，多数情况下可行，但在证据稀疏时失败。
3. **关闭 thinking 消除了 repair 链路**: 在修复前的运行中，thinking 消耗 token 导致 JSON 截断 → `ValueError` → schema repair → `repair` 或 `fallback`。全局关闭 thinking 后，模型直接输出紧凑 JSON，消除了整条 repair 链路。
4. **双模型名分离是最具影响力的单次改动**。分离前，LoRA adapter 干扰教学响应生成；分离后，base 模型更忠实地遵循 help response schema，LoRA adapter 独立处理视觉事实抽取，各司其职。

## 6. 局限性

1. **单次 session，15 次 cycle**。尚未测量更长 session 及更多显示状态转换下的泛化表现。
2. **网络测试在非高峰时段**。此前 session 中观察到的校园网出口路由器问题（高峰期丢包导致 85% `RemoteProtocolError`）未复现。
3. **JPEG 压缩未经 PNG 基准对比**。LoRA adapter 使用 PNG 训练，虽经人工抽查未发现退化，但正式对比尚待完成。
4. **Few-shot 示例未做消融**。本报告无法分离 few-shot 示例对 93% model 率的独立贡献。
5. **`missing_conditions=[]` 回归**。部分请求的门控条件为空，阻碍 LLM 生成精确的"请满足 X"引导，迫使其从原始 VARS 数据推断条件。

## 7. 结论

v0.4 双阶段运行时部署在 15 次 help cycle 的实机测试中取得 93% 的 model-mode 率，较修复前 40% 的基准大幅提升。三项改动贡献了主要收益：(1) LoRA/模型名分离，消除 adapter 对文本生成路径的干扰；(2) 服务端全局关闭 thinking，消除 JSON 截断 → repair 链路；(3) 传输层重试，消除 `RemoteProtocolError`。

剩余失败模式为部分可观测条件下的 LLM 幻觉——这是小规模 base 模型的已知局限，未来应通过确定性守卫解决。基础设施栈（JPEG 压缩、连接池调优、Schema 放宽）已进入稳定可用状态。
