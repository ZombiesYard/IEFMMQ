# SimTutor Core (IEFMMQ)

面向 DCS F/A-18C Cold Start 训练场景的教学后端。项目采用“核心逻辑 + 端口 + 适配器”结构，支持：
- Mock 场景仿真与评分
- DCS-BIOS 实时/回放助教循环
- 仅 Overlay（高亮）安全执行边界
- 本地 BM25 检索与知识白名单策略（cold-start production）

## 当前能力概览

- `simtutor` CLI：`run / replay / score / batch / validate / replay-bios / model-config`
- `live_dcs.py`：端到端实时循环（`bios -> enrich -> help -> response mapping -> overlay`）
- 模型提供方：`stub`、`openai_compat`、`ollama`
- 事件日志：统一 JSONL，可回放、可打分、可做 schema 校验
- DCS 工具链：索引构建、DCS hook 安装、Telemetry/BIOS 监听与录制

## 目录结构

- `core/`：领域引擎（procedure/gating/scoring/overlay/knowledge/types）
- `ports/`：模型、知识、遥测等接口约束
- `adapters/`：模型适配、DCS 适配、事件写入、响应映射
- `simtutor/`：CLI 入口与 schemas
- `packs/fa18c_startup/`：训练包（步骤、taxonomy、ui_map、telemetry_map）
- `mock_scenarios/`：离线场景输入
- `tools/`：索引、安装 hook、监听/录制等工具
- `Doc/Evaluation/`：权威训练文档与评估材料

## 环境要求

- Python `3.10+`
- 建议在仓库根目录执行命令
- 本仓库示例统一使用 `python3`（很多环境没有 `python` 别名）

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

## 快速开始（离线 Mock）

1. 构建文档索引（默认索引 `Doc/Evaluation`）：

```bash
python3 -m tools.index_docs --output Doc/Evaluation/index.json
```

2. 运行单个 mock 场景：

```bash
python3 -m simtutor run \
  --pack packs/fa18c_startup/pack.yaml \
  --scenario mock_scenarios/correct_process.json \
  --output logs/run_demo.jsonl
```

3. 回放检查轨迹：

```bash
python3 -m simtutor replay logs/run_demo.jsonl --pack packs/fa18c_startup/pack.yaml
```

4. 评分：

```bash
python3 -m simtutor score \
  logs/run_demo.jsonl \
  --pack packs/fa18c_startup/pack.yaml \
  --taxonomy packs/fa18c_startup/taxonomy.yaml
```

5. 批量跑场景并输出 CSV：

```bash
python3 -m simtutor batch \
  --pack packs/fa18c_startup/pack.yaml \
  --taxonomy packs/fa18c_startup/taxonomy.yaml \
  --output-dir artifacts
```

默认生成：`artifacts/results.csv`

## CLI 速查

```bash
python3 -m simtutor -h
```

子命令：
- `validate`：按 schema 校验 JSONL
- `run`：跑 mock 场景
- `replay`：回放事件日志或 telemetry 日志
- `score`：按 taxonomy 评分
- `batch`：批处理 mock 场景并导出结果
- `model-config`：校验模型环境变量（不输出敏感信息）
- `replay-bios`：将 DCS-BIOS JSONL 走完整助教流程

## 模型接入（与 `model_access.md` 保持一致）

支持的 provider：
- `openai_compat`（vLLM / llama.cpp / TGI 等 OpenAI 兼容接口）
- `ollama`
- `stub`（无外部依赖，便于测试）

关键环境变量：
- `SIMTUTOR_MODEL_PROVIDER=ollama|openai_compat|stub`
- `SIMTUTOR_MODEL_NAME`
- `SIMTUTOR_MODEL_BASE_URL`
- `SIMTUTOR_MODEL_TIMEOUT_S`
- `SIMTUTOR_LANG=zh|en`
- `SIMTUTOR_MODEL_API_KEY`（`openai_compat` 必需）

最小本地 Stub 配置示例：

```bash
export SIMTUTOR_MODEL_PROVIDER=stub
export SIMTUTOR_MODEL_TIMEOUT_S=20
export SIMTUTOR_LANG=zh
```

校验当前配置：

```bash
python3 -m simtutor model-config
```

## Live DCS 助教循环

### 1) 回放 BIOS（推荐先离线验证）

`replay-bios` 默认开启 `--dry-run-overlay`（安全模式，不真实发高亮）。

```bash
python3 -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --model-provider stub \
  --auto-help-once \
  --output logs/replay_bios_demo.jsonl
```

常用参数：
- `--speed 1.0`：按原始时间回放；`--speed 0`：尽快跑完
- `--no-dry-run-overlay`：关闭安全模式，真实发送 overlay
- `--stdin-help`：按回车触发 help
- `--help-udp-port <port>`：启用 UDP help trigger

### 2) 实时监听 DCS-BIOS UDP

```bash
export SIMTUTOR_MODEL_API_KEY="<your_key>"

python3 live_dcs.py \
  --host 0.0.0.0 \
  --port 7790 \
  --model-provider openai_compat \
  --model-name Qwen3-8B-Instruct \
  --model-base-url http://127.0.0.1:8000 \
  --model-api-key "${SIMTUTOR_MODEL_API_KEY}" \
  --knowledge-index Doc/Evaluation/index.json \
  --rag-top-k 5 \
  --cold-start-production \
  --knowledge-source-policy knowledge_source_policy.yaml \
  --auto-help-every-n-frames 20 \
  --output logs/live_dcs_live.jsonl
```

说明：
- `live_dcs.py` 的 `--dry-run-overlay` 默认关闭（传入才启用）
- `--cold-start-production` 开启后，知识源策略必须有效
- 若你在 PowerShell 下运行，环境变量写法是 `$env:SIMTUTOR_MODEL_API_KEY="..."`

## Knowledge Index 与白名单策略

### 索引

- 默认路径：`Doc/Evaluation/index.json`
- 可用 `tools.index_docs` 重新构建

### Knowledge Source Policy（`knowledge_source_policy.yaml`）

- 开关：`--cold-start-production / --no-cold-start-production`
- 环境默认：`SIMTUTOR_COLD_START_PRODUCTION=1|0`
- `--knowledge-source-policy <path>`：任意模式下都可启用过滤
- 在 cold-start production 模式下：
  - 若未显式传路径，会尝试仓库根目录 `knowledge_source_policy.yaml`
  - 缺失或无效会在启动阶段直接报错退出
- `allow[].line_range` 会在运行时裁剪 snippet 行范围

## DCS 相关工具

### 安装 DCS Hook

```bash
python3 -m tools.install_dcs_hook --dcs-variant DCS
```

可选：
- `--saved-games <path>` 手动指定 Saved Games 路径
- `--no-export` 不补丁 `Export.lua`

### Telemetry 监听/录制

```bash
python3 -m tools.listen_dcs_telemetry --host 0.0.0.0 --port 7780
python3 -m tools.record_dcs_telemetry --output logs/dcs_telemetry.jsonl --duration 30 --print
```

### 发送假 Telemetry（联调）

```bash
python3 -m tools.send_fake_dcs_telemetry --host 127.0.0.1 --port 7780 --count 20 --hz 20
```

### DCS-BIOS 原始流解码

```bash
python3 -m tools.listen_dcs_bios_raw --aircraft FA-18C_hornet
```

一次性抓取更完整快照（常用）：

```bash
python3 -m tools.listen_dcs_bios_raw \
  --aircraft FA-18C_hornet \
  --once --wait 15 --min-keys 500 \
  --output artifacts/dcs_bios_frame_once.json
```

## Schema 校验

```bash
python3 -m simtutor validate logs/run_demo.jsonl --schema event
```

可选 schema 名称（来自 `simtutor/schemas/v1` 与 `v2`）：
- `event`
- `observation`
- `tutor_request`
- `tutor_response`
- `dcs_observation`
- `dcs_bios_frame`
- `telemetry_frame`
- `dcs_overlay_command`
- `dcs_overlay_ack`
- `dcs_hello`
- `dcs_caps`

## 测试

```bash
python3 -m pytest -q
```

当前仓库测试文件数量：`60`（`tests/`）。

## 常见问题

- `python: command not found`
  - 直接使用 `python3`。

- `model-config` 报 `Missing required env: SIMTUTOR_MODEL_PROVIDER`
  - 先设置最小模型环境变量，再执行 `python3 -m simtutor model-config`。

- 启动 live/replay 时提示 cold-start policy 错误
  - 检查 `knowledge_source_policy.yaml` 与 `Doc/Evaluation/index.json` 是否匹配（`doc_id/chunk_id/line_range`）。

## 参考文档

- `model_access.md`
- `help_flow.md`
- `help_flow_en.md`
- `Doc/Evaluation/fa18c_startup_master.md`
- `Doc/Evaluation/Appendix - Training Task Syllabus.md`
- `Doc/Evaluation/fa18c_error_coding_guide.md`
- `Doc/Evaluation/fa18c_scoring_sheet_template.md`
- `Doc/Evaluation/fa18c_coldstart_quiz.md`
- `Doc/Evaluation/fa18c_nasatlx_vr.md`
