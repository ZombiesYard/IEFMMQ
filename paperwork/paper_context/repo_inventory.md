# Repository Inventory

> 目标：给后续英文论文写作提供一份“以仓库为准”的系统实现清单。  
> 范围：只描述当前仓库里**已经实现或已经存档**的内容；不把 proposal 中尚未完成的 VR / 用户研究写成已实现系统。

## 1. 仓库总体定位

当前仓库实现的是一个面向 DCS F/A-18C cold-start 任务的研究原型，核心是：

1. 遥测驱动的 procedural tutoring runtime
2. 基于 cockpit screenshot 的 VLM visual fact extraction
3. 可回放、可记录、可 benchmark 的实验与分析链路

关键总览证据：

- `README.md`
- `pyproject.toml`
- `live_dcs.py`
- `simtutor/__main__.py`

## 2. 代码层级与职责

### 2.1 `core/`：领域逻辑与中立数据结构

`core/` 保存与具体 simulator / model provider 解耦的核心逻辑。

关键模块：

- `core/procedure.py`：procedure engine，维护 step 激活、完成、阻塞等状态
- `core/gating.py`：pack gate / rule evaluation
- `core/scoring.py`：事件日志评分
- `core/interaction_metrics.py`：交互统计
- `core/vision_facts.py`：vision fact ontology 的加载、聚合、过期、step binding
- `core/types.py`：v1 runtime types，如 `TutorRequest`、`TutorResponse`、`Observation`、`Event`
- `core/types_v2.py`：v2 vision / telemetry / DCS 相关结构
- `core/knowledge.py`：BM25 retrieval
- `core/security.py`：日志与 model I/O 的敏感信息处理
- `core/vars.py`：变量解析与投影
- `core/step_hint.py`：deterministic step inference
- `core/step_signal_metadata.py`：step observability / visual confirmation 元数据
- `core/help_cycle_audit.py`：help cycle 完整性审计
- `core/help_failure.py`：help failure 模式分类
- `core/constants.py`

证据路径：

- `core/procedure.py`
- `core/gating.py`
- `core/scoring.py`
- `core/vision_facts.py`
- `core/types.py`
- `core/types_v2.py`
- `core/knowledge.py`
- `tests/test_procedure_engine.py`
- `tests/test_gating_engine.py`
- `tests/test_scoring_engine.py`
- `tests/test_vision_fact_state.py`

### 2.2 `ports/`：稳定抽象接口

`ports/` 定义运行时依赖的抽象边界，支持 clean / hexagonal architecture 的写法。

关键接口：

- `ports/model_port.py`
- `ports/knowledge_port.py`
- `ports/vision_port.py`
- `ports/telemetry.py`

这些接口说明系统在设计上把 reasoning、knowledge、vision、telemetry 当作可替换依赖，而不是硬编码在单个脚本里。

### 2.3 `adapters/`：外部依赖与 runtime glue

`adapters/` 是实现最密集的目录，连接 DCS、model provider、knowledge index、vision pipeline 和 runtime orchestration。

#### A. 模型与响应处理

- `adapters/base_help_model.py`：help-generation 基类
- `adapters/openai_compat_model.py`：OpenAI-compatible model adapter
- `adapters/ollama_model.py`：Ollama adapter
- `adapters/model_stub.py`：deterministic test stub
- `adapters/help_response_parser.py`：help JSON 解析
- `adapters/json_extract.py`：鲁棒 JSON 提取
- `adapters/response_mapping.py`：model output 映射到 runtime response

证据路径：

- `adapters/base_help_model.py`
- `adapters/openai_compat_model.py`
- `adapters/json_extract.py`
- `adapters/response_mapping.py`
- `tests/test_base_help_model.py`
- `tests/test_json_extract.py`
- `tests/test_response_mapping.py`

#### B. 遥测与 DCS 集成

- `adapters/telemetry_pipeline.py`：BIOS observation enrichment、delta sanitization、state completion latching
- `adapters/delta_sanitizer.py`
- `adapters/delta_aggregator.py`
- `adapters/dcs_adapter.py`
- `adapters/dcs_bios/receiver.py`
- `adapters/dcs_bios/bios_ui_map.py`
- `adapters/dcs/overlay/sender.py`
- `adapters/dcs/telemetry/receiver.py`

证据路径：

- `adapters/telemetry_pipeline.py`
- `adapters/delta_sanitizer.py`
- `adapters/delta_aggregator.py`
- `adapters/dcs_bios/receiver.py`
- `adapters/dcs/overlay/sender.py`
- `tests/test_telemetry_pipeline.py`
- `tests/adapters/test_dcs_bios_receiver.py`
- `tests/adapters/test_dcs_overlay_sender.py`

#### C. Vision / VLM path

- `adapters/vision_fact_extractor.py`：核心 VLM fact extraction runtime
- `adapters/vision_fact_prompting.py`：13-fact prompt builder
- `adapters/openai_compat_multimodal.py`：multimodal payload construction
- `adapters/vision_prompting.py`：vision layout / region prompt
- `adapters/vision_frames.py`：frame manifest 与 frame root handling
- `adapters/vision_sync.py`：pre-trigger / trigger frame selection 与同步
- `adapters/vision_capture_trigger.py`：采帧触发

证据路径：

- `adapters/vision_fact_extractor.py`
- `adapters/vision_fact_prompting.py`
- `adapters/openai_compat_multimodal.py`
- `adapters/vision_prompting.py`
- `adapters/vision_frames.py`
- `adapters/vision_sync.py`
- `tests/test_vision_fact_extractor.py`
- `tests/test_vision_fact_prompting.py`
- `tests/test_vision_frame_ingestor.py`
- `tests/test_vision_sync.py`

#### D. Knowledge / grounding

- `adapters/knowledge_local.py`：本地 BM25 adapter
- `adapters/knowledge_source_policy.py`：source allowlist / whitelist policy
- `adapters/source_chunk_refs.py`
- `adapters/evidence_refs.py`
- `adapters/prompting.py`：将 step、telemetry、recent actions、knowledge、vision facts 组装进 prompt

证据路径：

- `adapters/knowledge_local.py`
- `adapters/knowledge_source_policy.py`
- `adapters/prompting.py`
- `knowledge_source_policy.yaml`
- `tests/test_knowledge_retrieval.py`
- `tests/test_knowledge_source_policy.py`

## 3. `simtutor/`：CLI、schema registry、replay/run 入口

`simtutor/` 提供 CLI 入口与 schema registry。

关键文件：

- `simtutor/__main__.py`：CLI commands，包括 `run`、`replay`、`score`、`record-vlm`、`replay-bios` 等
- `simtutor/runner.py`：simulation runner 与 replay validation
- `simtutor/replay_eval.py`
- `simtutor/config.py`
- `simtutor/schemas/v1/*.schema.json`
- `simtutor/schemas/v2/*.json`

其中 v2 schema 明确覆盖：

- telemetry frame
- DCS observation / caps / hello / overlay command / overlay ack
- vision observation
- vision fact observation

证据路径：

- `simtutor/__main__.py`
- `simtutor/runner.py`
- `simtutor/schemas/v1/tutor_response.schema.json`
- `simtutor/schemas/v2/telemetry_frame.json`
- `simtutor/schemas/v2/vision_observation.json`
- `simtutor/schemas/v2/vision_fact_observation.json`

## 4. `live_dcs.py`：live runtime orchestration

`live_dcs.py` 是当前实现系统的关键运行时汇总入口。它把以下部分接起来：

1. DCS-BIOS receiver
2. telemetry enrichment
3. knowledge retrieval
4. model help generation
5. optional VLM visual fact extraction
6. overlay action execution
7. event logging

这也是“当前实现系统”和“proposal 中的最初 MVP”之间最大差异最明显的地方。

证据路径：

- `live_dcs.py`
- `adapters/action_executor.py`
- `adapters/telemetry_pipeline.py`
- `adapters/vision_fact_extractor.py`

## 5. `packs/fa18c_startup/`：pack-driven procedure configuration

当前仓库只有一个主要 procedure pack：`fa18c_startup`。

关键配置：

- `packs/fa18c_startup/pack.yaml`：pack 元数据、priority steps、source documents、vision settings
- `packs/fa18c_startup/step_registry.yaml`：步骤定义
- `packs/fa18c_startup/taxonomy.yaml`：error taxonomy / scoring
- `packs/fa18c_startup/ui_map.yaml`：overlay targets
- `packs/fa18c_startup/telemetry_map.yaml`
- `packs/fa18c_startup/delta_policy.yaml`
- `packs/fa18c_startup/bios_to_ui.yaml`
- `packs/fa18c_startup/vision_facts.yaml`：13-fact ontology 与 step bindings
- `packs/fa18c_startup/vision_layout.yaml`

从 `pack.yaml` 可直接看出当前系统同时依赖：

- BIOS priority steps
- vision priority steps
- manual / out-of-layout steps

这说明系统已经不是“只靠 RAG 输出一步文字卡片”，而是一个多信号融合 runtime。

证据路径：

- `packs/fa18c_startup/pack.yaml`
- `packs/fa18c_startup/vision_facts.yaml`
- `packs/fa18c_startup/vision_layout.yaml`
- `packs/fa18c_startup/step_registry.yaml`

## 6. `tools/`：数据、训练、benchmark、安装与运维脚本

### 6.1 数据与标注链路

- `tools/capture_vlm_dataset.py`
- `tools/capture_vision_sidecar.py`
- `tools/generate_vlm_prelabels.py`
- `tools/generate_vlm_prelabels_en.py`
- `tools/generate_vlm_prelabels_zh.py`
- `tools/export_vision_sft_dataset.py`
- `tools/repair_vlm_capture_session.py`

另外还有辅助分析工具：

- `tools/help_failure_stats.py`：help failure 统计
- `tools/build_coldstart_state_matrix.py`：cold-start state matrix 构建
- `tools/convert_to_predictions.py`：prediction 格式转换

这条链路对应的研究流程是：

capture -> prelabel -> human review -> SFT export

### 6.2 训练与 benchmark

- `tools/train_qwen35_vlm_unsloth.py`
- `tools/train_gemma4_vlm_unsloth.py`
- `tools/benchmark_qwen35_vlm_facts.py`
- `tools/benchmark_gemma4_vlm_facts.py`

### 6.3 DCS runtime 相关工具

- `tools/install_dcs_hook.py`
- `tools/install_dcs_monitor_setup.py`
- `tools/record_dcs_telemetry.py`
- `tools/listen_dcs_bios_raw.py`
- `tools/send_help_hotkey.py`

### 6.4 文档与索引

- `tools/index_docs.py`
- `tools/regenerate_eval_docs.py`

证据路径：

- `tools/export_vision_sft_dataset.py`
- `tools/train_qwen35_vlm_unsloth.py`
- `tools/train_gemma4_vlm_unsloth.py`
- `tools/benchmark_qwen35_vlm_facts.py`
- `tools/install_dcs_hook.py`
- `tools/index_docs.py`

## 7. `DCS/Scripts/`：Lua-side simulator integration

仓库包含 DCS 侧脚本：

- `DCS/Scripts/SimTutor/SimTutor.lua`
- `DCS/Scripts/SimTutor/SimTutor Function.lua`
- `DCS/Scripts/SimTutor/SimTutorDcsBiosHub.lua`
- `DCS/Scripts/Hooks/VRHilite.lua`
- `DCS/Scripts/Hooks/SimTutorHighlight.lua`
- `DCS/Scripts/Export.lua`

这些文件表明当前实现系统已经具备：

1. DCS 侧 state export
2. overlay / highlight 侧回写
3. DCS-BIOS / hook 级集成

证据路径：

- `DCS/Scripts/SimTutor/SimTutor.lua`
- `DCS/Scripts/Hooks/VRHilite.lua`
- `DCS/Scripts/Export.lua`

## 8. 数据、模型与实验产物

### 8.1 数据集统计快照

仓库当前直接跟踪的是各数据集目录下的 `stats.json`，包括：

- `datasets/vision_sft/stats.json`
- `datasets/vision_sft_holdout_run002/stats.json`
- `datasets/vision_sft_holdout_run002_newfacts/stats.json`
- `datasets/vision_sft_holdout_run004_random/stats.json`
- `datasets/vision_sft_run003/stats.json`
- `datasets/vision_sft_run005_composition_rebalance/stats.json`

这些文件是当前最稳定、最可信的数据集规模与分布来源。

### 8.2 Benchmark artifacts

`benchmarks/qwen35_vlm_finetune/` 下已存有多个 benchmark 目录。每个主目录通常包含：

- `benchmark_summary.json`
- `metrics_base.json`
- `metrics_lora.json`
- `comparison.json`
- `predictions_base.jsonl`
- `predictions_lora.jsonl`
- `errors_base.jsonl`
- `errors_lora.jsonl`
- `fact_scores.csv`
- `report.md`
- `charts/*.png`

证据路径：

- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/benchmark_summary.json`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/report.md`
- `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run004_random_v2/metrics_lora.json`

### 8.3 模型 artifact

Qwen artifacts：

- `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/train_summary.json`
- `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_v1/train_summary.json`
- `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/train_summary.json`
- `models/qwen35_vlm_lora/README.md`

Gemma artifacts：

- `models/gemma4_vlm_lora/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/train_summary.json`
- `models/gemma4_vlm_lora/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/README.md`

## 9. 测试与可验证性

仓库含大量 pytest 测试，覆盖：

- procedure / gating / scoring
- telemetry pipeline
- DCS adapters
- vision fact extraction
- training / export / benchmark scripts
- CLI 和 replay path

代表性测试文件：

- `tests/test_live_dcs.py`
- `tests/test_export_vision_sft_dataset.py`
- `tests/test_benchmark_qwen35_vlm_facts.py`
- `tests/test_capture_vlm_dataset.py`
- `tests/test_vision_fact_contracts.py`
- `tests/integration/test_coldstart_help_loop.py`

## 10. 当前仓库反映出的系统边界

当前仓库**已经实现**：

- desktop / DCS-based tutoring runtime
- telemetry-grounded prompting
- optional VLM visual fact extraction
- LoRA fine-tuning and benchmark pipeline
- replay / scoring / logging artifacts

当前仓库**尚未提供可直接引用的已完成实证结果**：

- VR Quest 3 runtime 数据
- human-subject evaluation 数据
- runtime 数据收集框架下的完整 human-study table / logs

这部分应在后续论文里作为 planned work / placeholder evaluation，而不是 current technical results。

## 11. 当前实现系统的数据流

当前仓库可支持的主数据流可以总结为：

```text
DCS-BIOS telemetry + composite-panel screenshot
-> telemetry / vision adapters
-> normalized vars + visual facts
-> prompt construction + BM25 grounding
-> structured help response
-> overlay / logging / replay / benchmark artifacts
```

关键证据路径：

- `DCS/Scripts/Export.lua`
- `adapters/dcs_bios/receiver.py`
- `adapters/telemetry_pipeline.py`
- `adapters/vision_fact_extractor.py`
- `adapters/prompting.py`
- `adapters/knowledge_local.py`
- `adapters/action_executor.py`
- `simtutor/__main__.py`

