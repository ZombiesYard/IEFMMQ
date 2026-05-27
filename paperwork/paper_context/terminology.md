# Terminology

> 目标：统一论文术语与代码级名称，避免同一个概念在 paper 中反复换名。  
> 建议：论文正文优先使用左栏术语，括号内或方法章节中第一次出现时再映射到代码名。

| 论文术语 | 含义 | 代码/配置名 | 主要证据 |
|---|---|---|---|
| SimTutor | 整个研究原型与 runtime 框架 | `simtutor/`; `live_dcs.py` | `README.md`; `simtutor/__main__.py` |
| procedure pack | 面向某一 procedure 的配置包 | `packs/fa18c_startup/` | `packs/fa18c_startup/pack.yaml` |
| step registry | 过程步骤定义与元数据 | `step_registry.yaml` | `packs/fa18c_startup/step_registry.yaml` |
| gating rule | 控制某一步是否允许激活/完成的规则 | `precondition_gates`, `completion_gates`; `core/gating.py` | `packs/fa18c_startup/pack.yaml`; `core/gating.py` |
| telemetry frame | 结构化遥测记录 | `telemetry_frame` schema | `simtutor/schemas/v2/telemetry_frame.json` |
| DCS-BIOS raw frame | 从 DCS-BIOS 收到的原始 BIOS JSONL / UDP 载荷 | `dcs_bios_raw.jsonl`; `DcsBiosRawReceiver` | `simtutor/schemas/v2/dcs_bios_frame.json`; `replay_eval/.../dcs_bios_raw.jsonl` |
| telemetry enrichment | 从 BIOS 数据生成更稳定 runtime vars 的过程 | `enrich_bios_observation` | `adapters/telemetry_pipeline.py` |
| delta sanitization | 遥测去噪、去抖、阈值过滤 | `DeltaSanitizer`; `delta_policy.yaml` | `adapters/delta_sanitizer.py`; `packs/fa18c_startup/delta_policy.yaml` |
| delta aggregation | 将高频变化压缩为 prompt-safe 摘要 | `DeltaAggregator` | `adapters/delta_aggregator.py` |
| grounding query | 用于 retrieval 的查询串 | `build_grounding_query` | `adapters/knowledge_local.py` |
| source policy | 知识片段白名单 / allowlist 规则 | `KnowledgeSourcePolicy` | `adapters/knowledge_source_policy.py`; `knowledge_source_policy.yaml` |
| knowledge retrieval | 从本地文档索引中检索 grounding snippets | `LocalKnowledgeAdapter`; `BM25Retriever` | `adapters/knowledge_local.py`; `core/knowledge.py` |
| help cycle | 一次帮助触发、推理、输出、执行、记录的完整周期 | `help_cycle_id` | `live_dcs.py`; `core/help_cycle_audit.py` |
| structured help response | 模型输出的结构化帮助对象 | `TutorResponse`; help JSON | `core/types.py`; `adapters/response_mapping.py` |
| overlay target | cockpit 上被高亮/清除/脉冲提示的 UI target | `ui_targets`; `overlay_targets` | `packs/fa18c_startup/pack.yaml`; `adapters/action_executor.py` |
| replay evaluation | 基于已存日志或 BIOS/vision 产物进行离线运行与分析 | `replay`; `replay-bios`; `replay_eval/` | `simtutor/__main__.py`; `simtutor/runner.py` |
| vision observation | 来自 frame sidecar / screenshot 的视觉输入对象 | `VisionObservation` | `simtutor/schemas/v2/vision_observation.json`; `core/types_v2.py` |
| candidate frame | 参与当前 VLM 判断的一个或两个 frame | `candidate_frames` | `adapters/vision_fact_extractor.py`; `adapters/vision_sync.py` |
| pre-trigger frame | 触发前的候选图像 | `pre_trigger_frame` 语义，见 prompt 说明 | `adapters/vision_fact_prompting.py`; `adapters/vision_sync.py` |
| trigger frame | 触发时刻图像 | `trigger_frame` 语义，见 prompt 说明 | `adapters/vision_fact_prompting.py`; `adapters/vision_sync.py` |
| VLM-ready frame | 为模型准备过的图像 artifact | `*_vlm.png` / `artifact_kind=vlm_ready` | `tests/test_vision_frame_ingestor.py`; `replay_eval/.../artifacts/*_vlm.png` |
| vision fact | 从截图中抽取的结构化视觉命题 | `VisionFact`; `facts` array | `core/types_v2.py`; `adapters/vision_fact_extractor.py` |
| vision fact ontology | 当前允许的 visual facts 集合 | `vision_facts.yaml` | `packs/fa18c_startup/vision_facts.yaml` |
| fact state | 单个 visual fact 的三值标签 | `seen`, `not_seen`, `uncertain` | `adapters/vision_fact_extractor.py`; `tools/export_vision_sft_dataset.py` |
| 8-fact v1 | 早期 visual fact ontology | `datasets/vision_sft/stats.json` 中的 fact set | `datasets/vision_sft/stats.json` |
| 13-fact v2 | 当前 runtime-aligned visual fact ontology | `CORE_FACT_IDS`; `vision_facts.yaml` | `tools/export_vision_sft_dataset.py`; `packs/fa18c_startup/vision_facts.yaml` |
| visual fact prompt | 指导模型输出 facts 的 prompt | `build_vision_fact_prompt` | `adapters/vision_fact_prompting.py` |
| multimodal request | image + text 的模型请求 | `build_multimodal_image_contents` | `adapters/openai_compat_multimodal.py` |
| LoRA adapter | 轻量微调后的 adapter 权重 | `models/.../adapter`; `train_summary.json` | `models/qwen35_vlm_lora/README.md`; `models/gemma4_vlm_lora/.../train_summary.json` |
| base model | 未加载 adapter 的底座模型 | `Qwen/Qwen3.5-9B-Base`; `google/gemma-4-31B` 或其 unsloth variant | `tools/train_qwen35_vlm_unsloth.py`; `tools/train_gemma4_vlm_unsloth.py` |
| contaminated dev set | 与训练数据同源、仅用于回归检查的数据 | `benchmark_kind=contaminated_dev_set` | `benchmarks/qwen35_vlm_finetune/base_vs_lora_current180_v1/benchmark_summary.json` |
| holdout set | 未参与训练的独立评估集 | `benchmark_kind=heldout_new_session` / `holdout_run002_newfacts` / `holdout_run004_random` | `benchmarks/qwen35_vlm_finetune/*/benchmark_summary.json` |
| benchmark summary | 某次 benchmark 的摘要指标与对比结果 | `benchmark_summary.json` | `benchmarks/qwen35_vlm_finetune/*/benchmark_summary.json` |
| runtime contract | runtime 期望的输入/输出结构与 fact contract | schema + prompt + downstream bindings | `adapters/vision_fact_extractor.py`; `adapters/vision_fact_prompting.py`; `packs/fa18c_startup/vision_facts.yaml` |
| deterministic step inference | 不依赖 LLM、基于 pack rules + vars 进行的局部步骤推理 | `step_inference.py`; `core/step_hint.py` | `adapters/step_inference.py`; `core/step_hint.py` |
| evidence grounding | overlay target 必须有来自 var/gate/rag/delta/visual 的 evidence ref | `evidence_refs.py`; overlay evidence validation | `adapters/evidence_refs.py`; `adapters/response_mapping.py:_validate_overlay_evidence` |
| vision layout | 三种 cockpit display region 的组合布局定义 | `vision_layout.yaml`; `layout_id=fa18c_composite_panel_v2` | `packs/fa18c_startup/vision_layout.yaml` |
| sticky fact | 一旦 seen 就不会被后续 not_seen/uncertain 覆盖的 visual fact（在 TTL 有效期内） | `sticky=true`; `_STICKY_PRESERVE_STATES` | `packs/fa18c_startup/vision_facts.yaml`; `core/vision_facts.py:merge_vision_fact_observation` |
| one-of fact collection | 多个 fact 中至少一个为 seen 即满足的 step binding 规则 | `any_of` in step_bindings | `packs/fa18c_startup/vision_facts.yaml:step_bindings`; `core/vision_facts.py:facts_satisfy_step_binding` |
| composition rebalance | Run-005 中针对 class-imbalanced facts 的重采样策略 | Run-005 composition rebalance | `datasets/vision_sft_run005_composition_rebalance/stats.json` |
| bilingual SFT | 同一张图 + 同样 labels，分别生成 en 和 zh 两份 SFT rows | `exported_languages=[en, zh]` | `tools/export_vision_sft_dataset.py`; `data/vision_sft_run003/stats.json` |

## 建议的论文用词规范

### 1. 优先使用的统称

- 用 `visual fact extraction`，不要在正文里来回切换成 `VQA labeling`、`screen classification`、`fact parsing`
- 用 `telemetry-grounded tutoring runtime`，不要和 `RAG backend` 混为一谈
- 用 `procedure pack` 指代 `packs/fa18c_startup/` 这一类配置包

### 2. 需要刻意区分的概念

- `base model` 不等于 `runtime model`
- `LoRA adapter` 不等于完整模型权重
- `vision fact` 不等于最终 tutor decision
- `holdout benchmark` 不等于 human-subject evaluation
- `proposal-level VR tutor` 不等于当前仓库里的 implemented runtime

### 3. 后续英文写作时的术语边界

建议在英文论文里坚持以下表达：

- `visual facts`：用于中间表征
- `telemetry-grounded`：强调 runtime 不是纯视觉驱动
- `current technical results`：只指仓库里已有 artifact 支撑的结果
- `planned runtime data collection and human-subject evaluation`：专门容纳未来 VR / 用户研究主线

