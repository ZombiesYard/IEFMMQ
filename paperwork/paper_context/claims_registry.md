# Claims Registry

> 规则：只要 claim 涉及当前已实现系统或当前已有 benchmark，才允许标 `verified` 或 `likely`。  
> VR / runtime 数据收集 / human-subject evaluation 相关主张保留为论文核心占位，但统一标 `TODO`。

| Claim ID | Claim statement | Evidence files | Status | Suitable paper section | Risk of overclaiming |
|---|---|---|---|---|---|
| C-ARCH-01 | 当前仓库实现了以 `core/` + `ports/` + `adapters/` 为主的分层/hexagonal 风格架构，而不是单体脚本。 | `pyproject.toml`; `core/procedure.py`; `ports/model_port.py`; `adapters/telemetry_pipeline.py`; `README.md` | verified | Methods / System Context | low |
| C-ARCH-02 | 当前 runtime 能把 telemetry、knowledge retrieval、optional vision facts 和 overlay execution 接到同一条 live DCS 帮助链路中。 | `live_dcs.py`; `adapters/telemetry_pipeline.py`; `adapters/knowledge_local.py`; `adapters/vision_fact_extractor.py`; `adapters/action_executor.py` | verified | Methods | low |
| C-ARCH-03 | 当前系统的帮助输出是结构化 response，并且下游有显式 mapping / safety handling，而不是直接把 model 文本原样执行。 | `core/types.py`; `adapters/response_mapping.py`; `adapters/help_response_parser.py`; `core/security.py` | verified | Methods / Discussion | low |
| C-PROC-01 | 当前仓库已有 pack-driven 的 F/A-18C cold-start procedure 配置，而不是完全硬编码步骤顺序。 | `packs/fa18c_startup/pack.yaml`; `packs/fa18c_startup/step_registry.yaml`; `packs/fa18c_startup/taxonomy.yaml` | verified | Methods | low |
| C-PROC-02 | 当前 procedure runtime 同时使用 telemetry priority steps、vision priority steps 与 manual/out-of-layout steps 的区分。 | `packs/fa18c_startup/pack.yaml`; `packs/fa18c_startup/vision_facts.yaml` | verified | Methods | medium |
| C-TELEM-01 | 当前仓库实现了 telemetry sanitization / aggregation pipeline，用于把 DCS-BIOS 数据转成更稳定的 runtime state。 | `adapters/telemetry_pipeline.py`; `adapters/delta_sanitizer.py`; `adapters/delta_aggregator.py`; `tests/test_telemetry_pipeline.py` | verified | Methods / System Context | low |
| C-RAG-01 | 当前实现使用本地 BM25 retrieval 和 source policy，而不是无约束地向模型暴露整个文档库。 | `adapters/knowledge_local.py`; `knowledge_source_policy.yaml`; `tools/index_docs.py`; `core/knowledge.py` | verified | Methods | low |
| C-VLM-01 | 当前仓库已实现 runtime 侧 structured visual fact extraction path。 | `adapters/vision_fact_extractor.py`; `adapters/vision_fact_prompting.py`; `adapters/openai_compat_multimodal.py`; `packs/fa18c_startup/vision_facts.yaml` | verified | Methods | low |
| C-VLM-02 | 当前 runtime visual contract 已围绕 13-fact ontology 组织，而不是仅停留在早期 8-fact 目标。 | `packs/fa18c_startup/vision_facts.yaml`; `adapters/vision_fact_prompting.py`; `tools/benchmark_qwen35_vlm_facts.py` | verified | Methods / Current Technical Results | low |
| C-VLM-03 | 当前仓库保存了从 capture 到 benchmark 的完整 VLM adaptation artifact chain。 | `tools/capture_vlm_dataset.py`; `tools/generate_vlm_prelabels.py`; `tools/export_vision_sft_dataset.py`; `tools/train_qwen35_vlm_unsloth.py`; `tools/benchmark_qwen35_vlm_facts.py`; `benchmarks/qwen35_vlm_finetune/` | verified | Methods | low |
| C-VLM-03a | holdout 集 (Run-002, Run-004) 与训练数据之间的 exact overlap 经验证为 0。 | `Doc/Vision/Reports/vlm_backbone_comparison_summary_EN.md` | verified | Methods / Current Technical Results | low |
| C-VLM-03b | Run-003 + Run-005x2 的数据配方对 Qwen 与 Gemma 两个 backbone 均有效。 | `Doc/Vision/Reports/vlm_backbone_comparison_summary_EN.md`; `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run002_newfacts_v2/metrics_*.json` | likely | Current Technical Results | medium |
| C-VLM-04 | 13-fact Qwen adapter 在 holdout `run002_newfacts` 上显著优于 base model。 | `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/metrics_base.json`; `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/metrics_lora.json`; `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/benchmark_summary.json` | verified | Current Technical Results | low |
| C-VLM-05 | 13-fact Qwen adapter 在 holdout `run004_random` 上也显著优于 base model，并且 JSON/schema validity 更稳定。 | `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run004_random_v1/metrics_base.json`; `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run004_random_v1/metrics_lora.json` | verified | Current Technical Results | low |
| C-VLM-06 | 13-fact Gemma adapter 也存在可观 benchmark 改进，但是否作为论文主实验线仍需人工决定。 | `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run002_newfacts_v2/metrics_base.json`; `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run002_newfacts_v2/metrics_lora.json`; `benchmarks/qwen35_vlm_finetune/gemma4_31b_run003_plus_run005x2_vs_base_holdout_run004_random_v2/metrics_lora.json` | likely | Current Technical Results / Discussion | medium |
| C-VLM-07 | 8-fact v1 与 13-fact v2 代表了两代不同实验线，后者更接近当前 runtime contract。 | `datasets/vision_sft/stats.json`; `datasets/vision_sft_holdout_run002/stats.json`; `datasets/vision_sft_holdout_run002_newfacts/stats.json`; `packs/fa18c_startup/vision_facts.yaml` | verified | Methods / Current Technical Results | low |
| C-DATA-01 | 当前仓库已经存有多个 reviewed dataset statistics，可支撑数据集来源、规模与 fact distribution 描述。 | `datasets/vision_sft/stats.json`; `datasets/vision_sft_run003/stats.json`; `datasets/vision_sft_run005_composition_rebalance/stats.json`; `datasets/vision_sft_holdout_run004_random/stats.json` | verified | Methods | low |
| C-TRAIN-01 | 当前仓库保存了可直接引用的训练配置摘要，包括 Qwen 与 Gemma 的 train rows、epoch、learning rate 和 LoRA 参数。 | `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/train_summary.json`; `models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_run003_plus_run005x2_v1/train_summary.json`; `models/gemma4_vlm_lora/full_gemma4_31b_base_bilingual_run003_plus_run005x2_v1/train_summary.json` | verified | Methods | low |
| C-RUNTIME-01 | 当前仓库已具备 replay / logging / benchmark 方向的 runtime evidence infrastructure，但还不是最终的人因实验数据收集系统。 | `simtutor/runner.py`; `replay_eval/fa18c_startup_v04/suite.yaml`; `replay_eval/fa18c_startup_v04/cases`; `core/event_store.py` | likely | Methods / Discussion | medium |
| C-PROPOSAL-01 | thesis 的原始核心目标是 VR-first grounded tutoring with human-subject validation。 | `ThesisProposal1128.pdf`; `paperwork/paper_context/proposal_summary.md` | TODO | Introduction / Discussion | medium |
| C-PROPOSAL-02 | VR / Quest 3 / in-headset tutoring 仍然是论文核心叙事的一部分，但当前仓库还没有足够证据把它写成已实现系统。 | `ThesisProposal1128.pdf`; `paperwork/paper_context/system_delta_from_proposal.md`; `paperwork/paper_context/todos.md` | TODO | Introduction / Planned Runtime Data Collection & Human-Subject Evaluation | high |
| C-HUMAN-01 | grounded tutoring system 将改善 learning gains、procedural performance 或 workload。 | `ThesisProposal1128.pdf`; `paperwork/paper_context/todos.md` | TODO | Planned Runtime Data Collection & Human-Subject Evaluation / Discussion | high |
| C-HUMAN-02 | 将来的人因实验会比较 `video+notes`、`ungrounded text-only LLM` 与 grounded tutor。 | `ThesisProposal1128.pdf`; `paperwork/paper_context/proposal_summary.md`; `paperwork/paper_context/todos.md` | TODO | Planned Runtime Data Collection & Human-Subject Evaluation | high |
| C-HUMAN-03 | 将采集 NASA-TLX、retention、transfer、completion time 等指标。 | `ThesisProposal1128.pdf`; `Doc/Evaluation/fa18c_nasatlx_vr.md`; `paperwork/paper_context/todos.md` | TODO | Planned Runtime Data Collection & Human-Subject Evaluation | high |
| C-DATA-COLLECT-01 | runtime 数据收集系统会被完善到足以支持未来用户研究。 | `paperwork/paper_context/todos.md`; `live_dcs.py`; `simtutor/__main__.py` | TODO | Planned Runtime Data Collection & Human-Subject Evaluation | high |
| C-NARRATIVE-01 | 当前论文最稳妥的写法是“先报告已实现技术结果，再保留 VR/human-study 作为核心占位章节”。 | `paperwork/paper_context/system_delta_from_proposal.md`; `paperwork/paper_context/paper_outline.md` | likely | Whole paper framing | low |

## 使用说明

后续写英文论文时，建议按下面的纪律使用这张表：

1. 只有 `verified` 项可以进入 “Current Technical Results” 的结果句。
2. `likely` 项可以进入方法或讨论，但语气应保守，并尽量附局限性说明。
3. `TODO` 项只能进入占位章节、future work、planned evaluation、research agenda，不能写成已经完成的实验发现。

