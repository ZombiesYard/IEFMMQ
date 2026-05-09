# System Delta from Proposal

> 对比对象：`ThesisProposal1128.pdf` vs 当前仓库实现  
> 原则：proposal 是 thesis vision；仓库代码与实验产物才是当前实现证据。

## 1. 总体判断

proposal 的核心是：

- VR-first
- Quest 3
- RAG+LLM
- in-headset step cards
- human-subject evaluation

当前仓库的核心则是：

- desktop / DCS runtime
- telemetry-grounded help generation
- optional VLM visual fact extraction
- LoRA fine-tuning + benchmark pipeline
- replay / logging / benchmark artifacts

因此，当前论文必须同时写清两条线：

1. **当前已实现并已有技术证据的系统线**
2. **仍然作为 thesis 核心目标、但尚待 runtime 数据收集与实验设计落地的 VR / human-study 线**

## 2. 哪些内容被保留了

| Proposal 中的内容 | 当前状态 | 证据 |
|---|---|---|
| procedural learning / sequence-sensitive task 作为研究对象 | 保留 | `README.md`, `packs/fa18c_startup/pack.yaml` |
| DCS World 作为实验平台 | 保留 | `DCS/Scripts/Export.lua`, `live_dcs.py` |
| grounded guidance 而非纯自由对话 | 保留并强化 | `adapters/prompting.py`, `adapters/knowledge_local.py`, `knowledge_source_policy.yaml` |
| 结构化帮助输出，而不是长篇自然语言 | 保留并工程化 | `core/types.py`, `adapters/response_mapping.py` |
| 利用 simulator state 约束推理 | 保留并大幅扩展 | `adapters/telemetry_pipeline.py`, `packs/fa18c_startup/telemetry_map.yaml` |

## 3. 哪些内容被新增了

| 当前仓库新增内容 | Proposal 是否已有 | 证据 |
|---|---|---|
| telemetry delta sanitization / aggregation pipeline | 否 | `adapters/telemetry_pipeline.py`, `adapters/delta_sanitizer.py`, `adapters/delta_aggregator.py` |
| replay / JSONL logging / scoring | 否 | `simtutor/runner.py`, `core/event_store.py`, `core/scoring.py` |
| procedure pack + gating engine | 仅有概念，没有当前实现细节 | `packs/fa18c_startup/*.yaml`, `core/gating.py` |
| VLM visual fact extraction runtime | 否 | `adapters/vision_fact_extractor.py`, `adapters/vision_fact_prompting.py` |
| capture -> prelabel -> review -> export 数据链 | 否 | `tools/capture_vlm_dataset.py`, `tools/generate_vlm_prelabels.py`, `tools/export_vision_sft_dataset.py` |
| Qwen / Gemma LoRA fine-tuning | 否 | `tools/train_qwen35_vlm_unsloth.py`, `tools/train_gemma4_vlm_unsloth.py` |
| benchmark artifact pipeline | 否 | `tools/benchmark_qwen35_vlm_facts.py`, `benchmarks/qwen35_vlm_finetune/*` |
| 8-fact -> 13-fact ontology 演化 | 否 | `datasets/vision_sft/stats.json`, `packs/fa18c_startup/vision_facts.yaml` |

这部分是当前系统最重要的“actual contribution delta”。

## 4. 哪些内容被删除或弱化了

| Proposal 中的内容 | 当前仓库状态 | 说明 |
|---|---|---|
| Meta Quest 3 in-headset runtime | 未见已实现 runtime 证据 | 当前系统不是可直接描述成“已实现 VR tutor” |
| gaze / hand / voice multimodal interaction | 未见当前 runtime 代码证据 | 当前可确认的 help trigger 主要是 hotkey / trigger path |
| VR step cards as current output UI | 当前主输出是 DCS overlay / highlight path | `adapters/action_executor.py`, `DCS/Scripts/Hooks/VRHilite.lua` |
| 以 human-subject study 为主的 results 章节 | 尚未有结果数据 | 只能保留为 planned evaluation |
| 避免 heavy-weight runtime perception | 当前实际上新增了 runtime VLM path | proposal 与实现在这里发生了根本偏移 |

## 5. 需要在论文里重写的叙事

### 5.1 平台叙事必须重写

不能把当前系统写成：

- “我们已经实现了 Quest 3 VR tutoring runtime”

当前更准确的写法应是：

- 当前仓库实现了一个 desktop / DCS-centered tutoring runtime
- VR 与 human-subject evaluation 仍是 thesis-level target and planned next phase

### 5.2 方法叙事必须重写

proposal 的方法叙事是：

- RAG+LLM tutor
- read-only state flags
- VR delivery
- user study

当前仓库的方法叙事应改成：

- telemetry-grounded runtime
- pack-driven step logic
- optional VLM visual fact extraction
- LoRA adaptation experiments
- automated benchmark first
- planned runtime data collection and future human-subject evaluation second

### 5.3 结果叙事必须拆成两段

论文不能把“proposal 的验证目标”和“当前已经有的数据”混写。

推荐拆法：

1. **Current Technical Results**
   - 只写仓库里已有 benchmark、training artifact、data pipeline、runtime implementation
2. **Planned Runtime Data Collection & Human-Subject Evaluation**
   - 写 VR / user study / NASA-TLX / retention / learning gains 的实验框架与待收集数据

## 6. 当前最重要的新增主线：VLM adaptation

proposal 中没有 VLM adaptation 主线；但当前仓库里，这条线已经形成完整闭环：

```text
capture
-> VLM prelabel
-> human review
-> SFT export
-> LoRA fine-tuning
-> base-vs-LoRA benchmark
```

证据路径：

- `tools/capture_vlm_dataset.py`
- `tools/generate_vlm_prelabels.py`
- `tools/export_vision_sft_dataset.py`
- `tools/train_qwen35_vlm_unsloth.py`
- `tools/train_gemma4_vlm_unsloth.py`
- `tools/benchmark_qwen35_vlm_facts.py`
- `benchmarks/qwen35_vlm_finetune/run003_plus_run005x2_vs_base_holdout_run002_newfacts_v1/benchmark_summary.json`

这意味着论文至少要从“VR tutoring system only”改写成：

- a tutoring-system thesis with a substantial VLM adaptation and benchmarking contribution

## 7. 8-fact 到 13-fact 的 ontology 演化

proposal 里并没有当前这种显式 visual-fact ontology 设计。

当前仓库已经出现两代 ontology：

1. 8-fact v1：`datasets/vision_sft/stats.json`
2. 13-fact v2：`packs/fa18c_startup/vision_facts.yaml`

这会影响论文写法：

- 旧 8-fact 结果更适合写成 early baseline / historical experiment
- 新 13-fact 结果更适合写成 current runtime-aligned experimental line

## 8. 对最终论文 framing 的建议

### 8.1 可以保留的 thesis-level framing

以下内容可以继续作为论文主线保留：

- grounded procedural tutoring
- VR / immersive procedural learning motivation
- need for runtime data collection and human evaluation

### 8.2 不能越界的地方

以下内容不能写成已完成事实：

- 已完成 Quest 3 runtime deployment
- 已完成 VR participant study
- 已完成 NASA-TLX / retention results

### 8.3 建议的论文叙事结构

建议把论文叙事改成三层：

1. **Original thesis vision**
   - VR-first, human-evaluation-centered research agenda
2. **Current implemented system**
   - desktop/DCS runtime + telemetry-grounded help + VLM adaptation pipeline
3. **Next implementation phase**
   - runtime data collection completion + VR/human-subject evaluation

## 9. 一句话结论

proposal 没有失效，但已经不再等于当前系统。  
当前仓库更像是：**为最终 VR / 用户研究 thesis 主线铺设了一套已经具备技术证据的 runtime 与 VLM adaptation 基线**。

