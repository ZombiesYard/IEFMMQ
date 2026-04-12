# SimTutor

[![Build Status](https://github.com/ZombiesYard/IEFMMQ/actions/workflows/ci-guard.yml/badge.svg)](https://github.com/ZombiesYard/IEFMMQ/actions/workflows/ci-guard.yml)
[![Code Coverage](https://codecov.io/gh/ZombiesYard/IEFMMQ/branch/main/graph/badge.svg)](https://codecov.io/gh/ZombiesYard/IEFMMQ)
[![Last Commit](https://img.shields.io/github/last-commit/ZombiesYard/IEFMMQ?logo=github)](https://github.com/ZombiesYard/IEFMMQ/commits/main)
[![Documentation](https://img.shields.io/badge/docs-wiki-0EA5E9?logo=readthedocs&logoColor=white)](https://github.com/ZombiesYard/IEFMMQ/tree/main/docs/wiki)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Lua](https://img.shields.io/badge/Lua-hooks-2C2D72?logo=lua&logoColor=white)](https://www.lua.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Qwen3.5](https://img.shields.io/badge/base-Qwen3.5--9B-111827)](https://huggingface.co/Qwen)
[![VLM](https://img.shields.io/badge/task-vision--language%20facts-06B6D4)](https://huggingface.co/docs/transformers/tasks/image_text_to_text)
[![LoRA](https://img.shields.io/badge/fine--tuning-LoRA-F97316)](https://arxiv.org/abs/2106.09685)
[![Unsloth](https://img.shields.io/badge/training-Unsloth-111827)](https://unsloth.ai/)
[![PEFT](https://img.shields.io/badge/adapter-PEFT-7C3AED)](https://huggingface.co/docs/peft/)
[![TRL](https://img.shields.io/badge/SFT-TRL-10B981)](https://huggingface.co/docs/trl/)
[![Label Studio](https://img.shields.io/badge/annotation-Label%20Studio-FF5C35)](https://labelstud.io/)
[![JSONL](https://img.shields.io/badge/data-JSONL-64748B)](https://jsonlines.org/)

SimTutor is a research prototype for **structured cockpit visual fact extraction** and tutoring-oriented state reasoning in the DCS F/A-18C cold-start task. The project combines a simulator-facing telemetry/vision pipeline, replayable tutoring logic, human-reviewed VLM datasets, and a fine-tuned Qwen3.5 LoRA adapter for extracting cockpit visual facts from composite display screenshots.

The repository provides a fine-tuned **LoRA adapter**, datasets, benchmark artifacts, and reproducible tools. It **does not redistribute `Qwen/Qwen3.5-9B-Base`**. The base model must be obtained separately under its upstream terms.

## What This Repository Contains

- A simulator-agnostic tutoring core with clean architecture boundaries across `core/`, `ports/`, and `adapters/`.
- DCS-facing Lua/Python integration for telemetry, overlays, viewport capture, replay, and visual sidecars.
- A cockpit VLM data pipeline: screenshot capture, Qwen pre-labeling, Label Studio review, SFT export, LoRA fine-tuning, and benchmark.
- A Qwen3.5-9B LoRA adapter trained for structured cockpit visual fact extraction.
- Academic-style reports and benchmark charts for the current fine-tuning experiment.

## Architecture

```text
DCS cockpit state
  ├─ DCS-BIOS telemetry stream
  └─ composite-panel screenshot
        ↓
SimTutor adapters
  ├─ telemetry normalization
  ├─ VLM-ready frame rendering
  └─ visual fact extraction
        ↓
Core reasoning
  ├─ procedure pack
  ├─ state/fact interpretation
  └─ tutor response planning
        ↓
Replay logs, benchmark artifacts, and optional simulator overlay output
```

The VLM does not produce final tutoring decisions directly. It produces structured visual facts that downstream reasoning can inspect, replay, score, and combine with telemetry.

## Cockpit Visual Facts

In this project, a visual `fact` is a structured proposition extracted from one cockpit panel image. It is an intermediate representation between VLM perception and downstream task reasoning.

Each fact has one of:

- `seen`
- `not_seen`
- `uncertain`

The current first-pass ontology contains eight facts:

| fact_id | Plain meaning |
|---|---|
| `fcs_page_visible` | FCS page is visible |
| `bit_root_page_visible` | BIT root/menu page is visible |
| `bit_page_failure_visible` | BIT failure list page is visible |
| `right_ddi_fcsmc_page_visible` | Right DDI shows the FCS-MC page |
| `right_ddi_in_test_visible` | Right DDI clearly shows an IN TEST state |
| `fcs_bit_result_visible` | Final FCS BIT result is visible |
| `ins_alignment_page_visible` | AMPCD shows the INS alignment page |
| `ins_go` | INS GO indication is visible |

The reports discuss why this ontology is useful for a first controlled experiment, and why `ins_go` should be decomposed into lower-level visual evidence in future work.

## Fine-Tuned LoRA Adapter

Provided artifact:

```text
models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/adapter
```

Base model:

```text
Qwen/Qwen3.5-9B-Base
```

Important: this repository provides the **adapter only**. It does not include the Qwen base weights.

Training stack:

```text
Unsloth + PEFT LoRA + TRL SFTTrainer
```

Quick loading sketch:

```python
from peft import PeftModel
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B-Base",
    max_seq_length=4096,
    load_in_4bit=True,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(
    model,
    "models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/adapter",
)
model.eval()
```

See the [adapter model card](models/qwen35_vlm_lora/full_qwen35_9b_base_bilingual_v1/README.md) for training details.

## Current Benchmark Snapshot

Run-002 is an independent heldout session not used for LoRA-v1 training.

| Metric | Base | LoRA-v1 |
|---|---:|---:|
| JSON valid rate | 1.0000 | 1.0000 |
| Schema valid rate | 1.0000 | 1.0000 |
| Fact accuracy | 0.7600 | 0.9150 |
| Seen F1 | 0.4714 | 0.8380 |
| Sample exact match | 0.0600 | 0.3600 |
| Critical false positives | 13 | 15 |

The adapter improves most facts substantially, but `ins_go` shows a concentrated false-positive failure mode. This is documented as a target-design issue rather than a solved task.

Artifacts:

- [English technical report](Doc/Vision/Reports/qwen35_vlm_finetune_report_EN.md)
- [Chinese technical report](Doc/Vision/Reports/qwen35_vlm_finetune_report_ZH.md)
- [Benchmark artifacts](benchmarks/qwen35_vlm_finetune/)
- [Run-001 SFT dataset](datasets/vision_sft/)
- [Run-002 holdout dataset](datasets/vision_sft_holdout_run002/)

## Quick Start

### Linux / WSL

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest -q
```

Run a deterministic mock scenario:

```bash
python -m tools.index_docs --output Doc/Evaluation/index.json
python -m simtutor run \
  --pack packs/fa18c_startup/pack.yaml \
  --scenario mock_scenarios/correct_process.json \
  --output logs/run_demo.jsonl
```

Replay and score:

```bash
python -m simtutor replay logs/run_demo.jsonl --pack packs/fa18c_startup/pack.yaml
python -m simtutor score logs/run_demo.jsonl \
  --pack packs/fa18c_startup/pack.yaml \
  --taxonomy packs/fa18c_startup/taxonomy.yaml
```

## Documentation

Detailed commands are intentionally kept out of the front page:

- [Wiki index](docs/wiki/README.md)
- [Quickstart](docs/wiki/quickstart.md)
- [DCS live operation](docs/wiki/dcs-live-operation.md)
- [VLM dataset pipeline](docs/wiki/vlm-dataset-pipeline.md)
- [Qwen3.5 VLM fine-tuning](docs/wiki/fine-tuning-qwen35-vlm.md)
- [Benchmarking](docs/wiki/benchmarking.md)
- [Replay and CLI reference](docs/wiki/replay-and-cli-reference.md)
- [Artifacts and large files](docs/wiki/artifacts-and-large-files.md)

## Repository Map

| Path | Purpose |
|---|---|
| `core/` | Domain logic, procedure engine, scoring, gating, overlay planning |
| `ports/` | Stable interfaces for model, knowledge, telemetry, and vision |
| `adapters/` | Simulator, model-provider, DCS-BIOS, and VLM adapters |
| `simtutor/` | CLI entrypoints, runtime config, schema registry |
| `packs/fa18c_startup/` | F/A-18C cold-start procedure pack and mappings |
| `DCS/Scripts/` | Lua-side simulator integration scripts |
| `tools/` | Capture, annotation, SFT export, training, benchmark, and install utilities |
| `datasets/` | Reviewed visual fact datasets and SFT JSONL exports |
| `benchmarks/` | Base-vs-LoRA benchmark outputs and charts |
| `models/` | LoRA adapter artifacts and model cards |
| `Doc/Vision/Reports/` | Academic reports and chart assets |

## Licensing Notes

Code in this repository follows the [Apache-2.0 license](LICENSE).

`Qwen/Qwen3.5-9B-Base` is not redistributed here and remains governed by its upstream license and model terms. Reuse of the LoRA adapter should consider both this repository license and the upstream base-model terms.
