# Quickstart

This page covers local setup and deterministic replay. Commands are for Linux/WSL unless marked otherwise.

## Install

```bash
cd <repo>
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run Tests

```bash
python -m pytest -q
```

## Build the Grounding Index

```bash
python -m tools.index_docs --output Doc/Evaluation/index.json
```

## Run a Mock Scenario

```bash
python -m simtutor run \
  --pack packs/fa18c_startup/pack.yaml \
  --scenario mock_scenarios/correct_process.json \
  --output logs/run_demo.jsonl
```

## Replay and Score

```bash
python -m simtutor replay logs/run_demo.jsonl --pack packs/fa18c_startup/pack.yaml
python -m simtutor score logs/run_demo.jsonl \
  --pack packs/fa18c_startup/pack.yaml \
  --taxonomy packs/fa18c_startup/taxonomy.yaml
```

## Replay DCS-BIOS Offline

Offline replay is the safest way to iterate because overlay commands can remain dry-run.

```bash
python -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --auto-help-once \
  --stdin-help \
  --dry-run-overlay
```

## OpenAI-Compatible Model Configuration

```bash
export SIMTUTOR_MODEL_PROVIDER=openai_compat
export SIMTUTOR_MODEL_BASE_URL=http://127.0.0.1:8000
export SIMTUTOR_MODEL_NAME=Qwen3-8B-Instruct
export SIMTUTOR_MODEL_TIMEOUT_S=20
export SIMTUTOR_MODEL_API_KEY=dummy
export SIMTUTOR_LANG=zh
```

Validate without printing secrets:

```bash
python -m simtutor model-config
```
