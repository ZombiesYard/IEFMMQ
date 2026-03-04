# SimTutor Core (IEFMMQ)

SimTutor Core is a tutoring backend for the DCS F/A-18C cold-start training scenario.
The project uses a clean "core + ports + adapters" architecture and supports:
- mock scenario simulation and scoring
- live/replay DCS-BIOS tutoring loop
- overlay-only (highlight) execution safety boundary
- local BM25 retrieval with cold-start knowledge source policy

## What Works Now

- `simtutor` CLI: `run / replay / score / batch / validate / replay-bios / model-config`
- `live_dcs.py`: end-to-end runtime (`bios -> enrich -> help -> response mapping -> overlay`)
- model providers: `stub`, `openai_compat`, `ollama`
- JSONL event logs for replay, scoring, and schema validation
- DCS utilities for indexing, hook installation, telemetry/BIOS listening, and recording

## Repository Layout

- `core/`: domain engines (`procedure`, `gating`, `scoring`, `overlay`, `knowledge`, `types`)
- `ports/`: model/knowledge/telemetry interfaces
- `adapters/`: model adapters, DCS adapters, event writers, response mapping
- `simtutor/`: CLI entrypoint and schemas
- `packs/fa18c_startup/`: pack config (`pack`, `taxonomy`, `ui_map`, `telemetry_map`)
- `mock_scenarios/`: offline scenario inputs
- `tools/`: indexing, hook install, listener/recorder tools
- `Doc/Evaluation/`: source training and evaluation documents

## Requirements

- Python `3.10+`
- Run commands from repo root
- Commands below use `python3` (many environments do not provide a `python` alias)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
```

## Quickstart (Offline Mock)

1. Build knowledge index (default input is `Doc/Evaluation`):

```bash
python3 -m tools.index_docs --output Doc/Evaluation/index.json
```

2. Run one mock scenario:

```bash
python3 -m simtutor run \
  --pack packs/fa18c_startup/pack.yaml \
  --scenario mock_scenarios/correct_process.json \
  --output logs/run_demo.jsonl
```

3. Replay trajectory:

```bash
python3 -m simtutor replay logs/run_demo.jsonl --pack packs/fa18c_startup/pack.yaml
```

4. Score the run:

```bash
python3 -m simtutor score \
  logs/run_demo.jsonl \
  --pack packs/fa18c_startup/pack.yaml \
  --taxonomy packs/fa18c_startup/taxonomy.yaml
```

5. Batch scenarios and export CSV:

```bash
python3 -m simtutor batch \
  --pack packs/fa18c_startup/pack.yaml \
  --taxonomy packs/fa18c_startup/taxonomy.yaml \
  --output-dir artifacts
```

Output: `artifacts/results.csv`

## CLI Overview

```bash
python3 -m simtutor -h
```

Subcommands:
- `validate`: validate JSONL logs against schema
- `run`: run one mock scenario
- `replay`: replay event logs or telemetry logs
- `score`: score a run with taxonomy
- `batch`: run multiple scenarios and export results
- `model-config`: validate model env configuration (non-sensitive output)
- `replay-bios`: replay DCS-BIOS JSONL through the live tutor pipeline

## Model Providers

Supported providers:
- `openai_compat` (vLLM / llama.cpp / TGI via OpenAI-compatible APIs)
- `ollama`
- `stub` (deterministic local fallback for testing)

Main env vars:
- `SIMTUTOR_MODEL_PROVIDER=ollama|openai_compat|stub`
- `SIMTUTOR_MODEL_NAME`
- `SIMTUTOR_MODEL_BASE_URL`
- `SIMTUTOR_MODEL_TIMEOUT_S`
- `SIMTUTOR_LANG=zh|en`
- `SIMTUTOR_MODEL_API_KEY` (`openai_compat` required)

Minimal local stub setup:

```bash
export SIMTUTOR_MODEL_PROVIDER=stub
export SIMTUTOR_MODEL_TIMEOUT_S=20
export SIMTUTOR_LANG=zh
```

Check current model config:

```bash
python3 -m simtutor model-config
```

## Live DCS Tutor Loop

### Replay BIOS First (recommended)

`replay-bios` runs in safe mode by default (`--dry-run-overlay` enabled).

```bash
python3 -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --model-provider stub \
  --auto-help-once \
  --output logs/replay_bios_demo.jsonl
```

Common options:
- `--speed 1.0`: realtime pacing, `--speed 0`: max speed
- `--no-dry-run-overlay`: actually send overlay commands
- `--stdin-help`: press Enter to trigger help
- `--help-udp-port <port>`: enable UDP help trigger

### Live DCS-BIOS UDP

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

Notes:
- `live_dcs.py` does not enable `--dry-run-overlay` by default.
- In cold-start production mode, a valid knowledge source policy is mandatory.
- In PowerShell, use `$env:SIMTUTOR_MODEL_API_KEY="..."`.

## Knowledge Index and Source Policy

### Index

- Default index path: `Doc/Evaluation/index.json`
- Rebuild with `tools.index_docs`

### Knowledge Source Policy (`knowledge_source_policy.yaml`)

- Flags: `--cold-start-production / --no-cold-start-production`
- Env default: `SIMTUTOR_COLD_START_PRODUCTION=1|0`
- `--knowledge-source-policy <path>` enables whitelist filtering in any mode
- In cold-start production mode:
  - if no path is provided, runtime falls back to repo `knowledge_source_policy.yaml`
  - invalid/missing policy fails fast at startup
- `allow[].line_range` is enforced at runtime by clipping snippet lines

## DCS Tools

### Install DCS Hook

```bash
python3 -m tools.install_dcs_hook --dcs-variant DCS
```

Optional:
- `--saved-games <path>`
- `--no-export` (skip `Export.lua` patch)

### Telemetry Listen/Record

```bash
python3 -m tools.listen_dcs_telemetry --host 0.0.0.0 --port 7780
python3 -m tools.record_dcs_telemetry --output logs/dcs_telemetry.jsonl --duration 30 --print
```

### Send Fake Telemetry

```bash
python3 -m tools.send_fake_dcs_telemetry --host 127.0.0.1 --port 7780 --count 20 --hz 20
```

### Decode Raw DCS-BIOS Stream

```bash
python3 -m tools.listen_dcs_bios_raw --aircraft FA-18C_hornet
```

One-shot fuller snapshot:

```bash
python3 -m tools.listen_dcs_bios_raw \
  --aircraft FA-18C_hornet \
  --once --wait 15 --min-keys 500 \
  --output artifacts/dcs_bios_frame_once.json
```

## Schema Validation

```bash
python3 -m simtutor validate logs/run_demo.jsonl --schema event
```

Available schema names:
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

## Tests

```bash
python3 -m pytest -q
```

## Troubleshooting

- `python: command not found`
  - Use `python3`.

- `model-config` shows `Missing required env: SIMTUTOR_MODEL_PROVIDER`
  - Export required model env vars first.

- cold-start policy startup errors in live/replay modes
  - Verify `knowledge_source_policy.yaml` matches `Doc/Evaluation/index.json`
    (`doc_id/chunk_id/line_range`).

## References

- `READMEZH.md` (Chinese README)
- `model_access.md`
- `help_flow.md`
- `help_flow_en.md`
Replay BIOS via `simtutor` CLI (default safe mode with dry-run overlay):
```sh
python -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --speed 1.0 \
  --help-udp-port 7794 \
  --model-provider stub \
  --output logs/replay_bios.jsonl
```
Notes:
- `--speed 1.0`: realtime pacing by frame `t_wall`; `--speed 0`: max speed.
- `replay-bios` defaults to `--dry-run-overlay`; use `--no-dry-run-overlay` only when you really want to send overlay commands.

Grounding metadata (in `tutor_request` / `tutor_response.payload.metadata`):
- `grounding_snippet_ids`: snippet ids actually injected into prompt.
- `grounding_missing`: `true` when no retrieval grounding is applied (e.g., index unavailable, RAG disabled via `rag_top_k<=0`, or retrieval error); flow degrades safely without crash.
- `context.gates`: deterministic gate results (`allowed|blocked`, `reason_code`, `reason`); valid gate evidence refs are exactly `GATES.<gate_id>` where `<gate_id>` is a key in `context.gates` (for example `GATES.S05.precondition`).

### Overlay action evidence protocol
Evidence protocol hard gate: overlay actions are rejected and logged in response metadata if any target lacks verifiable `overlay.evidence` refs, or any evidence item is malformed, type/ref mismatched, or cites unknown refs (allowed prefixes: `VARS.*` / `GATES.*` / `RECENT_UI_TARGETS.*` / `DELTA_KEYS.*` / `RAG_SNIPPETS.*`).

## Source Documents (authoritative)
- `Doc/Evaluation/fa18c_startup_master.md`
- `Doc/Evaluation/Appendix - Training Task Syllabus.md`
- `Doc/Evaluation/fa18c_error_coding_guide.md`
- `Doc/Evaluation/fa18c_scoring_sheet_template.md`
- `Doc/Evaluation/fa18c_coldstart_quiz.md`
- `Doc/Evaluation/fa18c_nasatlx_vr.md`
