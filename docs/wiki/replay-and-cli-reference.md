# Replay and CLI Reference

This page keeps frequently used CLI examples that used to live in the front-page README.

## Top-Level Entrypoints

Linux/WSL:

```bash
python -m simtutor --help
python live_dcs.py --help
python -m tools.capture_vlm_dataset --help
python -m tools.generate_vlm_prelabels --help
```

## Replay With Vision Sidecar Frames

Linux/WSL:

```bash
python -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --auto-help-once \
  --vision-saved-games-dir "<saved-games-dir>" \
  --vision-session-id sess-replay \
  --dry-run-overlay
```

## Replay With a Local OpenAI-Compatible Endpoint

Linux/WSL:

```bash
export SIMTUTOR_MODEL_PROVIDER=openai_compat
export SIMTUTOR_MODEL_BASE_URL=http://127.0.0.1:8000
export SIMTUTOR_MODEL_NAME=Qwen3-8B-Instruct
export SIMTUTOR_MODEL_ENABLE_MULTIMODAL=0
export SIMTUTOR_MODEL_TIMEOUT_S=20
export SIMTUTOR_MODEL_API_KEY=dummy
export SIMTUTOR_LANG=zh

python -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --model-provider openai_compat \
  --auto-help-once \
  --stdin-help
```

## Common Environment Variables

| Variable | Purpose |
|---|---|
| `SIMTUTOR_MODEL_PROVIDER` | `stub`, `openai_compat`, or `ollama` |
| `SIMTUTOR_MODEL_NAME` | Model identifier |
| `SIMTUTOR_MODEL_BASE_URL` | OpenAI-compatible or Ollama base URL |
| `SIMTUTOR_MODEL_ENABLE_MULTIMODAL` | Enables multimodal model input where supported |
| `SIMTUTOR_MODEL_TIMEOUT_S` | Model timeout in seconds |
| `SIMTUTOR_MODEL_API_KEY` | Provider API key or local dummy token |
| `SIMTUTOR_LANG` | `zh` or `en` |
| `SIMTUTOR_COLD_START_PRODUCTION` | Cold-start production-mode switch |
| `SIMTUTOR_LOG_RAW_LLM_TEXT` | Raw model text logging for debugging |
| `SIMTUTOR_PRINT_MODEL_IO` | Terminal prompt/reply debug printing |

## DCS-BIOS Replay

Linux/WSL:

```bash
python -m simtutor replay-bios \
  --input logs/dcs_bios_raw.jsonl \
  --speed 1.0 \
  --pack packs/fa18c_startup/pack.yaml \
  --ui-map packs/fa18c_startup/ui_map.yaml \
  --telemetry-map packs/fa18c_startup/telemetry_map.yaml \
  --bios-to-ui packs/fa18c_startup/bios_to_ui.yaml \
  --knowledge-index Doc/Evaluation/index.json \
  --auto-help-once \
  --stdin-help \
  --dry-run-overlay
```

Use `--print-model-io` only when you intentionally want to inspect full prompts and decoded model replies.

## Extract a Live Help Fixture

When a live run fails, copy the reported `request_id` or `help_cycle_id` and extract a compact replay/debug fixture from the JSONL runtime log:

```bash
python -m simtutor extract-live-fixture \
  --input logs/live_dcs.jsonl \
  --request-id "<request-id>" \
  --output-dir artifacts/live_fixtures
```

This writes:

```text
artifacts/live_fixtures/<request-id>.fixture.json
```

Use `--output` instead of `--output-dir` when you want an exact file path:

```bash
python -m simtutor extract-live-fixture \
  --input logs/live_dcs.jsonl \
  --help-cycle-id "<help-cycle-id>" \
  --output artifacts/live_fixtures/s19_failure.fixture.json
```

The generated fixture has schema version `live_help_replay_fixture.v1` and is intended for regression-test authoring and failure inspection. Important top-level sections:

| Field | Purpose |
|---|---|
| `extraction_id` | The id supplied on the CLI. |
| `request_id` | The real request id from the log when available. |
| `help_cycle_id` | The normalized live help cycle id. |
| `source_log` | Source path, event count, and malformed JSONL lines skipped during extraction. |
| `cycle.tutor_request` | Sanitized tutor request payload for the selected help cycle. |
| `cycle.tutor_response` | Sanitized tutor response payload for the selected help cycle. |
| `cycle.events` | Compact cycle-related event bundle, including overlay events. |
| `context.observations` | The request observation or nearby observation frames. |
| `context.vision` | Vision frame ids, fact summary, and response-side vision metadata. |
| `context.vision_fact_observations` | Matching raw/parsed vision fact observation events when present. |
| `context.evidence_packet_summary` | Compact EvidencePacket summary from request context or harness trace. |
| `context.telemetry_window_digest` | Telemetry window digest when present in the evidence summary. |
| `model_io` | Logged model response objects and raw-text presence metadata. |
| `harness` | Candidate steps, harness trace, model decision, validator result, repair result, and final action plan. |
| `expectations` | Stable assertion seeds: final step id, overlay target ids, VLM call status, LLM decision status, and final response source. |

Typical regression loop:

```text
live request_id -> extract fixture -> inspect expectations/harness trace -> write replay/eval assertion -> fix -> keep fixture with test
```

Malformed JSONL lines do not stop extraction; they are recorded under `source_log.malformed_lines`. If the id is not found, the command exits with an `[EXTRACT_LIVE_FIXTURE] error:` message.

## Experiment Export and Analysis

From a raw live/replay JSONL log, first freeze one participant trial into study-ready CSV artifacts:

```bash
python -m simtutor experiment-export logs/live_dcs.jsonl \
  --study-id fa18c-thesis-pilot \
  --participant-id P01 \
  --condition with_tutor \
  --trial-id T01 \
  --group novice \
  --experimenter-id E01 \
  --questionnaire questionnaires/P01_pre.yml \
  --recording-ref recordings/P01_T01.mp4 \
  --model-provider openai_compat \
  --model-name Qwen3-8B-Instruct \
  --vision-model-name Qwen3.5-VL-9B \
  --prompt-version v0.4 \
  --scenario-profile airfield \
  --dcs-mission cold-start.miz \
  --dcs-aircraft FA-18C \
  --monitor-setup native-viewports \
  --output-dir artifacts/experiments \
  --strict
```

Repeat `experiment-export` for each participant/trial. Then combine all exported participant/trial folders into thesis-ready summary tables and optional figures:

```bash
python -m simtutor experiment-analyze artifacts/experiments \
  --output-dir artifacts/analysis/fa18c-thesis-pilot
```

This writes:

```text
artifacts/analysis/fa18c-thesis-pilot/study_summary.csv
artifacts/analysis/fa18c-thesis-pilot/condition_summary.csv
artifacts/analysis/fa18c-thesis-pilot/step_accuracy_by_condition.csv
artifacts/analysis/fa18c-thesis-pilot/help_quality_summary.csv
artifacts/analysis/fa18c-thesis-pilot/fig_completion_rate.png
artifacts/analysis/fa18c-thesis-pilot/fig_task_time.png
artifacts/analysis/fa18c-thesis-pilot/fig_step_accuracy.png
artifacts/analysis/fa18c-thesis-pilot/fig_help_requests.png
```

If `matplotlib` is not installed, the CSV files are still written and the command reports that figures were skipped. Use `--no-figures` when only tables are needed:

```bash
python -m simtutor experiment-analyze artifacts/experiments \
  --output-dir artifacts/analysis/fa18c-thesis-pilot \
  --no-figures
```
