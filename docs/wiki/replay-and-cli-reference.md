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
