# Model Provider Access (v0.3)

This document freezes model-provider access in a provider-agnostic way so core
logic does not change when switching runtime backends.

## Provider A (Default for v0.3): Ollama

- Use case: zero-cost local loop validation.
- Provider value: `SIMTUTOR_MODEL_PROVIDER=ollama`
- Typical local endpoint: `http://127.0.0.1:11434`
- API key: not required.

## Provider B (Migration Target): OpenAI-compatible API

- Use case: vLLM / llama.cpp server / TGI, all through OpenAI-compatible APIs.
- Provider value: `SIMTUTOR_MODEL_PROVIDER=openai_compat`
- Base URL required: `SIMTUTOR_MODEL_BASE_URL` (example: `http://127.0.0.1:8000/v1`)
- API key required: `SIMTUTOR_MODEL_API_KEY`

## Stub Provider

- Provider value: `SIMTUTOR_MODEL_PROVIDER=stub`
- Purpose: deterministic fallback/testing without external model service.

## Unified Environment Variables

- `SIMTUTOR_MODEL_PROVIDER=ollama|openai_compat|stub`
- `SIMTUTOR_MODEL_NAME` (Ollama tag or OpenAI-compatible model name)
- `SIMTUTOR_MODEL_BASE_URL` (required for `openai_compat`, optional for `ollama`)
- `SIMTUTOR_MODEL_TIMEOUT_S` (positive number)
- `SIMTUTOR_LANG=zh|en`
- `SIMTUTOR_MODEL_API_KEY` (required for `openai_compat`)

## Runtime Validation and Startup Output

- Missing required env values raise clear errors with env variable names.
- Startup output must be non-sensitive and should include:
  - provider
  - model name
  - timeout
  - language
  - base URL (if available)
- API keys must never appear in logs or exception messages.

