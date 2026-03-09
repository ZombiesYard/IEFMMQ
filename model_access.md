# Model Provider Access (v0.3)

This document freezes model-provider access in a provider-agnostic way so core
logic does not change when switching runtime backends.

## Provider A (Primary): OpenAI-compatible API (Qwen3 Focus)

- Use case: vLLM / llama.cpp server / TGI, all through OpenAI-compatible APIs.
- Provider value: `SIMTUTOR_MODEL_PROVIDER=openai_compat`
- Base URL required: `SIMTUTOR_MODEL_BASE_URL` (example: `http://127.0.0.1:8000/v1`)
- API key required: `SIMTUTOR_MODEL_API_KEY`
- Default model if `SIMTUTOR_MODEL_NAME` is omitted: `Qwen3-8B-Instruct`
- When `request.context["vision"]` carries synchronized frame artifacts, `openai_compat` can send them as OpenAI-compatible `image_url` content for Qwen3.5 VLM and automatically falls back to text-only if the multimodal path fails.

## Provider B (Fallback): Ollama

- Use case: local fallback / compatibility.
- Provider value: `SIMTUTOR_MODEL_PROVIDER=ollama`
- Typical local endpoint: `http://127.0.0.1:11434`
- API key: not required.
- Default model if `SIMTUTOR_MODEL_NAME` is omitted: `qwen3:8b`

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

