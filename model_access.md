# Model Provider Access (v0.3)

This document freezes model-provider access in a provider-agnostic way so core
logic does not change when switching runtime backends.

## Provider A (Primary): OpenAI-compatible API (Qwen3 Focus)

- Use case: vLLM / llama.cpp server / TGI, all through OpenAI-compatible APIs.
- Provider value: `SIMTUTOR_MODEL_PROVIDER=openai_compat`
- Base URL required: `SIMTUTOR_MODEL_BASE_URL` (example: `http://127.0.0.1:8000`)
- API key required: `SIMTUTOR_MODEL_API_KEY`
- Default model if `SIMTUTOR_MODEL_NAME` is omitted: `Qwen3-8B-Instruct`
- Set `SIMTUTOR_MODEL_ENABLE_MULTIMODAL=1` (or CLI `--model-enable-multimodal`) to allow synchronized vision frames to be sent as OpenAI-compatible `image_url` content for Qwen3.5 VLM.
- Only local frame files under the configured `Saved Games/.../SimTutor/frames` root are accepted for multimodal input; remote `http(s)` URLs and inline `data:` URLs are rejected, and local files are guarded by a max file-size limit before base64 encoding.
- Note: this repository appends `/v1/chat/completions` itself, so `SIMTUTOR_MODEL_BASE_URL` should be the provider root rather than an OpenAI SDK-style `/v1` base URL.

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
- `SIMTUTOR_MODEL_ENABLE_MULTIMODAL` (optional boolean; default off, enable only for VLM-capable `openai_compat` models)
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

