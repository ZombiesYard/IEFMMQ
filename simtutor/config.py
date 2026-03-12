"""
Runtime model-provider configuration for SimTutor.

This module keeps provider selection and env validation outside domain core so
provider migration (Ollama -> OpenAI-compatible) does not require core changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Mapping

from core.security import redact_url_for_log, validate_model_base_url_security


SUPPORTED_PROVIDERS = ("openai_compat", "stub", "ollama")
SUPPORTED_LANGS = ("zh", "en")

DEFAULT_MODEL_NAME_OPENAI_COMPAT = "Qwen3-8B-Instruct"
DEFAULT_MODEL_NAME_OLLAMA = "qwen3:8b"
DEFAULT_MODEL_NAME_STUB = "qwen3-stub"

ENV_MODEL_PROVIDER = "SIMTUTOR_MODEL_PROVIDER"
ENV_MODEL_NAME = "SIMTUTOR_MODEL_NAME"
ENV_MODEL_BASE_URL = "SIMTUTOR_MODEL_BASE_URL"
ENV_MODEL_TIMEOUT_S = "SIMTUTOR_MODEL_TIMEOUT_S"
ENV_LANG = "SIMTUTOR_LANG"
ENV_MODEL_API_KEY = "SIMTUTOR_MODEL_API_KEY"


class ModelConfigError(ValueError):
    """Raised when required model access settings are missing or invalid."""


@dataclass(frozen=True)
class ModelAccessConfig:
    provider: str
    model_name: str
    timeout_s: float
    lang: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)

    def public_startup_info(self) -> str:
        parts = [
            f"provider={self.provider}",
            f"model={self.model_name}",
            f"timeout_s={self.timeout_s:g}",
            f"lang={self.lang}",
        ]
        if self.base_url:
            parts.append(f"base_url={redact_url_for_log(self.base_url)}")
        return " ".join(parts)


def _required_env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if value is None or not value.strip():
        raise ModelConfigError(f"Missing required env: {name}")
    return value.strip()


def _optional_env(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _default_model_for_provider(provider: str) -> str:
    if provider == "openai_compat":
        return DEFAULT_MODEL_NAME_OPENAI_COMPAT
    if provider == "ollama":
        return DEFAULT_MODEL_NAME_OLLAMA
    return DEFAULT_MODEL_NAME_STUB


def load_model_access_config(env: Mapping[str, str] | None = None) -> ModelAccessConfig:
    source = os.environ if env is None else env

    provider = _required_env(source, ENV_MODEL_PROVIDER)
    if provider not in SUPPORTED_PROVIDERS:
        allowed = "|".join(SUPPORTED_PROVIDERS)
        raise ModelConfigError(
            f"Invalid {ENV_MODEL_PROVIDER}: {provider!r}. Allowed values: {allowed}"
        )

    model_name = _optional_env(source, ENV_MODEL_NAME) or _default_model_for_provider(provider)
    timeout_raw = _required_env(source, ENV_MODEL_TIMEOUT_S)
    lang = _required_env(source, ENV_LANG)
    if lang not in SUPPORTED_LANGS:
        allowed = "|".join(SUPPORTED_LANGS)
        raise ModelConfigError(f"Invalid {ENV_LANG}: {lang!r}. Allowed values: {allowed}")

    try:
        timeout_s = float(timeout_raw)
    except ValueError as exc:
        raise ModelConfigError(
            f"Invalid {ENV_MODEL_TIMEOUT_S}: {timeout_raw!r}. Must be a positive number."
        ) from exc
    if timeout_s <= 0:
        raise ModelConfigError(
            f"Invalid {ENV_MODEL_TIMEOUT_S}: {timeout_raw!r}. Must be a positive number."
        )
    if math.isnan(timeout_s):
        raise ModelConfigError(
            f"Invalid {ENV_MODEL_TIMEOUT_S}: {timeout_raw!r}. Must be a positive number."
        )
    if math.isinf(timeout_s):
        raise ModelConfigError(
            f"Invalid {ENV_MODEL_TIMEOUT_S}: {timeout_raw!r}. Must be a positive number."
        )

    base_url = _optional_env(source, ENV_MODEL_BASE_URL)
    api_key: str | None = None

    if provider == "ollama":
        base_url = base_url or "http://127.0.0.1:11434"
    elif provider == "openai_compat":
        if not base_url:
            raise ModelConfigError(
                f"Missing required env for provider=openai_compat: {ENV_MODEL_BASE_URL}"
            )
        api_key = _required_env(source, ENV_MODEL_API_KEY)
    else:
        # Stub mode does not use remote model access settings; ignore leftover env
        # values so unrelated provider config does not break deterministic runs.
        base_url = None

    if provider in {"openai_compat", "ollama"} and base_url:
        try:
            validate_model_base_url_security(base_url, provider=provider)
        except ValueError as exc:
            raise ModelConfigError(str(exc)) from exc

    return ModelAccessConfig(
        provider=provider,
        model_name=model_name,
        timeout_s=timeout_s,
        lang=lang,
        base_url=base_url,
        api_key=api_key,
    )


__all__ = [
    "DEFAULT_MODEL_NAME_OPENAI_COMPAT",
    "DEFAULT_MODEL_NAME_OLLAMA",
    "DEFAULT_MODEL_NAME_STUB",
    "ENV_LANG",
    "ENV_MODEL_API_KEY",
    "ENV_MODEL_BASE_URL",
    "ENV_MODEL_NAME",
    "ENV_MODEL_PROVIDER",
    "ENV_MODEL_TIMEOUT_S",
    "ModelAccessConfig",
    "ModelConfigError",
    "SUPPORTED_LANGS",
    "SUPPORTED_PROVIDERS",
    "load_model_access_config",
]
