"""
Runtime model-provider configuration for SimTutor.

This module keeps provider selection and env validation outside domain core so
provider migration (Ollama -> OpenAI-compatible) does not require core changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os,math
from typing import Mapping
from urllib.parse import SplitResult, urlsplit, urlunsplit


SUPPORTED_PROVIDERS = ("ollama", "openai_compat", "stub")
SUPPORTED_LANGS = ("zh", "en")

ENV_MODEL_PROVIDER = "SIMTUTOR_MODEL_PROVIDER"
ENV_MODEL_NAME = "SIMTUTOR_MODEL_NAME"
ENV_MODEL_BASE_URL = "SIMTUTOR_MODEL_BASE_URL"
ENV_MODEL_TIMEOUT_S = "SIMTUTOR_MODEL_TIMEOUT_S"
ENV_LANG = "SIMTUTOR_LANG"
ENV_MODEL_API_KEY = "SIMTUTOR_MODEL_API_KEY"


class ModelConfigError(ValueError):
    """Raised when required model access settings are missing or invalid."""


def _redact_url_for_log(url: str) -> str:
    value = url.strip()
    if not value:
        return value

    parsed = urlsplit(value)
    if not parsed.scheme and not parsed.netloc and "@" in value:
        # Support host forms without scheme, e.g. user:pass@host:port/path.
        parsed = urlsplit(f"//{value}")

    if not parsed.scheme and not parsed.netloc:
        return value

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    safe_netloc = host
    if parsed.port is not None:
        safe_netloc = f"{safe_netloc}:{parsed.port}" if safe_netloc else ""

    safe_parts = SplitResult(
        scheme=parsed.scheme,
        netloc=safe_netloc,
        path=parsed.path,
        query="",
        fragment="",
    )
    return urlunsplit(safe_parts)


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
            parts.append(f"base_url={_redact_url_for_log(self.base_url)}")
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


def load_model_access_config(env: Mapping[str, str] | None = None) -> ModelAccessConfig:
    source = os.environ if env is None else env

    provider = _required_env(source, ENV_MODEL_PROVIDER)
    if provider not in SUPPORTED_PROVIDERS:
        allowed = "|".join(SUPPORTED_PROVIDERS)
        raise ModelConfigError(
            f"Invalid {ENV_MODEL_PROVIDER}: {provider!r}. Allowed values: {allowed}"
        )

    model_name = _required_env(source, ENV_MODEL_NAME)
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

    return ModelAccessConfig(
        provider=provider,
        model_name=model_name,
        timeout_s=timeout_s,
        lang=lang,
        base_url=base_url,
        api_key=api_key,
    )


__all__ = [
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
