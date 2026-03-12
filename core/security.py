"""
Security helpers for transport validation and log-safe redaction.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Mapping
from urllib.parse import SplitResult, urlsplit, urlunsplit


_ABS_WIN_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/][^\s,;]+)")
_ABS_POSIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])(/(?:[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*))")
_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_HOST_PORT_RE = re.compile(
    r"\b(?:localhost|(?:\d{1,3}\.){3}\d{1,3}|\[[0-9A-Fa-f:]+\]|[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?::\d{2,5})\b"
)
_HOST_OR_HOST_PATH_RE = re.compile(
    r"\b(?:localhost|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})(?:/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*)?\b",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]+")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_PROMPT_LEAK_MARKERS = (
    "allowed_step_ids",
    "allowed_overlay_targets",
    "allowed_evidence_refs",
    "deterministic_step_hint",
    "overlay_target_policy",
    "evidence_sources",
    "response_format",
    "json_schema",
    "system prompt",
)


class ModelTransportSecurityError(ValueError):
    """Raised when a remote model endpoint does not satisfy transport requirements."""


def redact_url_for_log(url: str) -> str:
    value = url.strip()
    if not value:
        return value

    parsed = urlsplit(value)
    if not parsed.scheme and not parsed.netloc and "@" in value:
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


def _is_local_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.strip().lower().strip("[]")
    if not normalized:
        return False
    if normalized in {"localhost", "localhost.localdomain"}:
        return True
    try:
        parsed = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return bool(parsed.is_loopback or parsed.is_private or parsed.is_link_local)


def validate_model_base_url_security(base_url: str, *, provider: str) -> None:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ModelTransportSecurityError(
            f"{provider} base URL must use http or https: {redact_url_for_log(base_url)}"
        )
    if parsed.scheme == "https":
        return
    if _is_local_hostname(parsed.hostname):
        return
    raise ModelTransportSecurityError(
        f"{provider} remote base URL must use https: {redact_url_for_log(base_url)}"
    )


def redact_sensitive_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    redacted = _URL_RE.sub("[REDACTED_URL]", value)
    redacted = _HOST_PORT_RE.sub("[REDACTED_ENDPOINT]", redacted)
    redacted = _HOST_OR_HOST_PATH_RE.sub("[REDACTED_ENDPOINT]", redacted)
    redacted = _BEARER_RE.sub("Bearer [REDACTED_SECRET]", redacted)
    redacted = _OPENAI_KEY_RE.sub("[REDACTED_SECRET]", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[REDACTED_SECRET]", redacted)
    redacted = _ABS_WIN_PATH_RE.sub("[REDACTED_PATH]", redacted)
    redacted = _ABS_POSIX_PATH_RE.sub("[REDACTED_PATH]", redacted)
    return redacted


def looks_like_prompt_leak(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    markers = sum(1 for marker in _PROMPT_LEAK_MARKERS if marker in lowered)
    if markers >= 2:
        return True
    return "ignore previous instructions" in lowered or "reveal system prompt" in lowered


def sanitize_public_model_text(value: Any, *, lang: str) -> Any:
    if not isinstance(value, str):
        return value
    redacted = redact_sensitive_text(value)
    if looks_like_prompt_leak(redacted):
        if lang == "en":
            return "Potential prompt/source leakage was blocked. Re-trigger Help after confirming the current step."
        return "已拦截可能的提示词/源输出泄露。请确认当前步骤后重新触发 Help。"
    return redacted


def sanitize_help_response_for_log(help_obj: Mapping[str, Any], *, lang: str) -> dict[str, Any]:
    sanitized = dict(help_obj)

    explanations = sanitized.get("explanations")
    if isinstance(explanations, list):
        sanitized["explanations"] = [
            sanitize_public_model_text(item, lang=lang) for item in explanations if isinstance(item, str)
        ]

    overlay = sanitized.get("overlay")
    if isinstance(overlay, Mapping):
        sanitized_overlay = dict(overlay)
        evidence_raw = overlay.get("evidence")
        if isinstance(evidence_raw, list):
            sanitized_evidence: list[dict[str, Any]] = []
            for item in evidence_raw:
                if not isinstance(item, Mapping):
                    continue
                normalized = dict(item)
                if "quote" in normalized:
                    normalized["quote"] = "[REDACTED_SOURCE_QUOTE]"
                sanitized_evidence.append(normalized)
            sanitized_overlay["evidence"] = sanitized_evidence
        sanitized["overlay"] = sanitized_overlay

    return sanitized


__all__ = [
    "ModelTransportSecurityError",
    "redact_sensitive_text",
    "redact_url_for_log",
    "sanitize_help_response_for_log",
    "sanitize_public_model_text",
    "validate_model_base_url_security",
]
