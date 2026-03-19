"""
Shared payload helpers for help-triggered vision sidecar captures.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

DEFAULT_VISION_CAPTURE_TRIGGER_HOST = "127.0.0.1"
DEFAULT_VISION_CAPTURE_TRIGGER_PORT = 7795
_CAPTURE_INTENTS = {"capture_vision", "vision_capture", "help"}


def build_capture_request_payload(*, session_id: str, reason: str = "help") -> bytes:
    normalized_session_id = str(session_id).strip()
    if not normalized_session_id:
        raise ValueError("session_id must be a non-empty string")
    normalized_reason = str(reason).strip() or "help"
    payload = {
        "intent": "capture_vision",
        "session_id": normalized_session_id,
        "reason": normalized_reason,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")


def parse_capture_request_payload(
    payload: bytes | str,
    *,
    expected_session_id: str | None = None,
) -> dict[str, Any] | None:
    text = payload.decode("utf-8", errors="ignore") if isinstance(payload, bytes) else str(payload)
    stripped = text.strip()
    if not stripped:
        return None
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        if stripped.lower() != "help" or expected_session_id is not None:
            return None
        return {
            "intent": "capture_vision",
            "session_id": None,
            "reason": "help",
        }
    if not isinstance(decoded, Mapping):
        return None
    intent = decoded.get("intent")
    if not isinstance(intent, str) or intent.strip().lower() not in _CAPTURE_INTENTS:
        return None
    session_id = decoded.get("session_id")
    if expected_session_id is not None:
        if not isinstance(session_id, str) or session_id.strip() != expected_session_id:
            return None
    reason = decoded.get("reason")
    return {
        "intent": "capture_vision",
        "session_id": session_id.strip() if isinstance(session_id, str) and session_id.strip() else None,
        "reason": reason.strip() if isinstance(reason, str) and reason.strip() else "help",
    }


__all__ = [
    "DEFAULT_VISION_CAPTURE_TRIGGER_HOST",
    "DEFAULT_VISION_CAPTURE_TRIGGER_PORT",
    "build_capture_request_payload",
    "parse_capture_request_payload",
]
