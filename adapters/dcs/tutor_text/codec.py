from __future__ import annotations

import json
from typing import Any, Mapping
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker

from simtutor.schemas import load_schema

_CMD_VALIDATOR = Draft202012Validator(load_schema("dcs_tutor_text_command"), format_checker=FormatChecker())
_ACK_VALIDATOR = Draft202012Validator(load_schema("dcs_tutor_text_ack"), format_checker=FormatChecker())


def validate_command(payload: Mapping[str, Any]) -> None:
    errors = sorted(_CMD_VALIDATOR.iter_errors(payload), key=lambda e: e.path)
    if errors:
        err = errors[0]
        location = ".".join([str(p) for p in err.path]) or "<root>"
        raise ValueError(f"dcs_tutor_text_command invalid at {location}: {err.message}")


def validate_ack(payload: Mapping[str, Any]) -> None:
    errors = sorted(_ACK_VALIDATOR.iter_errors(payload), key=lambda e: e.path)
    if errors:
        err = errors[0]
        location = ".".join([str(p) for p in err.path]) or "<root>"
        raise ValueError(f"dcs_tutor_text_ack invalid at {location}: {err.message}")


def command_from_message(
    text: str,
    *,
    display_time_s: float = 12.0,
    clear_view: bool = False,
    cmd_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    display = float(display_time_s)
    if not (0.0 < display < float("inf")):
        raise ValueError(f"display_time_s must be a finite positive number, got {display}")
    payload = {
        "schema_version": "v2",
        "cmd_id": cmd_id or str(uuid4()),
        "text": text.strip(),
        "display_time_s": display,
        "clear_view": bool(clear_view),
    }
    validate_command(payload)
    return payload


def encode_command(payload: Mapping[str, Any]) -> bytes:
    validate_command(payload)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decode_command(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Command payload must be a JSON object")
    validate_command(payload)
    return payload


def encode_ack(payload: Mapping[str, Any]) -> bytes:
    validate_ack(payload)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decode_ack(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Ack payload must be a JSON object")
    validate_ack(payload)
    return payload
