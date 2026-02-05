from __future__ import annotations

import json
from importlib import resources
from typing import Any, Mapping
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker

from core.overlay import OverlayIntent

_SCHEMA_PACKAGE = "simtutor.schemas.v2"
_CMD_SCHEMA = "dcs_overlay_command.json"
_ACK_SCHEMA = "dcs_overlay_ack.json"


def _load_schema(name: str) -> Mapping[str, Any]:
    schema_path = resources.files(_SCHEMA_PACKAGE) / name
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


_CMD_VALIDATOR = Draft202012Validator(_load_schema(_CMD_SCHEMA), format_checker=FormatChecker())
_ACK_VALIDATOR = Draft202012Validator(_load_schema(_ACK_SCHEMA), format_checker=FormatChecker())


def validate_command(payload: Mapping[str, Any]) -> None:
    errors = sorted(_CMD_VALIDATOR.iter_errors(payload), key=lambda e: e.path)
    if errors:
        err = errors[0]
        location = ".".join([str(p) for p in err.path]) or "<root>"
        raise ValueError(f"dcs_overlay_command invalid at {location}: {err.message}")


def validate_ack(payload: Mapping[str, Any]) -> None:
    errors = sorted(_ACK_VALIDATOR.iter_errors(payload), key=lambda e: e.path)
    if errors:
        err = errors[0]
        location = ".".join([str(p) for p in err.path]) or "<root>"
        raise ValueError(f"dcs_overlay_ack invalid at {location}: {err.message}")


def command_from_intent(intent: OverlayIntent, cmd_id: str | None = None) -> dict:
    if intent.intent == "clear":
        action = "clear"
    elif intent.intent == "highlight":
        action = "highlight"
    else:
        raise ValueError(f"Unsupported overlay intent: {intent.intent!r}")
    return {
        "schema_version": "v2",
        "cmd_id": cmd_id or str(uuid4()),
        "action": action,
        "target": intent.element_id,
    }


def encode_command(payload: Mapping[str, Any]) -> bytes:
    validate_command(payload)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decode_command(data: bytes) -> dict:
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


def decode_ack(data: bytes) -> dict:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Ack payload must be a JSON object")
    validate_ack(payload)
    return payload

