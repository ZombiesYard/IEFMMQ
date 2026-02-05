from __future__ import annotations

import json
import socket
import time
from importlib import resources
from typing import Any, Mapping, Optional, Callable

from jsonschema import Draft202012Validator, FormatChecker

from core.types import Event

_SCHEMA_PACKAGE = "simtutor.schemas.v2"
_HELLO_SCHEMA = "dcs_hello.json"
_CAPS_SCHEMA = "dcs_caps.json"


def _load_schema(name: str) -> Mapping[str, Any]:
    schema_path = resources.files(_SCHEMA_PACKAGE) / name
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


_HELLO_VALIDATOR = Draft202012Validator(_load_schema(_HELLO_SCHEMA), format_checker=FormatChecker())
_CAPS_VALIDATOR = Draft202012Validator(_load_schema(_CAPS_SCHEMA), format_checker=FormatChecker())


def build_hello(version: str, requested: Optional[dict] = None) -> dict:
    payload = {
        "schema_version": "v2",
        "client": "simtutor",
        "version": version,
        "requested": requested or {},
    }
    validate_hello(payload)
    return payload


def validate_hello(payload: Mapping[str, Any]) -> None:
    errors = sorted(_HELLO_VALIDATOR.iter_errors(payload), key=lambda e: e.path)
    if errors:
        err = errors[0]
        location = ".".join([str(p) for p in err.path]) or "<root>"
        raise ValueError(f"dcs_hello invalid at {location}: {err.message}")


def validate_caps(payload: Mapping[str, Any]) -> None:
    errors = sorted(_CAPS_VALIDATOR.iter_errors(payload), key=lambda e: e.path)
    if errors:
        err = errors[0]
        location = ".".join([str(p) for p in err.path]) or "<root>"
        raise ValueError(f"dcs_caps invalid at {location}: {err.message}")


def encode(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decode(data: bytes) -> dict:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object")
    return payload


def negotiate(
    host: str = "127.0.0.1",
    port: int = 7793,
    timeout: float = 1.0,
    version: str = "0.2.0",
    requested: Optional[dict] = None,
    session_id: Optional[str] = None,
    event_sink: Optional[Callable[[Event], None]] = None,
) -> Optional[dict]:
    hello = build_hello(version=version, requested=requested)
    data = encode(hello)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(data, (host, port))
        try:
            resp, _ = sock.recvfrom(4096)
        except socket.timeout:
            return None
        caps = decode(resp)
        validate_caps(caps)
        if event_sink:
            event_sink(
                Event(
                    kind="capabilities_negotiated",
                    payload=caps,
                    t_wall=time.time(),
                    session_id=session_id,
                )
            )
        return caps


def apply_caps_to_overlay_sender(sender: Any, caps: Mapping[str, Any]) -> None:
    """
    Convenience helper to adapt overlay behavior based on negotiated caps.
    """
    sender.enabled = bool(caps.get("overlay"))
    sender.ack_enabled = bool(caps.get("overlay_ack"))
