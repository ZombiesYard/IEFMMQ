from __future__ import annotations

import json

import pytest

from adapters.dcs.overlay.codec import (
    command_from_intent,
    decode_ack,
    decode_command,
    encode_ack,
    encode_command,
)
from core.overlay import OverlayIntent


def test_command_round_trip() -> None:
    cmd = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        "action": "highlight",
        "target": "pnt_331",
    }
    data = encode_command(cmd)
    decoded = decode_command(data)
    assert decoded["cmd_id"] == cmd["cmd_id"]
    assert decoded["action"] == "highlight"
    assert decoded["target"] == "pnt_331"


def test_ack_round_trip() -> None:
    ack = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "ok",
    }
    data = encode_ack(ack)
    decoded = decode_ack(data)
    assert decoded["status"] == "ok"


def test_command_from_intent_highlight() -> None:
    intent = OverlayIntent(intent="highlight", target="battery_switch", element_id="pnt_331")
    cmd = command_from_intent(intent, cmd_id="123e4567-e89b-12d3-a456-426614174000")
    assert cmd["schema_version"] == "v2"
    assert cmd["cmd_id"] == "123e4567-e89b-12d3-a456-426614174000"
    assert cmd["action"] == "highlight"
    assert cmd["target"] == "pnt_331"


def test_command_from_intent_clear() -> None:
    intent = OverlayIntent(intent="clear", target="battery_switch", element_id="pnt_331")
    cmd = command_from_intent(intent, cmd_id="123e4567-e89b-12d3-a456-426614174000")
    assert cmd["action"] == "clear"


def test_command_from_intent_rejects_pulse() -> None:
    intent = OverlayIntent(intent="pulse", target="battery_switch", element_id="pnt_331")
    with pytest.raises(ValueError, match="Unsupported overlay intent"):
        command_from_intent(intent)


def test_decode_invalid_command_raises() -> None:
    payload = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        "action": "highlight",
    }
    data = json.dumps(payload).encode("utf-8")
    with pytest.raises(ValueError, match="dcs_overlay_command invalid"):
        decode_command(data)


def test_decode_invalid_ack_raises() -> None:
    payload = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
    }
    data = json.dumps(payload).encode("utf-8")
    with pytest.raises(ValueError, match="dcs_overlay_ack invalid"):
        decode_ack(data)
