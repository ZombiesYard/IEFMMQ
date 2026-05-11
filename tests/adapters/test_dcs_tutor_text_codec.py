from __future__ import annotations

import json

import pytest

from adapters.dcs.tutor_text.codec import (
    command_from_message,
    decode_ack,
    decode_command,
    encode_ack,
    encode_command,
)


def test_command_round_trip() -> None:
    cmd = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        "text": "Tutor: turn on APU.",
        "display_time_s": 8.0,
        "clear_view": False,
    }
    data = encode_command(cmd)
    decoded = decode_command(data)
    assert decoded["cmd_id"] == cmd["cmd_id"]
    assert decoded["text"] == "Tutor: turn on APU."
    assert decoded["display_time_s"] == 8.0
    assert decoded["clear_view"] is False


def test_ack_round_trip() -> None:
    ack = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "ok",
    }
    data = encode_ack(ack)
    decoded = decode_ack(data)
    assert decoded["status"] == "ok"


def test_command_from_message_builds_valid_payload() -> None:
    cmd = command_from_message(
        "Tutor: turn on APU.",
        display_time_s=6.5,
        clear_view=True,
        cmd_id="123e4567-e89b-12d3-a456-426614174000",
    )
    assert cmd["schema_version"] == "v2"
    assert cmd["cmd_id"] == "123e4567-e89b-12d3-a456-426614174000"
    assert cmd["text"] == "Tutor: turn on APU."
    assert cmd["display_time_s"] == 6.5
    assert cmd["clear_view"] is True


def test_command_from_message_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        command_from_message("   ")


def test_decode_invalid_command_raises() -> None:
    payload = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        "display_time_s": 8.0,
        "clear_view": False,
    }
    data = json.dumps(payload).encode("utf-8")
    with pytest.raises(ValueError, match="dcs_tutor_text_command invalid"):
        decode_command(data)


def test_decode_invalid_ack_raises() -> None:
    payload = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
    }
    data = json.dumps(payload).encode("utf-8")
    with pytest.raises(ValueError, match="dcs_tutor_text_ack invalid"):
        decode_ack(data)


def test_command_from_message_rejects_zero_display_time() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        command_from_message("hello", display_time_s=0.0)


def test_command_from_message_rejects_negative_display_time() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        command_from_message("hello", display_time_s=-1.0)


def test_command_from_message_rejects_inf_display_time() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        command_from_message("hello", display_time_s=float("inf"))


def test_command_from_message_rejects_nan_display_time() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        command_from_message("hello", display_time_s=float("nan"))


def test_decode_command_rejects_non_object_json() -> None:
    data = json.dumps(["not", "an", "object"]).encode("utf-8")
    with pytest.raises(ValueError, match="Command payload must be a JSON object"):
        decode_command(data)


def test_decode_ack_rejects_non_object_json() -> None:
    data = json.dumps(["not", "an", "object"]).encode("utf-8")
    with pytest.raises(ValueError, match="Ack payload must be a JSON object"):
        decode_ack(data)


def test_decode_command_rejects_malformed_bytes() -> None:
    with pytest.raises(ValueError, match="Invalid JSON payload"):
        decode_command(b"not json")


def test_decode_ack_rejects_malformed_bytes() -> None:
    with pytest.raises(ValueError, match="Invalid JSON payload"):
        decode_ack(b"not json")
