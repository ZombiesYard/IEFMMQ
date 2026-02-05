from __future__ import annotations

from adapters.dcs.overlay.codec import (
    decode_ack,
    decode_command,
    encode_ack,
    encode_command,
)


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
