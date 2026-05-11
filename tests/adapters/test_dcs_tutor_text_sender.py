from __future__ import annotations

import json
import socket

from adapters.dcs.tutor_text.sender import DcsTutorTextSender
from tests.adapters.socket_stubs import DummySocket


class AckingSocket(DummySocket):
    def __init__(self, responses: list[bytes | Exception]) -> None:
        super().__init__()
        self._responses = list(responses)

    def recvfrom(self, _size: int):
        if not self._responses:
            raise socket.timeout
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, ("127.0.0.1", 7783)


def decode_tutor_text_command(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def test_sender_sends_command_and_waits_for_ack(monkeypatch) -> None:
    ack_payload = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "ok",
    }
    dummy = AckingSocket([json.dumps(ack_payload).encode("utf-8")])
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender(host="127.0.0.1", port=7783)

    result = sender.send_text(
        "Tutor: turn on APU.",
        display_time_s=7.0,
        clear_view=False,
        cmd_id="123e4567-e89b-12d3-a456-426614174000",
    )

    assert result["status"] == "ok"
    assert len(dummy.sent) == 1
    cmd = decode_tutor_text_command(dummy.sent[0][0])
    assert cmd["text"] == "Tutor: turn on APU."
    assert cmd["display_time_s"] == 7.0
    assert cmd["clear_view"] is False


def test_sender_reports_timeout_as_failure(monkeypatch) -> None:
    dummy = AckingSocket([socket.timeout()])
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender(host="127.0.0.1", port=7783, timeout=0.1)

    result = sender.send_text("Tutor: turn on APU.")

    assert result["status"] == "failed"
    assert result["failure_class"] == "ack_timeout"


def test_sender_reports_disabled_without_sending(monkeypatch) -> None:
    dummy = AckingSocket([])
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender(host="127.0.0.1", port=7783, enabled=False)

    result = sender.send_text("Tutor: turn on APU.")

    assert result["status"] == "failed"
    assert result["failure_class"] == "sender_disabled"
    assert dummy.sent == []
