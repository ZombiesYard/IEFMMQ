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


def test_sender_skip_ack_when_expect_ack_false(monkeypatch) -> None:
    dummy = AckingSocket([])
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender(host="127.0.0.1", port=7783)

    result = sender.send_text("Tutor: turn on APU.", expect_ack=False)

    assert result["status"] == "ok"
    assert result["ack_skipped"] is True
    assert len(dummy.sent) == 1


def test_sender_reports_transport_error_on_send(monkeypatch) -> None:
    class SendFailingSocket(DummySocket):
        def sendto(self, data, server):
            raise OSError("send failed")

    dummy = SendFailingSocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender(host="127.0.0.1", port=7783)

    result = sender.send_text("Tutor: turn on APU.")

    assert result["status"] == "failed"
    assert result["failure_class"] == "transport_error"
    assert "send failed" in result["reason"]


def test_sender_reports_transport_error_on_recv(monkeypatch) -> None:
    class RecvFailingSocket(DummySocket):
        def recvfrom(self, _size: int):
            raise OSError("recv failed")

    dummy = RecvFailingSocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender(host="127.0.0.1", port=7783)

    result = sender.send_text("Tutor: turn on APU.")

    assert result["status"] == "failed"
    assert result["failure_class"] == "transport_error"
    assert "recv failed" in result["reason"]


def test_sender_reports_invalid_ack_on_malformed_response(monkeypatch) -> None:
    dummy = AckingSocket([b"not valid json"])
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender(host="127.0.0.1", port=7783)

    result = sender.send_text("Tutor: turn on APU.")

    assert result["status"] == "failed"
    assert result["failure_class"] == "invalid_ack"


def test_sender_reports_remote_failure(monkeypatch) -> None:
    ack_payload = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "failed",
        "reason": "DCS internal error",
    }
    dummy = AckingSocket([json.dumps(ack_payload).encode("utf-8")])
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender(host="127.0.0.1", port=7783)

    result = sender.send_text(
        "Tutor: turn on APU.",
        cmd_id="123e4567-e89b-12d3-a456-426614174000",
    )

    assert result["status"] == "failed"
    assert result["failure_class"] == "remote_failure"
    assert result["reason"] == "DCS internal error"


def test_sender_rejects_mismatched_ack_cmd_id(monkeypatch) -> None:
    ack_payload = {
        "schema_version": "v2",
        "cmd_id": "00000000-0000-0000-0000-000000000000",
        "status": "ok",
    }
    dummy = AckingSocket([json.dumps(ack_payload).encode("utf-8")])
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender(host="127.0.0.1", port=7783)

    result = sender.send_text(
        "Tutor: turn on APU.",
        cmd_id="123e4567-e89b-12d3-a456-426614174000",
    )

    assert result["status"] == "failed"
    assert result["failure_class"] == "invalid_ack"
    assert "cmd_id mismatch" in result["reason"]


def test_sender_close(monkeypatch) -> None:
    close_called = []

    class CloseTrackingSocket(DummySocket):
        def close(self):
            close_called.append(True)

    dummy = CloseTrackingSocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsTutorTextSender()
    sender.close()

    assert close_called == [True]
