from __future__ import annotations

import socket

from adapters.dcs.overlay.ack_receiver import DcsOverlayAckReceiver
from adapters.dcs.overlay.codec import encode_ack


class DummySocket:
    def __init__(self, packets: list[bytes]):
        self._packets = list(packets)
        self.timeout = None

    def bind(self, server) -> None:
        self.server = server

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def recvfrom(self, size: int):
        if self._packets:
            return self._packets.pop(0), ("127.0.0.1", 9999)
        raise socket.timeout

    def close(self) -> None:
        return None


def _make_receiver(monkeypatch, packets: list[bytes]) -> DcsOverlayAckReceiver:
    dummy = DummySocket(packets)
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    return DcsOverlayAckReceiver(host="127.0.0.1", port=0)


def test_recv_decodes_ack(monkeypatch) -> None:
    ack = {
        "schema_version": "v2",
        "cmd_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "ok",
    }
    receiver = _make_receiver(monkeypatch, [encode_ack(ack)])
    received = receiver.recv()
    assert received is not None
    assert received["cmd_id"] == ack["cmd_id"]
    assert received["status"] == "ok"


def test_wait_for_buffers_out_of_order(monkeypatch) -> None:
    ack_a = {
        "schema_version": "v2",
        "cmd_id": "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa",
        "status": "ok",
    }
    ack_b = {
        "schema_version": "v2",
        "cmd_id": "bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb",
        "status": "failed",
    }
    receiver = _make_receiver(monkeypatch, [encode_ack(ack_b)])
    assert receiver.wait_for(ack_a["cmd_id"], timeout=0.01) is None
    received = receiver.wait_for(ack_b["cmd_id"], timeout=0.0)
    assert received is not None
    assert received["cmd_id"] == ack_b["cmd_id"]
    assert received["status"] == "failed"


def test_wait_for_immediate_match(monkeypatch) -> None:
    ack = {
        "schema_version": "v2",
        "cmd_id": "cccccccc-cccc-4ccc-cccc-cccccccccccc",
        "status": "ok",
    }
    receiver = _make_receiver(monkeypatch, [encode_ack(ack)])
    received = receiver.wait_for(ack["cmd_id"], timeout=0.01)
    assert received is not None
    assert received["cmd_id"] == ack["cmd_id"]


def test_to_event_status_mapping(monkeypatch) -> None:
    receiver = _make_receiver(monkeypatch, [])
    ok_ack = {
        "schema_version": "v2",
        "cmd_id": "dddddddd-dddd-4ddd-dddd-dddddddddddd",
        "status": "ok",
    }
    failed_ack = {
        "schema_version": "v2",
        "cmd_id": "eeeeeeee-eeee-4eee-eeee-eeeeeeeeeeee",
        "status": "failed",
    }
    ok_event = receiver.to_event(ok_ack, intent="highlight", target="pnt_001")
    failed_event = receiver.to_event(failed_ack, intent="highlight", target="pnt_001")
    assert ok_event.kind == "overlay_applied"
    assert failed_event.kind == "overlay_failed"
