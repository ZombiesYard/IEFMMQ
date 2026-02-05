from __future__ import annotations

import socket
import threading

from adapters.dcs.caps.handshake import (
    apply_caps_to_overlay_sender,
    build_hello,
    decode,
    encode,
    negotiate,
    validate_caps,
    validate_hello,
)


def test_hello_schema_validates() -> None:
    hello = build_hello(version="0.2.0", requested={"telemetry": True})
    validate_hello(hello)


def test_caps_schema_validates() -> None:
    caps = {
        "schema_version": "v2",
        "telemetry": True,
        "overlay": False,
        "overlay_ack": False,
        "clickable_actions": False,
        "vlm_frame": False,
    }
    validate_caps(caps)


def test_negotiate_round_trip() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    host, port = server.getsockname()
    server.settimeout(1.0)

    caps_payload = {
        "schema_version": "v2",
        "telemetry": True,
        "overlay": True,
        "overlay_ack": False,
        "clickable_actions": False,
        "vlm_frame": False,
    }

    def server_thread():
        try:
            data, addr = server.recvfrom(4096)
            _ = decode(data)
            server.sendto(encode(caps_payload), addr)
        finally:
            server.close()

    th = threading.Thread(target=server_thread, daemon=True)
    th.start()

    caps = negotiate(host=host, port=port, timeout=1.0)
    assert caps is not None
    assert caps["overlay"] is True


def test_apply_caps_to_sender() -> None:
    class Dummy:
        enabled = True
        ack_enabled = True

    sender = Dummy()
    caps = {
        "overlay": False,
        "overlay_ack": False,
    }
    apply_caps_to_overlay_sender(sender, caps)
    assert sender.enabled is False
    assert sender.ack_enabled is False


def test_negotiate_emits_event() -> None:
    events = []
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    host, port = server.getsockname()
    server.settimeout(1.0)

    caps_payload = {
        "schema_version": "v2",
        "telemetry": True,
        "overlay": True,
        "overlay_ack": True,
        "clickable_actions": False,
        "vlm_frame": False,
    }

    def server_thread():
        try:
            data, addr = server.recvfrom(4096)
            _ = decode(data)
            server.sendto(encode(caps_payload), addr)
        finally:
            server.close()

    th = threading.Thread(target=server_thread, daemon=True)
    th.start()

    caps = negotiate(
        host=host,
        port=port,
        timeout=1.0,
        session_id="sess-1",
        event_sink=events.append,
    )
    assert caps is not None
    assert events
    assert events[0].kind == "capabilities_negotiated"
