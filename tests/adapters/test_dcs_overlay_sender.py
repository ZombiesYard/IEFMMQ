from __future__ import annotations

import socket

from adapters.dcs.overlay.sender import DcsOverlaySender
from core.overlay import OverlayIntent
from core.types import Event
from tests.adapters.socket_stubs import DummySocket, decode_overlay_command


class FakeAckReceiver:
    def __init__(self, status: str = "ok") -> None:
        self.status = status
        self.last_cmd_id: str | None = None

    def wait_for(self, cmd_id: str, timeout: float = 1.0) -> dict:
        self.last_cmd_id = cmd_id
        return {
            "schema_version": "v2",
            "cmd_id": cmd_id,
            "status": self.status,
        }

    def to_event(self, ack: dict, *, intent: str | None = None, target: str | None = None) -> Event:
        payload = dict(ack)
        if intent:
            payload["intent"] = intent
        if target:
            payload["target"] = target
        kind = "overlay_applied" if ack.get("status") == "ok" else "overlay_failed"
        return Event(kind=kind, payload=payload, t_wall=0.0)


class SequencedAckReceiver:
    def __init__(self, responses: list[dict | None]) -> None:
        self._responses = list(responses)
        self.wait_calls: list[str] = []

    def wait_for(self, cmd_id: str, timeout: float = 1.0) -> dict | None:
        self.wait_calls.append(cmd_id)
        if not self._responses:
            return None
        response = self._responses.pop(0)
        if response is None:
            return None
        payload = dict(response)
        payload.setdefault("schema_version", "v2")
        payload.setdefault("cmd_id", cmd_id)
        return payload

    def to_event(self, ack: dict, *, intent: str | None = None, target: str | None = None) -> Event:
        payload = dict(ack)
        if intent:
            payload["intent"] = intent
        if target:
            payload["target"] = target
        kind = "overlay_applied" if ack.get("status") == "ok" else "overlay_failed"
        return Event(kind=kind, payload=payload, t_wall=0.0)


def test_sender_sends_command_and_emits_events(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    events: list[Event] = []
    ack_receiver = FakeAckReceiver(status="ok")
    sender = DcsOverlaySender(
        host="127.0.0.1",
        port=7781,
        ack_receiver=ack_receiver,
        event_sink=events.append,
    )
    sender.push_event_metadata({"help_cycle_id": "cycle-1", "generation_mode": "model"})
    intent = OverlayIntent(intent="highlight", target="battery_switch", element_id="pnt_331")
    try:
        ack = sender.send_intent(intent, expect_ack=True)
    finally:
        sender.pop_event_metadata()

    assert ack is not None
    assert len(dummy.sent) == 1
    cmd = decode_overlay_command(dummy.sent[0][0])
    assert cmd["action"] == "highlight"
    assert cmd["target"] == "pnt_331"
    assert ack_receiver.last_cmd_id == cmd["cmd_id"]

    kinds = [evt.kind for evt in events]
    assert "overlay_requested" in kinds
    assert "overlay_applied" in kinds
    assert all(evt.metadata["help_cycle_id"] == "cycle-1" for evt in events)
    assert all(evt.metadata["generation_mode"] == "model" for evt in events)
    assert all(evt.payload["help_cycle_id"] == "cycle-1" for evt in events)


def test_sender_auto_clear_on_switch(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsOverlaySender(host="127.0.0.1", port=7781, auto_clear=True, ack_receiver=None)
    sender.send_intent(
        OverlayIntent(intent="highlight", target="first", element_id="pnt_100"),
        expect_ack=False,
    )
    sender.send_intent(
        OverlayIntent(intent="highlight", target="second", element_id="pnt_200"),
        expect_ack=False,
    )

    assert len(dummy.sent) == 3
    cmds = [decode_overlay_command(item[0]) for item in dummy.sent]
    assert cmds[0]["action"] == "highlight"
    assert cmds[0]["target"] == "pnt_100"
    assert cmds[1]["action"] == "clear"
    assert cmds[1]["target"] == "pnt_100"
    assert cmds[2]["action"] == "highlight"
    assert cmds[2]["target"] == "pnt_200"


def test_sender_disabled_emits_failed_event(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    events: list[Event] = []
    sender = DcsOverlaySender(
        host="127.0.0.1",
        port=7781,
        enabled=False,
        ack_receiver=None,
        event_sink=events.append,
    )
    intent = OverlayIntent(intent="highlight", target="battery_switch", element_id="pnt_331")
    ack = sender.send_intent(intent, expect_ack=True)

    assert ack is None
    assert dummy.sent == []
    assert len(events) == 1
    assert events[0].kind == "overlay_failed"
    assert events[0].payload["reason"] == "overlay disabled"
    assert events[0].payload["target"] == "pnt_331"


def test_sender_retries_idempotently_and_classifies_ack_timeout(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    events: list[Event] = []
    ack_receiver = SequencedAckReceiver([None, None, None])
    sender = DcsOverlaySender(
        host="127.0.0.1",
        port=7781,
        ack_receiver=ack_receiver,
        event_sink=events.append,
        ack_retry_count=2,
    )

    intent = OverlayIntent(intent="highlight", target="battery_switch", element_id="pnt_331")
    ack = sender.send_intent(intent, expect_ack=True)

    assert ack is not None
    assert "schema_version" not in ack
    assert ack["status"] == "failed"
    assert ack["failure_class"] == "ack_timeout"
    assert ack["attempt_count"] == 3
    assert len(dummy.sent) == 3
    cmds = [decode_overlay_command(item[0]) for item in dummy.sent]
    assert {cmd["cmd_id"] for cmd in cmds} == {cmds[0]["cmd_id"]}
    assert all(cmd["action"] == "highlight" for cmd in cmds)
    assert all(cmd["target"] == "pnt_331" for cmd in cmds)
    assert ack_receiver.wait_calls == [cmds[0]["cmd_id"]] * 3

    requested = [event for event in events if event.kind == "overlay_requested"]
    failed = [event for event in events if event.kind == "overlay_failed"]
    assert len(requested) == 3
    assert len(failed) == 1
    assert failed[0].payload["failure_class"] == "ack_timeout"
    assert failed[0].payload["attempt_count"] == 3


def test_sender_accepts_late_ack_during_retry_window(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    events: list[Event] = []
    ack_receiver = SequencedAckReceiver([None, {"status": "ok"}])
    sender = DcsOverlaySender(
        host="127.0.0.1",
        port=7781,
        ack_receiver=ack_receiver,
        event_sink=events.append,
        ack_retry_count=2,
    )

    intent = OverlayIntent(intent="highlight", target="battery_switch", element_id="pnt_331")
    ack = sender.send_intent(intent, expect_ack=True)

    assert ack is not None
    assert "schema_version" not in ack
    assert ack["status"] == "ok"
    assert ack["attempt_count"] == 2
    assert len(dummy.sent) == 2
    cmds = [decode_overlay_command(item[0]) for item in dummy.sent]
    assert {cmd["cmd_id"] for cmd in cmds} == {cmds[0]["cmd_id"]}
    applied = [event for event in events if event.kind == "overlay_applied"]
    assert len(applied) == 1
    assert "schema_version" not in applied[0].payload
    assert applied[0].payload["attempt_count"] == 2


def test_sender_uses_sender_session_id_for_remote_and_local_ack_results(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    events: list[Event] = []
    sender = DcsOverlaySender(
        ack_receiver=SequencedAckReceiver([{"status": "ok"}, None, None]),
        event_sink=events.append,
        session_id="sender-session",
        ack_retry_count=1,
    )

    sender.send_intent(OverlayIntent(intent="highlight", target="battery_switch", element_id="pnt_331"), expect_ack=True)
    sender.send_intent(OverlayIntent(intent="clear", target="battery_switch", element_id="pnt_331"), expect_ack=True)

    ack_events = [event for event in events if event.kind in {"overlay_applied", "overlay_failed"}]
    assert len(ack_events) == 2
    assert all(event.session_id == "sender-session" for event in ack_events)
