from __future__ import annotations

import json
import socket

from adapters.action_executor import OverlayActionExecutor
from adapters.dcs.overlay.sender import DcsOverlaySender
from core.types import Event


class DummySocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self._timeout: float | None = None

    def settimeout(self, timeout: float) -> None:
        self._timeout = timeout

    def gettimeout(self) -> float | None:
        return self._timeout

    def sendto(self, data: bytes, server) -> None:
        self.sent.append((data, server))

    def close(self) -> None:
        return None


def _decode_cmd(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def test_executor_maps_target_and_sends_highlight_udp(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    events: list[Event] = []
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False, event_sink=events.append)
    executor = OverlayActionExecutor(sender=sender, event_sink=events.append, max_targets=1)

    report = executor.execute_actions(
        [
            {
                "type": "overlay",
                "intent": "clear",
                "target": "apu_switch",
                "element_id": "pnt_hacked",
                "ttl_s": 99,
            }
        ]
    )

    assert len(dummy.sent) == 1
    cmd = _decode_cmd(dummy.sent[0][0])
    assert cmd["action"] == "highlight"
    assert cmd["target"] == "pnt_375"
    assert len(report.executed) == 1
    assert report.executed[0]["target"] == "apu_switch"
    assert any(evt.kind == "overlay_requested" for evt in events)


def test_executor_rejects_non_overlay_action_and_records_event(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    events: list[Event] = []
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False, event_sink=events.append)
    executor = OverlayActionExecutor(sender=sender, event_sink=events.append)

    report = executor.execute_actions([{"type": "click", "target": "apu_switch"}])

    assert dummy.sent == []
    assert len(report.rejected) == 1
    assert report.rejected[0]["reason"] == "rejected_non_overlay_action"
    assert any(evt.kind == "overlay_failed" for evt in events)


def test_executor_max_targets_only_applies_to_overlay_execution(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False)
    executor = OverlayActionExecutor(sender=sender, max_targets=1)

    report = executor.execute_actions(
        [
            {"type": "overlay", "target": "apu_switch"},
            {"type": "click", "target": "apu_switch"},
            {"type": "overlay", "target": "battery_switch"},
        ]
    )

    assert len(dummy.sent) == 1
    cmd = _decode_cmd(dummy.sent[0][0])
    assert cmd["target"] == "pnt_375"
    assert len(report.executed) == 1
    assert len(report.rejected) == 1
    assert report.rejected[0]["reason"] == "rejected_non_overlay_action"
    assert len(report.dropped) == 1
    assert report.dropped[0]["target"] == "battery_switch"


def test_executor_dry_run_reports_preview_without_udp(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False)
    executor = OverlayActionExecutor(sender=sender, dry_run=True, ttl_s=1.25, pulse_enabled=True)

    report = executor.execute_actions([{"type": "overlay", "target": "apu_switch"}])

    assert dummy.sent == []
    assert len(report.dry_run) == 1
    assert report.dry_run[0]["target"] == "apu_switch"
    assert report.dry_run[0]["element_id"] == "pnt_375"
    assert report.dry_run[0]["ttl_s"] == 1.25
    assert report.dry_run[0]["pulse_enabled"] is True


def test_executor_system_pulse_controls_clear_by_ttl(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    monkeypatch.setattr("adapters.action_executor.time.sleep", lambda _x: None)
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False)
    executor = OverlayActionExecutor(sender=sender, pulse_enabled=True, ttl_s=0.5)

    report = executor.execute_actions([{"type": "overlay", "target": "apu_switch", "intent": "pulse"}])

    assert len(report.executed) == 1
    assert len(dummy.sent) == 2
    cmds = [_decode_cmd(item[0]) for item in dummy.sent]
    assert cmds[0]["action"] == "highlight"
    assert cmds[0]["target"] == "pnt_375"
    assert cmds[1]["action"] == "clear"
    assert cmds[1]["target"] == "pnt_375"
