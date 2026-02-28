from __future__ import annotations

import socket
from pathlib import Path

import adapters.action_executor as action_executor_module
import pytest
from adapters.action_executor import OverlayActionExecutor, execute_overlay_actions
from adapters.dcs.overlay.sender import DcsOverlaySender
from core.types import Event
from tests.adapters.socket_stubs import DummySocket, decode_overlay_command


def test_executor_maps_target_and_sends_highlight_udp(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    events: list[Event] = []
    sink = events.append
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False, event_sink=sink)
    executor = OverlayActionExecutor(sender=sender, event_sink=sink, max_targets=1)

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
    cmd = decode_overlay_command(dummy.sent[0][0])
    assert cmd["action"] == "highlight"
    assert cmd["target"] == "pnt_375"
    assert len(report.executed) == 1
    assert report.executed[0]["target"] == "apu_switch"
    overlay_requested_count = sum(1 for evt in events if evt.kind == "overlay_requested")
    assert overlay_requested_count == 1


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


def test_executor_rejects_invalid_action_payload(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False)
    executor = OverlayActionExecutor(sender=sender)

    report = executor.execute_actions(["bad-action-payload"])

    assert dummy.sent == []
    assert len(report.rejected) == 1
    assert report.rejected[0]["reason"] == "invalid_action_payload"


def test_executor_rejects_invalid_overlay_target(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False)
    executor = OverlayActionExecutor(sender=sender)

    report = executor.execute_actions([{"type": "overlay", "target": ""}])

    assert dummy.sent == []
    assert len(report.rejected) == 1
    assert report.rejected[0]["reason"] == "invalid_overlay_target"


def test_executor_rejects_overlay_target_not_in_allowlist(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False)
    executor = OverlayActionExecutor(sender=sender)

    report = executor.execute_actions([{"type": "overlay", "target": "unknown_target"}])

    assert dummy.sent == []
    assert len(report.rejected) == 1
    assert report.rejected[0]["reason"] == "overlay_target_not_in_allowlist"


def test_executor_pack_ui_targets_narrows_allowlist(monkeypatch, tmp_path: Path) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(
        "pack_id: test_pack\n"
        "version: v1\n"
        "ui_targets:\n"
        "  - battery_switch\n",
        encoding="utf-8",
    )
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False)
    executor = OverlayActionExecutor(sender=sender, pack_path=pack_path)

    report = executor.execute_actions([{"type": "overlay", "target": "apu_switch"}])

    assert dummy.sent == []
    assert len(report.rejected) == 1
    assert report.rejected[0]["reason"] == "overlay_target_not_in_allowlist"


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
    cmd = decode_overlay_command(dummy.sent[0][0])
    assert cmd["target"] == "pnt_375"
    assert len(report.executed) == 1
    assert len(report.rejected) == 1
    assert report.rejected[0]["reason"] == "rejected_non_overlay_action"
    assert len(report.dropped) == 1
    assert report.dropped[0]["target"] == "battery_switch"


def test_executor_dry_run_reports_preview_without_udp(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    events: list[Event] = []
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False)
    executor = OverlayActionExecutor(sender=sender, dry_run=True, ttl_s=1.25, pulse_enabled=True, event_sink=events.append)

    report = executor.execute_actions([{"type": "overlay", "target": "apu_switch"}])

    assert dummy.sent == []
    assert len(report.dry_run) == 1
    assert report.dry_run[0]["target"] == "apu_switch"
    assert report.dry_run[0]["element_id"] == "pnt_375"
    assert report.dry_run[0]["ttl_s"] == 1.25
    assert report.dry_run[0]["pulse_enabled"] is True
    assert any(evt.kind == "overlay_dry_run" for evt in events)


def test_executor_system_pulse_controls_clear_by_ttl(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    monkeypatch.setattr("adapters.action_executor.time.sleep", lambda _x: None)
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False)
    executor = OverlayActionExecutor(sender=sender, pulse_enabled=True, ttl_s=0.5)

    report = executor.execute_actions([{"type": "overlay", "target": "apu_switch"}])

    assert len(report.executed) == 1
    assert len(dummy.sent) == 2
    cmds = [decode_overlay_command(item[0]) for item in dummy.sent]
    assert cmds[0]["action"] == "highlight"
    assert cmds[0]["target"] == "pnt_375"
    assert cmds[1]["action"] == "clear"
    assert cmds[1]["target"] == "pnt_375"


def test_executor_does_not_mutate_external_sender_event_sink(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False, event_sink=None)
    events: list[Event] = []

    executor = OverlayActionExecutor(sender=sender, event_sink=events.append)
    report = executor.execute_actions([{"type": "overlay", "target": "apu_switch"}])

    assert len(report.executed) == 1
    assert sender.event_sink is None
    assert any(evt.kind == "overlay_requested" for evt in events)


def test_executor_rejects_when_external_sender_disabled(monkeypatch) -> None:
    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = DcsOverlaySender(auto_clear=False, ack_enabled=False, enabled=False)
    executor = OverlayActionExecutor(sender=sender)

    report = executor.execute_actions([{"type": "overlay", "target": "apu_switch"}])

    assert dummy.sent == []
    assert report.executed == []
    assert len(report.rejected) == 1
    assert report.rejected[0]["reason"] == "overlay_sender_disabled"


def test_execute_overlay_actions_with_external_sender_returns_report_and_keeps_sender_open(monkeypatch) -> None:
    class TrackingSender(DcsOverlaySender):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.closed = False

        def close(self) -> None:
            self.closed = True
            super().close()

    dummy = DummySocket()
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: dummy)
    sender = TrackingSender(auto_clear=False, ack_enabled=False)

    report = execute_overlay_actions(
        [{"type": "overlay", "target": "apu_switch"}],
        sender=sender,
    )

    assert set(report.keys()) == {"executed", "rejected", "dropped", "dry_run"}
    assert len(report["executed"]) == 1
    assert sender.closed is False
    sender.close()


def test_execute_overlay_actions_without_sender_owns_sender_lifecycle(monkeypatch) -> None:
    class FakeOwnedSender:
        instances: list["FakeOwnedSender"] = []

        def __init__(self, session_id=None, event_sink=None):
            self.session_id = session_id
            self.event_sink = event_sink
            self.enabled = True
            self.closed = False
            self.send_calls: list[tuple[object, bool]] = []
            FakeOwnedSender.instances.append(self)

        def send_intent(self, intent, expect_ack: bool = True):
            self.send_calls.append((intent, expect_ack))
            return {"schema_version": "v2", "cmd_id": "owned-cmd", "status": "ok"}

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(action_executor_module, "DcsOverlaySender", FakeOwnedSender)
    report = execute_overlay_actions([{"type": "overlay", "target": "apu_switch"}], sender=None, expect_ack=False)

    assert set(report.keys()) == {"executed", "rejected", "dropped", "dry_run"}
    assert len(report["executed"]) == 1
    assert len(FakeOwnedSender.instances) == 1
    owned_sender = FakeOwnedSender.instances[0]
    assert owned_sender.closed is True
    assert len(owned_sender.send_calls) == 1
    assert owned_sender.send_calls[0][1] is False


def test_executor_pack_path_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    class NoopSender:
        def __init__(self):
            self.enabled = True
            self.event_sink = None

    bad_pack = tmp_path / "pack_non_mapping.yaml"
    bad_pack.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pack must be a mapping"):
        OverlayActionExecutor(sender=NoopSender(), pack_path=bad_pack)


def test_executor_pack_path_rejects_ui_targets_not_list(tmp_path: Path) -> None:
    class NoopSender:
        def __init__(self):
            self.enabled = True
            self.event_sink = None

    bad_pack = tmp_path / "pack_ui_targets_not_list.yaml"
    bad_pack.write_text("pack_id: x\nversion: v1\nui_targets: apu_switch\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pack.ui_targets must be a list"):
        OverlayActionExecutor(sender=NoopSender(), pack_path=bad_pack)


def test_executor_pack_path_rejects_invalid_ui_targets_items(tmp_path: Path) -> None:
    class NoopSender:
        def __init__(self):
            self.enabled = True
            self.event_sink = None

    bad_pack = tmp_path / "pack_ui_targets_bad_items.yaml"
    bad_pack.write_text("pack_id: x\nversion: v1\nui_targets:\n  - apu_switch\n  - ''\n  - 123\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pack.ui_targets must contain non-empty strings"):
        OverlayActionExecutor(sender=NoopSender(), pack_path=bad_pack)
