from __future__ import annotations

import socket
import time
from typing import Any, Callable, Mapping, Optional, Sequence
from uuid import uuid4

from core.help_cycle_audit import normalize_help_cycle_audit_fields
from core.overlay import OverlayIntent
from core.types import Event

from adapters.dcs.overlay.ack_receiver import DcsOverlayAckReceiver
from adapters.dcs.overlay.codec import command_from_intent, encode_command


class DcsOverlaySender:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7781,
        ack_receiver: Optional[DcsOverlayAckReceiver] = None,
        timeout: float = 0.5,
        auto_clear: bool = True,
        enabled: bool = True,
        ack_enabled: bool = True,
        session_id: str | None = None,
        event_sink: Optional[Callable[[Event], None]] = None,
        ack_retry_count: int = 1,
    ) -> None:
        self.server = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.ack_receiver = ack_receiver
        # auto_clear only clears when switching targets (not on every highlight).
        self.auto_clear = auto_clear
        self.enabled = enabled
        self.ack_enabled = ack_enabled
        self.ack_retry_count = max(0, int(ack_retry_count))
        self.session_id = session_id
        self.event_sink = event_sink
        self._last_target: Optional[str] = None
        self._active_targets: list[str] = []
        self._preserve_targets_stack: list[set[str]] = []
        self._event_metadata_stack: list[dict[str, Any]] = []

    def close(self) -> None:
        self.sock.close()

    def _emit_event(self, event: Event) -> None:
        current_trace = self._current_event_metadata()
        if current_trace:
            event.payload = dict(event.payload)
            merged_meta = dict(event.metadata)
            merged_meta.update(current_trace)
            event.metadata = merged_meta
            if isinstance(event.payload, dict):
                for key, value in current_trace.items():
                    if key not in event.payload:
                        event.payload[key] = value
        if self.event_sink:
            self.event_sink(event)

    def _current_event_metadata(self) -> dict[str, Any]:
        if not self._event_metadata_stack:
            return {}
        return dict(self._event_metadata_stack[-1])

    def push_event_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        self._event_metadata_stack.append(normalize_help_cycle_audit_fields(metadata))

    def pop_event_metadata(self) -> None:
        if self._event_metadata_stack:
            self._event_metadata_stack.pop()

    def push_preserve_targets(self, targets: Sequence[str] | None) -> None:
        preserve = {
            item
            for item in (targets or [])
            if isinstance(item, str) and item
        }
        self._preserve_targets_stack.append(preserve)

    def pop_preserve_targets(self) -> None:
        if self._preserve_targets_stack:
            self._preserve_targets_stack.pop()

    def _current_preserve_targets(self) -> set[str]:
        if not self._preserve_targets_stack:
            return set()
        return set(self._preserve_targets_stack[-1])

    def _track_active_target(self, target: str) -> None:
        self._active_targets = [item for item in self._active_targets if item != target]
        self._active_targets.append(target)
        self._last_target = target

    def _forget_active_target(self, target: str) -> None:
        self._active_targets = [item for item in self._active_targets if item != target]
        self._last_target = self._active_targets[-1] if self._active_targets else None

    def _send_command(self, payload: dict) -> None:
        data = encode_command(payload)
        self.sock.sendto(data, self.server)

    def _record_request(self, payload: dict, *, attempt_count: int = 1) -> None:
        event_payload = dict(payload)
        event_payload["attempt_count"] = attempt_count
        if attempt_count > 1:
            event_payload["is_retry"] = True
        evt = Event(
            kind="overlay_requested",
            payload=event_payload,
            t_wall=time.time(),
            session_id=self.session_id,
        )
        self._emit_event(evt)

    def _build_failed_result(
        self,
        *,
        cmd: dict,
        intent: OverlayIntent,
        attempt_count: int,
        failure_class: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "cmd_id": cmd["cmd_id"],
            "status": "failed",
            "reason": reason,
            "failure_class": failure_class,
            "attempt_count": attempt_count,
            "action": cmd["action"],
            "intent": intent.intent,
            "target": intent.element_id,
        }

    def _build_failed_result_for_command(
        self,
        *,
        cmd: Mapping[str, Any],
        intent: str,
        target: str,
        attempt_count: int,
        failure_class: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "cmd_id": cmd["cmd_id"],
            "status": "failed",
            "reason": reason,
            "failure_class": failure_class,
            "attempt_count": attempt_count,
            "action": cmd["action"],
            "intent": intent,
            "target": target,
        }

    def _decorate_ack_result(self, ack: Mapping[str, Any], *, cmd: dict, intent: OverlayIntent, attempt_count: int) -> dict[str, Any]:
        result = {
            "cmd_id": cmd["cmd_id"],
            "status": ack.get("status"),
            "reason": ack.get("reason"),
            "attempt_count": attempt_count,
            "action": cmd["action"],
            "intent": intent.intent,
            "target": intent.element_id,
        }
        if ack.get("status") == "failed":
            result["failure_class"] = "remote_failure"
        return result

    def _emit_ack_result(self, ack: Mapping[str, Any]) -> None:
        payload = dict(ack)
        kind = "overlay_applied" if payload.get("status") == "ok" else "overlay_failed"
        event = Event(
            kind=kind,
            payload=payload,
            t_wall=time.time(),
            session_id=self.session_id,
        )
        self._emit_event(event)

    def _should_track_last_target_after_ack(self, intent: OverlayIntent, ack: Mapping[str, Any]) -> bool:
        status = ack.get("status")
        if status == "ok":
            return True
        failure_class = ack.get("failure_class")
        if failure_class == "ack_timeout" and intent.intent == "highlight":
            return True
        return False

    def send_intent(self, intent: OverlayIntent, expect_ack: bool = True) -> Optional[dict]:
        if not self.enabled:
            evt = Event(
                kind="overlay_failed",
                payload={
                    "status": "failed",
                    "failure_class": "overlay_disabled",
                    "reason": "overlay disabled",
                    "intent": intent.intent,
                    "target": intent.element_id,
                },
                t_wall=time.time(),
                session_id=self.session_id,
            )
            self._emit_event(evt)
            return None
        target = intent.element_id
        if self.auto_clear and intent.intent == "highlight":
            preserve_targets = self._current_preserve_targets()
            stale_targets = [
                item
                for item in self._active_targets
                if item != target and item not in preserve_targets
            ]
            for stale_target in stale_targets:
                clear_payload = {
                    "schema_version": "v2",
                    "cmd_id": str(uuid4()),
                    "action": "clear",
                    "target": stale_target,
                }
                self._record_request(clear_payload)
                try:
                    self._send_command(clear_payload)
                except OSError as exc:
                    clear_failure = self._build_failed_result_for_command(
                        cmd=clear_payload,
                        intent="clear",
                        target=stale_target,
                        attempt_count=1,
                        failure_class="transport_error",
                        reason=str(exc),
                    )
                    self._emit_ack_result(clear_failure)
                    return clear_failure
                self._forget_active_target(stale_target)

        cmd = command_from_intent(intent)
        ack_expected = self.ack_enabled and expect_ack and self.ack_receiver is not None
        total_attempts = self.ack_retry_count + 1 if ack_expected else 1
        ack: Optional[dict[str, Any]] = None
        for attempt in range(1, total_attempts + 1):
            self._record_request(cmd, attempt_count=attempt)
            try:
                self._send_command(cmd)
            except OSError as exc:
                ack = self._build_failed_result(
                    cmd=cmd,
                    intent=intent,
                    attempt_count=attempt,
                    failure_class="transport_error",
                    reason=str(exc),
                )
                self._emit_ack_result(ack)
                return ack
            if not ack_expected:
                break
            wait_timeout = self.sock.gettimeout()
            if wait_timeout is None:
                wait_timeout = 0.5
            received = self.ack_receiver.wait_for(cmd["cmd_id"], timeout=wait_timeout)
            if received:
                ack = self._decorate_ack_result(received, cmd=cmd, intent=intent, attempt_count=attempt)
                self._emit_ack_result(ack)
                break
        if ack_expected and ack is None:
            ack = self._build_failed_result(
                cmd=cmd,
                intent=intent,
                attempt_count=total_attempts,
                failure_class="ack_timeout",
                reason="overlay ack timeout",
            )
            self._emit_ack_result(ack)

        if not ack_expected:
            if intent.intent == "highlight":
                self._track_active_target(target)
            elif intent.intent == "clear":
                self._forget_active_target(target)
            return None

        if ack and self._should_track_last_target_after_ack(intent, ack):
            if intent.intent == "highlight":
                self._track_active_target(target)
            elif intent.intent == "clear":
                self._forget_active_target(target)
        return ack

    def __enter__(self) -> "DcsOverlaySender":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
