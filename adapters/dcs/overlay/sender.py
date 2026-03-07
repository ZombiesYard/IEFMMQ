from __future__ import annotations

import socket
import time
from typing import Any, Callable, Mapping, Optional
from uuid import uuid4

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
    ) -> None:
        self.server = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.ack_receiver = ack_receiver
        # auto_clear only clears when switching targets (not on every highlight).
        self.auto_clear = auto_clear
        self.enabled = enabled
        self.ack_enabled = ack_enabled
        self.session_id = session_id
        self.event_sink = event_sink
        self._last_target: Optional[str] = None
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
            help_cycle_id = current_trace.get("help_cycle_id")
            if (
                isinstance(help_cycle_id, str)
                and help_cycle_id
                and isinstance(event.payload, dict)
                and "help_cycle_id" not in event.payload
            ):
                event.payload["help_cycle_id"] = help_cycle_id
            generation_mode = current_trace.get("generation_mode")
            if (
                isinstance(generation_mode, str)
                and generation_mode
                and isinstance(event.payload, dict)
                and "generation_mode" not in event.payload
            ):
                event.payload["generation_mode"] = generation_mode
        if self.event_sink:
            self.event_sink(event)

    def _current_event_metadata(self) -> dict[str, Any]:
        if not self._event_metadata_stack:
            return {}
        return dict(self._event_metadata_stack[-1])

    def push_event_metadata(self, metadata: Mapping[str, Any] | None) -> None:
        if not isinstance(metadata, Mapping):
            self._event_metadata_stack.append({})
            return
        normalized: dict[str, Any] = {}
        help_cycle_id = metadata.get("help_cycle_id")
        if isinstance(help_cycle_id, str) and help_cycle_id:
            normalized["help_cycle_id"] = help_cycle_id
        generation_mode = metadata.get("generation_mode")
        if isinstance(generation_mode, str) and generation_mode:
            normalized["generation_mode"] = generation_mode
        self._event_metadata_stack.append(normalized)

    def pop_event_metadata(self) -> None:
        if self._event_metadata_stack:
            self._event_metadata_stack.pop()

    def _send_command(self, payload: dict) -> None:
        data = encode_command(payload)
        self.sock.sendto(data, self.server)

    def _record_request(self, payload: dict) -> None:
        evt = Event(
            kind="overlay_requested",
            payload=payload,
            t_wall=time.time(),
            session_id=self.session_id,
        )
        self._emit_event(evt)

    def send_intent(self, intent: OverlayIntent, expect_ack: bool = True) -> Optional[dict]:
        if not self.enabled:
            evt = Event(
                kind="overlay_failed",
                payload={
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
        if self.auto_clear and intent.intent == "highlight" and self._last_target and self._last_target != target:
            clear_payload = {
                "schema_version": "v2",
                "cmd_id": str(uuid4()),
                "action": "clear",
                "target": self._last_target,
            }
            self._record_request(clear_payload)
            self._send_command(clear_payload)

        cmd = command_from_intent(intent)
        self._record_request(cmd)
        self._send_command(cmd)

        if intent.intent == "highlight":
            self._last_target = target
        elif intent.intent == "clear":
            self._last_target = None

        if not self.ack_enabled or not expect_ack or not self.ack_receiver:
            return None
        wait_timeout = self.sock.gettimeout()
        if wait_timeout is None:
            wait_timeout = 0.5
        # Note: ack_receiver has its own socket timeout for individual recv calls.
        ack = self.ack_receiver.wait_for(cmd["cmd_id"], timeout=wait_timeout)
        if ack:
            event = self.ack_receiver.to_event(ack, intent=intent.intent, target=target)
            self._emit_event(event)
        return ack

    def __enter__(self) -> "DcsOverlaySender":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
