from __future__ import annotations

import socket
import time
from typing import Optional, Callable
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
        session_id: str | None = None,
        event_sink: Optional[Callable[[Event], None]] = None,
    ) -> None:
        self.server = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.ack_receiver = ack_receiver
        # auto_clear only clears when switching targets (not on every highlight).
        self.auto_clear = auto_clear
        self.session_id = session_id
        self.event_sink = event_sink
        self._last_target: Optional[str] = None

    def close(self) -> None:
        self.sock.close()

    def _emit_event(self, event: Event) -> None:
        if self.event_sink:
            self.event_sink(event)

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

        if not expect_ack or not self.ack_receiver:
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
