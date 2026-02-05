from __future__ import annotations

import socket
import time
from typing import Optional

from core.types import Event

from adapters.dcs.overlay.codec import decode_ack


class DcsOverlayAckReceiver:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 7782,
        timeout: float = 0.2,
        session_id: str | None = None,
    ) -> None:
        self.server = (host, port)
        self.session_id = session_id
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.server)
        self.sock.settimeout(timeout)

    def close(self) -> None:
        self.sock.close()

    def recv(self) -> Optional[dict]:
        try:
            data, _ = self.sock.recvfrom(4096)
        except socket.timeout:
            return None
        except OSError:
            return None
        try:
            return decode_ack(data)
        except ValueError:
            return None

    def wait_for(self, cmd_id: str, timeout: float = 1.0) -> Optional[dict]:
        deadline = time.time() + max(0.0, timeout)
        while time.time() <= deadline:
            ack = self.recv()
            if ack and ack.get("cmd_id") == cmd_id:
                return ack
        return None

    def to_event(self, ack: dict, *, intent: str | None = None, target: str | None = None) -> Event:
        status = ack.get("status")
        kind = "overlay_applied" if status == "ok" else "overlay_failed"
        payload = dict(ack)
        if intent:
            payload["intent"] = intent
        if target:
            payload["target"] = target
        return Event(kind=kind, payload=payload, t_wall=time.time(), session_id=self.session_id)

    def __enter__(self) -> "DcsOverlayAckReceiver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
