from __future__ import annotations

from collections import OrderedDict
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
        completed_cache_size: int = 2048,
        pending_cache_size: int = 2048,
    ) -> None:
        self.server = (host, port)
        self.session_id = session_id
        self._pending_cache_size = max(1, int(pending_cache_size))
        self._pending: OrderedDict[str, dict] = OrderedDict()
        self._completed_cache_size = max(1, int(completed_cache_size))
        self._completed: OrderedDict[str, None] = OrderedDict()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.server)
        self.sock.settimeout(timeout)

    def close(self) -> None:
        self.sock.close()

    def _remember_completed(self, cmd_id: str) -> None:
        self._completed.pop(cmd_id, None)
        self._completed[cmd_id] = None
        while len(self._completed) > self._completed_cache_size:
            self._completed.popitem(last=False)

    def _remember_pending(self, cmd_id: str, ack: dict) -> None:
        self._pending.pop(cmd_id, None)
        self._pending[cmd_id] = ack
        while len(self._pending) > self._pending_cache_size:
            self._pending.popitem(last=False)

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

    def _pop_pending(self, cmd_id: str) -> Optional[dict]:
        ack = self._pending.pop(cmd_id, None)
        if ack is not None:
            self._remember_completed(cmd_id)
        return ack

    def wait_for(self, cmd_id: str, timeout: float = 1.0) -> Optional[dict]:
        if cmd_id in self._completed:
            return None
        pending = self._pop_pending(cmd_id)
        if pending:
            return pending
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            ack = self.recv()
            if not ack:
                continue
            ack_cmd_id = ack.get("cmd_id")
            if not isinstance(ack_cmd_id, str) or not ack_cmd_id:
                continue
            if ack_cmd_id in self._completed:
                continue
            if ack_cmd_id == cmd_id:
                self._remember_completed(cmd_id)
                return ack
            self._remember_pending(ack_cmd_id, ack)
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
