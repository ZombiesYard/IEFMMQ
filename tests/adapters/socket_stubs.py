from __future__ import annotations

import json


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


def decode_overlay_command(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


__all__ = ["DummySocket", "decode_overlay_command"]
