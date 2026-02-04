"""
Telemetry port defines interfaces for high-frequency sensor frames.
"""

from __future__ import annotations

from typing import Protocol

from core.types_v2 import TelemetryFrame


class TelemetryPort(Protocol):
    def start(self, session_id: str) -> None:
        ...

    def poll(self) -> list[TelemetryFrame]:
        ...

    def stop(self) -> None:
        ...

