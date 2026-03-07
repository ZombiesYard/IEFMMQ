"""
Vision port reserves a frame-oriented interface for future VLM adapters.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types_v2 import VisionObservation


@runtime_checkable
class VisionPort(Protocol):
    def start(self, session_id: str) -> None:
        ...

    def poll(self) -> list[VisionObservation]:
        ...

    def stop(self) -> None:
        ...
