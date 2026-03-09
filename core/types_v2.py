"""
Versioned data contracts for simulator-specific telemetry (v2).
"""

from __future__ import annotations

from itertools import count
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
from uuid import uuid4

CONTRACT_VERSION_V2 = "v2"
_VISION_FRAME_SEQ = count()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid4())


def _next_frame_id() -> str:
    capture_wall_ms = int(time.time() * 1000)
    frame_seq = next(_VISION_FRAME_SEQ) % 1_000_000
    return f"{capture_wall_ms}_{frame_seq:06d}"


@dataclass
class DcsObservation:
    schema_version: str = CONTRACT_VERSION_V2
    seq: int = 0
    sim_time: float = 0.0
    aircraft: str = "Unknown"
    cockpit: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetryFrame:
    schema_version: str = CONTRACT_VERSION_V2
    seq: int = 0
    t_wall: float = 0.0
    source: str = "derived"
    t_sim: Optional[float] = None
    session_id: Optional[str] = None
    aircraft: Optional[str] = None
    lo: Optional[Dict[str, Any]] = None
    bios: Optional[Dict[str, Any]] = None
    cockpit_args: Optional[Dict[str, float]] = None
    vars: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisionObservation:
    schema_version: str = CONTRACT_VERSION_V2
    frame_id: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now_iso)
    source: str = "vision"
    session_id: Optional[str] = None
    observation_ref: Optional[str] = None
    channel: Optional[str] = None
    layout_id: Optional[str] = None
    image_uri: Optional[str] = None
    source_image_path: Optional[str] = None
    mime_type: Optional[str] = None
    capture_wall_ms: Optional[int] = None
    frame_seq: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    source_session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["DcsObservation", "TelemetryFrame", "VisionObservation", "CONTRACT_VERSION_V2"]

