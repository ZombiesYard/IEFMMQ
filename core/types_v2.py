"""
Versioned data contracts for simulator-specific telemetry (v2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

CONTRACT_VERSION_V2 = "v2"


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
    aircraft: Optional[str] = None
    lo: Optional[Dict[str, Any]] = None
    bios: Optional[Dict[str, Any]] = None
    cockpit_args: Optional[Dict[str, float]] = None
    vars: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["DcsObservation", "TelemetryFrame", "CONTRACT_VERSION_V2"]

