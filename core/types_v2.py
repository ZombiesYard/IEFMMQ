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


__all__ = ["DcsObservation", "CONTRACT_VERSION_V2"]

