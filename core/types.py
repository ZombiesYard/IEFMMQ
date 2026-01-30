"""
Versioned data contracts for simulator-agnostic messaging.

These dataclasses mirror the JSON Schemas in `simtutor/schemas/v1`. They provide a
type-safe representation for internal code while keeping serialization stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

CONTRACT_VERSION = "v1"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid4())


@dataclass
class Observation:
    observation_id: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now_iso)
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    version: str = CONTRACT_VERSION
    procedure_hint: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TutorRequest:
    request_id: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now_iso)
    actor: str = "learner"
    intent: str = ""
    version: str = CONTRACT_VERSION
    message: Optional[str] = None
    observation_ref: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TutorResponse:
    response_id: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now_iso)
    actor: str = "tutor"
    status: str = "ok"
    version: str = CONTRACT_VERSION
    in_reply_to: Optional[str] = None
    message: Optional[str] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Event:
    event_id: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now_iso)
    kind: str = "custom"
    payload: Dict[str, Any] = field(default_factory=dict)
    version: str = CONTRACT_VERSION
    related_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["Observation", "TutorRequest", "TutorResponse", "Event", "CONTRACT_VERSION"]
