"""
Typed per-step harness contracts.

The domain object is intentionally independent from live runtime modules so
adapters, replay evaluation, and tests can consume step contracts directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecoveryActionPolicy:
    kind: str
    allowed_targets: tuple[str, ...] = ()
    fallback_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "allowed_targets": list(self.allowed_targets),
        }
        if self.fallback_message is not None:
            out["fallback_message"] = self.fallback_message
        return out


@dataclass(frozen=True)
class SignalQualityRequirement:
    min_confidence: float | None = None
    freshness_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_confidence": self.min_confidence,
            "freshness_ms": self.freshness_ms,
        }


@dataclass(frozen=True)
class StepHarnessSpec:
    step_id: str
    required_observables: tuple[str, ...]
    optional_observables: tuple[str, ...]
    telemetry_facts: tuple[str, ...]
    vision_facts: tuple[str, ...]
    recent_action_facts: tuple[str, ...]
    completion_predicates: tuple[str, ...]
    latch_semantics: str
    allowed_overlay_targets: tuple[str, ...]
    forbidden_overlay_targets: tuple[str, ...]
    recovery_policy: RecoveryActionPolicy
    observability_status: str
    signal_quality: SignalQualityRequirement
    requires_visual_confirmation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "required_observables": list(self.required_observables),
            "optional_observables": list(self.optional_observables),
            "telemetry_facts": list(self.telemetry_facts),
            "vision_facts": list(self.vision_facts),
            "recent_action_facts": list(self.recent_action_facts),
            "completion_predicates": list(self.completion_predicates),
            "latch_semantics": self.latch_semantics,
            "allowed_overlay_targets": list(self.allowed_overlay_targets),
            "forbidden_overlay_targets": list(self.forbidden_overlay_targets),
            "recovery_policy": self.recovery_policy.to_dict(),
            "observability_status": self.observability_status,
            "signal_quality": self.signal_quality.to_dict(),
            "requires_visual_confirmation": self.requires_visual_confirmation,
        }


__all__ = [
    "RecoveryActionPolicy",
    "SignalQualityRequirement",
    "StepHarnessSpec",
]
