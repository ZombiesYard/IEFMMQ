"""
Shared step-signal metadata enums and helpers.

These values are pack-level metadata for deterministic hints and visual
confirmation signals. They are distinct from HelpResponse overlay evidence
types validated by llm_schema.
"""

from __future__ import annotations

from typing import Sequence

STEP_OBSERVABILITY_VALUES = frozenset({"observable", "partial", "unobservable"})
LEGACY_STEP_OBSERVABILITY_ALIASES = {
    "partially": "partial",
    "unknown": "unobservable",
}
STEP_EVIDENCE_REQUIREMENT_VALUES = frozenset({"var", "gate", "delta", "rag", "visual"})


def normalize_observability_status(observability: str | None) -> str | None:
    if observability in STEP_OBSERVABILITY_VALUES:
        return observability
    if isinstance(observability, str):
        return LEGACY_STEP_OBSERVABILITY_ALIASES.get(observability)
    return None


def compute_requires_visual_confirmation(
    observability: str | None,
    step_evidence_requirements: Sequence[str] | None,
) -> bool:
    requirements = set(step_evidence_requirements or [])
    status = normalize_observability_status(observability)
    return bool(status in {"partial", "unobservable"} or "visual" in requirements)


__all__ = [
    "LEGACY_STEP_OBSERVABILITY_ALIASES",
    "STEP_OBSERVABILITY_VALUES",
    "STEP_EVIDENCE_REQUIREMENT_VALUES",
    "compute_requires_visual_confirmation",
    "normalize_observability_status",
]
