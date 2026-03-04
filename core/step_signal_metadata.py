"""
Shared step-signal metadata enums and helpers.

These values are pack-level metadata for deterministic hints and visual
confirmation signals. They are distinct from HelpResponse overlay evidence
types validated by llm_schema.
"""

from __future__ import annotations

from typing import Sequence

STEP_OBSERVABILITY_VALUES = frozenset({"observable", "partially", "unknown"})
STEP_EVIDENCE_REQUIREMENT_VALUES = frozenset({"var", "gate", "delta", "rag", "visual"})


def compute_requires_visual_confirmation(
    observability: str | None,
    step_evidence_requirements: Sequence[str] | None,
) -> bool:
    requirements = set(step_evidence_requirements or [])
    return bool(observability in {"partially", "unknown"} or "visual" in requirements)


__all__ = [
    "STEP_OBSERVABILITY_VALUES",
    "STEP_EVIDENCE_REQUIREMENT_VALUES",
    "compute_requires_visual_confirmation",
]
