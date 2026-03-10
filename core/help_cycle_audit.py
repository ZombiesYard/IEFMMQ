"""
Shared help-cycle audit field normalization.
"""

from __future__ import annotations

from typing import Any, Mapping

HELP_CYCLE_AUDIT_FIELDS: tuple[str, ...] = (
    "help_cycle_id",
    "generation_mode",
    "vision_used",
    "frame_id",
    "sync_delta_ms",
    "vision_fact_summary",
    "fused_step_id",
    "fused_missing_conditions",
    "vision_fallback_reason",
    "layout_id",
)


def normalize_help_cycle_audit_fields(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    for key in HELP_CYCLE_AUDIT_FIELDS:
        if key not in raw:
            continue
        value = raw.get(key)
        if isinstance(value, Mapping):
            normalized[key] = dict(value)
        elif isinstance(value, list):
            normalized[key] = list(value)
        elif isinstance(value, tuple):
            normalized[key] = list(value)
        else:
            normalized[key] = value
    return normalized


__all__ = ["HELP_CYCLE_AUDIT_FIELDS", "normalize_help_cycle_audit_fields"]
