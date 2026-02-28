from __future__ import annotations

from typing import Any

from adapters.delta_aggregator import DeltaSummary
from adapters.delta_sanitizer import SanitizedDelta


def build_recent_actions(source: SanitizedDelta | DeltaSummary, *, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Convert sanitized delta artifacts into prompt-friendly recent_actions.

    ST-010b: input must be SanitizedDelta or DeltaSummary (raw delta is not accepted).
    """
    limit = max(0, int(top_k))
    if isinstance(source, SanitizedDelta):
        out: list[dict[str, Any]] = []
        for idx, (key, value) in enumerate(source.kept.items()):
            if idx >= limit:
                break
            out.append({"k": key, "to": value})
        return out

    if isinstance(source, DeltaSummary):
        out = []
        for idx, row in enumerate(source.recent_key_changes_topk):
            if idx >= limit:
                break
            out.append(
                {
                    "k": row.get("key"),
                    "to": row.get("value"),
                    "ui_targets": list(row.get("ui_targets") or []),
                    "importance": row.get("importance"),
                }
            )
        return out

    raise TypeError("build_recent_actions expects SanitizedDelta or DeltaSummary")


__all__ = ["build_recent_actions"]
