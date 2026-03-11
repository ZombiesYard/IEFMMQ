from __future__ import annotations

from typing import Any, Mapping


def hint_items(raw: Any) -> tuple[Any, ...]:
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    return ()


def hint_has_hard_blocker(missing_conditions: Any, gate_blockers: Any) -> bool:
    normalized_missing = hint_items(missing_conditions)
    if any(isinstance(item, str) and item for item in normalized_missing):
        return True
    normalized_blockers = hint_items(gate_blockers)
    return any(
        (isinstance(item, str) and item)
        or (
            isinstance(item, Mapping)
            and any(
                isinstance(item.get(key), str) and item.get(key)
                for key in ("ref", "reason_code", "reason")
            )
        )
        for item in normalized_blockers
    )


__all__ = ["hint_has_hard_blocker", "hint_items"]
