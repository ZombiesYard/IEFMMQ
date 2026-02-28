from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from adapters.dcs_bios.bios_ui_map import BiosUiMapper
from adapters.delta_sanitizer import DeltaPolicy, SanitizedDelta
from core.types import Event


@dataclass
class DeltaSummary:
    recent_ui_targets: list[str]
    recent_key_changes_topk: list[dict[str, Any]]
    dropped_stats: dict[str, Any]
    raw_changes_total: int
    kept_changes_total: int
    window_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "recent_ui_targets": list(self.recent_ui_targets),
            "recent_key_changes_topk": list(self.recent_key_changes_topk),
            "dropped_stats": dict(self.dropped_stats),
            "raw_changes_total": self.raw_changes_total,
            "kept_changes_total": self.kept_changes_total,
            "window_size": self.window_size,
        }


def aggregate_delta_window(
    window: Iterable[SanitizedDelta],
    *,
    policy: DeltaPolicy,
    mapper: BiosUiMapper | None = None,
) -> DeltaSummary:
    deltas = list(window)
    raw_total = 0
    kept_total = 0
    dropped_total = 0
    dropped_by_reason: dict[str, int] = {}
    recent_ui_targets: list[str] = []
    seen_targets: set[str] = set()
    by_key: dict[str, dict[str, Any]] = {}

    for idx, sanitized in enumerate(deltas):
        raw_total += sanitized.raw_count
        kept_total += sanitized.kept_count
        dropped_total += sanitized.dropped_count

        for reason, count in sanitized.dropped_by_reason.items():
            dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + int(count)

        for key, value in sanitized.kept.items():
            targets = mapper.targets_for_key(key) if mapper else []
            for target in targets:
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                recent_ui_targets.append(target)

            item = by_key.get(key)
            if item is None:
                item = {
                    "key": key,
                    "value": value,
                    "count": 0,
                    "ui_targets": targets,
                    "first_idx": idx,
                    "last_idx": idx,
                }
                by_key[key] = item
            item["value"] = value
            item["count"] += 1
            item["ui_targets"] = targets
            item["last_idx"] = idx

    rows: list[dict[str, Any]] = []
    for key, item in by_key.items():
        score = int(item["count"])
        if key in policy.important_bios_keys:
            score += 100
        if item["ui_targets"]:
            score += 10
        rows.append(
            {
                "key": key,
                "value": item["value"],
                "count": int(item["count"]),
                "ui_targets": list(item["ui_targets"]),
                "importance": score,
                "first_idx": int(item["first_idx"]),
                "last_idx": int(item["last_idx"]),
            }
        )

    rows.sort(key=lambda x: (-int(x["importance"]), int(x["first_idx"]), -int(x["last_idx"])))
    topk = rows[: policy.max_changes_per_window]
    # internal sort helpers are removed from final payload
    for row in topk:
        row.pop("first_idx", None)
        row.pop("last_idx", None)

    dropped_stats = {
        "dropped_total": dropped_total,
        "dropped_by_reason": dropped_by_reason,
    }
    return DeltaSummary(
        recent_ui_targets=recent_ui_targets[: policy.max_changes_per_window],
        recent_key_changes_topk=topk,
        dropped_stats=dropped_stats,
        raw_changes_total=raw_total,
        kept_changes_total=kept_total,
        window_size=len(deltas),
    )


class DeltaAggregator:
    def __init__(
        self,
        policy: DeltaPolicy,
        *,
        mapper: BiosUiMapper | None = None,
        window_size: int = 20,
    ) -> None:
        self.policy = policy
        self.mapper = mapper
        self.window_size = max(1, int(window_size))
        self._window: deque[SanitizedDelta] = deque(maxlen=self.window_size)

    def add(self, sanitized: SanitizedDelta) -> DeltaSummary:
        self._window.append(sanitized)
        return aggregate_delta_window(self._window, policy=self.policy, mapper=self.mapper)


def emit_delta_sanitized_event(
    summary: DeltaSummary,
    *,
    related_id: str | None,
    event_sink: Callable[[Event], None] | None = None,
) -> Event:
    event = Event(
        kind="delta_sanitized",
        related_id=related_id,
        payload={
            "window_size": summary.window_size,
            "raw_changes_total": summary.raw_changes_total,
            "kept_changes_total": summary.kept_changes_total,
            "recent_ui_targets_count": len(summary.recent_ui_targets),
            "recent_key_changes_count": len(summary.recent_key_changes_topk),
            "dropped_stats": summary.dropped_stats,
        },
    )
    if event_sink is not None:
        event_sink(event)
    return event


__all__ = [
    "DeltaAggregator",
    "DeltaSummary",
    "aggregate_delta_window",
    "emit_delta_sanitized_event",
]
