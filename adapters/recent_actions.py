from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
from typing import Mapping

from adapters.delta_aggregator import DeltaSummary
from adapters.delta_sanitizer import SanitizedDelta
from adapters.dcs_bios.bios_ui_map import BiosUiMapper


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


DEFAULT_RECENT_DELTA_WINDOW_S = 8.0
DEFAULT_RECENT_DELTA_MAX_ITEMS = 20
DEFAULT_RECENT_UI_TARGETS_MAX_ITEMS = 8


def _coerce_wall_time(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("t_wall must be int/float")
    return float(value)


def _coerce_seq(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _normalize_positive_float(value: Any, *, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        f = float(value)
        if f > 0:
            return f
    return fallback


def _normalize_positive_int(value: Any, *, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return fallback


def _normalize_targets(raw: Any) -> list[str]:
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, Mapping):
        raw_targets = raw.get("targets")
        if isinstance(raw_targets, str):
            candidates = [raw_targets]
        elif isinstance(raw_targets, (list, tuple)):
            candidates = list(raw_targets)
        else:
            candidates = []
    elif isinstance(raw, (list, tuple)):
        candidates = list(raw)
    else:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _targets_for_key(bios_to_ui: BiosUiMapper | Mapping[str, Any], key: str) -> list[str]:
    if isinstance(bios_to_ui, BiosUiMapper):
        return bios_to_ui.targets_for_key(key)

    mappings = bios_to_ui.get("mappings") if isinstance(bios_to_ui, Mapping) else None
    if isinstance(mappings, Mapping):
        raw = mappings.get(key)
    else:
        raw = bios_to_ui.get(key) if isinstance(bios_to_ui, Mapping) else None
    return _normalize_targets(raw)


def _coerce_nonnegative_limit(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int-like, not bool")
    try:
        return max(0, int(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be int-like") from exc


def _extract_frame_delta(item: Mapping[str, Any]) -> tuple[float | None, int | None, Mapping[str, Any]]:
    delta = item.get("delta")
    if isinstance(delta, Mapping):
        t_wall = item.get("t_wall")
        seq = item.get("seq")
        return (
            float(t_wall) if isinstance(t_wall, (int, float)) and not isinstance(t_wall, bool) else None,
            _coerce_seq(seq),
            delta,
        )

    key = item.get("k") if isinstance(item.get("k"), str) else item.get("key")
    if not isinstance(key, str) or not key:
        return None, None, {}
    t_wall = item.get("t_wall")
    seq = item.get("seq")
    return (
        float(t_wall) if isinstance(t_wall, (int, float)) and not isinstance(t_wall, bool) else None,
        _coerce_seq(seq),
        {key: item.get("to")},
    )


@dataclass(frozen=True)
class RecentActionsConfig:
    window_s: float = DEFAULT_RECENT_DELTA_WINDOW_S
    max_items: int = DEFAULT_RECENT_DELTA_MAX_ITEMS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "window_s",
            _normalize_positive_float(self.window_s, fallback=DEFAULT_RECENT_DELTA_WINDOW_S),
        )
        object.__setattr__(
            self,
            "max_items",
            _normalize_positive_int(self.max_items, fallback=DEFAULT_RECENT_DELTA_MAX_ITEMS),
        )


@dataclass(frozen=True)
class RecentDeltaFrame:
    t_wall: float
    delta: dict[str, Any]
    seq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_wall": self.t_wall,
            "seq": self.seq,
            "delta": dict(self.delta),
        }


class RecentDeltaRingBuffer:
    """
    Hold recent sanitized delta frames in a bounded time window.

    Trimming is applied by:
    - `window_s`: keep only latest N seconds
    - `max_items`: keep only latest N frames
    """

    def __init__(
        self,
        *,
        window_s: float = DEFAULT_RECENT_DELTA_WINDOW_S,
        max_items: int = DEFAULT_RECENT_DELTA_MAX_ITEMS,
    ) -> None:
        self.config = RecentActionsConfig(window_s=window_s, max_items=max_items)
        self._frames: deque[RecentDeltaFrame] = deque()

    def add_delta(
        self,
        delta: Mapping[str, Any],
        *,
        t_wall: float,
        seq: int | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(delta, Mapping):
            raise TypeError("delta must be a mapping")
        wall = _coerce_wall_time(t_wall)

        normalized: dict[str, Any] = {}
        for key, value in delta.items():
            if not isinstance(key, str) or not key:
                continue
            normalized[key] = value

        if normalized:
            self._frames.append(
                RecentDeltaFrame(
                    t_wall=wall,
                    seq=_coerce_seq(seq),
                    delta=normalized,
                )
            )
        self._trim(now_t_wall=wall)
        return [frame.to_dict() for frame in self._frames]

    def add_sanitized_delta(self, sanitized: SanitizedDelta) -> list[dict[str, Any]]:
        if not isinstance(sanitized, SanitizedDelta):
            raise TypeError("sanitized must be a SanitizedDelta")
        if sanitized.t_wall is None:
            raise ValueError("sanitized.t_wall is required for RecentDeltaRingBuffer window trimming")
        return self.add_delta(sanitized.kept, t_wall=sanitized.t_wall, seq=sanitized.seq)

    def snapshot(self, *, now_t_wall: float | None = None) -> list[dict[str, Any]]:
        if now_t_wall is None:
            if not self._frames:
                return []
            now = self._frames[-1].t_wall
        else:
            now = _coerce_wall_time(now_t_wall)
        self._trim(now_t_wall=now)
        return [frame.to_dict() for frame in self._frames]

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def _trim(self, *, now_t_wall: float) -> None:
        cutoff = now_t_wall - self.config.window_s
        while self._frames and self._frames[0].t_wall < cutoff:
            self._frames.popleft()
        while len(self._frames) > self.config.max_items:
            self._frames.popleft()


def project_recent_ui_targets(
    recent_deltas: list[Mapping[str, Any]],
    bios_to_ui: BiosUiMapper | Mapping[str, Any],
    *,
    max_items: int = DEFAULT_RECENT_UI_TARGETS_MAX_ITEMS,
) -> list[str]:
    """
    Project recent delta frames into highlightable UI targets.

    Ordering is stable and recency-first:
    - newer frames first
    - preserve key order within each frame
    - preserve mapping target order for each key
    - de-duplicate by first occurrence
    """
    limit = _coerce_nonnegative_limit(max_items, name="max_items")
    if not recent_deltas or limit == 0:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in reversed(recent_deltas):
        if not isinstance(item, Mapping):
            continue
        _, _, delta = _extract_frame_delta(item)
        for key in delta.keys():
            if not isinstance(key, str) or not key:
                continue
            for target in _targets_for_key(bios_to_ui, key):
                if target in seen:
                    continue
                seen.add(target)
                out.append(target)
                if len(out) >= limit:
                    return out
    return out


def build_recent_button_signal(
    recent_deltas: list[Mapping[str, Any]],
    bios_to_ui: BiosUiMapper | Mapping[str, Any],
    *,
    max_items: int = DEFAULT_RECENT_UI_TARGETS_MAX_ITEMS,
) -> dict[str, Any]:
    recent_buttons = project_recent_ui_targets(recent_deltas, bios_to_ui, max_items=max_items)
    return {
        "current_button": recent_buttons[0] if recent_buttons else None,
        "recent_buttons": recent_buttons,
    }


def build_prompt_recent_deltas(
    recent_deltas: list[Mapping[str, Any]],
    bios_to_ui: BiosUiMapper | Mapping[str, Any],
    *,
    max_items: int = DEFAULT_RECENT_DELTA_MAX_ITEMS,
) -> list[dict[str, Any]]:
    """
    Convert recent delta frames to prompt-friendly rows (newest first).
    """
    limit = _coerce_nonnegative_limit(max_items, name="max_items")
    if not recent_deltas or limit == 0:
        return []

    out: list[dict[str, Any]] = []
    for item in reversed(recent_deltas):
        if not isinstance(item, Mapping):
            continue
        t_wall, seq, delta = _extract_frame_delta(item)
        if not delta:
            continue
        for key, value in delta.items():
            if not isinstance(key, str) or not key:
                continue
            targets = _targets_for_key(bios_to_ui, key)
            if targets:
                for target in targets:
                    out.append(
                        {
                            "k": key,
                            "to": value,
                            "mapped_ui_target": target,
                            "action": "delta",
                            "seq": seq,
                            "t_wall": t_wall,
                        }
                    )
                    if len(out) >= limit:
                        return out
            else:
                out.append(
                    {
                        "k": key,
                        "to": value,
                        "action": "delta",
                        "seq": seq,
                        "t_wall": t_wall,
                    }
                )
                if len(out) >= limit:
                    return out
    return out


__all__ = [
    "DEFAULT_RECENT_DELTA_MAX_ITEMS",
    "DEFAULT_RECENT_DELTA_WINDOW_S",
    "DEFAULT_RECENT_UI_TARGETS_MAX_ITEMS",
    "RecentActionsConfig",
    "RecentDeltaFrame",
    "RecentDeltaRingBuffer",
    "build_prompt_recent_deltas",
    "build_recent_actions",
    "build_recent_button_signal",
    "project_recent_ui_targets",
]
