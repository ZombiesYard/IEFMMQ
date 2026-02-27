from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class DeltaPolicyError(ValueError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_policy_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "delta_policy.yaml"


def _to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class DeltaPolicy:
    ignore_bios_prefixes: tuple[str, ...] = ()
    ignore_bios_keys: frozenset[str] = frozenset()
    debounce_ms_by_key: dict[str, int] = field(default_factory=dict)
    epsilon_by_key: dict[str, float] = field(default_factory=dict)
    max_changes_per_window: int = 12
    important_bios_keys: frozenset[str] = frozenset()

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "DeltaPolicy":
        p = Path(path) if path else _default_policy_path()
        try:
            raw_text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise DeltaPolicyError(f"delta policy read failed: {p}") from exc
        try:
            raw = yaml.safe_load(raw_text) or {}
        except yaml.YAMLError as exc:
            raise DeltaPolicyError(f"delta policy contains invalid YAML: {p}") from exc
        if not isinstance(raw, dict):
            raise DeltaPolicyError(f"delta policy must be mapping: {p}")

        prefixes = tuple(
            x for x in raw.get("ignore_bios_prefixes", []) if isinstance(x, str) and x
        )
        keys = frozenset(x for x in raw.get("ignore_bios_keys", []) if isinstance(x, str) and x)

        debounce_ms: dict[str, int] = {}
        for key, value in (raw.get("debounce_ms_by_key") or {}).items():
            if isinstance(key, str) and key and isinstance(value, (int, float)):
                debounce_ms[key] = max(0, int(value))

        epsilon: dict[str, float] = {}
        for key, value in (raw.get("epsilon_by_key") or {}).items():
            if isinstance(key, str) and key and isinstance(value, (int, float)):
                epsilon[key] = max(0.0, float(value))

        max_changes = raw.get("max_changes_per_window", 12)
        if not isinstance(max_changes, (int, float)):
            max_changes = 12
        max_changes_int = max(1, int(max_changes))

        important = frozenset(
            x for x in (raw.get("important_bios_keys") or []) if isinstance(x, str) and x
        )
        return cls(
            ignore_bios_prefixes=prefixes,
            ignore_bios_keys=keys,
            debounce_ms_by_key=debounce_ms,
            epsilon_by_key=epsilon,
            max_changes_per_window=max_changes_int,
            important_bios_keys=important,
        )

    def should_ignore_key(self, key: str) -> bool:
        if key in self.ignore_bios_keys:
            return True
        return any(key.startswith(prefix) for prefix in self.ignore_bios_prefixes)

    def debounce_ms_for(self, key: str) -> int:
        return int(self.debounce_ms_by_key.get(key, 0))

    def epsilon_for(self, key: str) -> float:
        return float(self.epsilon_by_key.get(key, 0.0))


@dataclass
class SanitizedDelta:
    kept: dict[str, Any]
    dropped_by_reason: dict[str, int]
    raw_count: int
    kept_count: int
    dropped_count: int
    seq: int | None = None
    t_wall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "t_wall": self.t_wall,
            "kept": dict(self.kept),
            "raw_count": self.raw_count,
            "kept_count": self.kept_count,
            "dropped_count": self.dropped_count,
            "dropped_by_reason": dict(self.dropped_by_reason),
        }


class DeltaSanitizer:
    def __init__(self, policy: DeltaPolicy):
        self.policy = policy
        self._last_emitted_ms: dict[str, int] = {}
        self._last_seen_value: dict[str, Any] = {}
        self._pending: dict[str, tuple[Any, int]] = {}

    def sanitize_delta(
        self,
        raw_delta: Mapping[str, Any] | None,
        *,
        t_wall: float | None = None,
        seq: int | None = None,
    ) -> SanitizedDelta:
        if not isinstance(raw_delta, Mapping):
            raw_delta = {}
        now_ms = int(t_wall * 1000) if isinstance(t_wall, (int, float)) else None
        kept: dict[str, Any] = {}
        dropped: dict[str, int] = {}
        touched_keys: set[str] = set()

        for key, value in raw_delta.items():
            if not isinstance(key, str) or not key:
                dropped["invalid_key"] = dropped.get("invalid_key", 0) + 1
                continue
            touched_keys.add(key)

            if self.policy.should_ignore_key(key):
                dropped["blacklist"] = dropped.get("blacklist", 0) + 1
                self._last_seen_value[key] = value
                continue

            prev_value = self._last_seen_value.get(key)
            eps = self.policy.epsilon_for(key)
            if eps > 0:
                prev_num = _to_number(prev_value)
                curr_num = _to_number(value)
                if prev_num is not None and curr_num is not None and abs(curr_num - prev_num) < eps:
                    dropped["epsilon"] = dropped.get("epsilon", 0) + 1
                    self._last_seen_value[key] = value
                    continue

            debounce_ms = self.policy.debounce_ms_for(key)
            if debounce_ms > 0 and now_ms is not None:
                last_emit = self._last_emitted_ms.get(key)
                if last_emit is not None and (now_ms - last_emit) < debounce_ms:
                    self._pending[key] = (value, now_ms)
                    dropped["debounce"] = dropped.get("debounce", 0) + 1
                    self._last_seen_value[key] = value
                    continue

            kept[key] = value
            self._last_seen_value[key] = value
            if now_ms is not None:
                self._last_emitted_ms[key] = now_ms
            self._pending.pop(key, None)

        # Flush pending values after debounce window closes, while avoiding duplicates.
        if now_ms is not None:
            for key, (value, _seen_ms) in list(self._pending.items()):
                if key in touched_keys:
                    continue
                debounce_ms = self.policy.debounce_ms_for(key)
                last_emit = self._last_emitted_ms.get(key)
                if last_emit is None or (now_ms - last_emit) < debounce_ms:
                    continue
                kept[key] = value
                self._last_seen_value[key] = value
                self._last_emitted_ms[key] = now_ms
                del self._pending[key]

        raw_count = len(raw_delta)
        kept_count = len(kept)
        dropped_count = max(0, raw_count - kept_count)
        return SanitizedDelta(
            kept=kept,
            dropped_by_reason=dropped,
            raw_count=raw_count,
            kept_count=kept_count,
            dropped_count=dropped_count,
            seq=seq,
            t_wall=float(t_wall) if isinstance(t_wall, (int, float)) else None,
        )


def sanitize_delta(
    raw_delta: Mapping[str, Any] | None,
    policy: DeltaPolicy,
    *,
    t_wall: float | None = None,
    seq: int | None = None,
) -> SanitizedDelta:
    return DeltaSanitizer(policy).sanitize_delta(raw_delta, t_wall=t_wall, seq=seq)


__all__ = [
    "DeltaPolicy",
    "DeltaPolicyError",
    "DeltaSanitizer",
    "SanitizedDelta",
    "sanitize_delta",
]
