"""
Shared buffering and help-cycle frame selection for live/replay vision sidecars.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any, Callable, Iterable

from core.types_v2 import VisionObservation
from ports.vision_port import VisionPort

DEFAULT_LIVE_SYNC_WINDOW_MS = 250
DEFAULT_REPLAY_SYNC_WINDOW_MS = 100
DEFAULT_LIVE_TRIGGER_WAIT_MS = 250
_POLL_SLEEP_S = 0.02


def _capture_wall_ms(observation: VisionObservation) -> int | None:
    if isinstance(observation.capture_wall_ms, int) and observation.capture_wall_ms >= 0:
        return observation.capture_wall_ms
    timestamp = observation.timestamp
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def _frame_payload(
    observation: VisionObservation,
    *,
    role: str,
    trigger_wall_ms: int,
    sync_status: str,
) -> dict[str, Any]:
    capture_wall_ms = _capture_wall_ms(observation)
    sync_delta_ms = None if capture_wall_ms is None else capture_wall_ms - trigger_wall_ms
    return {
        "role": role,
        "frame_id": observation.frame_id,
        "capture_wall_ms": capture_wall_ms,
        "frame_seq": observation.frame_seq,
        "layout_id": observation.layout_id,
        "channel": observation.channel,
        "image_uri": observation.image_uri,
        "source_image_path": observation.source_image_path,
        "width": observation.width,
        "height": observation.height,
        "sync_status": sync_status,
        "sync_delta_ms": sync_delta_ms,
        "frame_stale": bool(sync_delta_ms is not None and sync_delta_ms < 0),
        "sync_miss_reason": None,
    }


def _primary_sync_payload(
    observation: VisionObservation | None,
    *,
    trigger_wall_ms: int,
    sync_status: str | None,
) -> dict[str, Any]:
    if observation is None or sync_status is None:
        return {
            "frame_id": None,
            "sync_status": None,
            "sync_delta_ms": None,
            "frame_stale": None,
        }
    payload = _frame_payload(
        observation,
        role="resolved_frame",
        trigger_wall_ms=trigger_wall_ms,
        sync_status=sync_status,
    )
    return {
        "frame_id": payload["frame_id"],
        "sync_status": payload["sync_status"],
        "sync_delta_ms": payload["sync_delta_ms"],
        "frame_stale": payload["frame_stale"],
    }


def _observation_anchor_wall_ms(*, observation_t_wall_s: float | None, trigger_wall_ms: int) -> int:
    if isinstance(observation_t_wall_s, (int, float)):
        return int(round(float(observation_t_wall_s) * 1000.0))
    return int(trigger_wall_ms)


def _trigger_sync_status(*, capture_wall_ms: int, trigger_wall_ms: int) -> str:
    if capture_wall_ms == trigger_wall_ms:
        return "matched_exact"
    return "matched_future_fallback"


@dataclass(frozen=True)
class HelpCycleVisionSelection:
    status: str
    observation_ref: str | None
    observation_seq: int | None
    observation_t_wall_s: float | None
    observation_t_wall_ms: int
    trigger_wall_ms: int
    sync_window_ms: int
    vision_used: bool
    frame_id: str | None
    sync_status: str | None
    sync_delta_ms: int | None
    frame_stale: bool | None
    frame_ids: list[str]
    selected_frames: list[dict[str, Any]]
    pre_trigger_frame: dict[str, Any] | None
    trigger_frame: dict[str, Any] | None
    sync_miss_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observation_ref": self.observation_ref,
            "observation_seq": self.observation_seq,
            "observation_t_wall_s": self.observation_t_wall_s,
            "observation_t_wall_ms": self.observation_t_wall_ms,
            "trigger_wall_ms": self.trigger_wall_ms,
            "sync_window_ms": self.sync_window_ms,
            "vision_used": self.vision_used,
            "frame_id": self.frame_id,
            "sync_status": self.sync_status,
            "sync_delta_ms": self.sync_delta_ms,
            "frame_stale": self.frame_stale,
            "frame_ids": list(self.frame_ids),
            "selected_frames": [dict(item) for item in self.selected_frames],
            "pre_trigger_frame": dict(self.pre_trigger_frame) if self.pre_trigger_frame is not None else None,
            "trigger_frame": dict(self.trigger_frame) if self.trigger_frame is not None else None,
            "sync_miss_reason": self.sync_miss_reason,
        }


def _select_primary_observation(
    observations: Iterable[tuple[int, str, VisionObservation]],
    *,
    trigger_wall_ms: int,
    sync_window_ms: int,
) -> tuple[VisionObservation | None, str | None]:
    latest_past: tuple[int, VisionObservation] | None = None
    earliest_future: tuple[int, VisionObservation] | None = None
    for capture_wall_ms, _frame_id, observation in observations:
        delta_ms = capture_wall_ms - trigger_wall_ms
        if abs(delta_ms) > sync_window_ms:
            continue
        if delta_ms <= 0:
            latest_past = (capture_wall_ms, observation)
            continue
        if earliest_future is None:
            earliest_future = (capture_wall_ms, observation)

    if latest_past is not None:
        capture_wall_ms, observation = latest_past
        if capture_wall_ms == trigger_wall_ms:
            return observation, "matched_exact"
        return observation, "matched_past"
    if earliest_future is not None:
        return earliest_future[1], "matched_future_fallback"
    return None, None


def select_help_cycle_frames(
    observations: Iterable[VisionObservation],
    *,
    trigger_wall_ms: int,
    sync_window_ms: int,
    observation_ref: str | None = None,
    observation_seq: int | None = None,
    observation_t_wall_s: float | None = None,
) -> HelpCycleVisionSelection:
    window_ms = max(0, int(sync_window_ms))
    observation_t_wall_ms = _observation_anchor_wall_ms(
        observation_t_wall_s=observation_t_wall_s,
        trigger_wall_ms=trigger_wall_ms,
    )
    normalized: list[tuple[int, str, VisionObservation]] = []
    seen_frame_ids: set[str] = set()
    for observation in observations:
        capture_wall_ms = _capture_wall_ms(observation)
        if capture_wall_ms is None:
            continue
        if observation.frame_id in seen_frame_ids:
            continue
        seen_frame_ids.add(observation.frame_id)
        normalized.append((capture_wall_ms, observation.frame_id, observation))
    normalized.sort(key=lambda item: (item[0], item[1]))

    resolved_observation, resolved_sync_status = _select_primary_observation(
        normalized,
        trigger_wall_ms=trigger_wall_ms,
        sync_window_ms=window_ms,
    )
    resolved_sync = _primary_sync_payload(
        resolved_observation,
        trigger_wall_ms=trigger_wall_ms,
        sync_status=resolved_sync_status,
    )

    pre_observation: VisionObservation | None = None
    trigger_observation: VisionObservation | None = None
    for capture_wall_ms, _frame_id, observation in normalized:
        delta_ms = capture_wall_ms - trigger_wall_ms
        if delta_ms < 0 and abs(delta_ms) <= window_ms:
            pre_observation = observation
            continue
        if delta_ms >= 0 and delta_ms <= window_ms and trigger_observation is None:
            trigger_observation = observation

    pre_payload = (
        _frame_payload(
            pre_observation,
            role="pre_trigger_frame",
            trigger_wall_ms=trigger_wall_ms,
            sync_status="matched_past",
        )
        if pre_observation is not None
        else None
    )
    trigger_payload = (
        _frame_payload(
            trigger_observation,
            role="trigger_frame",
            trigger_wall_ms=trigger_wall_ms,
            sync_status=_trigger_sync_status(
                capture_wall_ms=_capture_wall_ms(trigger_observation) or trigger_wall_ms,
                trigger_wall_ms=trigger_wall_ms,
            ),
        )
        if trigger_observation is not None
        else None
    )

    selected_frames = [
        item
        for item in (pre_payload, trigger_payload)
        if item is not None
    ]
    frame_ids = [str(item["frame_id"]) for item in selected_frames if isinstance(item.get("frame_id"), str)]

    if pre_payload is not None and trigger_payload is not None:
        status = "available"
        sync_miss_reason = None
    elif selected_frames:
        status = "partial"
        if pre_payload is None:
            sync_miss_reason = "missing_pre_trigger_frame"
        else:
            sync_miss_reason = "missing_trigger_frame"
    else:
        status = "vision_unavailable"
        sync_miss_reason = "no_frame_within_window"

    return HelpCycleVisionSelection(
        status=status,
        observation_ref=observation_ref,
        observation_seq=observation_seq,
        observation_t_wall_s=observation_t_wall_s,
        observation_t_wall_ms=observation_t_wall_ms,
        trigger_wall_ms=int(trigger_wall_ms),
        sync_window_ms=window_ms,
        vision_used=bool(selected_frames),
        frame_id=resolved_sync["frame_id"],
        sync_status=resolved_sync["sync_status"],
        sync_delta_ms=resolved_sync["sync_delta_ms"],
        frame_stale=resolved_sync["frame_stale"],
        frame_ids=frame_ids,
        selected_frames=selected_frames,
        pre_trigger_frame=pre_payload,
        trigger_frame=trigger_payload,
        sync_miss_reason=sync_miss_reason,
    )


class BufferedVisionSession:
    def __init__(
        self,
        *,
        vision_port: VisionPort,
        session_id: str,
        sync_window_ms: int,
        trigger_wait_ms: int = 0,
        retention_ms: int | None = None,
        live_mode: bool,
        observation_sink: Callable[[VisionObservation], None] | None = None,
    ) -> None:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("vision session_id must be a non-empty string")
        self.vision_port = vision_port
        self.session_id = session_id
        self.sync_window_ms = max(0, int(sync_window_ms))
        self.trigger_wait_ms = max(0, int(trigger_wait_ms))
        self.retention_ms = (
            max(0, int(retention_ms))
            if isinstance(retention_ms, int)
            else max(2000, self.sync_window_ms * 4 + self.trigger_wait_ms)
        )
        self.live_mode = bool(live_mode)
        self.observation_sink = observation_sink
        self._history: list[VisionObservation] = []
        self._frame_ids: set[str] = set()
        self.vision_port.start(session_id)

    @staticmethod
    def _history_key(observation: VisionObservation) -> tuple[int, str]:
        capture_wall_ms = _capture_wall_ms(observation)
        return ((capture_wall_ms if capture_wall_ms is not None else -1), observation.frame_id)

    def _insert_observation(self, observation: VisionObservation) -> None:
        key = self._history_key(observation)
        if not self._history:
            self._history.append(observation)
            return
        if key >= self._history_key(self._history[-1]):
            self._history.append(observation)
            return
        keys = [self._history_key(item) for item in self._history]
        insert_at = bisect_left(keys, key)
        self._history.insert(insert_at, observation)

    def _prune_history(self) -> None:
        if not self._history or self.retention_ms <= 0:
            return
        latest_capture_wall_ms = _capture_wall_ms(self._history[-1])
        if latest_capture_wall_ms is None:
            return
        cutoff_wall_ms = latest_capture_wall_ms - self.retention_ms
        prune_count = 0
        for observation in self._history:
            capture_wall_ms = _capture_wall_ms(observation)
            if capture_wall_ms is None or capture_wall_ms < cutoff_wall_ms:
                prune_count += 1
                continue
            break
        if prune_count <= 0:
            return
        removed = self._history[:prune_count]
        self._history = self._history[prune_count:]
        for observation in removed:
            self._frame_ids.discard(observation.frame_id)

    def poll(self) -> list[VisionObservation]:
        observations = self.vision_port.poll()
        added: list[VisionObservation] = []
        for observation in observations:
            if observation.frame_id in self._frame_ids:
                continue
            self._frame_ids.add(observation.frame_id)
            self._insert_observation(observation)
            added.append(observation)
            if self.observation_sink is not None:
                self.observation_sink(observation)
        self._prune_history()
        return added

    def select_for_help(self, *, trigger_wall_s: float) -> HelpCycleVisionSelection:
        trigger_wall_ms = int(round(float(trigger_wall_s) * 1000.0))
        deadline = time.monotonic() + (self.trigger_wait_ms / 1000.0)
        while True:
            self.poll()
            selection = select_help_cycle_frames(
                self._history,
                trigger_wall_ms=trigger_wall_ms,
                sync_window_ms=self.sync_window_ms,
            )
            if not self.live_mode:
                return selection
            if selection.frame_id is not None:
                return selection
            if time.monotonic() >= deadline:
                return selection
            time.sleep(_POLL_SLEEP_S)

    def close(self) -> None:
        self.vision_port.stop()


__all__ = [
    "BufferedVisionSession",
    "DEFAULT_LIVE_SYNC_WINDOW_MS",
    "DEFAULT_LIVE_TRIGGER_WAIT_MS",
    "DEFAULT_REPLAY_SYNC_WINDOW_MS",
    "HelpCycleVisionSelection",
    "select_help_cycle_frames",
]
