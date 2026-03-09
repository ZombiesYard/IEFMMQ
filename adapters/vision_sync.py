"""
Shared buffering and help-cycle frame selection for live/replay vision sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any, Iterable

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


@dataclass(frozen=True)
class HelpCycleVisionSelection:
    status: str
    trigger_wall_ms: int
    sync_window_ms: int
    vision_used: bool
    frame_ids: list[str]
    selected_frames: list[dict[str, Any]]
    pre_trigger_frame: dict[str, Any] | None
    trigger_frame: dict[str, Any] | None
    sync_miss_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "trigger_wall_ms": self.trigger_wall_ms,
            "sync_window_ms": self.sync_window_ms,
            "vision_used": self.vision_used,
            "frame_ids": list(self.frame_ids),
            "selected_frames": [dict(item) for item in self.selected_frames],
            "pre_trigger_frame": dict(self.pre_trigger_frame) if self.pre_trigger_frame is not None else None,
            "trigger_frame": dict(self.trigger_frame) if self.trigger_frame is not None else None,
            "sync_miss_reason": self.sync_miss_reason,
        }


def select_help_cycle_frames(
    observations: Iterable[VisionObservation],
    *,
    trigger_wall_ms: int,
    sync_window_ms: int,
) -> HelpCycleVisionSelection:
    window_ms = max(0, int(sync_window_ms))
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

    pre_observation: VisionObservation | None = None
    trigger_observation: VisionObservation | None = None
    for capture_wall_ms, _frame_id, observation in normalized:
        delta_ms = capture_wall_ms - trigger_wall_ms
        if delta_ms <= 0 and abs(delta_ms) <= window_ms:
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
            sync_status="matched_future",
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
        trigger_wall_ms=int(trigger_wall_ms),
        sync_window_ms=window_ms,
        vision_used=bool(selected_frames),
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
        live_mode: bool,
    ) -> None:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("vision session_id must be a non-empty string")
        self.vision_port = vision_port
        self.session_id = session_id
        self.sync_window_ms = max(0, int(sync_window_ms))
        self.trigger_wait_ms = max(0, int(trigger_wait_ms))
        self.live_mode = bool(live_mode)
        self._history: list[VisionObservation] = []
        self._frame_ids: set[str] = set()
        self.vision_port.start(session_id)

    def poll(self) -> list[VisionObservation]:
        observations = self.vision_port.poll()
        added: list[VisionObservation] = []
        for observation in observations:
            if observation.frame_id in self._frame_ids:
                continue
            self._frame_ids.add(observation.frame_id)
            self._history.append(observation)
            added.append(observation)
        self._history.sort(key=lambda item: ((_capture_wall_ms(item) or -1), item.frame_id))
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
            if selection.trigger_frame is not None:
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
