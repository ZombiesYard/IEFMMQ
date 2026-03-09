from __future__ import annotations

from core.types_v2 import VisionObservation

from adapters.vision_sync import BufferedVisionSession, select_help_cycle_frames


def _vision_obs(frame_id: str, capture_wall_ms: int) -> VisionObservation:
    return VisionObservation(
        frame_id=frame_id,
        source="vision_test",
        capture_wall_ms=capture_wall_ms,
        frame_seq=int(frame_id.rsplit("_", 1)[1]),
        layout_id="fa18c_composite_panel_v2",
        channel="composite_panel",
        image_uri=f"/tmp/{frame_id}.png",
    )


def test_select_help_cycle_frames_returns_pre_and_trigger_frames() -> None:
    selection = select_help_cycle_frames(
        [
            _vision_obs("1772872444950_000122", 1772872444950),
            _vision_obs("1772872445010_000123", 1772872445010),
        ],
        trigger_wall_ms=1772872445000,
        sync_window_ms=250,
    )

    assert selection.status == "available"
    assert selection.frame_ids == ["1772872444950_000122", "1772872445010_000123"]
    assert selection.pre_trigger_frame is not None
    assert selection.pre_trigger_frame["sync_status"] == "matched_past"
    assert selection.pre_trigger_frame["sync_delta_ms"] == -50
    assert selection.trigger_frame is not None
    assert selection.trigger_frame["sync_status"] == "matched_future"
    assert selection.trigger_frame["sync_delta_ms"] == 10


def test_select_help_cycle_frames_marks_partial_when_trigger_frame_missing() -> None:
    selection = select_help_cycle_frames(
        [
            _vision_obs("1772872444950_000122", 1772872444950),
        ],
        trigger_wall_ms=1772872445000,
        sync_window_ms=250,
    )

    assert selection.status == "partial"
    assert selection.pre_trigger_frame is not None
    assert selection.trigger_frame is None
    assert selection.sync_miss_reason == "missing_trigger_frame"
    assert selection.frame_ids == ["1772872444950_000122"]


def test_select_help_cycle_frames_marks_vision_unavailable_when_out_of_window() -> None:
    selection = select_help_cycle_frames(
        [
            _vision_obs("1772872444500_000120", 1772872444500),
        ],
        trigger_wall_ms=1772872445000,
        sync_window_ms=100,
    )

    assert selection.status == "vision_unavailable"
    assert selection.pre_trigger_frame is None
    assert selection.trigger_frame is None
    assert selection.frame_ids == []
    assert selection.sync_miss_reason == "no_frame_within_window"


def test_buffered_vision_session_waits_for_trigger_frame_in_live_mode(monkeypatch) -> None:
    class SequencedVisionPort:
        def __init__(self) -> None:
            self.poll_calls = 0

        def start(self, session_id: str) -> None:
            assert session_id == "sess-live"

        def poll(self) -> list[VisionObservation]:
            self.poll_calls += 1
            if self.poll_calls == 1:
                return [_vision_obs("1772872444950_000122", 1772872444950)]
            if self.poll_calls == 2:
                return [_vision_obs("1772872445010_000123", 1772872445010)]
            return []

        def stop(self) -> None:
            return

    sleeps: list[float] = []
    monkeypatch.setattr("adapters.vision_sync.time.sleep", lambda value: sleeps.append(float(value)))

    session = BufferedVisionSession(
        vision_port=SequencedVisionPort(),
        session_id="sess-live",
        sync_window_ms=250,
        trigger_wait_ms=50,
        live_mode=True,
    )
    try:
        selection = session.select_for_help(trigger_wall_s=1772872445.0)
    finally:
        session.close()

    assert selection.status == "available"
    assert selection.frame_ids == ["1772872444950_000122", "1772872445010_000123"]
    assert sleeps

