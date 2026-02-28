import pytest

from adapters.delta_aggregator import DeltaSummary
from adapters.delta_sanitizer import SanitizedDelta
from adapters.recent_actions import (
    RecentDeltaRingBuffer,
    build_prompt_recent_deltas,
    build_recent_actions,
    build_recent_button_signal,
    project_recent_ui_targets,
)


def _bios_to_ui_mapping() -> dict:
    return {
        "mappings": {
            "BATTERY_SW": ["battery_switch"],
            "ENGINE_CRANK_SW": ["eng_crank_switch"],
            "COMM1_CHAN": {
                "targets": [
                    "ufc_comm1_channel_selector_rotate",
                    "ufc_comm1_channel_selector_pull",
                ]
            },
        }
    }


def test_build_recent_actions_from_sanitized_delta() -> None:
    sanitized = SanitizedDelta(
        kept={"BATTERY_SW": 2, "ENGINE_CRANK_SW": 1},
        dropped_by_reason={},
        raw_count=2,
        kept_count=2,
        dropped_count=0,
        seq=1,
        t_wall=1.0,
    )
    out = build_recent_actions(sanitized, top_k=1)
    assert out == [{"k": "BATTERY_SW", "to": 2}]


def test_build_recent_actions_from_delta_summary() -> None:
    summary = DeltaSummary(
        recent_ui_targets=["battery_switch"],
        recent_key_changes_topk=[
            {"key": "BATTERY_SW", "value": 2, "ui_targets": ["battery_switch"], "importance": 110},
            {"key": "ENGINE_CRANK_SW", "value": 1, "ui_targets": ["eng_crank_switch"], "importance": 100},
        ],
        dropped_stats={"dropped_total": 3, "dropped_by_reason": {"blacklist": 3}},
        raw_changes_total=5,
        kept_changes_total=2,
        window_size=2,
    )
    out = build_recent_actions(summary, top_k=2)
    assert out[0]["k"] == "BATTERY_SW"
    assert out[0]["ui_targets"] == ["battery_switch"]
    assert out[1]["k"] == "ENGINE_CRANK_SW"


def test_build_recent_actions_rejects_raw_delta_mapping() -> None:
    with pytest.raises(TypeError, match="SanitizedDelta or DeltaSummary"):
        build_recent_actions({"BATTERY_SW": 2})  # type: ignore[arg-type]


def test_recent_delta_ring_buffer_trims_by_window_and_max_items() -> None:
    ring = RecentDeltaRingBuffer(window_s=3.0, max_items=2)

    ring.add_delta({"BATTERY_SW": 2}, t_wall=10.0, seq=1)
    ring.add_delta({"COMM1_CHAN": 11}, t_wall=11.0, seq=2)

    snap_after_count_trim = ring.add_delta({"ENGINE_CRANK_SW": 1}, t_wall=12.0, seq=3)
    assert [item["seq"] for item in snap_after_count_trim] == [2, 3]

    snap_after_window_trim = ring.add_delta({"BATTERY_SW": 0}, t_wall=15.2, seq=4)
    assert [item["seq"] for item in snap_after_window_trim] == [4]


def test_project_recent_ui_targets_is_recency_first_and_stable() -> None:
    recent = [
        {"t_wall": 10.0, "seq": 1, "delta": {"BATTERY_SW": 2, "COMM1_CHAN": 11}},
        {"t_wall": 11.0, "seq": 2, "delta": {"ENGINE_CRANK_SW": 1, "COMM1_CHAN": 12}},
    ]

    targets = project_recent_ui_targets(recent, _bios_to_ui_mapping(), max_items=8)

    assert targets == [
        "eng_crank_switch",
        "ufc_comm1_channel_selector_rotate",
        "ufc_comm1_channel_selector_pull",
        "battery_switch",
    ]


def test_build_prompt_recent_deltas_and_button_signal_respects_caps() -> None:
    recent = [
        {"t_wall": 10.0, "seq": 1, "delta": {"BATTERY_SW": 2}},
        {"t_wall": 11.0, "seq": 2, "delta": {"ENGINE_CRANK_SW": 1, "COMM1_CHAN": 12}},
    ]

    rows = build_prompt_recent_deltas(recent, _bios_to_ui_mapping(), max_items=3)
    assert len(rows) == 3
    assert rows[0]["mapped_ui_target"] == "eng_crank_switch"
    assert rows[1]["mapped_ui_target"] == "ufc_comm1_channel_selector_rotate"
    assert rows[2]["mapped_ui_target"] == "ufc_comm1_channel_selector_pull"

    signal = build_recent_button_signal(recent, _bios_to_ui_mapping(), max_items=3)
    assert signal["current_button"] == "eng_crank_switch"
    assert signal["recent_buttons"] == [
        "eng_crank_switch",
        "ufc_comm1_channel_selector_rotate",
        "ufc_comm1_channel_selector_pull",
    ]
