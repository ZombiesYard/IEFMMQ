import pytest

from adapters.delta_aggregator import DeltaSummary
from adapters.delta_sanitizer import SanitizedDelta
from adapters.recent_actions import build_recent_actions


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
