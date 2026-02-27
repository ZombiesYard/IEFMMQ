from pathlib import Path

from adapters.dcs_bios.bios_ui_map import BiosUiMapper
from adapters.delta_aggregator import DeltaAggregator, aggregate_delta_window, emit_delta_sanitized_event
from adapters.delta_sanitizer import DeltaPolicy, SanitizedDelta


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = REPO_ROOT / "packs" / "fa18c_startup"


def _mapper() -> BiosUiMapper:
    return BiosUiMapper.from_yaml(PACK_DIR / "bios_to_ui.yaml", PACK_DIR / "ui_map.yaml")


def test_aggregator_enforces_topk_budget_and_collects_targets() -> None:
    policy = DeltaPolicy(
        max_changes_per_window=2,
        important_bios_keys=frozenset({"BATTERY_SW", "ENGINE_CRANK_SW"}),
    )
    window = [
        SanitizedDelta(
            kept={"BATTERY_SW": 2, "RANDOM_A": 1},
            dropped_by_reason={"blacklist": 2},
            raw_count=4,
            kept_count=2,
            dropped_count=2,
            seq=1,
            t_wall=1.0,
        ),
        SanitizedDelta(
            kept={"ENGINE_CRANK_SW": 1, "RANDOM_B": 2},
            dropped_by_reason={"debounce": 1},
            raw_count=3,
            kept_count=2,
            dropped_count=1,
            seq=2,
            t_wall=2.0,
        ),
    ]

    summary = aggregate_delta_window(window, policy=policy, mapper=_mapper())

    assert len(summary.recent_key_changes_topk) == 2
    keys = [row["key"] for row in summary.recent_key_changes_topk]
    assert "BATTERY_SW" in keys
    assert "ENGINE_CRANK_SW" in keys
    assert "battery_switch" in summary.recent_ui_targets
    assert "eng_crank_switch" in summary.recent_ui_targets
    assert summary.dropped_stats["dropped_total"] == 3
    assert summary.dropped_stats["dropped_by_reason"]["blacklist"] == 2
    assert summary.dropped_stats["dropped_by_reason"]["debounce"] == 1


def test_delta_aggregator_stateful_add_and_event_payload_is_stats_only() -> None:
    policy = DeltaPolicy(max_changes_per_window=3)
    agg = DeltaAggregator(policy, mapper=_mapper(), window_size=3)

    summary = agg.add(
        SanitizedDelta(
            kept={"BATTERY_SW": 2},
            dropped_by_reason={},
            raw_count=1,
            kept_count=1,
            dropped_count=0,
            seq=1,
            t_wall=1.0,
        )
    )
    summary = agg.add(
        SanitizedDelta(
            kept={"ENGINE_CRANK_SW": 1},
            dropped_by_reason={"epsilon": 2},
            raw_count=3,
            kept_count=1,
            dropped_count=2,
            seq=2,
            t_wall=2.0,
        )
    )

    event = emit_delta_sanitized_event(summary, related_id="obs-1")
    assert event.kind == "delta_sanitized"
    assert event.related_id == "obs-1"
    assert "dropped_stats" in event.payload
    assert "recent_key_changes_count" in event.payload
    assert "recent_key_changes_topk" not in event.payload
