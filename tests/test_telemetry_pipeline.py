import json
from collections import OrderedDict
from pathlib import Path

import pytest

from adapters.delta_aggregator import DeltaAggregator
from adapters.delta_sanitizer import DeltaPolicy, DeltaSanitizer
from adapters.dcs_bios.bios_ui_map import BiosUiMapper
import adapters.telemetry_pipeline as telemetry_pipeline
from adapters.telemetry_pipeline import TelemetryDebugCache, enrich_bios_observation
from core.types import Event
from core.types import Observation
from core.vars import VarResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = REPO_ROOT / "packs" / "fa18c_startup"


def _resolver() -> VarResolver:
    return VarResolver.from_yaml(PACK_DIR / "telemetry_map.yaml")


def _mapper() -> BiosUiMapper:
    return BiosUiMapper.from_yaml(PACK_DIR / "bios_to_ui.yaml", PACK_DIR / "ui_map.yaml")


def _delta_policy() -> DeltaPolicy:
    return DeltaPolicy.from_yaml(PACK_DIR / "delta_policy.yaml")


def test_enrich_bios_observation_compacts_payload_and_keeps_metadata() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "schema_version": "v2",
            "seq": 42,
            "t_wall": 123.45,
            "aircraft": "FA-18C_hornet",
            "bios": {
                "BATTERY_SW": 2,
                "EXT_PWR_SW": 1,
                "L_GEN_SW": 1,
                "R_GEN_SW": 1,
                "ENGINE_CRANK_SW": 1,
                "IFEI_RPM_R": " 64",
                "IFEI_RPM_L": " 63",
                "IFEI_TEMP_R": "320",
                "IFEI_TEMP_L": "300",
                "IFEI_FF_R": "500",
                "IFEI_FF_L": "480",
                "IFEI_OIL_PRESS_R": "80",
                "IFEI_OIL_PRESS_L": "78",
                "EXT_NOZZLE_POS_R": 50010,
                "EXT_NOZZLE_POS_L": 49000,
            },
            "delta": {
                "BATTERY_SW": 2,
                "ENGINE_CRANK_SW": 1,
                "IFEI_RPM_R": " 64",
                "IFEI_RPM_L": " 63",
            },
        },
        metadata={"seq": 42, "gap": 0, "delta_count": 4, "from_addr": "127.0.0.1:5000"},
    )

    enriched = enrich_bios_observation(obs, _resolver(), mapper=_mapper())

    assert set(enriched.payload.keys()) == {"seq", "t_wall", "vars", "delta_summary", "recent_ui_targets"}
    assert "bios" not in enriched.payload
    assert "delta" not in enriched.payload
    assert enriched.payload["seq"] == 42
    assert enriched.payload["t_wall"] == 123.45
    assert enriched.payload["vars"]["rpm_r"] == 64
    assert enriched.payload["vars"]["rpm_r_gte_25"] is True
    assert enriched.payload["vars"]["fire_test_complete"] is False
    assert enriched.payload["vars"]["engine_crank_right_complete"] is True
    assert enriched.payload["vars"]["apu_start_support_complete"] is True
    assert enriched.payload["vars"]["throttle_r_idle_complete"] is True
    assert enriched.payload["delta_summary"]["delta_count"] == 4
    assert "battery_switch" in enriched.payload["recent_ui_targets"]
    assert "eng_crank_switch" in enriched.payload["recent_ui_targets"]

    assert enriched.metadata["seq"] == 42
    assert enriched.metadata["gap"] == 0
    assert enriched.metadata["delta_count"] == 4
    assert "bios_hash" not in enriched.metadata

    json.dumps(enriched.to_dict(), ensure_ascii=False)


def test_enrich_bios_observation_includes_pack_gate_vars_used_by_live_inference() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 43,
            "t_wall": 124.45,
            "bios": {
                "BATTERY_SW": 2,
                "EXT_PWR_SW": 1,
                "L_GEN_SW": 1,
                "R_GEN_SW": 1,
                "APU_READY_LT": 1,
                "ENGINE_CRANK_SW": 1,
                "RADALT_MIN_HEIGHT_PTR": 12000,
            },
            "delta": {"BATTERY_SW": 2, "APU_READY_LT": 1},
        },
        metadata={"seq": 43, "gap": 0, "delta_count": 2, "from_addr": "127.0.0.1:5000"},
    )

    enriched = enrich_bios_observation(obs, _resolver(), mapper=_mapper())

    assert enriched.payload["vars"]["apu_start_support_complete"] is True
    assert enriched.payload["vars"]["radar_altimeter_bug_value"] == 12000


def test_enrich_bios_observation_includes_probe_state_for_s19_progression() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 44,
            "t_wall": 125.45,
            "bios": {
                "BATTERY_SW": 2,
                "L_GEN_SW": 1,
                "R_GEN_SW": 1,
                "PROBE_SW": 1,
                "EXT_REFUEL_PROBE": 65535,
                "LAUNCH_BAR_SW": 1,
            },
            "delta": {"PROBE_SW": 1, "EXT_REFUEL_PROBE": 65535},
        },
        metadata={"seq": 44, "gap": 0, "delta_count": 2, "from_addr": "127.0.0.1:5000"},
    )

    enriched = enrich_bios_observation(obs, _resolver(), mapper=_mapper())

    assert enriched.payload["vars"]["probe_switch_value"] == 1
    assert enriched.payload["vars"]["ext_refuel_probe_value"] == 65535
    assert enriched.payload["vars"]["probe_extended"] is True
    assert enriched.payload["vars"]["probe_cycle_complete"] is True
    assert enriched.payload["vars"]["launch_bar_switch_value"] == 1


def test_enrich_bios_observation_latches_probe_cycle_complete_after_extension() -> None:
    extend_obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 45,
            "t_wall": 200.0,
            "bios": {
                "BATTERY_SW": 2,
                "PROBE_SW": 1,
                "EXT_REFUEL_PROBE": 65535,
            },
            "delta": {"EXT_REFUEL_PROBE": 65535},
        },
        metadata={"seq": 45, "gap": 0, "delta_count": 1, "from_addr": "127.0.0.1:5000"},
    )
    retract_obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 46,
            "t_wall": 201.0,
            "bios": {
                "BATTERY_SW": 2,
                "PROBE_SW": 1,
                "EXT_REFUEL_PROBE": 0,
            },
            "delta": {"EXT_REFUEL_PROBE": 0},
        },
        metadata={"seq": 46, "gap": 0, "delta_count": 1, "from_addr": "127.0.0.1:5000"},
    )

    extend_enriched = enrich_bios_observation(
        extend_obs,
        _resolver(),
        mapper=_mapper(),
        delta_stream_id="probe-cycle",
    )
    retract_enriched = enrich_bios_observation(
        retract_obs,
        _resolver(),
        mapper=_mapper(),
        delta_stream_id="probe-cycle",
    )

    assert extend_enriched.payload["vars"]["probe_cycle_complete"] is True
    assert retract_enriched.payload["vars"]["probe_extended"] is False
    assert retract_enriched.payload["vars"]["probe_cycle_complete"] is True


def test_enrich_bios_observation_supports_tag_hook_and_debug_cache() -> None:
    obs = Observation(
        source="dcs_bios",
        tags=["live"],
        payload={
            "seq": 3,
            "t_wall": 3.0,
            "bios": {"BATTERY_SW": 2, "IFEI_RPM_R": " 10"},
            "delta": {"BATTERY_SW": 2},
        },
        metadata={"seq": 3, "gap": 1, "delta_count": 1},
    )
    cache = TelemetryDebugCache()

    def tag_hook(_obs: Observation, payload: dict) -> list[str]:
        if payload["vars"].get("battery_on"):
            return ["power_on"]
        return []

    enriched = enrich_bios_observation(
        obs,
        _resolver(),
        mapper=_mapper(),
        tag_hook=tag_hook,
        debug_cache=cache,
    )

    assert enriched.tags == ["live", "power_on"]
    assert cache.last_seq == 3
    assert cache.last_bios.get("BATTERY_SW") == 2
    assert cache.last_delta == {"BATTERY_SW": 2}
    assert cache.last_bios_hash == enriched.metadata["bios_hash"]


def test_enrich_bios_observation_can_force_bios_hash_without_debug_cache() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 9,
            "t_wall": 9.0,
            "bios": {"BATTERY_SW": 2},
            "delta": {"BATTERY_SW": 2},
        },
        metadata={"seq": 9, "gap": 0, "delta_count": 1},
    )

    enriched = enrich_bios_observation(
        obs,
        _resolver(),
        mapper=_mapper(),
        include_bios_hash=True,
    )

    assert isinstance(enriched.metadata.get("bios_hash"), str)
    assert len(enriched.metadata["bios_hash"]) == 64


def test_enrich_bios_observation_selected_var_keys_empty_means_select_none() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 10,
            "t_wall": 10.0,
            "bios": {"BATTERY_SW": 2, "IFEI_RPM_R": " 30"},
            "delta": {"BATTERY_SW": 2},
        },
    )

    enriched = enrich_bios_observation(
        obs,
        _resolver(),
        mapper=_mapper(),
        selected_var_keys=[],
    )

    assert enriched.payload["vars"] == {}


def test_enrich_bios_observation_tag_hook_none_treated_as_no_extra_tags() -> None:
    obs = Observation(
        source="dcs_bios",
        tags=["live"],
        payload={"seq": 11, "t_wall": 11.0, "bios": {"BATTERY_SW": 2}, "delta": {"BATTERY_SW": 2}},
    )

    enriched = enrich_bios_observation(
        obs,
        _resolver(),
        mapper=_mapper(),
        tag_hook=lambda _obs, _payload: None,
    )

    assert enriched.tags == ["live"]


def test_enrich_bios_observation_tag_hook_invalid_return_raises_clear_error() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={"seq": 12, "t_wall": 12.0, "bios": {"BATTERY_SW": 2}, "delta": {"BATTERY_SW": 2}},
    )

    with pytest.raises(TypeError, match="tag_hook must return a sequence of strings or None"):
        enrich_bios_observation(
            obs,
            _resolver(),
            mapper=_mapper(),
            tag_hook=lambda _obs, _payload: 123,  # type: ignore[return-value]
        )


def test_enrich_bios_observation_sanitizes_delta_and_emits_stats_event() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 101,
            "t_wall": 101.0,
            "bios": {
                "BATTERY_SW": 2,
                "IFEI_CLOCK_S": "59",
                "EXT_STROBE_LIGHTS": 1,
            },
            "delta": {
                "BATTERY_SW": 2,
                "IFEI_CLOCK_S": "59",
                "EXT_STROBE_LIGHTS": 1,
            },
        },
        metadata={"seq": 101, "gap": 0, "delta_count": 3},
    )
    policy = _delta_policy()
    sanitizer = DeltaSanitizer(policy)
    aggregator = DeltaAggregator(policy, mapper=_mapper(), window_size=5)
    events: list[Event] = []

    enriched = enrich_bios_observation(
        obs,
        _resolver(),
        mapper=_mapper(),
        delta_policy=policy,
        delta_sanitizer=sanitizer,
        delta_aggregator=aggregator,
        delta_event_sink=lambda event: events.append(event),
    )

    summary = enriched.payload["delta_summary"]
    assert summary["raw_delta_count"] == 3
    assert summary["delta_count"] == 1
    assert summary["dropped_stats"]["dropped_total"] == 2
    assert summary["dropped_stats"]["dropped_by_reason"]["blacklist"] == 2
    assert summary["recent_key_changes_topk"][0]["key"] == "BATTERY_SW"
    assert enriched.payload["recent_ui_targets"] == ["battery_switch"]
    assert enriched.metadata["delta_dropped_count"] == 2

    assert len(events) == 1
    assert events[0].kind == "delta_sanitized"
    assert "recent_key_changes_topk" not in events[0].payload


def test_enrich_bios_observation_separates_per_obs_and_window_delta_summary() -> None:
    policy = _delta_policy()
    sanitizer = DeltaSanitizer(policy)
    aggregator = DeltaAggregator(policy, mapper=_mapper(), window_size=5)

    obs1 = Observation(
        source="dcs_bios",
        payload={
            "seq": 111,
            "t_wall": 111.0,
            "bios": {"BATTERY_SW": 2, "IFEI_CLOCK_S": "01"},
            "delta": {"BATTERY_SW": 2, "IFEI_CLOCK_S": "01"},
        },
    )
    obs2 = Observation(
        source="dcs_bios",
        payload={
            "seq": 112,
            "t_wall": 112.0,
            "bios": {"ENGINE_CRANK_SW": 1},
            "delta": {"ENGINE_CRANK_SW": 1},
        },
    )

    enrich_bios_observation(
        obs1,
        _resolver(),
        mapper=_mapper(),
        delta_policy=policy,
        delta_sanitizer=sanitizer,
        delta_aggregator=aggregator,
    )
    second = enrich_bios_observation(
        obs2,
        _resolver(),
        mapper=_mapper(),
        delta_policy=policy,
        delta_sanitizer=sanitizer,
        delta_aggregator=aggregator,
    )

    assert second.payload["delta_summary"]["dropped_stats"]["dropped_total"] == 0
    assert second.payload["delta_window_summary"]["dropped_stats"]["dropped_total"] == 1
    per_obs_keys = [row["key"] for row in second.payload["delta_summary"]["recent_key_changes_topk"]]
    win_keys = [row["key"] for row in second.payload["delta_window_summary"]["recent_key_changes_topk"]]
    assert per_obs_keys == ["ENGINE_CRANK_SW"]
    assert "BATTERY_SW" in win_keys
    assert second.metadata["delta_dropped_count"] == 0


def test_enrich_bios_observation_default_sanitizer_keeps_state_across_calls(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_DEFAULT_DELTA_POLICY", None)
    monkeypatch.setattr(telemetry_pipeline, "_DEFAULT_DELTA_SANITIZER", None)
    monkeypatch.setattr(telemetry_pipeline, "_POLICY_SCOPED_SANITIZERS", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_SHARED_DELTA_SANITIZER_LOCKS", {})

    obs1 = Observation(
        source="dcs_bios",
        payload={
            "seq": 201,
            "t_wall": 1.000,
            "bios": {"SAI_RATE_OF_TURN": 0},
            "delta": {"SAI_RATE_OF_TURN": 0},
        },
    )
    obs2 = Observation(
        source="dcs_bios",
        payload={
            "seq": 202,
            "t_wall": 1.100,
            "bios": {"SAI_RATE_OF_TURN": 20},
            "delta": {"SAI_RATE_OF_TURN": 20},
        },
    )

    first = enrich_bios_observation(obs1, _resolver(), mapper=_mapper())
    second = enrich_bios_observation(obs2, _resolver(), mapper=_mapper())

    assert first.payload["delta_summary"]["delta_count"] == 1
    assert second.payload["delta_summary"]["delta_count"] == 0
    assert second.metadata["delta_dropped_count"] == 1


def test_enrich_bios_observation_policy_scoped_sanitizer_keeps_state_across_calls(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_POLICY_SCOPED_SANITIZERS", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_SHARED_DELTA_SANITIZER_LOCKS", {})
    policy = DeltaPolicy(debounce_ms_by_key={"SAI_RATE_OF_TURN": 300}, epsilon_by_key={})

    obs1 = Observation(
        source="dcs_bios",
        payload={
            "seq": 211,
            "t_wall": 1.000,
            "bios": {"SAI_RATE_OF_TURN": 0},
            "delta": {"SAI_RATE_OF_TURN": 0},
        },
    )
    obs2 = Observation(
        source="dcs_bios",
        payload={
            "seq": 212,
            "t_wall": 1.100,
            "bios": {"SAI_RATE_OF_TURN": 20},
            "delta": {"SAI_RATE_OF_TURN": 20},
        },
    )

    first = enrich_bios_observation(obs1, _resolver(), mapper=_mapper(), delta_policy=policy)
    second = enrich_bios_observation(obs2, _resolver(), mapper=_mapper(), delta_policy=policy)

    assert first.payload["delta_summary"]["delta_count"] == 1
    assert second.payload["delta_summary"]["delta_count"] == 0
    assert second.metadata["delta_dropped_count"] == 1


def test_enrich_bios_observation_default_cache_scoped_by_session_id(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_DEFAULT_DELTA_POLICY", None)
    monkeypatch.setattr(telemetry_pipeline, "_DEFAULT_DELTA_SANITIZER", None)
    monkeypatch.setattr(telemetry_pipeline, "_POLICY_SCOPED_SANITIZERS", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_SHARED_DELTA_SANITIZER_LOCKS", {})

    obs_a1 = Observation(
        source="dcs_bios",
        payload={
            "seq": 221,
            "t_wall": 1.000,
            "bios": {"SAI_RATE_OF_TURN": 0},
            "delta": {"SAI_RATE_OF_TURN": 0},
        },
        metadata={"session_id": "sess-a"},
    )
    obs_a2 = Observation(
        source="dcs_bios",
        payload={
            "seq": 222,
            "t_wall": 1.100,
            "bios": {"SAI_RATE_OF_TURN": 20},
            "delta": {"SAI_RATE_OF_TURN": 20},
        },
        metadata={"session_id": "sess-a"},
    )
    obs_b1 = Observation(
        source="dcs_bios",
        payload={
            "seq": 223,
            "t_wall": 1.100,
            "bios": {"SAI_RATE_OF_TURN": 20},
            "delta": {"SAI_RATE_OF_TURN": 20},
        },
        metadata={"session_id": "sess-b"},
    )

    first_a = enrich_bios_observation(obs_a1, _resolver(), mapper=_mapper())
    second_a = enrich_bios_observation(obs_a2, _resolver(), mapper=_mapper())
    first_b = enrich_bios_observation(obs_b1, _resolver(), mapper=_mapper())

    assert first_a.payload["delta_summary"]["delta_count"] == 1
    assert second_a.payload["delta_summary"]["delta_count"] == 0
    assert first_b.payload["delta_summary"]["delta_count"] == 1


def test_enrich_bios_observation_delta_stream_id_overrides_metadata_scope(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_DEFAULT_DELTA_POLICY", None)
    monkeypatch.setattr(telemetry_pipeline, "_DEFAULT_DELTA_SANITIZER", None)
    monkeypatch.setattr(telemetry_pipeline, "_POLICY_SCOPED_SANITIZERS", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_SHARED_DELTA_SANITIZER_LOCKS", {})

    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 231,
            "t_wall": 1.000,
            "bios": {"SAI_RATE_OF_TURN": 0},
            "delta": {"SAI_RATE_OF_TURN": 0},
        },
        metadata={"session_id": "shared-session"},
    )
    first = enrich_bios_observation(obs, _resolver(), mapper=_mapper(), delta_stream_id="stream-a")

    obs_next = Observation(
        source="dcs_bios",
        payload={
            "seq": 232,
            "t_wall": 1.100,
            "bios": {"SAI_RATE_OF_TURN": 20},
            "delta": {"SAI_RATE_OF_TURN": 20},
        },
        metadata={"session_id": "shared-session"},
    )
    second = enrich_bios_observation(obs_next, _resolver(), mapper=_mapper(), delta_stream_id="stream-b")

    assert first.payload["delta_summary"]["delta_count"] == 1
    assert second.payload["delta_summary"]["delta_count"] == 1


def test_enrich_bios_observation_latches_momentary_lights_test_completion_within_stream(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_LOADED", True)

    active = Observation(
        source="dcs_bios",
        payload={
            "seq": 501,
            "t_wall": 10.0,
            "bios": {
                "LIGHTS_TEST_SW": 1,
                "MASTER_CAUTION_LT": 1,
            },
            "delta": {"LIGHTS_TEST_SW": 1},
        },
        metadata={"session_id": "sess-lights"},
    )
    released = Observation(
        source="dcs_bios",
        payload={
            "seq": 502,
            "t_wall": 12.0,
            "bios": {
                "LIGHTS_TEST_SW": 0,
                "MASTER_CAUTION_LT": 0,
            },
            "delta": {"LIGHTS_TEST_SW": 0},
        },
        metadata={"session_id": "sess-lights"},
    )

    first = enrich_bios_observation(active, _resolver(), mapper=_mapper())
    second = enrich_bios_observation(released, _resolver(), mapper=_mapper())

    assert first.payload["vars"]["lights_test_active"] is True
    assert first.payload["vars"]["lights_test_complete"] is True
    assert "lights_test_complete" not in first.payload["vars"]["vars_source_missing"]
    assert second.payload["vars"]["lights_test_active"] is False
    assert second.payload["vars"]["annunciator_panel_activity"] is False
    assert second.payload["vars"]["lights_test_complete"] is True
    assert "lights_test_complete" not in second.payload["vars"]["vars_source_missing"]


def test_enrich_bios_observation_latched_momentary_completion_is_session_sticky_and_scoped(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_LOADED", True)

    active = Observation(
        source="dcs_bios",
        payload={
            "seq": 511,
            "t_wall": 20.0,
            "bios": {"FIRE_TEST_SW": 0},
            "delta": {"FIRE_TEST_SW": 0},
        },
        metadata={"session_id": "sess-fire-a"},
    )
    other_stream = Observation(
        source="dcs_bios",
        payload={
            "seq": 512,
            "t_wall": 21.0,
            "bios": {"FIRE_TEST_SW": 1},
            "delta": {"FIRE_TEST_SW": 1},
        },
        metadata={"session_id": "sess-fire-b"},
    )
    expired = Observation(
        source="dcs_bios",
        payload={
            "seq": 513,
            "t_wall": 40.5,
            "bios": {"FIRE_TEST_SW": 1},
            "delta": {"FIRE_TEST_SW": 1},
        },
        metadata={"session_id": "sess-fire-a"},
    )
    power_reset = Observation(
        source="dcs_bios",
        payload={
            "seq": 514,
            "t_wall": 41.0,
            "bios": {"BATTERY_SW": 0, "FIRE_TEST_SW": 1},
            "delta": {"BATTERY_SW": 0},
        },
        metadata={"session_id": "sess-fire-a"},
    )

    first = enrich_bios_observation(active, _resolver(), mapper=_mapper())
    isolated = enrich_bios_observation(other_stream, _resolver(), mapper=_mapper())
    sticky_result = enrich_bios_observation(expired, _resolver(), mapper=_mapper())
    reset_result = enrich_bios_observation(power_reset, _resolver(), mapper=_mapper())

    assert first.payload["vars"]["fire_test_active"] is True
    assert first.payload["vars"]["fire_test_complete"] is True
    assert isolated.payload["vars"]["fire_test_complete"] is False
    assert sticky_result.payload["vars"]["fire_test_active"] is False
    assert sticky_result.payload["vars"]["fire_test_complete"] is True
    assert reset_result.payload["vars"]["fire_test_complete"] is False


def test_enrich_bios_observation_latched_fire_test_clears_source_missing(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_LOADED", True)

    active = Observation(
        source="dcs_bios",
        payload={
            "seq": 514,
            "t_wall": 50.0,
            "bios": {"FIRE_TEST_SW": 2},
            "delta": {"FIRE_TEST_SW": 2},
        },
        metadata={"session_id": "sess-fire-latch"},
    )
    released = Observation(
        source="dcs_bios",
        payload={
            "seq": 515,
            "t_wall": 51.0,
            "bios": {"FIRE_TEST_SW": 1},
            "delta": {"FIRE_TEST_SW": 1},
        },
        metadata={"session_id": "sess-fire-latch"},
    )

    enrich_bios_observation(active, _resolver(), mapper=_mapper())
    released_result = enrich_bios_observation(released, _resolver(), mapper=_mapper())

    assert released_result.payload["vars"]["fire_test_complete"] is True
    assert "fire_test_complete" not in released_result.payload["vars"]["vars_source_missing"]


def test_enrich_bios_observation_keeps_engine_crank_left_complete_in_selected_vars() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 516,
            "t_wall": 52.0,
            "bios": {
                "BATTERY_SW": 2,
                "EXT_PWR_SW": 1,
                "L_GEN_SW": 1,
                "R_GEN_SW": 1,
                "ENGINE_CRANK_SW": 1,
                "IFEI_RPM_L": "64",
                "INT_THROTTLE_LEFT": 12345,
            },
            "delta": {"ENGINE_CRANK_SW": 1, "IFEI_RPM_L": "64"},
        },
        metadata={"session_id": "sess-left-engine-complete"},
    )

    enriched = enrich_bios_observation(obs, _resolver(), mapper=_mapper())

    vars_out = enriched.payload["vars"]
    assert vars_out["left_engine_idle_ready"] is True
    assert vars_out["engine_crank_left_complete"] is True
    assert "engine_crank_left_complete" not in vars_out["vars_source_missing"]


def test_enrich_bios_observation_keeps_obogs_subconditions_in_selected_vars() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 517,
            "t_wall": 53.0,
            "bios": {
                "OBOGS_SW": 1,
                "OXY_FLOW": 65535,
            },
            "delta": {"OBOGS_SW": 1, "OXY_FLOW": 65535},
        },
        metadata={"session_id": "sess-obogs"},
    )

    enriched = enrich_bios_observation(obs, _resolver(), mapper=_mapper())

    vars_out = enriched.payload["vars"]
    assert vars_out["obogs_switch_on"] is True
    assert vars_out["obogs_flow_on"] is True
    assert vars_out["obogs_ready"] is True
    assert "obogs_switch_on" not in vars_out["vars_source_missing"]
    assert "obogs_flow_on" not in vars_out["vars_source_missing"]


def test_enrich_bios_observation_latches_fcs_reset_complete_after_button_release(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_LOADED", True)

    pressed = Observation(
        source="dcs_bios",
        payload={
            "seq": 518,
            "t_wall": 54.0,
            "bios": {"FCS_RESET_BTN": 1},
            "delta": {"FCS_RESET_BTN": 1},
        },
        metadata={"session_id": "sess-fcs-reset"},
    )
    released = Observation(
        source="dcs_bios",
        payload={
            "seq": 519,
            "t_wall": 55.0,
            "bios": {"FCS_RESET_BTN": 0},
            "delta": {"FCS_RESET_BTN": 0},
        },
        metadata={"session_id": "sess-fcs-reset"},
    )

    first = enrich_bios_observation(pressed, _resolver(), mapper=_mapper())
    second = enrich_bios_observation(released, _resolver(), mapper=_mapper())

    assert first.payload["vars"]["fcs_reset_pressed"] is True
    assert first.payload["vars"]["fcs_reset_complete"] is True
    assert second.payload["vars"]["fcs_reset_pressed"] is False
    assert second.payload["vars"]["fcs_reset_complete"] is True
    assert "fcs_reset_complete" not in second.payload["vars"]["vars_source_missing"]


def test_enrich_bios_observation_latches_takeoff_trim_set_after_button_release(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_LOADED", True)

    pressed = Observation(
        source="dcs_bios",
        payload={
            "seq": 520,
            "t_wall": 56.0,
            "bios": {"TO_TRIM_BTN": 1},
            "delta": {"TO_TRIM_BTN": 1},
        },
        metadata={"session_id": "sess-to-trim"},
    )
    released = Observation(
        source="dcs_bios",
        payload={
            "seq": 521,
            "t_wall": 57.0,
            "bios": {"TO_TRIM_BTN": 0},
            "delta": {"TO_TRIM_BTN": 0},
        },
        metadata={"session_id": "sess-to-trim"},
    )

    first = enrich_bios_observation(pressed, _resolver(), mapper=_mapper())
    second = enrich_bios_observation(released, _resolver(), mapper=_mapper())

    assert first.payload["vars"]["takeoff_trim_pressed"] is True
    assert first.payload["vars"]["takeoff_trim_set"] is True
    assert second.payload["vars"]["takeoff_trim_pressed"] is False
    assert second.payload["vars"]["takeoff_trim_set"] is True
    assert "takeoff_trim_set" not in second.payload["vars"]["vars_source_missing"]


def test_enrich_bios_observation_restores_momentary_latches_across_process_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latch_path = tmp_path / "completion_latches.json"
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_PATH", latch_path)
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_LOADED", True)

    pressed = Observation(
        source="dcs_bios",
        payload={
            "seq": 530,
            "t_wall": 58.0,
            "bios": {"FCS_RESET_BTN": 1, "TO_TRIM_BTN": 1, "BATTERY_SW": 2},
            "delta": {"FCS_RESET_BTN": 1, "TO_TRIM_BTN": 1, "BATTERY_SW": 2},
        },
        metadata={"session_id": "sess-persisted-completions"},
    )
    restored = Observation(
        source="dcs_bios",
        payload={
            "seq": 531,
            "t_wall": 59.0,
            "bios": {"FCS_RESET_BTN": 0, "TO_TRIM_BTN": 0, "BATTERY_SW": 2},
            "delta": {"BATTERY_SW": 2},
        },
        metadata={"session_id": "sess-persisted-completions"},
    )

    first = enrich_bios_observation(pressed, _resolver(), mapper=_mapper())
    assert first.payload["vars"]["fcs_reset_complete"] is True
    assert first.payload["vars"]["takeoff_trim_set"] is True
    assert latch_path.exists()

    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_LOADED", False)

    second = enrich_bios_observation(restored, _resolver(), mapper=_mapper())
    assert second.payload["vars"]["fcs_reset_pressed"] is False
    assert second.payload["vars"]["takeoff_trim_pressed"] is False
    assert second.payload["vars"]["fcs_reset_complete"] is True
    assert second.payload["vars"]["takeoff_trim_set"] is True


def test_enrich_bios_observation_clears_persisted_momentary_latches_on_cold_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latch_path = tmp_path / "completion_latches.json"
    latch_path.write_text(
        json.dumps(
            {
                "sess-cold-start-clear": {
                    "fcs_reset_complete": True,
                    "takeoff_trim_set": True,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_PATH", latch_path)
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_LOADED", False)

    cold_start = Observation(
        source="dcs_bios",
        payload={
            "seq": 532,
            "t_wall": 60.0,
            "bios": {"BATTERY_SW": 1, "FCS_RESET_BTN": 0, "TO_TRIM_BTN": 0},
            "delta": {"BATTERY_SW": 1},
        },
        metadata={"session_id": "sess-cold-start-clear"},
    )

    enriched = enrich_bios_observation(cold_start, _resolver(), mapper=_mapper())
    assert enriched.payload["vars"]["battery_on"] is False
    assert enriched.payload["vars"]["fcs_reset_complete"] is False
    assert enriched.payload["vars"]["takeoff_trim_set"] is False


def test_save_completion_latches_skips_non_regular_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latch_path = tmp_path / "completion_latches.json"
    latch_path.mkdir()
    monkeypatch.setattr(telemetry_pipeline, "_COMPLETION_LATCHES_PATH", latch_path)
    monkeypatch.setattr(
        telemetry_pipeline,
        "_COMPLETION_LATCHES",
        OrderedDict({"sess-1": {"fcs_reset_complete": True}}),
    )

    telemetry_pipeline._save_completion_latches_to_disk()

    assert latch_path.is_dir()
    assert not any(latch_path.iterdir())

def test_enrich_bios_observation_missing_delta_count_aligns_to_kept_and_records_raw() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 401,
            "t_wall": 401.0,
            "bios": {"BATTERY_SW": 2, "IFEI_CLOCK_S": "59"},
            "delta": {"BATTERY_SW": 2, "IFEI_CLOCK_S": "59"},
        },
        metadata={"seq": 401, "gap": 0},
    )

    enriched = enrich_bios_observation(obs, _resolver(), mapper=_mapper())

    assert enriched.payload["delta_summary"]["delta_count"] == 1
    assert enriched.payload["delta_summary"]["raw_delta_count"] == 2
    assert enriched.metadata["delta_count"] == 1
    assert enriched.metadata["raw_delta_count"] == 2


def test_enrich_bios_observation_keeps_existing_delta_dropped_count_metadata() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={
            "seq": 402,
            "t_wall": 402.0,
            "bios": {"BATTERY_SW": 2, "IFEI_CLOCK_S": "59"},
            "delta": {"BATTERY_SW": 2, "IFEI_CLOCK_S": "59"},
        },
        metadata={"delta_dropped_count": 99},
    )

    enriched = enrich_bios_observation(obs, _resolver(), mapper=_mapper())
    assert enriched.metadata["delta_dropped_count"] == 99


def test_enrich_bios_observation_rejects_mismatched_policy_and_sanitizer() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={"seq": 301, "t_wall": 301.0, "bios": {"BATTERY_SW": 2}, "delta": {"BATTERY_SW": 2}},
    )
    policy_a = DeltaPolicy(max_changes_per_window=8)
    policy_b = DeltaPolicy(max_changes_per_window=12)
    sanitizer = DeltaSanitizer(policy_a)

    with pytest.raises(ValueError, match="delta_policy does not match delta_sanitizer.policy"):
        enrich_bios_observation(
            obs,
            _resolver(),
            mapper=_mapper(),
            delta_policy=policy_b,
            delta_sanitizer=sanitizer,
        )


def test_enrich_bios_observation_rejects_mismatched_policy_and_aggregator() -> None:
    obs = Observation(
        source="dcs_bios",
        payload={"seq": 302, "t_wall": 302.0, "bios": {"BATTERY_SW": 2}, "delta": {"BATTERY_SW": 2}},
    )
    policy_a = DeltaPolicy(max_changes_per_window=8)
    policy_b = DeltaPolicy(max_changes_per_window=12)
    aggregator = DeltaAggregator(policy_b, mapper=_mapper(), window_size=5)

    with pytest.raises(ValueError, match="delta_aggregator.policy does not match effective delta policy"):
        enrich_bios_observation(
            obs,
            _resolver(),
            mapper=_mapper(),
            delta_policy=policy_a,
            delta_aggregator=aggregator,
        )


def test_policy_scoped_sanitizer_cache_is_lru_bounded(monkeypatch) -> None:
    monkeypatch.setattr(telemetry_pipeline, "_POLICY_SCOPED_SANITIZERS", OrderedDict())
    monkeypatch.setattr(telemetry_pipeline, "_SHARED_DELTA_SANITIZER_LOCKS", {})
    monkeypatch.setattr(telemetry_pipeline, "_MAX_POLICY_SCOPED_SANITIZERS", 2)
    policy = DeltaPolicy(debounce_ms_by_key={"SAI_RATE_OF_TURN": 300})

    s1, _ = telemetry_pipeline._get_policy_scoped_sanitizer(policy, stream_id="s1")
    telemetry_pipeline._get_policy_scoped_sanitizer(policy, stream_id="s2")
    telemetry_pipeline._get_policy_scoped_sanitizer(policy, stream_id="s3")

    keys = list(telemetry_pipeline._POLICY_SCOPED_SANITIZERS.keys())
    assert len(keys) == 2
    assert (policy, "s1") not in keys
    assert (policy, "s2") in keys
    assert (policy, "s3") in keys
    assert s1 not in telemetry_pipeline._SHARED_DELTA_SANITIZER_LOCKS


def test_resolve_policy_and_sanitizer_returns_lock_for_caller_sanitizer() -> None:
    policy = DeltaPolicy()
    sanitizer = DeltaSanitizer(policy)

    _, resolved, lock1 = telemetry_pipeline._resolve_delta_policy_and_sanitizer(
        delta_policy=None,
        delta_sanitizer=sanitizer,
        delta_aggregator=None,
        delta_stream_id="stream-x",
    )
    _, _, lock2 = telemetry_pipeline._resolve_delta_policy_and_sanitizer(
        delta_policy=None,
        delta_sanitizer=sanitizer,
        delta_aggregator=None,
        delta_stream_id="stream-y",
    )

    assert resolved is sanitizer
    assert lock1 is not None
    assert lock2 is lock1
