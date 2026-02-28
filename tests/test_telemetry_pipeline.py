import json
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
    assert enriched.payload["delta_summary"]["delta_count"] == 4
    assert "battery_switch" in enriched.payload["recent_ui_targets"]
    assert "eng_crank_switch" in enriched.payload["recent_ui_targets"]

    assert enriched.metadata["seq"] == 42
    assert enriched.metadata["gap"] == 0
    assert enriched.metadata["delta_count"] == 4
    assert "bios_hash" not in enriched.metadata

    json.dumps(enriched.to_dict(), ensure_ascii=False)


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
    monkeypatch.setattr(telemetry_pipeline, "_POLICY_SCOPED_SANITIZERS", {})
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
    monkeypatch.setattr(telemetry_pipeline, "_POLICY_SCOPED_SANITIZERS", {})
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
    monkeypatch.setattr(telemetry_pipeline, "_POLICY_SCOPED_SANITIZERS", {})
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
    monkeypatch.setattr(telemetry_pipeline, "_POLICY_SCOPED_SANITIZERS", {})
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
