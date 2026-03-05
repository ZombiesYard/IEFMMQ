import json
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from core.types_v2 import TelemetryFrame
from core.vars import VarResolver, VarResolverError

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_TELEMETRY_MAP_PATH = REPO_ROOT / "packs" / "fa18c_startup" / "telemetry_map.yaml"
SAMPLE_FRAME_ONCE_PATH = REPO_ROOT / "artifacts" / "dcs_bios_frame_once.json"
SAMPLE_RAW_JSONL_PATH = REPO_ROOT / "logs" / "dcs_bios_raw_15s.jsonl"

EXPECTED_S11_S25_VAR_KEYS = {
    "rpm_l_gte_25",
    "rpm_l_in_range",
    "temp_l_in_range",
    "ff_l_in_range",
    "oil_l_in_range",
    "noz_l_in_range",
    "left_engine_nominal_start_params",
    "left_engine_idle_ready",
    "ins_mode_set",
    "ins_mode_cv_or_gnd",
    "radar_mode_value",
    "radar_mode_opr",
    "obogs_switch_on",
    "obogs_flow_on",
    "obogs_ready",
    "fcs_reset_pressed",
    "fcs_page_reviewed",
    "flap_mode_value",
    "flap_auto",
    "flap_half",
    "flap_full",
    "flap_configured",
    "takeoff_trim_pressed",
    "takeoff_trim_set",
    "fcs_bit_switch_up",
    "fcs_bit_complete",
    "probe_switch_value",
    "launch_bar_switch_value",
    "hook_handle_value",
    "pitot_heat_on",
    "four_down_complete",
    "parking_brake_pull_value",
    "parking_brake_rotate_value",
    "parking_brake_released",
    "ifei_up_pressed",
    "ifei_down_pressed",
    "bingo_fuel_set",
    "standby_pressure_set_0",
    "standby_pressure_set_1",
    "standby_pressure_set_2",
    "standby_altimeter_set",
    "radar_altimeter_bug_value",
    "radar_altimeter_bug_set",
    "sai_cage_value",
    "standby_attitude_uncaged",
    "hud_att_source_value",
    "attitude_source_auto",
}

EXPECTED_UNKNOWN_VALUE_KEYS = {
    "fcs_page_reviewed",
    "takeoff_trim_set",
    "fcs_bit_complete",
    "four_down_complete",
    "bingo_fuel_set",
    "standby_altimeter_set",
    "radar_altimeter_bug_set",
    "standby_attitude_uncaged",
    "attitude_source_auto",
}


def _tmp_dir() -> Path:
    base = Path("tests/.tmp_telemetry")
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_var_resolver_maps_bios_and_derived() -> None:
    mapping = {
        "vars": {
            "battery_on": "bios.BATTERY_SW == 2",
            "apu_on": "bios.APU_CONTROL_SW == 1",
            "power_available": "derived(vars.battery_on and vars.apu_on)",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    resolver = VarResolver.from_yaml(path)
    frame = TelemetryFrame(
        seq=1,
        t_wall=1.0,
        source="dcs_bios",
        bios={"BATTERY_SW": 2, "APU_CONTROL_SW": 1},
    )
    vars_out = resolver.resolve(frame)
    assert vars_out["battery_on"] is True
    assert vars_out["apu_on"] is True
    assert vars_out["power_available"] is True


def test_var_resolver_rejects_is_operator() -> None:
    """Test that 'is' comparison operator raises VarResolverError."""
    mapping = {
        "vars": {
            "invalid_check": "bios.VALUE is None",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    resolver = VarResolver.from_yaml(path)
    frame = TelemetryFrame(
        seq=1,
        t_wall=1.0,
        source="dcs_bios",
        bios={"VALUE": 1},
    )

    with pytest.raises(VarResolverError, match="Unsupported comparison operator.*Is"):
        resolver.resolve(frame)


def test_var_resolver_rejects_is_not_operator() -> None:
    """Test that 'is not' comparison operator raises VarResolverError."""
    mapping = {
        "vars": {
            "invalid_check": "bios.VALUE is not None",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    resolver = VarResolver.from_yaml(path)
    frame = TelemetryFrame(
        seq=1,
        t_wall=1.0,
        source="dcs_bios",
        bios={"VALUE": 1},
    )

    with pytest.raises(VarResolverError, match="Unsupported comparison operator.*IsNot"):
        resolver.resolve(frame)


def test_var_resolver_rejects_in_operator() -> None:
    """Test that 'in' comparison operator raises VarResolverError."""
    mapping = {
        "vars": {
            "invalid_check": "bios.VALUE in [1, 2, 3]",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    resolver = VarResolver.from_yaml(path)
    frame = TelemetryFrame(
        seq=1,
        t_wall=1.0,
        source="dcs_bios",
        bios={"VALUE": 1},
    )

    with pytest.raises(VarResolverError, match="Unsupported comparison operator.*In"):
        resolver.resolve(frame)


def test_var_resolver_rejects_not_in_operator() -> None:
    """Test that 'not in' comparison operator raises VarResolverError."""
    mapping = {
        "vars": {
            "invalid_check": "bios.VALUE not in [4, 5, 6]",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    resolver = VarResolver.from_yaml(path)
    frame = TelemetryFrame(
        seq=1,
        t_wall=1.0,
        source="dcs_bios",
        bios={"VALUE": 1},
    )

    with pytest.raises(VarResolverError, match="Unsupported comparison operator.*NotIn"):
        resolver.resolve(frame)


def test_var_resolver_num_cast_and_missing_helper() -> None:
    mapping = {
        "vars": {
            "rpm_r": "derived(num(bios.IFEI_RPM_R))",
            "rpm_r_gte_25": "derived(vars.rpm_r >= 25)",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    resolver = VarResolver.from_yaml(path)

    frame_ok = TelemetryFrame(
        seq=1,
        t_wall=1.0,
        source="dcs_bios",
        bios={"IFEI_RPM_R": " 27"},
    )
    vars_ok = resolver.resolve(frame_ok)
    assert vars_ok["rpm_r"] == 27
    assert vars_ok["rpm_r_gte_25"] is True
    assert vars_ok["vars_source_missing"] == []

    frame_missing = TelemetryFrame(seq=2, t_wall=2.0, source="dcs_bios", bios={})
    vars_missing = resolver.resolve(frame_missing)
    assert vars_missing["rpm_r"] is None
    assert vars_missing["rpm_r_gte_25"] is False
    assert "rpm_r" in vars_missing["vars_source_missing"]
    assert "rpm_r_gte_25" in vars_missing["vars_source_missing"]


def test_var_resolver_pack_map_resolves_from_dcs_bios_frame_once() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)
    frame = json.loads(SAMPLE_FRAME_ONCE_PATH.read_text(encoding="utf-8"))

    vars_out = resolver.resolve(frame)
    required = ("rpm_r", "rpm_l", "temp_r", "ff_r", "oil_r", "noz_r", "rpm_r_gte_25")
    for key in required:
        assert key in vars_out

    assert isinstance(vars_out["rpm_r"], (int, float))
    assert isinstance(vars_out["temp_r"], (int, float))
    assert vars_out["rpm_r"] >= 0
    assert vars_out["temp_r"] >= 0
    assert vars_out["rpm_r_gte_25"] == (vars_out["rpm_r"] >= 25)
    assert isinstance(vars_out["vars_source_missing"], list)
    for key in ("ff_r", "oil_r", "noz_r"):
        assert (vars_out[key] is None) == (key in vars_out["vars_source_missing"])


def test_var_resolver_pack_battery_on_requires_switch_value_2() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)
    frame_on = TelemetryFrame(
        seq=2,
        t_wall=2.0,
        source="dcs_bios",
        bios={"BATTERY_SW": 2},
    )
    vars_on = resolver.resolve(frame_on)
    assert vars_on["battery_on"] is True

    frame_off = TelemetryFrame(
        seq=99,
        t_wall=99.0,
        source="dcs_bios",
        bios={"BATTERY_SW": 1},
    )
    vars_off = resolver.resolve(frame_off)
    assert vars_off["battery_on"] is False

    frame_override = TelemetryFrame(
        seq=100,
        t_wall=100.0,
        source="dcs_bios",
        bios={"BATTERY_SW": 0},
    )
    vars_override = resolver.resolve(frame_override)
    assert vars_override["battery_on"] is False


def test_var_resolver_pack_engine_crank_switch_three_position_mapping() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame_left = TelemetryFrame(
        seq=201,
        t_wall=201.0,
        source="dcs_bios",
        bios={"ENGINE_CRANK_SW": 0},
    )
    vars_left = resolver.resolve(frame_left)
    assert vars_left["engine_crank_left"] is True
    assert vars_left["engine_crank_right"] is False

    frame_off = TelemetryFrame(
        seq=202,
        t_wall=202.0,
        source="dcs_bios",
        bios={"ENGINE_CRANK_SW": 1},
    )
    vars_off = resolver.resolve(frame_off)
    assert vars_off["engine_crank_left"] is False
    assert vars_off["engine_crank_right"] is False

    frame_right = TelemetryFrame(
        seq=203,
        t_wall=203.0,
        source="dcs_bios",
        bios={"ENGINE_CRANK_SW": 2},
    )
    vars_right = resolver.resolve(frame_right)
    assert vars_right["engine_crank_left"] is False
    assert vars_right["engine_crank_right"] is True


def test_var_resolver_pack_throttle_not_off_flags_track_internal_throttle_axes() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame_off = TelemetryFrame(
        seq=301,
        t_wall=301.0,
        source="dcs_bios",
        bios={"INT_THROTTLE_RIGHT": 0, "INT_THROTTLE_LEFT": 0},
    )
    vars_off = resolver.resolve(frame_off)
    assert vars_off["throttle_r_not_off"] is False
    assert vars_off["throttle_l_not_off"] is False

    frame_idle = TelemetryFrame(
        seq=302,
        t_wall=302.0,
        source="dcs_bios",
        bios={"INT_THROTTLE_RIGHT": 1, "INT_THROTTLE_LEFT": 2},
    )
    vars_idle = resolver.resolve(frame_idle)
    assert vars_idle["throttle_r_not_off"] is True
    assert vars_idle["throttle_l_not_off"] is True


def test_var_resolver_pack_map_resolves_from_raw_jsonl_samples() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)
    lines = [line for line in SAMPLE_RAW_JSONL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "expected non-empty bios raw jsonl sample"

    for line in lines[:30]:
        frame = json.loads(line)
        vars_out = resolver.resolve(frame)
        assert isinstance(vars_out["vars_source_missing"], list)
        for key in ("rpm_r", "rpm_l", "temp_r", "ff_r", "oil_r", "noz_r", "rpm_r_gte_25"):
            assert key in vars_out


def test_var_resolver_none_rule_counts_as_source_missing() -> None:
    mapping = {
        "vars": {
            "manual_placeholder": None,
            "manual_ready": "derived(vars.manual_placeholder == 1)",
        }
    }
    tmp_path = _tmp_dir()
    path = tmp_path / "telemetry_map.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    resolver = VarResolver.from_yaml(path)

    frame = TelemetryFrame(seq=1, t_wall=1.0, source="derived", bios={})
    vars_out = resolver.resolve(frame)

    assert vars_out["manual_placeholder"] is None
    assert vars_out["manual_ready"] is False
    assert "manual_placeholder" in vars_out["vars_source_missing"]
    assert "manual_ready" in vars_out["vars_source_missing"]


def test_var_resolver_pack_s11_s25_vars_are_present_and_unknown_is_explicit() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)
    frame = TelemetryFrame(
        seq=700,
        t_wall=700.0,
        source="dcs_bios",
        bios={
            "IFEI_RPM_L": " 66",
            "IFEI_TEMP_L": "300",
            "IFEI_FF_L": "500",
            "IFEI_OIL_PRESS_L": "70",
            "EXT_NOZZLE_POS_L": 50000,
            "INT_THROTTLE_LEFT": 1,
            "INS_SW": 4,
            "RADAR_SW": 2,
            "OBOGS_SW": 1,
            "OXY_FLOW": 65535,
            "FLAP_SW": 0,
            "EMERGENCY_PARKING_BRAKE_PULL": 0,
            "EMERGENCY_PARKING_BRAKE_ROTATE": 2,
            "RADALT_MIN_HEIGHT_PTR": 37431,
            "HUD_ATT_SW": 1,
        },
    )

    vars_out = resolver.resolve(frame)

    assert EXPECTED_S11_S25_VAR_KEYS.issubset(vars_out.keys())
    for key in EXPECTED_UNKNOWN_VALUE_KEYS:
        assert vars_out[key] == "unknown"


def test_var_resolver_pack_flap_semantics_for_0_1_2_none_and_missing() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame_auto = TelemetryFrame(seq=801, t_wall=801.0, source="dcs_bios", bios={"FLAP_SW": 0})
    vars_auto = resolver.resolve(frame_auto)
    assert vars_auto["flap_mode_value"] == 0
    assert vars_auto["flap_auto"] is True
    assert vars_auto["flap_half"] is False
    assert vars_auto["flap_full"] is False
    assert vars_auto["flap_configured"] is True

    frame_half = TelemetryFrame(seq=802, t_wall=802.0, source="dcs_bios", bios={"FLAP_SW": 1})
    vars_half = resolver.resolve(frame_half)
    assert vars_half["flap_mode_value"] == 1
    assert vars_half["flap_auto"] is False
    assert vars_half["flap_half"] is True
    assert vars_half["flap_full"] is False
    assert vars_half["flap_configured"] is True

    frame_full = TelemetryFrame(seq=803, t_wall=803.0, source="dcs_bios", bios={"FLAP_SW": 2})
    vars_full = resolver.resolve(frame_full)
    assert vars_full["flap_mode_value"] == 2
    assert vars_full["flap_auto"] is False
    assert vars_full["flap_half"] is False
    assert vars_full["flap_full"] is True
    assert vars_full["flap_configured"] is False

    frame_none = TelemetryFrame(seq=804, t_wall=804.0, source="dcs_bios", bios={"FLAP_SW": None})
    vars_none = resolver.resolve(frame_none)
    assert vars_none["flap_mode_value"] is None
    assert vars_none["flap_auto"] is False
    assert vars_none["flap_half"] is False
    assert vars_none["flap_full"] is False
    assert vars_none["flap_configured"] is False
    assert "flap_mode_value" in vars_none["vars_source_missing"]
    assert "flap_auto" in vars_none["vars_source_missing"]
    assert "flap_half" in vars_none["vars_source_missing"]
    assert "flap_full" in vars_none["vars_source_missing"]
    assert "flap_configured" in vars_none["vars_source_missing"]

    frame_missing = TelemetryFrame(seq=805, t_wall=805.0, source="dcs_bios", bios={})
    vars_missing = resolver.resolve(frame_missing)
    assert vars_missing["flap_mode_value"] is None
    assert vars_missing["flap_auto"] is False
    assert vars_missing["flap_half"] is False
    assert vars_missing["flap_full"] is False
    assert vars_missing["flap_configured"] is False
    assert "flap_mode_value" in vars_missing["vars_source_missing"]
    assert "flap_auto" in vars_missing["vars_source_missing"]
    assert "flap_half" in vars_missing["vars_source_missing"]
    assert "flap_full" in vars_missing["vars_source_missing"]
    assert "flap_configured" in vars_missing["vars_source_missing"]


def test_var_resolver_pack_composite_vars_propagate_source_missing() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)
    frame = TelemetryFrame(
        seq=900,
        t_wall=900.0,
        source="dcs_bios",
        bios={},
    )
    vars_out = resolver.resolve(frame)

    assert vars_out["left_engine_nominal_start_params"] is False
    assert vars_out["left_engine_idle_ready"] is False
    assert vars_out["core_avionics_online"] is False
    assert vars_out["obogs_ready"] is False
    assert "left_engine_nominal_start_params" in vars_out["vars_source_missing"]
    assert "left_engine_idle_ready" in vars_out["vars_source_missing"]
    assert "core_avionics_online" in vars_out["vars_source_missing"]
    assert "obogs_ready" in vars_out["vars_source_missing"]
