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
    "apu_start_support_complete",
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
    "fcs_reset_complete",
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
    "lights_test_active",
    "annunciator_panel_activity",
    "lights_test_complete",
    "comm1_freq_value",
    "comm2_freq_value",
    "comm1_channel_numeric",
    "comm2_channel_numeric",
    "comm1_freq_134_000",
    "ufc_comm1_pull_pressed",
    "ufc_key_1_pressed",
    "ufc_key_3_pressed",
    "ufc_key_4_pressed",
    "ufc_key_0_pressed",
    "ufc_ent_pressed",
    "ufc_scratchpad_number_display",
    "ufc_scratchpad_string_1_display",
    "ufc_scratchpad_string_2_display",
}

EXPECTED_UNKNOWN_VALUE_KEYS = {
    "fcs_page_reviewed",
    "fcs_bit_complete",
    "four_down_complete",
    "standby_altimeter_set",
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


def test_var_resolver_marks_probe_extended_from_switch_or_external_position() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    switch_extended = TelemetryFrame(
        seq=1,
        t_wall=1.0,
        source="dcs_bios",
        bios={"PROBE_SW": 0, "EXT_REFUEL_PROBE": 0},
    )
    vars_switch = resolver.resolve(switch_extended)
    assert vars_switch["probe_switch_value"] == 0
    assert vars_switch["probe_extended"] is True

    physically_extended = TelemetryFrame(
        seq=2,
        t_wall=2.0,
        source="dcs_bios",
        bios={"PROBE_SW": 1, "EXT_REFUEL_PROBE": 65535},
    )
    vars_physical = resolver.resolve(physically_extended)
    assert vars_physical["probe_switch_value"] == 1
    assert vars_physical["ext_refuel_probe_value"] == 65535
    assert vars_physical["probe_extended"] is True

    retracted = TelemetryFrame(
        seq=3,
        t_wall=3.0,
        source="dcs_bios",
        bios={"PROBE_SW": 1, "EXT_REFUEL_PROBE": 0},
    )
    vars_retracted = resolver.resolve(retracted)
    assert vars_retracted["probe_extended"] is False


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


def test_var_resolver_pack_fire_test_completion_requires_direct_switch_evidence() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=250,
        t_wall=250.0,
        source="dcs_bios",
        bios={
            "BATTERY_SW": 2,
            "L_GEN_SW": 1,
            "R_GEN_SW": 1,
            "APU_CONTROL_SW": 1,
            "APU_READY_LT": 1,
            "ENGINE_CRANK_SW": 0,
            "IFEI_RPM_R": 68,
            "IFEI_RPM_L": 65,
            "INS_SW": 1,
        },
    )

    vars_out = resolver.resolve(frame)

    assert vars_out["fire_test_complete"] is False
    assert vars_out["engine_crank_right_complete"] is True
    assert vars_out["throttle_r_idle_complete"] is False
    assert vars_out["bleed_air_cycle_complete"] is True
    assert "engine_crank_right_complete" not in vars_out["vars_source_missing"]
    assert "throttle_r_idle_complete" in vars_out["vars_source_missing"]
    assert "bleed_air_cycle_complete" not in vars_out["vars_source_missing"]
    assert "fire_test_complete" not in vars_out["vars_source_missing"]


def test_var_resolver_pack_fire_test_complete_does_not_accept_engine_start_progress() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=251,
        t_wall=251.0,
        source="dcs_bios",
        bios={
            "BATTERY_SW": 2,
            "L_GEN_SW": 1,
            "R_GEN_SW": 1,
            "ENGINE_CRANK_SW": 2,
        },
    )

    vars_out = resolver.resolve(frame)

    assert vars_out["engine_crank_right"] is True
    assert vars_out["fire_test_complete"] is False
    assert "fire_test_complete" not in vars_out["vars_source_missing"]


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
            "INS_SW": 2,
            "RADAR_SW": 2,
            "OBOGS_SW": 1,
            "OXY_FLOW": 65535,
            "FLAP_SW": 0,
            "EMERGENCY_PARKING_BRAKE_PULL": 0,
            "EMERGENCY_PARKING_BRAKE_ROTATE": 2,
            "IFEI_BINGO": "2500",
            "RADALT_MIN_HEIGHT_PTR": 37431,
            "SAI_ATT_WARNING_FLAG": 0,
            "HUD_ATT_SW": 1,
        },
    )

    vars_out = resolver.resolve(frame)

    assert EXPECTED_S11_S25_VAR_KEYS.issubset(vars_out.keys())
    for key in EXPECTED_UNKNOWN_VALUE_KEYS:
        assert vars_out[key] == "unknown"
    assert vars_out["bingo_fuel_set"] is True
    assert vars_out["radar_altimeter_bug_set"] is True
    assert vars_out["standby_attitude_uncaged"] is True
    assert vars_out["attitude_source_auto"] is True
    assert vars_out["ins_mode"] == 2
    assert vars_out["ins_mode_set"] is True
    assert vars_out["ins_mode_cv_or_gnd"] is True


def test_var_resolver_pack_ins_mode_matches_clickabledata_positions() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    off_frame = TelemetryFrame(seq=701, t_wall=701.0, source="dcs_bios", bios={"INS_SW": 0})
    cv_frame = TelemetryFrame(seq=702, t_wall=702.0, source="dcs_bios", bios={"INS_SW": 1})
    gnd_frame = TelemetryFrame(seq=703, t_wall=703.0, source="dcs_bios", bios={"INS_SW": 2})

    off_vars = resolver.resolve(off_frame)
    cv_vars = resolver.resolve(cv_frame)
    gnd_vars = resolver.resolve(gnd_frame)

    assert off_vars["ins_mode"] == 0
    assert off_vars["ins_mode_set"] is False
    assert off_vars["ins_mode_cv_or_gnd"] is False
    assert cv_vars["ins_mode"] == 1
    assert cv_vars["ins_mode_set"] is True
    assert cv_vars["ins_mode_cv_or_gnd"] is True
    assert gnd_vars["ins_mode"] == 2
    assert gnd_vars["ins_mode_set"] is True
    assert gnd_vars["ins_mode_cv_or_gnd"] is True


def test_var_resolver_pack_radar_mode_matches_clickabledata_positions() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    off_frame = TelemetryFrame(seq=704, t_wall=704.0, source="dcs_bios", bios={"RADAR_SW": 0})
    stby_frame = TelemetryFrame(seq=705, t_wall=705.0, source="dcs_bios", bios={"RADAR_SW": 1})
    opr_frame = TelemetryFrame(seq=706, t_wall=706.0, source="dcs_bios", bios={"RADAR_SW": 2})

    off_vars = resolver.resolve(off_frame)
    stby_vars = resolver.resolve(stby_frame)
    opr_vars = resolver.resolve(opr_frame)

    assert off_vars["radar_mode_value"] == 0
    assert off_vars["radar_on"] is False
    assert off_vars["radar_mode_opr"] is False
    assert stby_vars["radar_mode_value"] == 1
    assert stby_vars["radar_on"] is True
    assert stby_vars["radar_mode_opr"] is False
    assert opr_vars["radar_mode_value"] == 2
    assert opr_vars["radar_on"] is True
    assert opr_vars["radar_mode_opr"] is True


def test_var_resolver_pack_obogs_switch_matches_clickabledata_positions() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    on_frame = TelemetryFrame(seq=707, t_wall=707.0, source="dcs_bios", bios={"OBOGS_SW": 1, "OXY_FLOW": 65535})
    off_frame = TelemetryFrame(seq=708, t_wall=708.0, source="dcs_bios", bios={"OBOGS_SW": 0, "OXY_FLOW": 65535})

    on_vars = resolver.resolve(on_frame)
    off_vars = resolver.resolve(off_frame)

    assert on_vars["obogs_switch_on"] is True
    assert on_vars["obogs_flow_on"] is True
    assert on_vars["obogs_ready"] is True
    assert off_vars["obogs_switch_on"] is False
    assert off_vars["obogs_flow_on"] is True
    assert off_vars["obogs_ready"] is False
    assert on_vars["fcs_reset_complete"] is False


def test_var_resolver_pack_takeoff_trim_set_tracks_button_press() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    pressed = TelemetryFrame(seq=720, t_wall=720.0, source="dcs_bios", bios={"TO_TRIM_BTN": 1})
    pressed_vars = resolver.resolve(pressed)
    assert pressed_vars["takeoff_trim_pressed"] is True
    assert pressed_vars["takeoff_trim_set"] is True

    released = TelemetryFrame(seq=721, t_wall=721.0, source="dcs_bios", bios={"TO_TRIM_BTN": 0})
    released_vars = resolver.resolve(released)
    assert released_vars["takeoff_trim_pressed"] is False
    assert released_vars["takeoff_trim_set"] is False


def test_var_resolver_pack_lights_test_vars_follow_switch_and_annunciators() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    active = TelemetryFrame(
        seq=722,
        t_wall=722.0,
        source="dcs_bios",
        bios={
            "LIGHTS_TEST_SW": 1,
            "MASTER_CAUTION_LT": 1,
            "LH_ADV_GO": 1,
        },
    )
    active_vars = resolver.resolve(active)
    assert active_vars["lights_test_active"] is True
    assert active_vars["annunciator_panel_activity"] is True
    assert active_vars["lights_test_complete"] is True

    downstream = TelemetryFrame(
        seq=723,
        t_wall=723.0,
        source="dcs_bios",
        bios={
            "LIGHTS_TEST_SW": 0,
            "MASTER_CAUTION_LT": 0,
            "LH_ADV_GO": 0,
            "LEFT_DDI_BRT_CTL": 65535,
            "RIGHT_DDI_BRT_CTL": 65535,
            "AMPCD_BRT_CTL": 65535,
            "HUD_SYM_BRT": 65535,
        },
    )
    downstream_vars = resolver.resolve(downstream)
    assert downstream_vars["lights_test_active"] is False
    assert downstream_vars["annunciator_panel_activity"] is False
    assert downstream_vars["lights_test_complete"] is False


def test_var_resolver_pack_lights_test_complete_does_not_use_display_power_as_proxy() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=724,
        t_wall=724.0,
        source="dcs_bios",
        bios={
            "LEFT_DDI_BRT_CTL": 65535,
            "RIGHT_DDI_BRT_CTL": 65535,
            "AMPCD_BRT_CTL": 65535,
            "HUD_SYM_BRT": 65535,
        },
    )

    vars_out = resolver.resolve(frame)

    assert vars_out["left_ddi_on"] is True
    assert vars_out["right_ddi_on"] is True
    assert vars_out["lights_test_active"] is False
    assert vars_out["annunciator_panel_activity"] is False
    assert vars_out["lights_test_complete"] is False


def test_var_resolver_pack_lights_test_complete_does_not_use_generic_annunciators_as_proxy() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=725,
        t_wall=725.0,
        source="dcs_bios",
        bios={
            "MASTER_CAUTION_LT": 1,
            "LH_ADV_GO": 1,
            "LIGHTS_TEST_SW": 0,
        },
    )

    vars_out = resolver.resolve(frame)

    assert vars_out["annunciator_panel_activity"] is True
    assert vars_out["lights_test_active"] is False
    assert vars_out["lights_test_complete"] is False


def test_var_resolver_pack_right_engine_progress_does_not_follow_left_engine() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=725,
        t_wall=725.0,
        source="dcs_bios",
        bios={
            "ENGINE_CRANK_SW": 0,
            "IFEI_RPM_L": "64",
            "INT_THROTTLE_LEFT": 1,
            "IFEI_RPM_R": "0",
            "INT_THROTTLE_RIGHT": 0,
            "IFEI_TEMP_R": " 10",
            "IFEI_FF_R": "  0",
            "IFEI_OIL_PRESS_R": " 25",
            "EXT_NOZZLE_POS_R": 0,
        },
    )

    vars_out = resolver.resolve(frame)

    assert vars_out["engine_crank_left"] is True
    assert vars_out["engine_crank_right"] is False
    assert vars_out["engine_crank_right_complete"] is False
    assert vars_out["throttle_r_idle_complete"] is False


def test_var_resolver_pack_right_engine_idle_completion_requires_right_side_evidence() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=726,
        t_wall=726.0,
        source="dcs_bios",
        bios={
            "ENGINE_CRANK_SW": 2,
            "IFEI_RPM_R": "25",
            "INT_THROTTLE_RIGHT": 0,
            "IFEI_TEMP_R": " 10",
            "IFEI_FF_R": "  0",
            "IFEI_OIL_PRESS_R": " 25",
            "EXT_NOZZLE_POS_R": 0,
        },
    )

    vars_out = resolver.resolve(frame)

    assert vars_out["rpm_r_gte_25"] is True
    assert vars_out["throttle_r_not_off"] is False
    assert vars_out["throttle_r_idle_complete"] is False


def test_var_resolver_pack_ifei_fuel_flow_scales_display_by_100_for_nominal_window() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=727,
        t_wall=727.0,
        source="dcs_bios",
        bios={
            "IFEI_RPM_R": "64",
            "IFEI_TEMP_R": "319",
            "IFEI_FF_R": "7",
            "IFEI_OIL_PRESS_R": "60",
            "EXT_NOZZLE_POS_R": 50010,
        },
    )

    vars_out = resolver.resolve(frame)

    assert vars_out["ff_r"] == 700
    assert vars_out["ff_r_in_range"] is True
    assert vars_out["right_engine_nominal_start_params"] is True


def test_var_resolver_pack_power_and_left_engine_start_completion_follow_operational_signals() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=728,
        t_wall=728.0,
        source="dcs_bios",
        bios={
            "BATTERY_SW": 1,
            "EXT_PWR_SW": 1,
            "L_GEN_SW": 0,
            "R_GEN_SW": 0,
            "ENGINE_CRANK_SW": 1,
            "IFEI_RPM_L": "64",
            "INT_THROTTLE_LEFT": 12345,
        },
    )

    vars_out = resolver.resolve(frame)

    assert vars_out["power_available"] is True
    assert vars_out["engine_crank_left"] is False
    assert vars_out["engine_crank_left_complete"] is True
    assert vars_out["left_engine_idle_ready"] is True
    assert "engine_crank_left_complete" not in vars_out["vars_source_missing"]


def test_var_resolver_pack_radio_vars_follow_bios_exports() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=724,
        t_wall=724.0,
        source="dcs_bios",
        bios={
            "COMM1_FREQ": 25100,
            "COMM2_FREQ": 12750,
            "COMM1_CHANNEL_NUMERIC": 7,
            "COMM2_CHANNEL_NUMERIC": 11,
        },
    )
    vars_out = resolver.resolve(frame)
    assert vars_out["comm1_freq_value"] == 25100
    assert vars_out["comm2_freq_value"] == 12750
    assert vars_out["comm1_channel_numeric"] == 7
    assert vars_out["comm2_channel_numeric"] == 11


def test_var_resolver_pack_ufc_comm1_entry_vars_follow_bios_exports() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    frame = TelemetryFrame(
        seq=725,
        t_wall=725.0,
        source="dcs_bios",
        bios={
            "COMM1_FREQ": 13400,
            "UFC_COMM1_PULL": 1,
            "UFC_1": 0,
            "UFC_3": 1,
            "UFC_4": 0,
            "UFC_0": 0,
            "UFC_ENT": 0,
            "UFC_SCRATCHPAD_NUMBER_DISPLAY": "13      ",
            "UFC_SCRATCHPAD_STRING_1_DISPLAY": "1-",
            "UFC_SCRATCHPAD_STRING_2_DISPLAY": "-",
        },
    )

    vars_out = resolver.resolve(frame)

    assert vars_out["comm1_freq_134_000"] is True
    assert vars_out["ufc_comm1_pull_pressed"] is True
    assert vars_out["ufc_key_3_pressed"] is True
    assert vars_out["ufc_scratchpad_number_display"] == "13      "
    assert vars_out["ufc_scratchpad_string_1_display"] == "1-"
    assert vars_out["ufc_scratchpad_string_2_display"] == "-"


def test_var_resolver_pack_bingo_and_attitude_source_vars_follow_bios_state() -> None:
    resolver = VarResolver.from_yaml(PACK_TELEMETRY_MAP_PATH)

    configured = TelemetryFrame(
        seq=710,
        t_wall=710.0,
        source="dcs_bios",
        bios={
            "IFEI_BINGO": "2500",
            "RADALT_MIN_HEIGHT_PTR": 1200,
            "SAI_ATT_WARNING_FLAG": 0,
            "HUD_ATT_SW": 1,
        },
    )
    configured_vars = resolver.resolve(configured)
    assert configured_vars["bingo_fuel_set"] is True
    assert configured_vars["radar_altimeter_bug_set"] is True
    assert configured_vars["standby_attitude_uncaged"] is True
    assert configured_vars["attitude_source_auto"] is True

    unset = TelemetryFrame(
        seq=711,
        t_wall=711.0,
        source="dcs_bios",
        bios={
            "IFEI_BINGO": "     ",
            "RADALT_MIN_HEIGHT_PTR": 0,
            "SAI_ATT_WARNING_FLAG": 65535,
            "HUD_ATT_SW": 0,
        },
    )
    unset_vars = resolver.resolve(unset)
    assert unset_vars["bingo_fuel_set"] is False
    assert unset_vars["radar_altimeter_bug_set"] is False
    assert unset_vars["standby_attitude_uncaged"] is False
    assert unset_vars["attitude_source_auto"] is False


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
    assert "obogs_switch_on" in vars_out["vars_source_missing"]
    assert "obogs_flow_on" in vars_out["vars_source_missing"]
