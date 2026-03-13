from __future__ import annotations

from pathlib import Path

import pytest

from adapters.dcs_bios.bios_ui_map import BiosUiMapError, BiosUiMapper


BASE_DIR = Path(__file__).resolve().parent.parent.parent
BIOS_TO_UI_PATH = BASE_DIR / "packs" / "fa18c_startup" / "bios_to_ui.yaml"
UI_MAP_PATH = BASE_DIR / "packs" / "fa18c_startup" / "ui_map.yaml"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_pack_bios_to_ui_covers_critical_switches() -> None:
    mapper = BiosUiMapper.from_yaml(BIOS_TO_UI_PATH, UI_MAP_PATH)

    expected = {
        "BATTERY_SW": "battery_switch",
        "L_GEN_SW": "generator_left_switch",
        "R_GEN_SW": "generator_right_switch",
        "APU_CONTROL_SW": "apu_switch",
        "ENGINE_CRANK_SW": "eng_crank_switch",
        "FIRE_TEST_SW": "fire_test_switch",
        "BLEED_AIR_KNOB": "bleed_air_knob",
    }
    for bios_key, ui_target in expected.items():
        assert mapper.targets_for_key(bios_key) == [ui_target]


def test_pack_bios_to_ui_covers_cold_start_step_keys() -> None:
    mapper = BiosUiMapper.from_yaml(BIOS_TO_UI_PATH, UI_MAP_PATH)

    expected: dict[str, list[str]] = {
        "LIGHTS_TEST_SW": ["lights_test_button"],
        "LEFT_DDI_PB_05": ["left_mdi_pb5"],
        "LEFT_DDI_PB_15": ["left_mdi_pb15"],
        "LEFT_DDI_PB_18": ["left_mdi_pb18"],
        "RIGHT_DDI_PB_05": ["right_mdi_pb5"],
        "RIGHT_DDI_PB_18": ["right_mdi_pb18"],
        "COMM1_CHANNEL_NUMERIC": [
            "ufc_comm1_channel_selector_rotate",
            "ufc_comm1_channel_selector_pull",
        ],
        "COMM2_CHANNEL_NUMERIC": [
            "ufc_comm2_channel_selector_rotate",
            "ufc_comm2_channel_selector_pull",
        ],
        "UFC_COMM1_PULL": ["ufc_comm1_channel_selector_pull"],
        "UFC_COMM2_PULL": ["ufc_comm2_channel_selector_pull"],
        "OBOGS_SW": ["obogs_control_switch"],
        "OXY_FLOW": ["obogs_flow_knob"],
        "FCS_RESET_BTN": ["fcs_reset_button"],
        "TO_TRIM_BTN": ["takeoff_trim_button"],
        "FLAP_SW": ["flap_switch"],
        "PROBE_SW": ["refuel_probe_switch"],
        "LAUNCH_BAR_SW": ["launch_bar_switch"],
        "HOOK_LEVER": ["arresting_hook_handle"],
        "PITOT_HEAT_SW": ["pitot_heater_switch"],
        "EMERGENCY_PARKING_BRAKE_PULL": ["parking_brake_handle"],
        "EMERGENCY_PARKING_BRAKE_ROTATE": ["parking_brake_handle"],
        "IFEI_UP_BTN": ["ifei_up_button"],
        "IFEI_DWN_BTN": ["ifei_down_button"],
        "STBY_PRESS_SET_0": ["standby_altimeter_pressure_knob"],
        "STBY_PRESS_SET_1": ["standby_altimeter_pressure_knob"],
        "STBY_PRESS_SET_2": ["standby_altimeter_pressure_knob"],
        "RADALT_MIN_HEIGHT_PTR": ["radar_altimeter_bug_knob"],
        "SAI_CAGE": ["standby_attitude_cage_knob"],
        "HUD_ATT_SW": ["attitude_source_selector"],
    }

    for bios_key, ui_targets in expected.items():
        assert mapper.targets_for_key(bios_key) == ui_targets


def test_pack_bios_to_ui_maps_replay_style_keys_without_regression() -> None:
    mapper = BiosUiMapper.from_yaml(BIOS_TO_UI_PATH, UI_MAP_PATH)
    delta = {
        "LEFT_MDI_PB_5": 1,  # legacy key spelling
        "LEFT_DDI_PB_05": 1,  # current replay key spelling
        "LEFT_DDI_PB_15": 1,
        "LEFT_DDI_PB_18": 1,
        "FCS_RESET_BTN": 1,
        "COMM1_CHANNEL_NUMERIC": 3,
        "COMM2_CHAN": 11,
        "TO_TRIM_BTN": 1,
        "EMERGENCY_PARKING_BRAKE_PULL": 0,
        "NOT_MAPPED": 1,
    }

    assert mapper.map_delta(delta) == [
        "left_mdi_pb5",
        "left_mdi_pb15",
        "left_mdi_pb18",
        "fcs_reset_button",
        "ufc_comm1_channel_selector_rotate",
        "ufc_comm1_channel_selector_pull",
        "ufc_comm2_channel_selector_rotate",
        "ufc_comm2_channel_selector_pull",
        "takeoff_trim_button",
        "parking_brake_handle",
    ]


def test_map_delta_supports_one_to_many_and_stable_order() -> None:
    ui_map = FIXTURE_DIR / "ui_map_minimal.yaml"
    bios_map = FIXTURE_DIR / "bios_to_ui_one_to_many.yaml"
    mapper = BiosUiMapper.from_yaml(bios_map, ui_map)
    delta = {"KEY_B": 1, "UNKNOWN": 9, "KEY_A": 2, "KEY_C": 3}
    assert mapper.map_delta(delta) == ["t_b", "t_c", "t_a", "t_d"]


def test_map_delta_unknown_key_returns_empty_list() -> None:
    mapper = BiosUiMapper.from_yaml(BIOS_TO_UI_PATH, UI_MAP_PATH)
    assert mapper.map_delta({"NOT_EXIST": 1}) == []


def test_loader_rejects_target_not_in_ui_map() -> None:
    ui_map = FIXTURE_DIR / "ui_map_only_target.yaml"
    bios_map = FIXTURE_DIR / "bios_to_ui_bad_target.yaml"

    with pytest.raises(BiosUiMapError, match="unknown ui target"):
        BiosUiMapper.from_yaml(bios_map, ui_map)


def test_loader_wraps_missing_file_as_bios_ui_map_error() -> None:
    missing_bios_map = FIXTURE_DIR / "missing_bios_to_ui.yaml"

    with pytest.raises(BiosUiMapError, match="read failed"):
        BiosUiMapper.from_yaml(missing_bios_map, UI_MAP_PATH)


def test_loader_wraps_invalid_yaml_as_bios_ui_map_error() -> None:
    invalid_bios_map = FIXTURE_DIR / "bios_to_ui_invalid_yaml.yaml"

    with pytest.raises(BiosUiMapError, match="contains invalid YAML"):
        BiosUiMapper.from_yaml(invalid_bios_map, UI_MAP_PATH)
