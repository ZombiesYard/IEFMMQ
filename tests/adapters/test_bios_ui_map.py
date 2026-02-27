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
