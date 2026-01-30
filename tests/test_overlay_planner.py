from pathlib import Path
from uuid import uuid4

import pytest

from core.overlay import OverlayPlanner
from core.types import TutorResponse


UI_MAP = Path("packs/fa18c_startup/ui_map.yaml")
TEMP_DIR = Path("tests/.tmp_overlay")


def write_temp_ui_map(text: str) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMP_DIR / f"ui_map_{uuid4().hex}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_overlay_planner_maps_target():
    planner = OverlayPlanner(str(UI_MAP))
    intent = planner.plan("battery_switch", intent="highlight")
    action = intent.to_action()
    assert action["intent"] == "highlight"
    assert action["element_id"] == "pnt_404"
    assert action["type"] == "overlay"


def test_unknown_target_raises():
    planner = OverlayPlanner(str(UI_MAP))
    with pytest.raises(KeyError):
        planner.plan("nonexistent", intent="highlight")


def test_unknown_intent_raises():
    planner = OverlayPlanner(str(UI_MAP))
    with pytest.raises(ValueError, match="Unknown overlay intent"):
        planner.plan("battery_switch", intent="blink")


def test_tutor_response_can_carry_overlay_action():
    planner = OverlayPlanner(str(UI_MAP))
    overlay_action = planner.plan("apu_switch", intent="pulse").to_action()
    resp = TutorResponse(message="Turn on APU.", actions=[overlay_action]).to_dict()
    assert resp["actions"][0]["intent"] == "pulse"
    assert resp["actions"][0]["element_id"] == "pnt_375"


def test_ui_map_empty_file_rejected():
    ui_map = write_temp_ui_map("")
    with pytest.raises(ValueError, match="UI map must be a YAML mapping"):
        OverlayPlanner(str(ui_map))


def test_ui_map_non_mapping_root_rejected():
    ui_map = write_temp_ui_map("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="UI map must be a YAML mapping"):
        OverlayPlanner(str(ui_map))


def test_ui_map_bad_version_rejected():
    ui_map = write_temp_ui_map("version: v2\ncockpit_elements: {}\n")
    with pytest.raises(ValueError, match="Unsupported UI map version"):
        OverlayPlanner(str(ui_map))


def test_ui_map_cockpit_elements_type_rejected():
    ui_map = write_temp_ui_map("version: v1\ncockpit_elements: []\n")
    with pytest.raises(ValueError, match="cockpit_elements"):
        OverlayPlanner(str(ui_map))


def test_ui_map_default_overlay_type_rejected():
    ui_map = write_temp_ui_map("version: v1\ncockpit_elements: {}\ndefault_overlay: []\n")
    with pytest.raises(ValueError, match="default_overlay"):
        OverlayPlanner(str(ui_map))
