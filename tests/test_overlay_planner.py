from pathlib import Path

import pytest

from core.overlay import OverlayPlanner
from core.types import TutorResponse


UI_MAP = Path("packs/fa18c_startup/ui_map.yaml")


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


def test_tutor_response_can_carry_overlay_action():
    planner = OverlayPlanner(str(UI_MAP))
    overlay_action = planner.plan("apu_switch", intent="pulse").to_action()
    resp = TutorResponse(message="Turn on APU.", actions=[overlay_action]).to_dict()
    assert resp["actions"][0]["intent"] == "pulse"
    assert resp["actions"][0]["element_id"] == "pnt_375"
