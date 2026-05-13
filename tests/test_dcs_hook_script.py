from __future__ import annotations

from pathlib import Path


def test_simtutor_highlight_defines_as_lua_string_before_mission_do_script() -> None:
    hook_path = Path(__file__).resolve().parents[1] / "DCS" / "Scripts" / "Hooks" / "SimTutorHighlight.lua"
    content = hook_path.read_text(encoding="utf-8")

    as_lua_string_idx = content.index("local function as_lua_string")
    mission_do_script_idx = content.index("local function missionDoScript")

    assert as_lua_string_idx < mission_do_script_idx
