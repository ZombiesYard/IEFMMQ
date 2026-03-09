from __future__ import annotations

from pathlib import Path

import pytest

from tools.install_dcs_monitor_setup import (
    COMPOSITE_CANVAS_HEIGHT,
    COMPOSITE_CANVAS_WIDTH,
    MODE_EXTENDED_RIGHT,
    MODE_SINGLE_MONITOR,
    MODE_ULTRAWIDE_LEFT_STACK,
    MONITOR_SETUP_BASENAME,
    build_monitor_setup_plan,
    install_monitor_setup,
    resolve_main_dimensions,
)


def test_build_monitor_setup_plan_uses_normalized_strip_inside_extended_canvas() -> None:
    plan = build_monitor_setup_plan(main_width=1920, main_height=1080, mode=MODE_EXTENDED_RIGHT)

    assert plan.setup_name == MONITOR_SETUP_BASENAME
    assert plan.mode == MODE_EXTENDED_RIGHT
    assert plan.canvas_width == COMPOSITE_CANVAS_WIDTH
    assert plan.canvas_height == COMPOSITE_CANVAS_HEIGHT
    assert plan.total_width == 4480
    assert plan.total_height == 1440
    assert [(viewport.name, viewport.x, viewport.y, viewport.width, viewport.height) for viewport in plan.viewports] == [
        ("LEFT_MFCD", 2136, 24, 448, 448),
        ("CENTER_MFCD", 2136, 496, 448, 448),
        ("RIGHT_MFCD", 2136, 968, 448, 448),
    ]


def test_monitor_setup_lua_contains_extended_canvas_viewports() -> None:
    plan = build_monitor_setup_plan(main_width=2560, main_height=1440, mode=MODE_EXTENDED_RIGHT)
    lua_text = plan.lua_text

    assert "Viewports.Center =" in lua_text
    assert "_  = function(p) return p; end;" in lua_text
    assert "UIMainView = Viewports.Center" in lua_text
    assert "GU_MAIN_VIEWPORT = UIMainView" in lua_text
    assert "-- Recommended DCS resolution: 5120x1440" in lua_text
    assert "LEFT_MFCD =" in lua_text
    assert "CENTER_MFCD =" in lua_text
    assert "RIGHT_MFCD =" in lua_text
    assert "x = 2776;" in lua_text
    assert "y = 968;" in lua_text


def test_build_monitor_setup_plan_supports_single_monitor_mode() -> None:
    plan = build_monitor_setup_plan(main_width=1920, main_height=1080, mode=MODE_SINGLE_MONITOR)

    assert plan.mode == MODE_SINGLE_MONITOR
    assert plan.total_width == 1920
    assert plan.total_height == 1080
    assert plan.canvas_width == 660
    assert plan.canvas_height == 1080
    assert plan.main_width == 1260
    assert plan.main_height == 1080
    assert [(viewport.name, viewport.x, viewport.y, viewport.width, viewport.height) for viewport in plan.viewports] == [
        ("LEFT_MFCD", 162, 18, 336, 336),
        ("CENTER_MFCD", 162, 372, 336, 336),
        ("RIGHT_MFCD", 162, 726, 336, 336),
    ]


def test_single_monitor_lua_places_main_view_on_right_of_left_stack() -> None:
    plan = build_monitor_setup_plan(main_width=1920, main_height=1080, mode=MODE_SINGLE_MONITOR)
    lua_text = plan.lua_text

    assert "-- Recommended DCS resolution: 1920x1080" in lua_text
    assert "Description = 'SimTutor F/A-18C composite panel viewport PoC (single monitor normalized left-stack layout)'" in lua_text
    assert "    x = 660;" in lua_text
    assert "    y = 0;" in lua_text
    assert "    width = 1260;" in lua_text
    assert "    height = 1080;" in lua_text


def test_build_monitor_setup_plan_supports_ultrawide_left_stack_mode() -> None:
    plan = build_monitor_setup_plan(main_width=3440, main_height=1440, mode=MODE_ULTRAWIDE_LEFT_STACK)

    assert plan.mode == MODE_ULTRAWIDE_LEFT_STACK
    assert plan.total_width == 3440
    assert plan.total_height == 1440
    assert plan.canvas_width == 880
    assert plan.canvas_height == 1440
    assert plan.main_width == 2560
    assert plan.main_height == 1440
    assert [(viewport.name, viewport.x, viewport.y, viewport.width, viewport.height) for viewport in plan.viewports] == [
        ("LEFT_MFCD", 216, 24, 448, 448),
        ("CENTER_MFCD", 216, 496, 448, 448),
        ("RIGHT_MFCD", 216, 968, 448, 448),
    ]


def test_ultrawide_left_stack_lua_places_main_view_on_right() -> None:
    plan = build_monitor_setup_plan(main_width=3440, main_height=1440, mode=MODE_ULTRAWIDE_LEFT_STACK)
    lua_text = plan.lua_text

    assert "-- Recommended DCS resolution: 3440x1440" in lua_text
    assert "Description = 'SimTutor F/A-18C composite panel viewport PoC (ultrawide normalized left-stack layout)'" in lua_text
    assert "    x = 880;" in lua_text
    assert "    width = 2560;" in lua_text
    assert "CENTER_MFCD =" in lua_text


def test_ultrawide_and_single_monitor_share_same_normalized_solver_family() -> None:
    single = build_monitor_setup_plan(main_width=2560, main_height=1440, mode=MODE_SINGLE_MONITOR)
    ultrawide = build_monitor_setup_plan(main_width=2560, main_height=1440, mode=MODE_ULTRAWIDE_LEFT_STACK)
    assert single.viewports == ultrawide.viewports
    assert single.main_width == ultrawide.main_width


def test_install_monitor_setup_writes_saved_games_config_path(tmp_path: Path) -> None:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"

    result = install_monitor_setup(
        saved_games_dir=saved_games_dir,
        main_width=1920,
        main_height=1080,
        mode=MODE_EXTENDED_RIGHT,
    )

    expected_path = saved_games_dir / "Config" / "MonitorSetup" / f"{MONITOR_SETUP_BASENAME}.lua"
    assert result.monitor_setup_path == expected_path
    assert result.changed is True
    assert result.mode == MODE_EXTENDED_RIGHT
    assert expected_path.exists()
    assert "LEFT_MFCD" in expected_path.read_text(encoding="utf-8")


def test_install_monitor_setup_is_idempotent(tmp_path: Path) -> None:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"

    first = install_monitor_setup(
        saved_games_dir=saved_games_dir,
        main_width=1920,
        main_height=1080,
        mode=MODE_EXTENDED_RIGHT,
    )
    second = install_monitor_setup(
        saved_games_dir=saved_games_dir,
        main_width=1920,
        main_height=1080,
        mode=MODE_EXTENDED_RIGHT,
    )

    assert first.changed is True
    assert second.changed is False


def test_resolve_main_dimensions_uses_auto_detection_when_dimensions_missing(monkeypatch) -> None:
    monkeypatch.setattr("tools.install_dcs_monitor_setup.detect_main_resolution", lambda: (2560, 1440))

    assert resolve_main_dimensions(None, None) == (2560, 1440)


def test_install_monitor_setup_auto_detects_resolution_when_dimensions_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("tools.install_dcs_monitor_setup.detect_main_resolution", lambda: (1920, 1080))
    saved_games_dir = tmp_path / "Saved Games" / "DCS"

    result = install_monitor_setup(saved_games_dir=saved_games_dir, mode=MODE_EXTENDED_RIGHT)

    assert result.total_width == 4480
    assert result.total_height == 1440
    assert result.monitor_setup_path.exists()
    assert "-- Recommended DCS resolution: 4480x1440" in result.monitor_setup_path.read_text(encoding="utf-8")


def test_detect_main_resolution_raises_when_all_windows_detectors_fail(monkeypatch) -> None:
    monkeypatch.setattr("tools.install_dcs_monitor_setup._detect_resolution_windows", lambda: None)
    monkeypatch.setattr("tools.install_dcs_monitor_setup._detect_resolution_powershell", lambda: None)

    with pytest.raises(
        RuntimeError,
        match="failed to detect current screen resolution automatically; pass --main-width and --main-height explicitly",
    ):
        resolve_main_dimensions(None, None)


@pytest.mark.parametrize(
    ("main_width", "main_height"),
    [
        (0, 1080),
        (1920, 0),
        (-1, 1080),
    ],
)
def test_build_monitor_setup_plan_rejects_non_positive_dimensions(main_width: int, main_height: int) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        build_monitor_setup_plan(main_width=main_width, main_height=main_height)


def test_build_monitor_setup_plan_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported mode"):
        build_monitor_setup_plan(main_width=1920, main_height=1080, mode="unknown")


def test_resolve_main_dimensions_rejects_partial_manual_override() -> None:
    with pytest.raises(ValueError, match="main_width and main_height must be provided together"):
        resolve_main_dimensions(1920, None)


def test_single_monitor_rejects_too_narrow_screen() -> None:
    with pytest.raises(ValueError, match="too small"):
        build_monitor_setup_plan(main_width=639, main_height=1080, mode=MODE_SINGLE_MONITOR)
