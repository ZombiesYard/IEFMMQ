"""
Install a Saved Games monitor-setup Lua for the F/A-18C viewport PoC.

Supported modes:
- extended-right: place the 2560x1440 composite canvas to the right of the main viewport
- single-monitor: place LEFT_MFCD/CENTER_MFCD/RIGHT_MFCD on the top band of one screen
- ultrawide-left-stack: stack LEFT_MFCD/CENTER_MFCD/RIGHT_MFCD vertically in the extra left strip of an ultrawide screen
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from adapters.vision_prompting import load_vision_layout
from tools.install_dcs_hook import resolve_saved_games_dir

MONITOR_SETUP_BASENAME = "SimTutor_FA18C_CompositePanel_v1"
COMPOSITE_CANVAS_WIDTH = 2560
COMPOSITE_CANVAS_HEIGHT = 1440
TOP_ROW_CANONICAL_HEIGHT = 800
MIN_SINGLE_MONITOR_MAIN_HEIGHT = 280
MODE_EXTENDED_RIGHT = "extended-right"
MODE_SINGLE_MONITOR = "single-monitor"
MODE_ULTRAWIDE_LEFT_STACK = "ultrawide-left-stack"
ULTRAWIDE_STACK_MARGIN = 24
_VIEWPORT_REGION_TO_DCS_NAME = {
    "left_ddi": "LEFT_MFCD",
    "ampcd": "CENTER_MFCD",
    "right_ddi": "RIGHT_MFCD",
}


@dataclass(frozen=True)
class MonitorViewport:
    name: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class MonitorSetupPlan:
    setup_name: str
    mode: str
    main_width: int
    main_height: int
    total_width: int
    total_height: int
    canvas_width: int
    canvas_height: int
    viewports: tuple[MonitorViewport, ...]
    lua_text: str


@dataclass(frozen=True)
class InstallResult:
    monitor_setup_path: Path
    changed: bool
    mode: str
    total_width: int
    total_height: int


def _require_positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _expected_viewport_names() -> tuple[str, ...]:
    return tuple(_VIEWPORT_REGION_TO_DCS_NAME.values())


def _build_monitor_viewports_extended(main_width: int) -> tuple[MonitorViewport, ...]:
    layout = load_vision_layout()
    viewports: list[MonitorViewport] = []
    for region in layout["regions"]:
        region_id = region["region_id"]
        dcs_name = _VIEWPORT_REGION_TO_DCS_NAME.get(region_id)
        if dcs_name is None:
            continue
        viewports.append(
            MonitorViewport(
                name=dcs_name,
                x=main_width + int(region["x"]),
                y=int(region["y"]),
                width=int(region["width"]),
                height=int(region["height"]),
            )
        )
    expected_names = _expected_viewport_names()
    actual_names = tuple(viewport.name for viewport in viewports)
    if actual_names != expected_names:
        raise ValueError(f"unexpected viewport order: {actual_names!r}")
    return tuple(viewports)


def _single_monitor_scale(screen_width: int, screen_height: int) -> float:
    width_scale = screen_width / COMPOSITE_CANVAS_WIDTH
    height_scale = max(0.1, (screen_height - MIN_SINGLE_MONITOR_MAIN_HEIGHT) / TOP_ROW_CANONICAL_HEIGHT)
    return min(width_scale, height_scale)


def _build_monitor_viewports_single(screen_width: int, screen_height: int) -> tuple[MonitorViewport, ...]:
    layout = load_vision_layout()
    scale = _single_monitor_scale(screen_width, screen_height)
    top_band_height = int(round(TOP_ROW_CANONICAL_HEIGHT * scale))
    if screen_height - top_band_height < MIN_SINGLE_MONITOR_MAIN_HEIGHT:
        raise ValueError("screen height is too small for single-monitor viewport layout")
    viewports: list[MonitorViewport] = []
    for region in layout["regions"]:
        region_id = region["region_id"]
        dcs_name = _VIEWPORT_REGION_TO_DCS_NAME.get(region_id)
        if dcs_name is None:
            continue
        viewports.append(
            MonitorViewport(
                name=dcs_name,
                x=int(round(int(region["x"]) * scale)),
                y=int(round(int(region["y"]) * scale)),
                width=int(round(int(region["width"]) * scale)),
                height=int(round(int(region["height"]) * scale)),
            )
        )
    expected_names = _expected_viewport_names()
    actual_names = tuple(viewport.name for viewport in viewports)
    if actual_names != expected_names:
        raise ValueError(f"unexpected viewport order: {actual_names!r}")
    return tuple(viewports)


def _build_monitor_viewports_ultrawide_left_stack(screen_width: int, screen_height: int) -> tuple[MonitorViewport, ...]:
    main_view_width = int(round(screen_height * (16 / 9)))
    left_strip_width = screen_width - main_view_width
    if left_strip_width <= 0:
        raise ValueError("screen is not wide enough for ultrawide-left-stack layout")

    viewport_size = min(
        left_strip_width - 2 * ULTRAWIDE_STACK_MARGIN,
        (screen_height - 4 * ULTRAWIDE_STACK_MARGIN) // 3,
    )
    if viewport_size <= 0:
        raise ValueError("screen is too small for ultrawide-left-stack layout")

    x = left_strip_width // 2 - viewport_size // 2
    top_gap = (screen_height - viewport_size * 3) // 4
    if top_gap < ULTRAWIDE_STACK_MARGIN:
        top_gap = ULTRAWIDE_STACK_MARGIN
    y_positions = (
        top_gap,
        top_gap * 2 + viewport_size,
        top_gap * 3 + viewport_size * 2,
    )
    names = _expected_viewport_names()
    return tuple(
        MonitorViewport(name=name, x=x, y=y, width=viewport_size, height=viewport_size)
        for name, y in zip(names, y_positions)
    )


def build_monitor_setup_plan(*, main_width: int, main_height: int, mode: str = MODE_EXTENDED_RIGHT) -> MonitorSetupPlan:
    width = _require_positive_int(main_width, "main_width")
    height = _require_positive_int(main_height, "main_height")
    if mode == MODE_EXTENDED_RIGHT:
        viewports = _build_monitor_viewports_extended(width)
        total_width = width + COMPOSITE_CANVAS_WIDTH
        total_height = max(height, COMPOSITE_CANVAS_HEIGHT)
        effective_main_width = width
        effective_main_height = height
        canvas_width = COMPOSITE_CANVAS_WIDTH
        canvas_height = COMPOSITE_CANVAS_HEIGHT
    elif mode == MODE_SINGLE_MONITOR:
        viewports = _build_monitor_viewports_single(width, height)
        top_band_height = max(viewport.y + viewport.height for viewport in viewports)
        total_width = width
        total_height = height
        effective_main_width = width
        effective_main_height = height - top_band_height
        canvas_width = width
        canvas_height = top_band_height
    elif mode == MODE_ULTRAWIDE_LEFT_STACK:
        viewports = _build_monitor_viewports_ultrawide_left_stack(width, height)
        main_view_width = int(round(height * (16 / 9)))
        total_width = width
        total_height = height
        effective_main_width = main_view_width
        effective_main_height = height
        canvas_width = width - main_view_width
        canvas_height = height
    else:
        raise ValueError(f"unsupported mode: {mode}")
    lua_text = render_monitor_setup_lua(
        setup_name=MONITOR_SETUP_BASENAME,
        mode=mode,
        main_width=effective_main_width,
        main_height=effective_main_height,
        total_width=total_width,
        total_height=total_height,
        screen_width=width,
        screen_height=height,
        viewports=viewports,
    )
    return MonitorSetupPlan(
        setup_name=MONITOR_SETUP_BASENAME,
        mode=mode,
        main_width=effective_main_width,
        main_height=effective_main_height,
        total_width=total_width,
        total_height=total_height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        viewports=viewports,
        lua_text=lua_text,
    )


def render_monitor_setup_lua(
    *,
    setup_name: str,
    mode: str,
    main_width: int,
    main_height: int,
    total_width: int,
    total_height: int,
    screen_width: int,
    screen_height: int,
    viewports: tuple[MonitorViewport, ...],
) -> str:
    aspect = main_width / main_height
    if mode == MODE_EXTENDED_RIGHT:
        description = "SimTutor F/A-18C composite panel viewport PoC (extended desktop to the right)"
        placement_comment = f"-- Composite export canvas on the right: {COMPOSITE_CANVAS_WIDTH}x{COMPOSITE_CANVAS_HEIGHT}"
        center_x = 0
        center_y = 0
    elif mode == MODE_SINGLE_MONITOR:
        description = "SimTutor F/A-18C composite panel viewport PoC (single monitor top-row layout)"
        placement_comment = f"-- Top-row viewport band on a single screen: {screen_width}x{screen_height - main_height}"
        center_x = 0
        center_y = screen_height - main_height
    elif mode == MODE_ULTRAWIDE_LEFT_STACK:
        description = "SimTutor F/A-18C composite panel viewport PoC (ultrawide left-stack layout)"
        placement_comment = f"-- Left ultrawide strip for stacked MFCDs: {screen_width - main_width}x{screen_height}"
        center_x = screen_width - main_width
        center_y = 0
    else:
        raise ValueError(f"unsupported mode: {mode}")
    lines = [
        "_  = function(p) return p; end;",
        f"name = _('{setup_name}')",
        f"Description = '{description}'",
        "",
        f"-- Recommended DCS resolution: {total_width}x{total_height}",
        f"-- Main viewport: {main_width}x{main_height}",
        placement_comment,
        "",
        "Viewports = {}",
        "Viewports.Center =",
        "{",
        f"    x = {center_x};",
        f"    y = {center_y};",
        f"    width = {main_width};",
        f"    height = {main_height};",
        "    viewDx = 0;",
        "    viewDy = 0;",
        f"    aspect = {aspect:.10f};",
        "}",
        "",
        "UIMainView = Viewports.Center",
        "GU_MAIN_VIEWPORT = UIMainView",
        "",
    ]
    for viewport in viewports:
        lines.extend(
            [
                f"{viewport.name} =",
                "{",
                f"    x = {viewport.x};",
                f"    y = {viewport.y};",
                f"    width = {viewport.width};",
                f"    height = {viewport.height};",
                "}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _monitor_setup_path(saved_games_dir: Path) -> Path:
    return saved_games_dir / "Config" / "MonitorSetup" / f"{MONITOR_SETUP_BASENAME}.lua"


def install_monitor_setup(
    *,
    saved_games_dir: Path,
    main_width: int,
    main_height: int,
    mode: str = MODE_EXTENDED_RIGHT,
) -> InstallResult:
    plan = build_monitor_setup_plan(main_width=main_width, main_height=main_height, mode=mode)
    path = _monitor_setup_path(saved_games_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    changed = True
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == plan.lua_text:
            changed = False
    if changed:
        path.write_text(plan.lua_text, encoding="utf-8")
    return InstallResult(
        monitor_setup_path=path,
        changed=changed,
        mode=plan.mode,
        total_width=plan.total_width,
        total_height=plan.total_height,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install SimTutor DCS monitor setup for the F/A-18C viewport PoC.")
    parser.add_argument("--saved-games", type=str, default=None, help="Saved Games directory (defaults to <home>/Saved Games/<variant>).")
    parser.add_argument("--dcs-variant", type=str, default="DCS", help="DCS variant folder under Saved Games (e.g. DCS, DCS.openbeta).")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[MODE_EXTENDED_RIGHT, MODE_SINGLE_MONITOR, MODE_ULTRAWIDE_LEFT_STACK],
        default=MODE_EXTENDED_RIGHT,
        help="Layout mode: extended-right uses a second desktop region; single-monitor places three MFCDs on the top band; ultrawide-left-stack uses the extra left strip of an ultrawide screen.",
    )
    parser.add_argument("--main-width", type=int, required=True, help="Main screen width in pixels.")
    parser.add_argument("--main-height", type=int, required=True, help="Main screen height in pixels.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    saved_games_dir = resolve_saved_games_dir(args.saved_games, args.dcs_variant)

    try:
        result = install_monitor_setup(
            saved_games_dir=saved_games_dir,
            main_width=args.main_width,
            main_height=args.main_height,
            mode=args.mode,
        )
    except ValueError as exc:
        print(f"Install failed: {exc}")
        return 1

    print(
        "SimTutor monitor setup complete: "
        f"path={result.monitor_setup_path}, "
        f"mode={result.mode}, "
        f"changed={result.changed}, "
        f"recommended_resolution={result.total_width}x{result.total_height}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
