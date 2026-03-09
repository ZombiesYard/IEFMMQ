"""
Install a Saved Games monitor-setup Lua for the F/A-18C viewport PoC.

Supported modes:
- extended-right: place a fixed 2560x1440 debug canvas to the right of the main viewport
- single-monitor: solve the normalized left-stack layout against one screen
- ultrawide-left-stack: solve the same normalized left-stack layout against an ultrawide screen
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from adapters.vision_prompting import load_vision_layout, solve_layout_geometry
from tools.install_dcs_hook import resolve_saved_games_dir

# Keep the Saved Games monitor-setup filename stable for DCS Options/backward
# compatibility; the active visual contract version is tracked by layout_id in
# packs/fa18c_startup/vision_layout.yaml.
MONITOR_SETUP_BASENAME = "SimTutor_FA18C_CompositePanel_v1"
COMPOSITE_CANVAS_WIDTH = 2560
COMPOSITE_CANVAS_HEIGHT = 1440
MODE_EXTENDED_RIGHT = "extended-right"
MODE_SINGLE_MONITOR = "single-monitor"
MODE_ULTRAWIDE_LEFT_STACK = "ultrawide-left-stack"


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


def _normalize_detected_resolution(width: int, height: int, label: str) -> tuple[int, int]:
    return (_require_positive_int(width, f"{label}.width"), _require_positive_int(height, f"{label}.height"))


def _detect_resolution_windows() -> tuple[int, int] | None:
    if sys.platform != "win32":
        return None
    import ctypes

    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
    return _normalize_detected_resolution(user32.GetSystemMetrics(0), user32.GetSystemMetrics(1), "windows_api")


def _detect_resolution_powershell() -> tuple[int, int] | None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return None
    command = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$bounds=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
        "Write-Output ($bounds.Width.ToString() + ',' + $bounds.Height.ToString())"
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )
    if result.returncode != 0:
        return None
    output = result.stdout.strip().splitlines()
    if not output:
        return None
    match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", output[-1])
    if not match:
        return None
    return _normalize_detected_resolution(int(match.group(1)), int(match.group(2)), "powershell")


def _detect_resolution_xrandr() -> tuple[int, int] | None:
    if not os.environ.get("DISPLAY"):
        return None
    xrandr = shutil.which("xrandr")
    if not xrandr:
        return None
    result = subprocess.run(
        [xrandr, "--current"],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        match = re.search(r"(\d+)x(\d+)\s+\d+(?:\.\d+)?\*", line)
        if match:
            return _normalize_detected_resolution(int(match.group(1)), int(match.group(2)), "xrandr")
    return None


def _detect_resolution_tk() -> tuple[int, int] | None:
    try:
        import tkinter
    except Exception:
        return None
    try:
        root = tkinter.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
    except Exception:
        return None
    return _normalize_detected_resolution(width, height, "tk")


def detect_main_resolution() -> tuple[int, int]:
    # Prefer direct OS APIs first, then process-based fallbacks, then GUI probes.
    for detector in (
        _detect_resolution_windows,
        _detect_resolution_powershell,
        _detect_resolution_xrandr,
        _detect_resolution_tk,
    ):
        try:
            resolution = detector()
        except Exception:
            resolution = None
        if resolution is not None:
            return resolution
    raise RuntimeError(
        "failed to detect current screen resolution automatically; "
        "pass --main-width and --main-height explicitly"
    )


def resolve_main_dimensions(main_width: int | None, main_height: int | None) -> tuple[int, int]:
    if main_width is None and main_height is None:
        return detect_main_resolution()
    if (main_width is None) ^ (main_height is None):
        raise ValueError("main_width and main_height must be provided together")
    assert main_width is not None and main_height is not None
    return (
        _require_positive_int(main_width, "main_width"),
        _require_positive_int(main_height, "main_height"),
    )


def _region_id_to_dcs_name(region_id: str) -> str:
    mapping = {
        "left_ddi": "LEFT_MFCD",
        "ampcd": "CENTER_MFCD",
        "right_ddi": "RIGHT_MFCD",
    }
    try:
        return mapping[region_id]
    except KeyError as exc:
        raise ValueError(f"unsupported region id for DCS viewport export: {region_id}") from exc


def _region_viewports(solution: dict[str, object], *, x_offset: int = 0, y_offset: int = 0) -> tuple[MonitorViewport, ...]:
    regions = solution["regions"]
    assert isinstance(regions, list)
    return tuple(
        MonitorViewport(
            name=_region_id_to_dcs_name(region["region_id"]),
            x=x_offset + int(region["x"]),
            y=y_offset + int(region["y"]),
            width=int(region["width"]),
            height=int(region["height"]),
        )
        for region in regions
    )


def build_monitor_setup_plan(*, main_width: int, main_height: int, mode: str = MODE_EXTENDED_RIGHT) -> MonitorSetupPlan:
    width = _require_positive_int(main_width, "main_width")
    height = _require_positive_int(main_height, "main_height")
    layout = load_vision_layout()

    if mode == MODE_EXTENDED_RIGHT:
        # The debug canvas stays fixed at 2560x1440, but the three native
        # viewports inside that canvas are still solved from the normalized
        # left-strip contract so DCS export geometry and VLM crop geometry do
        # not drift apart.
        export_solution = solve_layout_geometry(
            layout,
            output_width=COMPOSITE_CANVAS_WIDTH,
            output_height=COMPOSITE_CANVAS_HEIGHT,
        )
        total_width = width + COMPOSITE_CANVAS_WIDTH
        total_height = max(height, COMPOSITE_CANVAS_HEIGHT)
        effective_main_x = 0
        effective_main_y = 0
        effective_main_width = width
        effective_main_height = height
        canvas_width = COMPOSITE_CANVAS_WIDTH
        canvas_height = COMPOSITE_CANVAS_HEIGHT
        viewports = _region_viewports(export_solution, x_offset=width)
    elif mode in {MODE_SINGLE_MONITOR, MODE_ULTRAWIDE_LEFT_STACK}:
        # Both single-monitor and ultrawide-left-stack intentionally share the
        # same normalized solver family; only the target screen size changes.
        solution = solve_layout_geometry(layout, output_width=width, output_height=height)
        main_rect = solution["main_view_rect"]
        strip_rect = solution["strip_rect"]
        assert isinstance(main_rect, dict)
        assert isinstance(strip_rect, dict)
        total_width = width
        total_height = height
        effective_main_x = int(main_rect["x"])
        effective_main_y = int(main_rect["y"])
        effective_main_width = int(main_rect["width"])
        effective_main_height = int(main_rect["height"])
        canvas_width = int(strip_rect["width"])
        canvas_height = int(strip_rect["height"])
        viewports = _region_viewports(solution)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    lua_text = render_monitor_setup_lua(
        setup_name=MONITOR_SETUP_BASENAME,
        mode=mode,
        main_x=effective_main_x,
        main_y=effective_main_y,
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
    main_x: int,
    main_y: int,
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
        placement_comment = f"-- Fixed debug export canvas on the right: {COMPOSITE_CANVAS_WIDTH}x{COMPOSITE_CANVAS_HEIGHT}"
    elif mode == MODE_SINGLE_MONITOR:
        description = "SimTutor F/A-18C composite panel viewport PoC (single monitor normalized left-stack layout)"
        placement_comment = f"-- Normalized export strip on a single screen: {screen_width - main_width}x{screen_height}"
    elif mode == MODE_ULTRAWIDE_LEFT_STACK:
        description = "SimTutor F/A-18C composite panel viewport PoC (ultrawide normalized left-stack layout)"
        placement_comment = f"-- Normalized export strip on an ultrawide screen: {screen_width - main_width}x{screen_height}"
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
        f"    x = {main_x};",
        f"    y = {main_y};",
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
    main_width: int | None = None,
    main_height: int | None = None,
    mode: str = MODE_EXTENDED_RIGHT,
) -> InstallResult:
    resolved_width, resolved_height = resolve_main_dimensions(main_width, main_height)
    plan = build_monitor_setup_plan(main_width=resolved_width, main_height=resolved_height, mode=mode)
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
        help="Layout mode: extended-right uses a fixed debug canvas; single-monitor and ultrawide-left-stack both resolve the same normalized left-stack layout against the target screen.",
    )
    parser.add_argument("--main-width", type=int, default=None, help="Main screen width in pixels. If omitted together with --main-height, the installer auto-detects the current primary-screen resolution.")
    parser.add_argument("--main-height", type=int, default=None, help="Main screen height in pixels. If omitted together with --main-width, the installer auto-detects the current primary-screen resolution.")
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
    except (RuntimeError, ValueError) as exc:
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
