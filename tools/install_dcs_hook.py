"""
Install SimTutor DCS Export hook files into a Saved Games DCS folder.

Usage:
  python -m tools.install_dcs_hook --saved-games "C:\\Users\\you\\Saved Games\\DCS"
  python -m tools.install_dcs_hook --dcs-variant DCS.openbeta
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

SIMTUTOR_EXPORT_SNIPPET = (
    "pcall(function() local SimTutorLfs=require('lfs');"
    "dofile(SimTutorLfs.writedir()..'Scripts/SimTutor/SimTutor.lua'); end)"
)
MONITOR_SETUP_BASENAME = "SimTutor_FA18C_CompositePanel_v1"
DEFAULT_VISION_LAYOUT_ID = "fa18c_composite_panel_v2"
DEFAULT_FRAME_CHANNEL = "composite_panel"
DEFAULT_FRAME_MANIFEST_NAME = "frames.jsonl"
DEFAULT_VISION_BACKGROUND_RGB = (15, 20, 24)
DEFAULT_OVERLAY_COMMAND_HOST = "127.0.0.1"
DEFAULT_OVERLAY_COMMAND_PORT = 7781
DEFAULT_OVERLAY_ACK_HOST = "127.0.0.1"
DEFAULT_OVERLAY_ACK_PORT = 7782


@dataclass(frozen=True)
class InstallResult:
    export_patched: bool
    export_backup: Path | None
    files_copied: bool
    config_written: bool = False
    config_path: Path | None = None
    monitor_setup_written: bool = False
    monitor_setup_path: Path | None = None
    vlm_frame_enabled: bool = False


@dataclass(frozen=True)
class PatchResult:
    changed: bool
    backup_path: Path | None


@dataclass(frozen=True)
class ConfigInstallResult:
    changed: bool
    path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _backup_file(path: Path) -> Path:
    backup_path = path.with_suffix(path.suffix + f".bak.{_timestamp()}")
    shutil.copy2(path, backup_path)
    return backup_path


def patch_export(export_path: Path, snippet: str = SIMTUTOR_EXPORT_SNIPPET) -> PatchResult:
    if export_path.exists():
        try:
            content = export_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"Export.lua is not valid UTF-8: {export_path}. "
                "Please fix or re-save the file before patching."
            ) from exc
    else:
        content = ""

    if "SimTutor/SimTutor.lua" in content or snippet in content:
        return PatchResult(changed=False, backup_path=None)

    _ensure_parent(export_path)
    backup_path = None
    if export_path.exists():
        backup_path = _backup_file(export_path)
        print(f"Backed up {export_path} to {backup_path}")

    new_content = content
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    new_content += snippet + "\n"
    export_path.write_text(new_content, encoding="utf-8")
    return PatchResult(changed=True, backup_path=backup_path)


def _lua_string(value: str) -> str:
    return f"[[{value}]]"


def _vision_frames_root(saved_games_dir: Path) -> Path:
    return saved_games_dir / "SimTutor" / "frames"


def _format_lua_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    parts = resolved.parts
    # This installer targets DCS on Windows. When run from WSL, convert `/mnt/<drive>/...`
    # mounts into Windows drive paths so the generated Lua config remains readable by DCS.
    # For native Windows paths, `str(resolved)` is already correct. We intentionally keep
    # the plain-path fallback for local tests that use temporary POSIX directories.
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha():
        drive = parts[2].upper()
        suffix = "\\".join(parts[3:])
        return f"{drive}:\\{suffix}" if suffix else f"{drive}:\\"
    return str(resolved)


def build_composite_panel_config(
    *,
    saved_games_dir: Path,
    frames_root: Path | None = None,
    monitor_setup_name: str = MONITOR_SETUP_BASENAME,
    layout_id: str = DEFAULT_VISION_LAYOUT_ID,
    channel: str = DEFAULT_FRAME_CHANNEL,
    background_rgb: Sequence[int] = DEFAULT_VISION_BACKGROUND_RGB,
    capture_width: int | None = None,
    capture_height: int | None = None,
    overlay_command_host: str = DEFAULT_OVERLAY_COMMAND_HOST,
    overlay_command_port: int = DEFAULT_OVERLAY_COMMAND_PORT,
    overlay_ack_host: str = DEFAULT_OVERLAY_ACK_HOST,
    overlay_ack_port: int = DEFAULT_OVERLAY_ACK_PORT,
    overlay_auto_clear: bool = True,
    overlay_hilite_id: int = 9101,
) -> str:
    effective_frames_root = (frames_root or _vision_frames_root(saved_games_dir)).expanduser().resolve()
    lua_frames_root = _format_lua_path(effective_frames_root)
    background = tuple(int(value) for value in background_rgb)
    if len(background) != 3 or any(value < 0 or value > 255 for value in background):
        raise ValueError("background_rgb must contain exactly three integers between 0 and 255")
    if (capture_width is None) ^ (capture_height is None):
        raise ValueError("capture_width and capture_height must be provided together")
    if not overlay_command_host:
        raise ValueError("overlay_command_host must be non-empty")
    if not overlay_ack_host:
        raise ValueError("overlay_ack_host must be non-empty")
    if overlay_command_port <= 0:
        raise ValueError("overlay_command_port must be positive")
    if overlay_ack_port <= 0:
        raise ValueError("overlay_ack_port must be positive")
    if overlay_hilite_id < 0:
        raise ValueError("overlay_hilite_id must be non-negative")

    lines = [
        "-- SimTutor composite-panel baseline for v0.4 vision bring-up.",
        "-- This file is safe to edit inside Saved Games after installation.",
        "return {",
        "    telemetry = {",
        '        host = "127.0.0.1",',
        "        port = 7780,",
        "        hz = 20,",
        "    },",
        "    handshake = {",
        '        host = "127.0.0.1",',
        "        port = 7793,",
        "    },",
        "    caps = {",
        "        telemetry = true,",
        "        overlay = false,",
        "        overlay_ack = false,",
        "        clickable_actions = false,",
        "        vlm_frame = true,",
        "    },",
        "    overlay = {",
        f'        command_host = "{overlay_command_host}",',
        f"        command_port = {int(overlay_command_port)},",
        f'        ack_host = "{overlay_ack_host}",',
        f"        ack_port = {int(overlay_ack_port)},",
        f"        auto_clear = {'true' if overlay_auto_clear else 'false'},",
        f"        hilite_id = {int(overlay_hilite_id)},",
        "    },",
        "    vision = {",
        "        enabled = true,",
        f'        layout_id = "{layout_id}",',
        f'        channel = "{channel}",',
        f"        output_root = {_lua_string(lua_frames_root)},",
        f'        manifest_name = "{DEFAULT_FRAME_MANIFEST_NAME}",',
        f'        monitor_setup = "{monitor_setup_name}",',
        f"        background_rgb = {{{background[0]}, {background[1]}, {background[2]}}},",
        '        region_order = {"LEFT_MFCD", "CENTER_MFCD", "RIGHT_MFCD"},',
    ]
    if capture_width is not None and capture_height is not None:
        lines.extend(
            [
                "        capture_resolution = {",
                f"            width = {capture_width},",
                f"            height = {capture_height},",
                "        },",
            ]
        )
    lines.extend(
        [
            "    },",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def install_composite_panel_config(
    *,
    saved_games_dir: Path,
    frames_root: Path | None = None,
    monitor_setup_name: str = MONITOR_SETUP_BASENAME,
    layout_id: str = DEFAULT_VISION_LAYOUT_ID,
    channel: str = DEFAULT_FRAME_CHANNEL,
    background_rgb: Sequence[int] = DEFAULT_VISION_BACKGROUND_RGB,
    capture_width: int | None = None,
    capture_height: int | None = None,
    overlay_command_host: str = DEFAULT_OVERLAY_COMMAND_HOST,
    overlay_command_port: int = DEFAULT_OVERLAY_COMMAND_PORT,
    overlay_ack_host: str = DEFAULT_OVERLAY_ACK_HOST,
    overlay_ack_port: int = DEFAULT_OVERLAY_ACK_PORT,
    overlay_auto_clear: bool = True,
    overlay_hilite_id: int = 9101,
) -> ConfigInstallResult:
    config_path = saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua"
    config_text = build_composite_panel_config(
        saved_games_dir=saved_games_dir,
        frames_root=frames_root,
        monitor_setup_name=monitor_setup_name,
        layout_id=layout_id,
        channel=channel,
        background_rgb=background_rgb,
        capture_width=capture_width,
        capture_height=capture_height,
        overlay_command_host=overlay_command_host,
        overlay_command_port=overlay_command_port,
        overlay_ack_host=overlay_ack_host,
        overlay_ack_port=overlay_ack_port,
        overlay_auto_clear=overlay_auto_clear,
        overlay_hilite_id=overlay_hilite_id,
    )
    _ensure_parent(config_path)
    changed = True
    if config_path.exists():
        existing = config_path.read_text(encoding="utf-8")
        if existing == config_text:
            changed = False
    if changed:
        config_path.write_text(config_text, encoding="utf-8")
    return ConfigInstallResult(changed=changed, path=config_path)


def install_scripting_files(source_root: Path, saved_games_dir: Path) -> bool:
    source_dir = source_root / "DCS" / "Scripts" / "SimTutor"
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source dir: {source_dir}")

    target_dir = saved_games_dir / "Scripts" / "SimTutor"
    target_dir.mkdir(parents=True, exist_ok=True)

    any_copied = False
    for name in ["SimTutor.lua", "SimTutor Function.lua"]:
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        dst = target_dir / name
        shutil.copy2(src, dst)
        any_copied = True
    return any_copied


def resolve_saved_games_dir(saved_games: str | None, dcs_variant: str) -> Path:
    if saved_games:
        return Path(saved_games).expanduser()
    return Path.home() / "Saved Games" / dcs_variant


def run_install(
    source_root: Path,
    saved_games_dir: Path,
    install_export: bool,
    *,
    install_composite_panel: bool = False,
    frame_output_root: Path | None = None,
    main_width: int | None = None,
    main_height: int | None = None,
    monitor_mode: str = "extended-right",
) -> InstallResult:
    files_copied = install_scripting_files(source_root, saved_games_dir)
    export_patched = False
    export_backup: Path | None = None
    config_written = False
    config_path: Path | None = None
    monitor_setup_written = False
    monitor_setup_path: Path | None = None

    if install_export:
        export_path = saved_games_dir / "Scripts" / "Export.lua"
        export_result = patch_export(export_path)
        export_patched = export_result.changed
        export_backup = export_result.backup_path

    if install_composite_panel:
        if (main_width is None) ^ (main_height is None):
            raise ValueError("main_width and main_height must be provided together")
        from tools.install_dcs_monitor_setup import install_monitor_setup

        monitor_result = install_monitor_setup(
            saved_games_dir=saved_games_dir,
            main_width=main_width,
            main_height=main_height,
            mode=monitor_mode,
        )
        monitor_setup_written = monitor_result.changed
        monitor_setup_path = monitor_result.monitor_setup_path
        capture_width = monitor_result.total_width
        capture_height = monitor_result.total_height
        config_result = install_composite_panel_config(
            saved_games_dir=saved_games_dir,
            frames_root=frame_output_root,
            capture_width=capture_width,
            capture_height=capture_height,
        )
        config_written = config_result.changed
        config_path = config_result.path

    return InstallResult(
        export_patched=export_patched,
        export_backup=export_backup,
        files_copied=files_copied,
        config_written=config_written,
        config_path=config_path,
        monitor_setup_written=monitor_setup_written,
        monitor_setup_path=monitor_setup_path,
        vlm_frame_enabled=install_composite_panel,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install SimTutor DCS hook files.")
    parser.add_argument(
        "--source-root",
        type=str,
        default=str(_repo_root()),
        help="Repo root containing DCS/ and adapters/.",
    )
    parser.add_argument(
        "--saved-games",
        type=str,
        default=None,
        help="Saved Games directory (defaults to <home>/Saved Games/<variant>).",
    )
    parser.add_argument(
        "--dcs-variant",
        type=str,
        default="DCS",
        help="DCS variant folder under Saved Games (e.g., DCS, DCS.openbeta).",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not patch Export.lua.",
    )
    parser.add_argument(
        "--install-composite-panel",
        action="store_true",
        help="Install SimTutorConfig.lua for the v0.4 composite-panel baseline and enable vlm_frame capability.",
    )
    parser.add_argument(
        "--frame-output-root",
        type=str,
        default=None,
        help="Optional override for the composite-panel frames root; defaults to <Saved Games>/<variant>/SimTutor/frames.",
    )
    parser.add_argument(
        "--monitor-mode",
        type=str,
        default="extended-right",
        help="Monitor-setup mode used by --install-composite-panel; if width/height are omitted on Windows the installer auto-detects the current primary-screen resolution.",
    )
    parser.add_argument(
        "--main-width",
        type=int,
        default=None,
        help="Optional main display width in pixels for monitor-setup installation. Omit together with --main-height to auto-detect on Windows, or pass explicitly on non-Windows shells.",
    )
    parser.add_argument(
        "--main-height",
        type=int,
        default=None,
        help="Optional main display height in pixels for monitor-setup installation. Omit together with --main-width to auto-detect on Windows, or pass explicitly on non-Windows shells.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser()
    saved_games_dir = resolve_saved_games_dir(args.saved_games, args.dcs_variant)
    frame_output_root = Path(args.frame_output_root).expanduser() if args.frame_output_root else None

    try:
        result = run_install(
            source_root=source_root,
            saved_games_dir=saved_games_dir,
            install_export=not args.no_export,
            install_composite_panel=args.install_composite_panel,
            frame_output_root=frame_output_root,
            main_width=args.main_width,
            main_height=args.main_height,
            monitor_mode=args.monitor_mode,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Install failed: {exc}")
        return 1

    print(
        "SimTutor install complete: "
        f"files_copied={result.files_copied}, "
        f"export_patched={result.export_patched}, "
        f"config_written={result.config_written}, "
        f"monitor_setup_written={result.monitor_setup_written}, "
        f"vlm_frame_enabled={result.vlm_frame_enabled}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
