"""Installer and preflight helpers for the SimTutor experiment launcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import tempfile
from typing import Callable, Iterable

from adapters.dcs.overlay.config import inspect_overlay_config
from adapters.dcs.installer import (
    DEFAULT_OVERLAY_ACK_PORT,
    DEFAULT_OVERLAY_COMMAND_PORT,
    DEFAULT_OVERLAY_HILITE_SLOT_COUNT,
    DEFAULT_TUTOR_TEXT_PORT,
    MONITOR_SETUP_BASENAME,
    SIMTUTOR_EXPORT_SNIPPET,
    InstallResult,
    resolve_saved_games_dir,
    run_install,
)
from adapters.vision_capture_trigger import DEFAULT_VISION_CAPTURE_TRIGGER_PORT
from simtutor.launcher_settings import LauncherSettings


TELEMETRY_PORT = 7780
HANDSHAKE_PORT = 7793


@dataclass(frozen=True)
class RequiredPort:
    name: str
    host: str
    port: int
    protocol: str = "udp"


REQUIRED_PORTS = (
    RequiredPort("dcs_telemetry", "127.0.0.1", TELEMETRY_PORT),
    RequiredPort("overlay_command", "127.0.0.1", DEFAULT_OVERLAY_COMMAND_PORT),
    RequiredPort("overlay_ack", "127.0.0.1", DEFAULT_OVERLAY_ACK_PORT),
    RequiredPort("tutor_text", "127.0.0.1", DEFAULT_TUTOR_TEXT_PORT),
    RequiredPort("dcs_handshake", "127.0.0.1", HANDSHAKE_PORT),
    RequiredPort("vision_capture_trigger", "127.0.0.1", DEFAULT_VISION_CAPTURE_TRIGGER_PORT),
)

DCS_LISTENER_PORT_NAMES = {"overlay_command", "tutor_text", "dcs_handshake"}


@dataclass(frozen=True)
class PreflightEntry:
    status: str
    code: str
    message: str
    path: Path | None = None


@dataclass(frozen=True)
class PreflightReport:
    entries: tuple[PreflightEntry, ...]

    @property
    def ok(self) -> bool:
        return all(entry.status != "error" for entry in self.entries)

    def to_text(self) -> str:
        lines: list[str] = []
        for entry in self.entries:
            suffix = f" ({entry.path})" if entry.path is not None else ""
            lines.append(f"{entry.status.upper()} {entry.code}: {entry.message}{suffix}")
        return "\n".join(lines)


PortChecker = Callable[[RequiredPort], bool]


def run_launcher_install(
    settings: LauncherSettings,
    *,
    source_root: str | Path | None = None,
    main_width: int | None = None,
    main_height: int | None = None,
    installer: Callable[..., InstallResult] = run_install,
) -> InstallResult:
    saved_games_dir = _saved_games_dir(settings)

    return installer(
        source_root=Path(source_root).expanduser() if source_root is not None else _repo_root(),
        saved_games_dir=saved_games_dir,
        install_export=True,
        install_composite_panel=True,
        main_width=main_width,
        main_height=main_height,
        monitor_mode=settings.monitor_mode,
    )


def run_preflight(
    settings: LauncherSettings,
    *,
    port_checker: PortChecker | None = None,
    required_ports: Iterable[RequiredPort] = REQUIRED_PORTS,
) -> PreflightReport:
    entries: list[PreflightEntry] = []
    saved_games_dir = _saved_games_dir(settings)

    _check_directory_exists(entries, "saved_games_exists", saved_games_dir, "Saved Games directory exists")
    _check_export_snippet(entries, saved_games_dir / "Scripts" / "Export.lua")
    _check_required_file(
        entries,
        "simtutor_lua_exists",
        saved_games_dir / "Scripts" / "SimTutor" / "SimTutor.lua",
        "Scripts/SimTutor/SimTutor.lua exists",
    )
    _check_required_file(
        entries,
        "simtutor_function_lua_exists",
        saved_games_dir / "Scripts" / "SimTutor" / "SimTutor Function.lua",
        "Scripts/SimTutor/SimTutor Function.lua exists",
    )
    _check_required_file(
        entries,
        "simtutor_highlight_lua_exists",
        saved_games_dir / "Scripts" / "Hooks" / "SimTutorHighlight.lua",
        "Scripts/Hooks/SimTutorHighlight.lua exists",
    )
    config_path = saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua"
    _check_required_file(
        entries,
        "simtutor_config_lua_exists",
        config_path,
        "Scripts/SimTutor/SimTutorConfig.lua exists",
    )
    _check_required_file(
        entries,
        "monitor_setup_lua_exists",
        saved_games_dir / "Config" / "MonitorSetup" / f"{MONITOR_SETUP_BASENAME}.lua",
        "monitor setup Lua exists",
    )
    _check_overlay_slots(entries, config_path)
    _check_output_directory(entries, settings.output_log_directory)
    _check_ports(entries, required_ports, port_checker or port_is_available)
    _check_metadata(entries, settings)
    return PreflightReport(tuple(entries))


def port_is_available(required: RequiredPort) -> bool:
    protocol = required.protocol.lower()
    if protocol == "udp":
        socket_type = socket.SOCK_DGRAM
    elif protocol == "tcp":
        socket_type = socket.SOCK_STREAM
    else:
        raise ValueError(f"unsupported port protocol: {required.protocol}")

    with socket.socket(socket.AF_INET, socket_type) as sock:
        try:
            sock.bind((required.host, required.port))
        except OSError:
            return False
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _saved_games_dir(settings: LauncherSettings) -> Path:
    raw = settings.saved_games_path
    saved_games = raw.strip() if isinstance(raw, str) and raw.strip() else None
    return resolve_saved_games_dir(saved_games, settings.dcs_variant)


def _pass(code: str, message: str, path: Path | None = None) -> PreflightEntry:
    return PreflightEntry(status="pass", code=code, message=message, path=path)


def _warn(code: str, message: str, path: Path | None = None) -> PreflightEntry:
    return PreflightEntry(status="warn", code=code, message=message, path=path)


def _error(code: str, message: str, path: Path | None = None) -> PreflightEntry:
    return PreflightEntry(status="error", code=code, message=message, path=path)


def _check_directory_exists(entries: list[PreflightEntry], code: str, path: Path, pass_message: str) -> None:
    if path.is_dir():
        entries.append(_pass(code, pass_message, path))
    else:
        entries.append(_error(code, f"missing directory: {path}", path))


def _check_required_file(entries: list[PreflightEntry], code: str, path: Path, pass_message: str) -> None:
    if path.is_file():
        entries.append(_pass(code, pass_message, path))
    else:
        entries.append(_error(code, f"missing file: {path.name}", path))


def _check_export_snippet(entries: list[PreflightEntry], export_path: Path) -> None:
    if not export_path.is_file():
        entries.append(_error("export_lua_contains_simtutor_snippet", "missing Export.lua", export_path))
        return
    try:
        content = export_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        entries.append(_error("export_lua_contains_simtutor_snippet", "Export.lua is not valid UTF-8", export_path))
        return
    if SIMTUTOR_EXPORT_SNIPPET in content:
        entries.append(
            _pass(
                "export_lua_contains_simtutor_snippet",
                "Scripts/Export.lua contains the SimTutor snippet",
                export_path,
            )
        )
    else:
        entries.append(
            _error(
                "export_lua_contains_simtutor_snippet",
                "Scripts/Export.lua does not contain the SimTutor snippet",
                export_path,
            )
        )


def _check_overlay_slots(entries: list[PreflightEntry], config_path: Path) -> None:
    inspection = inspect_overlay_config(config_path)
    if not inspection.exists:
        entries.append(_error("overlay_hilite_slots", "SimTutorConfig.lua is missing", config_path))
        return

    slot_count = inspection.declared_slot_count
    required_count = DEFAULT_OVERLAY_HILITE_SLOT_COUNT
    if inspection.hilite_ids_declared and slot_count >= required_count:
        entries.append(
            _pass(
                "overlay_hilite_slots",
                f"overlay.hilite_ids exposes {slot_count} highlight slots",
                config_path,
            )
        )
    else:
        entries.append(
            _error(
                "overlay_hilite_slots",
                f"overlay.hilite_ids exposes {slot_count} highlight slots; at least {required_count} required",
                config_path,
            )
        )


def _check_output_directory(entries: list[PreflightEntry], raw_path: str) -> None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        entries.append(_error("output_log_directory_writable", "output log directory is required"))
        return
    path = Path(raw_path).expanduser()
    if not path.is_dir():
        entries.append(_error("output_log_directory_writable", f"missing output log directory: {path}", path))
        return
    try:
        with tempfile.NamedTemporaryFile(prefix=".simtutor-preflight-", dir=path):
            pass
    except OSError as exc:
        entries.append(_error("output_log_directory_writable", f"output log directory is not writable: {exc}", path))
        return
    entries.append(_pass("output_log_directory_writable", "output log directory is writable", path))


def _check_ports(entries: list[PreflightEntry], required_ports: Iterable[RequiredPort], checker: PortChecker) -> None:
    for required in required_ports:
        code = f"port_{required.name}_available"
        try:
            available = checker(required)
        except Exception as exc:
            entries.append(_warn(code, f"could not verify {required.protocol.upper()} port {required.port}: {exc}"))
            continue
        if available:
            if required.name in DCS_LISTENER_PORT_NAMES:
                entries.append(
                    _pass(
                        code,
                        f"{required.protocol.upper()} port {required.port} is available; "
                        "DCS hook is not currently listening",
                    )
                )
            else:
                entries.append(_pass(code, f"{required.protocol.upper()} port {required.port} is available"))
        else:
            if required.name in DCS_LISTENER_PORT_NAMES:
                entries.append(
                    _pass(
                        code,
                        f"{required.protocol.upper()} port {required.port} is occupied; DCS hook appears to be listening",
                    )
                )
            else:
                entries.append(_error(code, f"{required.protocol.upper()} port {required.port} is already occupied"))


def _check_metadata(entries: list[PreflightEntry], settings: LauncherSettings) -> None:
    missing = [
        name
        for name in ("participant_id", "condition", "trial_id")
        if not isinstance(getattr(settings, name), str) or not getattr(settings, name).strip()
    ]
    if missing:
        entries.append(_error("experiment_metadata_filled", "missing formal-run metadata: " + ", ".join(missing)))
    else:
        entries.append(_pass("experiment_metadata_filled", "participant, condition, and trial metadata are filled"))


__all__ = [
    "PreflightEntry",
    "PreflightReport",
    "REQUIRED_PORTS",
    "RequiredPort",
    "port_is_available",
    "run_launcher_install",
    "run_preflight",
]
