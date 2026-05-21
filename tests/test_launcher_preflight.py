from __future__ import annotations

from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from simtutor.launcher import _run_install_action, _run_preflight_action
from simtutor.launcher_preflight import (
    PreflightEntry,
    PreflightReport,
    REQUIRED_PORTS,
    RequiredPort,
    run_launcher_install,
    run_preflight,
)
from simtutor.launcher_settings import LauncherSettings
from tools.install_dcs_hook import MONITOR_SETUP_BASENAME, SIMTUTOR_EXPORT_SNIPPET


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _settings(tmp_path: Path) -> LauncherSettings:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    (tmp_path / "logs").mkdir()
    return LauncherSettings(
        saved_games_path=str(saved_games_dir),
        dcs_variant="DCS",
        monitor_mode="extended-right",
        output_log_directory=str(tmp_path / "logs"),
        participant_id="P01",
        condition="with_tutor",
        trial_id="T01",
        max_overlay_targets=2,
    )


def _write_repo_scripting_files(repo_root: Path) -> None:
    _write(repo_root / "DCS" / "Scripts" / "SimTutor" / "SimTutor.lua", "-- SimTutor main\n")
    _write(repo_root / "DCS" / "Scripts" / "SimTutor" / "SimTutor Function.lua", "-- functions\n")
    _write(repo_root / "DCS" / "Scripts" / "Hooks" / "SimTutorHighlight.lua", "-- highlight\n")


def _write_installed_tree(saved_games_dir: Path) -> None:
    _write(saved_games_dir / "Scripts" / "Export.lua", SIMTUTOR_EXPORT_SNIPPET + "\n")
    _write(saved_games_dir / "Scripts" / "SimTutor" / "SimTutor.lua", "-- SimTutor main\n")
    _write(saved_games_dir / "Scripts" / "SimTutor" / "SimTutor Function.lua", "-- functions\n")
    _write(saved_games_dir / "Scripts" / "Hooks" / "SimTutorHighlight.lua", "-- highlight\n")
    _write(
        saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua",
        "return {\n"
        "    overlay = {\n"
        "        hilite_ids = {9101, 9102, 9103, 9104},\n"
        "    },\n"
        "}\n",
    )
    _write(saved_games_dir / "Config" / "MonitorSetup" / f"{MONITOR_SETUP_BASENAME}.lua", "-- monitor\n")


def _entry_by_code(report, code: str):
    return next(entry for entry in report.entries if entry.code == code)


def test_preflight_reports_installed_tree_as_pass(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    saved_games_dir = Path(settings.saved_games_path)
    _write_installed_tree(saved_games_dir)

    report = run_preflight(settings, port_checker=lambda _port: True)

    assert report.ok is True
    assert {entry.status for entry in report.entries} == {"pass"}
    assert "PASS saved_games_exists" in report.to_text()


def test_preflight_reports_missing_required_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    saved_games_dir = Path(settings.saved_games_path)
    _write_installed_tree(saved_games_dir)
    (saved_games_dir / "Scripts" / "SimTutor" / "SimTutor Function.lua").unlink()

    report = run_preflight(settings, port_checker=lambda _port: True)

    entry = _entry_by_code(report, "simtutor_function_lua_exists")
    assert report.ok is False
    assert entry.status == "error"
    assert "SimTutor Function.lua" in entry.message


def test_preflight_requires_exact_export_snippet(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    saved_games_dir = Path(settings.saved_games_path)
    _write_installed_tree(saved_games_dir)
    _write(
        saved_games_dir / "Scripts" / "Export.lua",
        "-- mentions Scripts/SimTutor/SimTutor.lua but does not install the hook\n",
    )

    report = run_preflight(settings, port_checker=lambda _port: True)

    entry = _entry_by_code(report, "export_lua_contains_simtutor_snippet")
    assert report.ok is False
    assert entry.status == "error"


def test_preflight_reports_missing_highlight_slots(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    saved_games_dir = Path(settings.saved_games_path)
    _write_installed_tree(saved_games_dir)
    _write(
        saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua",
        "return {\n"
        "    overlay = {\n"
        "        hilite_ids = {9101, 9102},\n"
        "    },\n"
        "}\n",
    )

    report = run_preflight(settings, port_checker=lambda _port: True)

    entry = _entry_by_code(report, "overlay_hilite_slots")
    assert report.ok is False
    assert entry.status == "error"
    assert "2" in entry.message
    assert "4" in entry.message


def test_preflight_reports_port_conflict(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    saved_games_dir = Path(settings.saved_games_path)
    _write_installed_tree(saved_games_dir)
    conflicted = next(port for port in REQUIRED_PORTS if port.name == "overlay_command")

    def fake_port_checker(port) -> bool:
        return port != conflicted

    report = run_preflight(settings, port_checker=fake_port_checker)

    entry = _entry_by_code(report, "port_overlay_command_available")
    assert report.ok is False
    assert entry.status == "error"
    assert str(conflicted.port) in entry.message


def test_preflight_reports_real_udp_port_conflict(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    saved_games_dir = Path(settings.saved_games_path)
    _write_installed_tree(saved_games_dir)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except PermissionError as exc:
        pytest.skip(f"socket creation is not permitted in this environment: {exc}")
    with sock:
        try:
            sock.bind(("127.0.0.1", 0))
        except PermissionError as exc:
            pytest.skip(f"socket bind is not permitted in this environment: {exc}")
        occupied_port = int(sock.getsockname()[1])

        report = run_preflight(
            settings,
            required_ports=(RequiredPort("demo", "127.0.0.1", occupied_port),),
        )

    entry = _entry_by_code(report, "port_demo_available")
    assert report.ok is False
    assert entry.status == "error"
    assert str(occupied_port) in entry.message


def test_launcher_install_reuses_existing_installers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    repo_root = tmp_path / "repo"
    _write_repo_scripting_files(repo_root)

    result = run_launcher_install(
        settings,
        source_root=repo_root,
        main_width=1920,
        main_height=1080,
    )

    saved_games_dir = Path(settings.saved_games_path)
    assert result.files_copied is True
    assert result.export_patched is True
    assert result.config_path == saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua"
    assert result.monitor_setup_path == saved_games_dir / "Config" / "MonitorSetup" / f"{MONITOR_SETUP_BASENAME}.lua"
    assert SIMTUTOR_EXPORT_SNIPPET in (saved_games_dir / "Scripts" / "Export.lua").read_text(encoding="utf-8")


def test_launcher_install_action_formats_summary_without_process_start(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    calls: list[LauncherSettings] = []

    def fake_installer(received: LauncherSettings):
        calls.append(received)
        return SimpleNamespace(
            files_copied=True,
            export_patched=False,
            config_written=True,
            monitor_setup_written=True,
            vlm_frame_enabled=True,
        )

    summary = _run_install_action(settings, installer=fake_installer)

    assert calls == [settings]
    assert "install complete" in summary
    assert "files_copied=True" in summary
    assert "vlm_frame_enabled=True" in summary


def test_launcher_preflight_action_formats_report_without_process_start(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = PreflightReport((PreflightEntry("pass", "demo", "ready"),))
    calls: list[LauncherSettings] = []

    def fake_runner(received: LauncherSettings) -> PreflightReport:
        calls.append(received)
        return report

    returned, text = _run_preflight_action(settings, runner=fake_runner)

    assert calls == [settings]
    assert returned is report
    assert text == "preflight report:\nPASS demo: ready"
