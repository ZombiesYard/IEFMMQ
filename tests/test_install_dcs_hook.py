from __future__ import annotations

from pathlib import Path

import pytest

from tools.install_dcs_hook import (
    MONITOR_SETUP_BASENAME,
    SIMTUTOR_EXPORT_SNIPPET,
    build_composite_panel_config,
    install_scripting_files,
    install_composite_panel_config,
    patch_export,
    run_install,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_patch_export_appends_line(tmp_path: Path) -> None:
    export_path = tmp_path / "Export.lua"
    original = "pcall(function() local TheWayLfs=require('lfs');dofile(TheWayLfs.writedir()..'Scripts/TheWay/TheWay.lua'); end)\n"
    _write(export_path, original)

    result = patch_export(export_path)
    assert result.changed is True
    content = export_path.read_text(encoding="utf-8")
    assert original.strip() in content
    assert SIMTUTOR_EXPORT_SNIPPET in content
    backups = list(tmp_path.glob("Export.lua.bak.*"))
    assert backups


def test_patch_export_idempotent(tmp_path: Path) -> None:
    export_path = tmp_path / "Export.lua"
    _write(export_path, "print('hello')\n")

    first = patch_export(export_path)
    second = patch_export(export_path)

    assert first.changed is True
    assert second.changed is False
    content = export_path.read_text(encoding="utf-8")
    assert content.count(SIMTUTOR_EXPORT_SNIPPET) == 1


def test_patch_export_invalid_utf8_raises(tmp_path: Path) -> None:
    export_path = tmp_path / "Export.lua"
    export_path.write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(RuntimeError, match="not valid UTF-8"):
        patch_export(export_path)


def test_patch_export_creates_when_missing(tmp_path: Path) -> None:
    export_path = tmp_path / "Export.lua"
    result = patch_export(export_path)
    assert result.changed is True
    assert result.backup_path is None
    content = export_path.read_text(encoding="utf-8")
    assert content == SIMTUTOR_EXPORT_SNIPPET + "\n"


def test_run_install_copies_and_export(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    simtutor_dir = repo_root / "DCS" / "Scripts" / "SimTutor"
    _write(simtutor_dir / "SimTutor.lua", "-- SimTutor main\n")
    _write(simtutor_dir / "SimTutor Function.lua", "-- SimTutor functions\n")

    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    export_path = saved_games_dir / "Scripts" / "Export.lua"
    _write(export_path, "print('existing')\n")

    result = run_install(
        source_root=repo_root,
        saved_games_dir=saved_games_dir,
        install_export=True,
    )

    assert result.files_copied is True
    assert result.export_patched is True
    assert result.export_backup is not None
    assert (saved_games_dir / "Scripts" / "SimTutor" / "SimTutor.lua").exists()
    assert (saved_games_dir / "Scripts" / "SimTutor" / "SimTutor Function.lua").exists()

    content = export_path.read_text(encoding="utf-8")
    assert SIMTUTOR_EXPORT_SNIPPET in content


def test_run_install_missing_source_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_install(
            source_root=tmp_path / "repo",
            saved_games_dir=tmp_path / "Saved Games" / "DCS",
            install_export=False,
        )


def test_run_install_no_export_copies_only(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    simtutor_dir = repo_root / "DCS" / "Scripts" / "SimTutor"
    _write(simtutor_dir / "SimTutor.lua", "-- SimTutor main\n")
    _write(simtutor_dir / "SimTutor Function.lua", "-- SimTutor functions\n")

    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    result = run_install(
        source_root=repo_root,
        saved_games_dir=saved_games_dir,
        install_export=False,
    )

    assert result.files_copied is True
    assert result.export_patched is False
    assert result.export_backup is None


def test_install_scripting_files_missing_one_file_raises(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    simtutor_dir = repo_root / "DCS" / "Scripts" / "SimTutor"
    _write(simtutor_dir / "SimTutor.lua", "-- SimTutor main\n")
    saved_games_dir = tmp_path / "Saved Games" / "DCS"

    with pytest.raises(FileNotFoundError, match="SimTutor Function.lua"):
        install_scripting_files(repo_root, saved_games_dir)


def test_build_composite_panel_config_enables_vlm_frame_and_frames_root(tmp_path: Path) -> None:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"

    config = build_composite_panel_config(
        saved_games_dir=saved_games_dir,
        capture_width=4480,
        capture_height=1440,
    )

    assert 'vlm_frame = true' in config
    assert f'monitor_setup = "{MONITOR_SETUP_BASENAME}"' in config
    assert 'layout_id = "fa18c_composite_panel_v2"' in config
    assert 'channel = "composite_panel"' in config
    assert str((saved_games_dir / "SimTutor" / "frames").resolve()) in config
    assert "width = 4480" in config
    assert "height = 1440" in config


def test_install_composite_panel_config_is_idempotent(tmp_path: Path) -> None:
    saved_games_dir = tmp_path / "Saved Games" / "DCS"

    first = install_composite_panel_config(saved_games_dir=saved_games_dir)
    second = install_composite_panel_config(saved_games_dir=saved_games_dir)

    assert first.changed is True
    assert second.changed is False
    assert first.path == saved_games_dir / "Scripts" / "SimTutor" / "SimTutorConfig.lua"
    assert 'vlm_frame = true' in first.path.read_text(encoding="utf-8")


def test_run_install_can_deploy_composite_panel_baseline_and_monitor_setup(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    simtutor_dir = repo_root / "DCS" / "Scripts" / "SimTutor"
    _write(simtutor_dir / "SimTutor.lua", "-- SimTutor main\n")
    _write(simtutor_dir / "SimTutor Function.lua", "-- SimTutor functions\n")

    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    result = run_install(
        source_root=repo_root,
        saved_games_dir=saved_games_dir,
        install_export=False,
        install_composite_panel=True,
        main_width=1920,
        main_height=1080,
        monitor_mode="extended-right",
    )

    assert result.files_copied is True
    assert result.vlm_frame_enabled is True
    assert result.config_path is not None
    assert result.monitor_setup_path is not None
    assert result.config_path.exists()
    assert result.monitor_setup_path.exists()
    config_text = result.config_path.read_text(encoding="utf-8")
    assert 'vlm_frame = true' in config_text
    assert 'output_root = [[' in config_text
    assert "width = 4480" in config_text
    assert "height = 1440" in config_text
    assert f'monitor_setup = "{MONITOR_SETUP_BASENAME}"' in config_text


def test_run_install_can_auto_detect_resolution_for_composite_panel(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    simtutor_dir = repo_root / "DCS" / "Scripts" / "SimTutor"
    _write(simtutor_dir / "SimTutor.lua", "-- SimTutor main\n")
    _write(simtutor_dir / "SimTutor Function.lua", "-- SimTutor functions\n")
    monkeypatch.setattr("tools.install_dcs_monitor_setup.detect_main_resolution", lambda: (3440, 1440))

    result = run_install(
        source_root=repo_root,
        saved_games_dir=tmp_path / "Saved Games" / "DCS",
        install_export=False,
        install_composite_panel=True,
        monitor_mode="ultrawide-left-stack",
    )

    assert result.monitor_setup_path is not None
    assert result.config_path is not None
    config_text = result.config_path.read_text(encoding="utf-8")
    assert "width = 3440" in config_text
    assert "height = 1440" in config_text
    assert result.monitor_setup_path.exists()


def test_run_install_rejects_partial_monitor_dimensions_for_composite_panel(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    simtutor_dir = repo_root / "DCS" / "Scripts" / "SimTutor"
    _write(simtutor_dir / "SimTutor.lua", "-- SimTutor main\n")
    _write(simtutor_dir / "SimTutor Function.lua", "-- SimTutor functions\n")

    with pytest.raises(ValueError, match="main_width and main_height must be provided together"):
        run_install(
            source_root=repo_root,
            saved_games_dir=tmp_path / "Saved Games" / "DCS",
            install_export=False,
            install_composite_panel=True,
            main_width=1920,
        )
