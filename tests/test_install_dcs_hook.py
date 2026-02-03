from __future__ import annotations

from pathlib import Path

import pytest

from tools.install_dcs_hook import (
    SIMTUTOR_EXPORT_SNIPPET,
    install_scripting_files,
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
