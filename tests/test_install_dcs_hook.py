from __future__ import annotations

from pathlib import Path

from tools.install_dcs_hook import (
    SIMTUTOR_EXPORT_SNIPPET,
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

    changed = patch_export(export_path)
    assert changed is True
    content = export_path.read_text(encoding="utf-8")
    assert original.strip() in content
    assert SIMTUTOR_EXPORT_SNIPPET in content


def test_patch_export_idempotent(tmp_path: Path) -> None:
    export_path = tmp_path / "Export.lua"
    _write(export_path, "print('hello')\n")

    first = patch_export(export_path)
    second = patch_export(export_path)

    assert first is True
    assert second is False
    content = export_path.read_text(encoding="utf-8")
    assert content.count(SIMTUTOR_EXPORT_SNIPPET) == 1


def test_run_install_copies_and_hook(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    simtutor_dir = repo_root / "DCS" / "Scripts" / "SimTutor"
    _write(simtutor_dir / "SimTutor.lua", "-- SimTutor main\n")
    _write(simtutor_dir / "SimTutor Function.lua", "-- SimTutor functions\n")

    hook_template = repo_root / "adapters" / "dcs" / "hooks" / "SimTutor.lua"
    _write(hook_template, "pcall(function() end)\n")

    saved_games_dir = tmp_path / "Saved Games" / "DCS"
    export_path = saved_games_dir / "Scripts" / "Export.lua"
    _write(export_path, "print('existing')\n")

    result = run_install(
        source_root=repo_root,
        saved_games_dir=saved_games_dir,
        install_export=True,
        install_hook=True,
    )

    assert result.files_copied is True
    assert result.export_patched is True
    assert result.hook_installed is True

    assert (saved_games_dir / "Scripts" / "SimTutor" / "SimTutor.lua").exists()
    assert (saved_games_dir / "Scripts" / "SimTutor" / "SimTutor Function.lua").exists()
    assert (saved_games_dir / "Scripts" / "Hooks" / "SimTutor.lua").exists()

    content = export_path.read_text(encoding="utf-8")
    assert SIMTUTOR_EXPORT_SNIPPET in content

