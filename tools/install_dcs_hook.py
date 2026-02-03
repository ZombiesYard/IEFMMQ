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

SIMTUTOR_EXPORT_SNIPPET = (
    "pcall(function() local SimTutorLfs=require('lfs');"
    "dofile(SimTutorLfs.writedir()..'Scripts/SimTutor/SimTutor.lua'); end)"
)


@dataclass(frozen=True)
class InstallResult:
    export_patched: bool
    export_backup: Path | None
    files_copied: bool


@dataclass(frozen=True)
class PatchResult:
    changed: bool
    backup_path: Path | None


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
) -> InstallResult:
    files_copied = install_scripting_files(source_root, saved_games_dir)
    export_patched = False
    export_backup: Path | None = None

    if install_export:
        export_path = saved_games_dir / "Scripts" / "Export.lua"
        export_result = patch_export(export_path)
        export_patched = export_result.changed
        export_backup = export_result.backup_path

    return InstallResult(
        export_patched=export_patched,
        export_backup=export_backup,
        files_copied=files_copied,
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
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser()
    saved_games_dir = resolve_saved_games_dir(args.saved_games, args.dcs_variant)

    try:
        result = run_install(
            source_root=source_root,
            saved_games_dir=saved_games_dir,
            install_export=not args.no_export,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Install failed: {exc}")
        return 1

    print(
        "SimTutor install complete: "
        f"files_copied={result.files_copied}, "
        f"export_patched={result.export_patched}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
