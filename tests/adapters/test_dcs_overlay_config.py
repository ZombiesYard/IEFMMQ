from __future__ import annotations

from pathlib import Path

from adapters.dcs.overlay.config import (
    build_multi_target_overlay_config_warning,
    inspect_overlay_config,
)


def test_inspect_overlay_config_detects_legacy_single_hilite_id(tmp_path: Path) -> None:
    config_path = tmp_path / "SimTutorConfig.lua"
    config_path.write_text(
        "return {\n"
        "    overlay = {\n"
        "        command_host = \"127.0.0.1\",\n"
        "        hilite_id = 9101,\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )

    inspection = inspect_overlay_config(config_path)

    assert inspection.exists is True
    assert inspection.hilite_id == 9101
    assert inspection.hilite_ids == ()
    assert inspection.declared_slot_count == 1


def test_multi_target_overlay_warning_flags_stale_single_slot_config(tmp_path: Path) -> None:
    config_path = tmp_path / "SimTutorConfig.lua"
    config_path.write_text(
        "return {\n"
        "    overlay = {\n"
        "        hilite_id = 9101,\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )

    warning = build_multi_target_overlay_config_warning(
        max_overlay_targets=2,
        config_path=config_path,
    )

    assert warning is not None
    assert "--max-overlay-targets=2" in warning
    assert "only 1 DCS highlight slot" in warning
    assert "overlay.hilite_ids" in warning
    assert "tools.install_dcs_hook" in warning


def test_multi_target_overlay_warning_flags_explicit_single_slot_config(tmp_path: Path) -> None:
    config_path = tmp_path / "SimTutorConfig.lua"
    config_path.write_text(
        "return {\n"
        "    overlay = {\n"
        "        hilite_id = 9101,\n"
        "        hilite_ids = {9101},\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )

    warning = build_multi_target_overlay_config_warning(
        max_overlay_targets=2,
        config_path=config_path,
    )

    assert warning is not None
    assert "only 1 DCS highlight slot" in warning


def test_inspect_overlay_config_counts_numeric_keyed_hilite_ids_as_values(tmp_path: Path) -> None:
    config_path = tmp_path / "SimTutorConfig.lua"
    config_path.write_text(
        "return {\n"
        "    overlay = {\n"
        "        hilite_ids = {[1] = 9101, [2] = 9102},\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )

    inspection = inspect_overlay_config(config_path)

    assert inspection.hilite_ids == (9101, 9102)
    assert inspection.declared_slot_count == 2


def test_multi_target_overlay_warning_accepts_explicit_two_slot_config(tmp_path: Path) -> None:
    config_path = tmp_path / "SimTutorConfig.lua"
    config_path.write_text(
        "return {\n"
        "    overlay = {\n"
        "        hilite_id = 9101,\n"
        "        hilite_ids = {9101, 9102},\n"
        "    },\n"
        "}\n",
        encoding="utf-8",
    )

    warning = build_multi_target_overlay_config_warning(
        max_overlay_targets=2,
        config_path=config_path,
    )

    assert warning is None
