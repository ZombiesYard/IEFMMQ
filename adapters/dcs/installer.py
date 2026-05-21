"""DCS installer facade used by application-facing launcher code."""

from __future__ import annotations

from tools.install_dcs_hook import (
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


__all__ = [
    "DEFAULT_OVERLAY_ACK_PORT",
    "DEFAULT_OVERLAY_COMMAND_PORT",
    "DEFAULT_OVERLAY_HILITE_SLOT_COUNT",
    "DEFAULT_TUTOR_TEXT_PORT",
    "MONITOR_SETUP_BASENAME",
    "SIMTUTOR_EXPORT_SNIPPET",
    "InstallResult",
    "resolve_saved_games_dir",
    "run_install",
]
