from __future__ import annotations

import subprocess

from simtutor.launcher import _fake_process_command
from simtutor.launcher_settings import LauncherSettings


def test_fake_process_command_escapes_profile_values() -> None:
    settings = LauncherSettings(participant_id="P'01", trial_id="T\"01")
    command = _fake_process_command(settings)

    result = subprocess.run(command, capture_output=True, check=False, text=True, timeout=5)

    assert result.returncode == 0
    assert "participant=P'01 trial=T\"01" in result.stdout
