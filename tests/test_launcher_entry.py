from __future__ import annotations

import subprocess

from simtutor.launcher import ADVANCED_FIELDS, DIRECTORY_FIELDS, FIELD_CHOICES, FILE_FIELDS, NORMAL_FIELDS, _fake_process_command
from simtutor.launcher_settings import LauncherSettings


def test_fake_process_command_escapes_profile_values() -> None:
    settings = LauncherSettings(participant_id="P'01", trial_id="T\"01")
    command = _fake_process_command(settings)

    result = subprocess.run(command, capture_output=True, check=False, text=True, timeout=5)

    assert result.returncode == 0
    assert "participant=P'01 trial=T\"01" in result.stdout


def test_launcher_normal_fields_hide_fixed_dcs_settings() -> None:
    names = {name for name, _label in NORMAL_FIELDS}

    assert "saved_games_path" not in names
    assert "dcs_variant" not in names
    assert "monitor_mode" in FIELD_CHOICES
    assert "language" in FIELD_CHOICES
    assert "scenario_profile" in FIELD_CHOICES


def test_launcher_advanced_non_path_non_model_fields_have_choices() -> None:
    path_fields = DIRECTORY_FIELDS | FILE_FIELDS
    model_text_fields = {"model_base_url", "text_model_name", "vision_model_name"}

    missing = [
        name
        for name, _label in ADVANCED_FIELDS
        if name not in path_fields and name not in model_text_fields and name not in FIELD_CHOICES
    ]

    assert missing == []


def test_launcher_choices_prioritize_dev_tunnel_vllm_profile() -> None:
    assert FIELD_CHOICES["model_provider"][0] == "openai_compat"
    assert FIELD_CHOICES["model_profile_mode"][0] == "remote_tunnel"
    assert FIELD_CHOICES["model_timeout_s"][0] == "60.0"
    assert FIELD_CHOICES["log_raw_llm_text"][0] == "true"
    assert FIELD_CHOICES["print_model_io"][0] == "true"
    assert FIELD_CHOICES["max_overlay_targets"][0] == "2"
    assert FIELD_CHOICES["ssh_tunnel_profile"][0] == "cloud-247-vllm"
    assert FIELD_CHOICES["ssh_user"][0] == "yz50"
    assert FIELD_CHOICES["ssh_host"][0] == "cloud-247.rz.tu-clausthal.de"
