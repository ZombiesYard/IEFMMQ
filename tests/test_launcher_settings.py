from __future__ import annotations

import json
from pathlib import Path

import pytest

from simtutor.launcher_settings import (
    LauncherSettings,
    LauncherSettingsError,
    default_saved_games_path,
    load_profile,
    load_settings,
    profiles_dir,
    save_profile,
    save_settings,
    settings_path,
    validate_settings,
)


def _valid_settings() -> LauncherSettings:
    return LauncherSettings(
        saved_games_path="C:/Users/test/Saved Games/DCS",
        dcs_variant="DCS",
        monitor_mode="single-monitor",
        language="zh",
        scenario_profile="airfield",
        output_log_directory="C:/SimTutor/logs",
        participant_id="P01",
        condition="with_tutor",
        trial_id="T01",
        model_provider="openai_compat",
        model_base_url="http://127.0.0.1:8000/v1",
        text_model_name="qwen36_27b",
        vision_model_name="simtutor-vision",
        model_timeout_s=30.0,
        max_overlay_targets=2,
        model_profile_mode="remote_direct",
    )


def test_settings_path_uses_windows_localappdata() -> None:
    env = {"LOCALAPPDATA": "C:/Users/test/AppData/Local"}

    assert settings_path(env=env, platform="win32") == Path("C:/Users/test/AppData/Local") / "SimTutor" / "settings.json"
    assert profiles_dir(env=env, platform="win32") == Path("C:/Users/test/AppData/Local") / "SimTutor" / "profiles"


def test_default_saved_games_path_uses_stable_dcs_directory() -> None:
    env = {"USERPROFILE": "C:/Users/test"}

    assert default_saved_games_path(env=env) == str(Path("C:/Users/test") / "Saved Games" / "DCS")


def test_launcher_defaults_match_dev_tunnel_vllm_profile() -> None:
    settings = LauncherSettings()

    assert settings.model_provider == "openai_compat"
    assert settings.model_profile_mode == "remote_tunnel"
    assert settings.model_base_url == "http://localhost:16324"
    assert settings.text_model_name == "simtutor-base"
    assert settings.vision_model_name == "simtutor-vision"
    assert settings.model_timeout_s == 60.0
    assert settings.log_raw_llm_text is True
    assert settings.print_model_io is True
    assert settings.max_overlay_targets == 2
    assert settings.ssh_tunnel_profile == "cloud-247-vllm"
    assert settings.ssh_user == "yz50"
    assert settings.ssh_host == "cloud-247.rz.tu-clausthal.de"
    assert settings.ssh_local_port == 16324
    assert settings.ssh_remote_host == "127.0.0.1"
    assert settings.ssh_remote_port == 6324


def test_settings_round_trip_json(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = _valid_settings()

    save_settings(settings, path)
    loaded = load_settings(path)

    assert loaded == settings
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["participant_id"] == "P01"
    assert "api_key" not in payload
    assert "password" not in payload
    assert "secret" not in payload


def test_load_settings_returns_defaults_when_missing(tmp_path: Path) -> None:
    loaded = load_settings(tmp_path / "missing.json")

    assert loaded == LauncherSettings()


def test_validate_settings_reports_required_fields() -> None:
    settings = LauncherSettings()
    settings.saved_games_path = ""

    with pytest.raises(LauncherSettingsError) as exc:
        validate_settings(settings)

    message = str(exc.value)
    assert "saved_games_path" in message
    assert "participant_id" in message
    assert "trial_id" in message


def test_validate_settings_rejects_invalid_numbers() -> None:
    settings = _valid_settings()
    settings.model_timeout_s = 0
    settings.max_overlay_targets = -1

    with pytest.raises(LauncherSettingsError) as exc:
        validate_settings(settings)

    assert "model_timeout_s" in str(exc.value)
    assert "max_overlay_targets" in str(exc.value)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf")])
def test_validate_settings_rejects_non_finite_timeout(timeout: float) -> None:
    settings = _valid_settings()
    settings.model_timeout_s = timeout

    with pytest.raises(LauncherSettingsError, match="model_timeout_s"):
        validate_settings(settings)


def test_load_settings_rejects_non_finite_timeout_constant(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"model_timeout_s": NaN}', encoding="utf-8")

    with pytest.raises(LauncherSettingsError, match="non-finite"):
        load_settings(path)


def test_validate_settings_reports_non_numeric_numbers() -> None:
    settings = _valid_settings()
    settings.model_timeout_s = "slow"  # type: ignore[assignment]
    settings.max_overlay_targets = "many"  # type: ignore[assignment]

    with pytest.raises(LauncherSettingsError) as exc:
        validate_settings(settings)

    message = str(exc.value)
    assert "model_timeout_s" in message
    assert "max_overlay_targets" in message


def test_validate_settings_rejects_invalid_raw_bios_port() -> None:
    settings = _valid_settings()
    settings.raw_bios_port = 70000

    with pytest.raises(LauncherSettingsError, match="raw_bios_port"):
        validate_settings(settings)


def test_openai_compat_requires_model_base_url() -> None:
    settings = _valid_settings()
    settings.model_base_url = ""

    with pytest.raises(LauncherSettingsError, match="model_base_url"):
        validate_settings(settings)


def test_remote_tunnel_requires_ssh_profile_fields() -> None:
    settings = _valid_settings()
    settings.model_profile_mode = "remote_tunnel"
    settings.model_base_url = ""
    settings.ssh_user = ""
    settings.ssh_host = ""

    with pytest.raises(LauncherSettingsError) as exc:
        validate_settings(settings)

    message = str(exc.value)
    assert "ssh_user" in message
    assert "ssh_host" in message


def test_local_stub_does_not_require_remote_base_url() -> None:
    settings = _valid_settings()
    settings.model_profile_mode = "local_stub"
    settings.model_provider = "stub"
    settings.model_base_url = ""

    validate_settings(settings)


def test_non_tunnel_profiles_ignore_blank_or_bad_ssh_ports() -> None:
    local_stub = _valid_settings()
    local_stub.model_profile_mode = "local_stub"
    local_stub.model_provider = "stub"
    local_stub.model_base_url = ""
    local_stub.ssh_local_port = ""  # type: ignore[assignment]
    local_stub.ssh_remote_port = "not-a-port"  # type: ignore[assignment]

    loaded = LauncherSettings.from_dict(local_stub.to_dict())
    validate_settings(loaded)

    remote_direct = _valid_settings()
    remote_direct.ssh_local_port = "not-a-port"  # type: ignore[assignment]
    remote_direct.ssh_remote_port = ""  # type: ignore[assignment]

    loaded = LauncherSettings.from_dict(remote_direct.to_dict())
    validate_settings(loaded)


def test_ollama_provider_remains_valid_for_existing_launcher_settings() -> None:
    settings = _valid_settings()
    settings.model_provider = "ollama"
    settings.model_profile_mode = "remote_direct"
    settings.model_base_url = "http://127.0.0.1:11434"

    validate_settings(settings)


def test_openai_compat_rejects_non_string_base_url() -> None:
    settings = _valid_settings()
    settings.model_base_url = 123  # type: ignore[assignment]

    with pytest.raises(LauncherSettingsError, match="model_base_url"):
        validate_settings(settings)


def test_load_settings_rejects_secret_fields(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    payload = _valid_settings().to_dict()
    payload["api_key"] = "sk-secret"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LauncherSettingsError, match="must not store secret field"):
        load_settings(path)


def test_profile_round_trip_uses_sanitized_profile_name(tmp_path: Path) -> None:
    settings = _valid_settings()

    path = save_profile("Pilot 01 / baseline", settings, directory=tmp_path)
    loaded = load_profile("Pilot 01 / baseline", directory=tmp_path)

    assert path.name == "Pilot_01_baseline.json"
    assert loaded == settings


def test_load_profile_reports_missing_profile(tmp_path: Path) -> None:
    with pytest.raises(LauncherSettingsError, match="profile not found"):
        load_profile("missing", directory=tmp_path)
