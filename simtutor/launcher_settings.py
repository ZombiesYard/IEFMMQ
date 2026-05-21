"""Settings persistence for the Windows SimTutor experiment launcher."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping


APP_DIR_NAME = "SimTutor"
SETTINGS_FILENAME = "settings.json"
PROFILES_DIR_NAME = "profiles"

SUPPORTED_LANGUAGES = ("zh", "en")
SUPPORTED_MODEL_PROVIDERS = ("stub", "openai_compat", "ollama")
SUPPORTED_MODEL_PROFILE_MODES = ("local_stub", "remote_direct", "remote_tunnel")

_SECRET_KEY_FRAGMENTS = ("api_key", "token", "password", "secret")


class LauncherSettingsError(ValueError):
    """Raised when launcher settings cannot be loaded, saved, or validated."""


@dataclass
class LauncherSettings:
    saved_games_path: str = ""
    dcs_variant: str = "DCS"
    monitor_mode: str = "single-monitor"
    language: str = "zh"
    scenario_profile: str = "airfield"
    output_log_directory: str = "logs"
    participant_id: str = ""
    condition: str = ""
    trial_id: str = ""
    model_provider: str = "stub"
    model_profile_mode: str = "local_stub"
    model_base_url: str = ""
    text_model_name: str = "Qwen3-8B-Instruct"
    vision_model_name: str = "simtutor-vision"
    model_timeout_s: float = 20.0
    model_enable_multimodal: bool = False
    max_overlay_targets: int = 1
    ssh_tunnel_profile: str = ""
    ssh_executable_path: str = ""
    ssh_user: str = ""
    ssh_host: str = ""
    ssh_local_port: int = 16324
    ssh_remote_host: str = "127.0.0.1"
    ssh_remote_port: int = 6324
    ssh_identity_file_path: str = ""
    preflight_profile: str = ""
    export_profile: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LauncherSettings":
        _reject_secret_keys(payload)
        known = {field.name for field in fields(cls)}
        values = {key: value for key, value in payload.items() if key in known}
        settings = cls(**values)
        try:
            settings.model_timeout_s = float(settings.model_timeout_s)
            settings.max_overlay_targets = int(settings.max_overlay_targets)
        except (TypeError, ValueError) as exc:
            raise LauncherSettingsError("model_timeout_s and max_overlay_targets must be numeric") from exc
        settings.model_enable_multimodal = _parse_bool(settings.model_enable_multimodal, "model_enable_multimodal")
        settings.ssh_local_port = _parse_profile_port(
            settings.ssh_local_port,
            "ssh_local_port",
            default=16324,
            required=settings.model_profile_mode == "remote_tunnel",
        )
        settings.ssh_remote_port = _parse_profile_port(
            settings.ssh_remote_port,
            "ssh_remote_port",
            default=6324,
            required=settings.model_profile_mode == "remote_tunnel",
        )
        if not math.isfinite(settings.model_timeout_s):
            raise LauncherSettingsError("model_timeout_s must be finite")
        return settings

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        _reject_secret_keys(payload)
        return payload


def settings_dir(
    *,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    source = os.environ if env is None else env
    platform_name = sys.platform if platform is None else platform
    if platform_name == "win32":
        base = source.get("LOCALAPPDATA", "").strip()
        if not base:
            raise LauncherSettingsError("LOCALAPPDATA is required on Windows")
        return Path(base) / APP_DIR_NAME
    if source.get("LOCALAPPDATA"):
        return Path(source["LOCALAPPDATA"]) / APP_DIR_NAME
    return (home or Path.home()) / ".simtutor"


def settings_path(
    *,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    return settings_dir(env=env, platform=platform, home=home) / SETTINGS_FILENAME


def profiles_dir(
    *,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    return settings_dir(env=env, platform=platform, home=home) / PROFILES_DIR_NAME


def load_settings(path: str | Path | None = None) -> LauncherSettings:
    resolved = Path(path) if path is not None else settings_path()
    if not resolved.exists():
        return LauncherSettings()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise LauncherSettingsError(f"invalid launcher settings JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise LauncherSettingsError(f"launcher settings must be a JSON object: {resolved}")
    return LauncherSettings.from_dict(payload)


def save_settings(settings: LauncherSettings, path: str | Path | None = None) -> Path:
    validate_settings(settings)
    resolved = Path(path) if path is not None else settings_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return resolved


def validate_settings(settings: LauncherSettings) -> None:
    errors: list[str] = []
    required = (
        "saved_games_path",
        "dcs_variant",
        "monitor_mode",
        "language",
        "scenario_profile",
        "output_log_directory",
        "participant_id",
        "condition",
        "trial_id",
        "model_provider",
        "model_profile_mode",
        "text_model_name",
        "vision_model_name",
    )
    for name in required:
        value = getattr(settings, name)
        if not isinstance(value, str) or not value.strip():
            errors.append(name)

    if settings.language not in SUPPORTED_LANGUAGES:
        errors.append("language")
    if settings.model_provider not in SUPPORTED_MODEL_PROVIDERS:
        errors.append("model_provider")
    if settings.model_profile_mode not in SUPPORTED_MODEL_PROFILE_MODES:
        errors.append("model_profile_mode")
    if settings.model_profile_mode == "remote_direct" and (
        not isinstance(settings.model_base_url, str) or not settings.model_base_url.strip()
    ):
        errors.append("model_base_url")
    if settings.model_profile_mode == "remote_tunnel":
        for name in ("ssh_user", "ssh_host", "ssh_remote_host"):
            value = getattr(settings, name)
            if not isinstance(value, str) or not value.strip():
                errors.append(name)
    try:
        timeout_s = float(settings.model_timeout_s)
    except (TypeError, ValueError):
        errors.append("model_timeout_s")
        timeout_s = 0.0
    try:
        max_overlay_targets = int(settings.max_overlay_targets)
    except (TypeError, ValueError):
        errors.append("max_overlay_targets")
        max_overlay_targets = -1

    if timeout_s <= 0 or not math.isfinite(timeout_s):
        errors.append("model_timeout_s")
    if max_overlay_targets < 0:
        errors.append("max_overlay_targets")
    if settings.model_profile_mode == "remote_tunnel":
        for name in ("ssh_local_port", "ssh_remote_port"):
            try:
                port = int(getattr(settings, name))
            except (TypeError, ValueError):
                errors.append(name)
                continue
            if port <= 0 or port > 65535:
                errors.append(name)

    if errors:
        raise LauncherSettingsError("invalid launcher settings: " + ", ".join(sorted(set(errors))))


def save_profile(name: str, settings: LauncherSettings, *, directory: str | Path | None = None) -> Path:
    resolved_dir = Path(directory) if directory is not None else profiles_dir()
    return save_settings(settings, resolved_dir / f"{_profile_slug(name)}.json")


def load_profile(name: str, *, directory: str | Path | None = None) -> LauncherSettings:
    resolved_dir = Path(directory) if directory is not None else profiles_dir()
    path = resolved_dir / f"{_profile_slug(name)}.json"
    if not path.exists():
        raise LauncherSettingsError(f"profile not found: {name}")
    return load_settings(path)


def _profile_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-")
    if not slug:
        raise LauncherSettingsError("profile name is required")
    return slug


def _reject_secret_keys(payload: Mapping[str, Any]) -> None:
    for key in payload:
        normalized = str(key).lower()
        if any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS):
            raise LauncherSettingsError(f"launcher settings must not store secret field: {key}")


def _reject_json_constant(value: str) -> None:
    raise LauncherSettingsError(f"invalid non-finite JSON number: {value}")


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise LauncherSettingsError(f"{name} must be a boolean")


def _parse_profile_port(value: Any, name: str, *, default: int, required: bool) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise LauncherSettingsError(f"{name} must be numeric")
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        if required:
            raise LauncherSettingsError(f"{name} must be numeric") from exc
        return default


__all__ = [
    "APP_DIR_NAME",
    "LauncherSettings",
    "LauncherSettingsError",
    "PROFILES_DIR_NAME",
    "SETTINGS_FILENAME",
    "load_profile",
    "load_settings",
    "profiles_dir",
    "save_profile",
    "save_settings",
    "settings_dir",
    "settings_path",
    "validate_settings",
]
