"""Remote model endpoint validation for the SimTutor launcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from simtutor.launcher_settings import LauncherSettings


@dataclass(frozen=True)
class ModelCheckEntry:
    status: str
    code: str
    message: str


@dataclass(frozen=True)
class ModelCheckReport:
    entries: tuple[ModelCheckEntry, ...]

    @property
    def ok(self) -> bool:
        return all(entry.status != "error" for entry in self.entries)

    def to_text(self) -> str:
        return "\n".join(f"{entry.status.upper()} {entry.code}: {entry.message}" for entry in self.entries)


@dataclass(frozen=True)
class ModelEndpointConfig:
    base_url: str
    text_model_name: str
    vision_model_name: str
    require_vision_model: bool
    timeout_s: float


def validate_launcher_model_profile(settings: LauncherSettings, *, client: Any | None = None) -> ModelCheckReport:
    if settings.model_profile_mode == "local_stub":
        return ModelCheckReport((ModelCheckEntry("pass", "model_profile_local_stub", "local stub profile selected"),))
    report = check_model_endpoint(endpoint_config_from_settings(settings), client=client)
    if settings.model_profile_mode == "remote_tunnel" and _has_endpoint_error(report):
        return ModelCheckReport(
            (
                *report.entries,
                ModelCheckEntry(
                    "warn",
                    "ssh_key_auth_hint",
                    "SSH key authentication must be configured; launcher password prompts are disabled.",
                ),
            )
        )
    return report


def endpoint_config_from_settings(settings: LauncherSettings) -> ModelEndpointConfig:
    base_url = settings.model_base_url
    if settings.model_profile_mode == "remote_tunnel" and not str(base_url).strip():
        base_url = f"http://127.0.0.1:{int(settings.ssh_local_port)}"
    return ModelEndpointConfig(
        base_url=_normalize_base_url(str(base_url)),
        text_model_name=settings.text_model_name,
        vision_model_name=settings.vision_model_name,
        require_vision_model=bool(settings.model_enable_multimodal),
        timeout_s=float(settings.model_timeout_s),
    )


def check_model_endpoint(config: ModelEndpointConfig, *, client: Any | None = None) -> ModelCheckReport:
    owns_client = client is None
    try:
        http_client = client or _make_http_client()
    except RuntimeError as exc:
        return ModelCheckReport((ModelCheckEntry("error", "http_client_missing", str(exc)),))
    entries: list[ModelCheckEntry] = []
    base_url = _normalize_base_url(config.base_url)

    try:
        try:
            response = http_client.get(f"{base_url}/health", timeout=config.timeout_s)
            response.raise_for_status()
        except Exception as exc:
            entries.append(ModelCheckEntry("error", "endpoint_health", f"GET /health failed: {exc}"))
            return ModelCheckReport(tuple(entries))
        entries.append(ModelCheckEntry("pass", "endpoint_health", "GET /health succeeded"))

        try:
            response = http_client.get(f"{base_url}/v1/models", timeout=config.timeout_s)
            response.raise_for_status()
            model_ids = _extract_model_ids(response.json())
        except Exception as exc:
            entries.append(ModelCheckEntry("error", "endpoint_models", f"GET /v1/models failed: {exc}"))
            return ModelCheckReport(tuple(entries))
        entries.append(ModelCheckEntry("pass", "endpoint_models", f"GET /v1/models returned {len(model_ids)} models"))

        text_name = config.text_model_name.strip()
        if text_name in model_ids:
            entries.append(ModelCheckEntry("pass", "text_model_available", f"text model available: {text_name}"))
        else:
            entries.append(ModelCheckEntry("error", "text_model_available", f"missing text model: {text_name}"))

        vision_name = config.vision_model_name.strip()
        if config.require_vision_model:
            if vision_name in model_ids:
                entries.append(ModelCheckEntry("pass", "vision_model_available", f"vision model available: {vision_name}"))
            else:
                entries.append(ModelCheckEntry("error", "vision_model_available", f"missing vision model: {vision_name}"))
        return ModelCheckReport(tuple(entries))
    finally:
        if owns_client and hasattr(http_client, "close"):
            http_client.close()


def _normalize_base_url(raw: str) -> str:
    value = raw.strip().rstrip("/")
    if value.lower().endswith("/v1"):
        value = value[:-3].rstrip("/")
    if not value:
        raise ValueError("model base URL is required")
    return value


def _extract_model_ids(body: Any) -> set[str]:
    if not isinstance(body, Mapping):
        raise ValueError("models response must be a JSON object")
    data = body.get("data")
    if not isinstance(data, list):
        raise ValueError("models response must include a data list")
    ids: set[str] = set()
    for item in data:
        if isinstance(item, Mapping) and isinstance(item.get("id"), str):
            ids.add(item["id"])
    return ids


def _has_endpoint_error(report: ModelCheckReport) -> bool:
    return any(
        entry.status == "error" and entry.code in {"endpoint_health", "endpoint_models"}
        for entry in report.entries
    )


def _make_http_client() -> Any:
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency: httpx. Install dependencies in the environment that starts the GUI "
            "with `python -m pip install httpx` or run `poetry install`."
        ) from exc
    return httpx.Client()


__all__ = [
    "ModelCheckEntry",
    "ModelCheckReport",
    "ModelEndpointConfig",
    "check_model_endpoint",
    "endpoint_config_from_settings",
    "validate_launcher_model_profile",
]
