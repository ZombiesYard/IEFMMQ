from __future__ import annotations

from typing import Any

from simtutor.launcher_model_check import (
    ModelEndpointConfig,
    check_model_endpoint,
    endpoint_config_from_settings,
    validate_launcher_model_profile,
)
from simtutor.launcher_settings import LauncherSettings


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "timeout": timeout})
        if not self.responses:
            raise RuntimeError("no fake response left")
        return self.responses.pop(0)


def test_check_model_endpoint_checks_health_and_models() -> None:
    client = FakeClient(
        [
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [{"id": "qwen36_27b"}, {"id": "simtutor-vision"}]}),
        ]
    )
    config = ModelEndpointConfig(
        base_url="http://localhost:16324",
        text_model_name="qwen36_27b",
        vision_model_name="simtutor-vision",
        require_vision_model=True,
        timeout_s=3.0,
    )

    report = check_model_endpoint(config, client=client)

    assert report.ok is True
    assert [call["url"] for call in client.calls] == [
        "http://localhost:16324/health",
        "http://localhost:16324/v1/models",
    ]


def test_check_model_endpoint_reports_missing_model_names() -> None:
    client = FakeClient(
        [
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [{"id": "qwen36_27b"}]}),
        ]
    )
    config = ModelEndpointConfig(
        base_url="http://localhost:16324",
        text_model_name="missing-text",
        vision_model_name="missing-vision",
        require_vision_model=True,
        timeout_s=3.0,
    )

    report = check_model_endpoint(config, client=client)

    assert report.ok is False
    assert "missing text model: missing-text" in report.to_text()
    assert "missing vision model: missing-vision" in report.to_text()


def test_validate_launcher_model_profile_local_stub_skips_remote_endpoint() -> None:
    settings = LauncherSettings(
        saved_games_path="C:/Users/test/Saved Games/DCS",
        output_log_directory="C:/SimTutor/logs",
        participant_id="P01",
        condition="with_tutor",
        trial_id="T01",
        model_provider="stub",
        model_profile_mode="local_stub",
        model_base_url="",
    )
    client = FakeClient([])

    report = validate_launcher_model_profile(settings, client=client)

    assert report.ok is True
    assert "local stub profile selected" in report.to_text()
    assert client.calls == []


def test_remote_tunnel_endpoint_defaults_to_local_forward_port() -> None:
    settings = LauncherSettings(
        model_provider="openai_compat",
        model_profile_mode="remote_tunnel",
        model_base_url="",
        ssh_user="pilot",
        ssh_host="cloud.example.test",
        ssh_local_port=16324,
    )

    config = endpoint_config_from_settings(settings)

    assert config.base_url == "http://127.0.0.1:16324"


def test_remote_tunnel_endpoint_failure_includes_key_auth_hint() -> None:
    settings = LauncherSettings(
        model_provider="openai_compat",
        model_profile_mode="remote_tunnel",
        model_base_url="http://localhost:16324",
        ssh_user="pilot",
        ssh_host="cloud.example.test",
    )
    client = FakeClient([FakeResponse({}, status_code=503)])

    report = validate_launcher_model_profile(settings, client=client)

    assert report.ok is False
    assert "password prompts are disabled" in report.to_text()


def test_check_model_endpoint_reports_missing_http_client(monkeypatch: Any) -> None:
    def fail_client() -> Any:
        raise RuntimeError("Missing Python dependency: httpx. Install dependencies.")

    monkeypatch.setattr("simtutor.launcher_model_check._make_http_client", fail_client)
    config = ModelEndpointConfig(
        base_url="http://localhost:16324",
        text_model_name="simtutor-base",
        vision_model_name="simtutor-vision",
        require_vision_model=True,
        timeout_s=3.0,
    )

    report = check_model_endpoint(config)

    assert report.ok is False
    assert "http_client_missing" in report.to_text()
    assert "httpx" in report.to_text()


def test_remote_tunnel_missing_model_does_not_include_key_auth_hint() -> None:
    settings = LauncherSettings(
        model_provider="openai_compat",
        model_profile_mode="remote_tunnel",
        model_base_url="http://localhost:16324",
        text_model_name="missing-text",
        ssh_user="pilot",
        ssh_host="cloud.example.test",
    )
    client = FakeClient(
        [
            FakeResponse({"status": "ok"}),
            FakeResponse({"data": [{"id": "other-model"}]}),
        ]
    )

    report = validate_launcher_model_profile(settings, client=client)

    assert report.ok is False
    assert "missing text model: missing-text" in report.to_text()
    assert "password prompts are disabled" not in report.to_text()
