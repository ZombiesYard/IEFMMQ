import pytest
import sys

from simtutor.config import (
    DEFAULT_MODEL_NAME_OLLAMA,
    DEFAULT_MODEL_NAME_OPENAI_COMPAT,
    ENV_LANG,
    ENV_MODEL_API_KEY,
    ENV_MODEL_BASE_URL,
    ENV_MODEL_NAME,
    ENV_MODEL_PROVIDER,
    ENV_MODEL_TIMEOUT_S,
    ModelConfigError,
    load_model_access_config,
)
from simtutor.__main__ import main


def _base_env() -> dict[str, str]:
    return {
        ENV_MODEL_PROVIDER: "ollama",
        ENV_MODEL_NAME: "qwen3:8b",
        ENV_MODEL_TIMEOUT_S: "30",
        ENV_LANG: "zh",
    }


def test_missing_provider_env_reports_name() -> None:
    env = _base_env()
    del env[ENV_MODEL_PROVIDER]

    with pytest.raises(ModelConfigError, match=ENV_MODEL_PROVIDER):
        load_model_access_config(env)


def test_openai_compat_requires_api_key_env_name() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "openai_compat"
    env[ENV_MODEL_BASE_URL] = "http://127.0.0.1:8000/v1"

    with pytest.raises(ModelConfigError, match=ENV_MODEL_API_KEY):
        load_model_access_config(env)


def test_openai_compat_requires_base_url_env_name() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "openai_compat"
    env[ENV_MODEL_API_KEY] = "sk-local-secret"

    with pytest.raises(ModelConfigError, match=ENV_MODEL_BASE_URL):
        load_model_access_config(env)


def test_invalid_provider_reports_allowed_values() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "foo"

    with pytest.raises(ModelConfigError, match=ENV_MODEL_PROVIDER):
        load_model_access_config(env)


def test_ollama_defaults_base_url_when_missing() -> None:
    cfg = load_model_access_config(_base_env())
    assert cfg.provider == "ollama"
    assert cfg.base_url == "http://127.0.0.1:11434"


def test_stub_provider_loads_without_key_and_base_url() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "stub"
    cfg = load_model_access_config(env)
    assert cfg.provider == "stub"
    assert cfg.api_key is None
    assert cfg.base_url is None


def test_stub_provider_ignores_leftover_base_url_env() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "stub"
    env[ENV_MODEL_BASE_URL] = "https:///broken"

    cfg = load_model_access_config(env)

    assert cfg.provider == "stub"
    assert cfg.base_url is None


def test_startup_info_is_non_sensitive() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "openai_compat"
    env[ENV_MODEL_BASE_URL] = "http://127.0.0.1:8000/v1"
    env[ENV_MODEL_API_KEY] = "sk-local-secret"
    env[ENV_LANG] = "en"
    env[ENV_MODEL_TIMEOUT_S] = "45"
    cfg = load_model_access_config(env)

    info = cfg.public_startup_info()
    assert "provider=openai_compat" in info
    assert "model=qwen3:8b" in info
    assert "timeout_s=45" in info
    assert "lang=en" in info
    assert "base_url=http://127.0.0.1:8000/v1" in info
    assert "sk-local-secret" not in info
    assert "api_key" not in info


def test_startup_info_redacts_base_url_credentials_query_and_fragment() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "openai_compat"
    env[ENV_MODEL_BASE_URL] = "https://user:pass@example.com:8443/v1?token=abc#frag"
    env[ENV_MODEL_API_KEY] = "sk-local-secret"

    cfg = load_model_access_config(env)
    info = cfg.public_startup_info()

    assert "user:pass@" not in info
    assert "token=abc" not in info
    assert "#frag" not in info
    assert "base_url=https://example.com:8443/v1" in info


def test_openai_compat_rejects_insecure_remote_http_base_url() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "openai_compat"
    env[ENV_MODEL_BASE_URL] = "http://api.example.com:8000/v1"
    env[ENV_MODEL_API_KEY] = "sk-local-secret"

    with pytest.raises(ModelConfigError, match="must use https"):
        load_model_access_config(env)


def test_openai_compat_allows_https_remote_base_url() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "openai_compat"
    env[ENV_MODEL_BASE_URL] = "https://api.example.com/v1"
    env[ENV_MODEL_API_KEY] = "sk-local-secret"

    cfg = load_model_access_config(env)
    assert cfg.base_url == "https://api.example.com/v1"


def test_openai_compat_rejects_base_url_without_hostname() -> None:
    env = _base_env()
    env[ENV_MODEL_PROVIDER] = "openai_compat"
    env[ENV_MODEL_BASE_URL] = "https:///v1"
    env[ENV_MODEL_API_KEY] = "sk-local-secret"

    with pytest.raises(ModelConfigError, match="must include a hostname"):
        load_model_access_config(env)


def test_cli_model_config_reports_missing_env(monkeypatch, capsys) -> None:
    for key in (
        ENV_MODEL_PROVIDER,
        ENV_MODEL_NAME,
        ENV_MODEL_BASE_URL,
        ENV_MODEL_TIMEOUT_S,
        ENV_LANG,
        ENV_MODEL_API_KEY,
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(sys, "argv", ["simtutor", "model-config"])
    code = main()
    out = capsys.readouterr().out

    assert code == 1
    assert ENV_MODEL_PROVIDER in out


def test_cli_model_config_prints_non_sensitive_info(monkeypatch, capsys) -> None:
    monkeypatch.setenv(ENV_MODEL_PROVIDER, "openai_compat")
    monkeypatch.setenv(ENV_MODEL_NAME, "Qwen3-8B-Instruct")
    monkeypatch.setenv(ENV_MODEL_BASE_URL, "http://127.0.0.1:8000/v1")
    monkeypatch.setenv(ENV_MODEL_TIMEOUT_S, "20")
    monkeypatch.setenv(ENV_LANG, "zh")
    monkeypatch.setenv(ENV_MODEL_API_KEY, "sk-super-secret")

    monkeypatch.setattr(sys, "argv", ["simtutor", "model-config"])
    code = main()
    out = capsys.readouterr().out

    assert code == 0
    assert "provider=openai_compat" in out
    assert "model=Qwen3-8B-Instruct" in out
    assert "timeout_s=20" in out
    assert "lang=zh" in out
    assert "sk-super-secret" not in out


def test_cli_model_config_redacts_sensitive_base_url_parts(monkeypatch, capsys) -> None:
    monkeypatch.setenv(ENV_MODEL_PROVIDER, "openai_compat")
    monkeypatch.setenv(ENV_MODEL_NAME, "Qwen3-8B-Instruct")
    monkeypatch.setenv(
        ENV_MODEL_BASE_URL, "https://alice:secret@api.example.local:8443/v1?api_key=abc#x"
    )
    monkeypatch.setenv(ENV_MODEL_TIMEOUT_S, "20")
    monkeypatch.setenv(ENV_LANG, "zh")
    monkeypatch.setenv(ENV_MODEL_API_KEY, "sk-super-secret")

    monkeypatch.setattr(sys, "argv", ["simtutor", "model-config"])
    code = main()
    out = capsys.readouterr().out

    assert code == 0
    assert "alice:secret@" not in out
    assert "api_key=abc" not in out
    assert "#x" not in out
    assert "base_url=https://api.example.local:8443/v1" in out


def test_model_name_defaults_by_provider_when_env_missing() -> None:
    env_openai = {
        ENV_MODEL_PROVIDER: "openai_compat",
        ENV_MODEL_BASE_URL: "http://127.0.0.1:8000/v1",
        ENV_MODEL_API_KEY: "sk-local-secret",
        ENV_MODEL_TIMEOUT_S: "30",
        ENV_LANG: "zh",
    }
    cfg_openai = load_model_access_config(env_openai)
    assert cfg_openai.model_name == DEFAULT_MODEL_NAME_OPENAI_COMPAT

    env_ollama = {
        ENV_MODEL_PROVIDER: "ollama",
        ENV_MODEL_TIMEOUT_S: "30",
        ENV_LANG: "zh",
    }
    cfg_ollama = load_model_access_config(env_ollama)
    assert cfg_ollama.model_name == DEFAULT_MODEL_NAME_OLLAMA
