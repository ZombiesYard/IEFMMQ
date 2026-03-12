from __future__ import annotations

import pytest

from core.security import (
    ModelTransportSecurityError,
    redact_sensitive_text,
    redact_url_for_log,
    validate_model_base_url_security,
)


def test_validate_model_base_url_security_rejects_http_dot_local_hostname() -> None:
    with pytest.raises(ModelTransportSecurityError, match="must use https"):
        validate_model_base_url_security("http://simtutor-box.local:8000/v1", provider="openai_compat")


def test_validate_model_base_url_security_allows_http_private_ip_for_split_model_topology() -> None:
    validate_model_base_url_security("http://10.0.0.42:8000/v1", provider="openai_compat")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("open /tmp before proceeding", "open [REDACTED_PATH] before proceeding"),
        ("read /etc for config", "read [REDACTED_PATH] for config"),
        ("connect to api.example.com immediately", "connect to [REDACTED_ENDPOINT] immediately"),
        ("fetch from api.example.com/v1/chat/completions", "fetch from [REDACTED_ENDPOINT]"),
        ("fetch from api.example.com:8443/v1/chat/completions", "fetch from [REDACTED_ENDPOINT]"),
    ],
)
def test_redact_sensitive_text_covers_single_segment_paths_and_bare_hosts(
    raw: str,
    expected: str,
) -> None:
    redacted = redact_sensitive_text(raw)
    assert redacted == expected


def test_redact_sensitive_text_keeps_non_path_slash_phrases() -> None:
    redacted = redact_sensitive_text("需要更多信息/请确认 apu_switch")
    assert redacted == "需要更多信息/请确认 apu_switch"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Visit https://api.example.com/v1/chat).", "Visit [REDACTED_URL])."),
        ("Use api.example.com/v1/chat).", "Use [REDACTED_ENDPOINT])."),
        ("Set api_key=sk-test-secret.", "Set api_key=[REDACTED_SECRET]."),
        ("Set token=abc123)", "Set token=[REDACTED_SECRET])"),
    ],
)
def test_redact_sensitive_text_preserves_trailing_punctuation(raw: str, expected: str) -> None:
    assert redact_sensitive_text(raw) == expected


def test_redact_url_for_log_strips_query_and_fragment_for_schemeless_url_like_values() -> None:
    redacted = redact_url_for_log("api.example.com/v1/chat?token=secret#frag")

    assert redacted == "//api.example.com/v1/chat"
