from __future__ import annotations

import pytest

from core.security import ModelTransportSecurityError, redact_sensitive_text, validate_model_base_url_security


def test_validate_model_base_url_security_rejects_http_dot_local_hostname() -> None:
    with pytest.raises(ModelTransportSecurityError, match="must use https"):
        validate_model_base_url_security("http://simtutor-box.local:8000/v1", provider="openai_compat")


def test_validate_model_base_url_security_allows_http_private_ip_for_split_model_topology() -> None:
    validate_model_base_url_security("http://10.0.0.42:8000/v1", provider="openai_compat")


@pytest.mark.parametrize(
    ("raw", "expected_marker"),
    [
        ("open /tmp before proceeding", "[REDACTED_PATH]"),
        ("read /etc for config", "[REDACTED_PATH]"),
        ("connect to api.example.com immediately", "[REDACTED_ENDPOINT]"),
        ("fetch from api.example.com/v1/chat/completions", "[REDACTED_ENDPOINT]"),
    ],
)
def test_redact_sensitive_text_covers_single_segment_paths_and_bare_hosts(
    raw: str,
    expected_marker: str,
) -> None:
    redacted = redact_sensitive_text(raw)
    assert expected_marker in redacted


def test_redact_sensitive_text_keeps_non_path_slash_phrases() -> None:
    redacted = redact_sensitive_text("需要更多信息/请确认 apu_switch")
    assert redacted == "需要更多信息/请确认 apu_switch"

