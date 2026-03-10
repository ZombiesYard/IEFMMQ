from __future__ import annotations

from core.help_failure import overlay_rejection_payload
from tools.help_failure_stats import summarize_help_failures


def test_summarize_help_failures_aggregates_primary_and_all_codes() -> None:
    events = [
        {
            "kind": "tutor_response",
            "payload": {
                "status": "error",
                "metadata": {
                    "provider": "openai_compat",
                    "failure_code": "json_extract_fail",
                    "failure_codes": ["json_extract_fail"],
                },
            },
        },
        {
            "kind": "tutor_response",
            "payload": {
                "status": "ok",
                "metadata": {
                    "provider": "openai_compat",
                    "failure_code": "allowlist_fail",
                    "failure_codes": ["allowlist_fail", "evidence_fail"],
                },
            },
        },
        {
            "kind": "overlay_rejected",
            "payload": {"failure_code": "evidence_fail"},
        },
    ]

    summary = summarize_help_failures(events)

    assert summary["total_tutor_responses"] == 2
    assert summary["primary_failure_code_counts"] == {
        "allowlist_fail": 1,
        "json_extract_fail": 1,
    }
    assert summary["all_failure_code_counts"] == {
        "allowlist_fail": 1,
        "evidence_fail": 1,
        "json_extract_fail": 1,
    }
    assert summary["status_counts"] == {"error": 1, "ok": 1}
    assert summary["provider_counts"] == {"openai_compat": 2}


def test_summarize_help_failures_falls_back_to_primary_failure_code_when_failure_codes_missing() -> None:
    events = [
        {
            "kind": "tutor_response",
            "payload": {
                "status": "error",
                "metadata": {
                    "provider": "openai_compat",
                    "failure_code": "schema_fail",
                },
            },
        },
        {
            "kind": "tutor_response",
            "payload": {
                "status": "error",
                "metadata": {
                    "provider": "ollama",
                    "failure_code": "model_http_fail",
                    "failure_codes": [None, ""],
                },
            },
        },
    ]

    summary = summarize_help_failures(events)

    assert summary["primary_failure_code_counts"] == {
        "model_http_fail": 1,
        "schema_fail": 1,
    }
    assert summary["all_failure_code_counts"] == {
        "model_http_fail": 1,
        "schema_fail": 1,
    }


def test_overlay_rejection_payload_keeps_overlay_failure_codes_separate_from_response_failure_codes() -> None:
    payload = overlay_rejection_payload(
        response_metadata={
            "provider": "openai_compat",
            "model": "Qwen3-8B",
            "failure_code": "json_extract_fail",
            "failure_codes": ["json_extract_fail"],
        },
        response_mapping={
            "overlay_rejected": True,
            "overlay_rejected_reasons": ["missing_overlay_evidence"],
            "mapping_errors": ["missing_overlay_evidence"],
        },
    )

    assert payload is not None
    assert payload["failure_code"] == "evidence_fail"
    assert payload["failure_codes"] == ["evidence_fail"]
    assert payload["response_failure_code"] == "json_extract_fail"
    assert payload["response_failure_codes"] == ["json_extract_fail"]


def test_summarize_help_failures_accepts_legacy_jsonl_without_vision_audit_fields() -> None:
    summary = summarize_help_failures(
        [
            {
                "kind": "tutor_response",
                "payload": {
                    "status": "ok",
                    "metadata": {
                        "provider": "legacy_stub",
                    },
                },
            }
        ]
    )

    assert summary["total_tutor_responses"] == 1
    assert summary["primary_failure_code_counts"] == {}
    assert summary["all_failure_code_counts"] == {}
    assert summary["provider_counts"] == {"legacy_stub": 1}
