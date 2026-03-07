from __future__ import annotations

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
