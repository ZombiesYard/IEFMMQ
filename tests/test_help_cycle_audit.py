from __future__ import annotations

from core.help_cycle_audit import normalize_help_cycle_audit_fields


def test_normalize_help_cycle_audit_fields_rejects_none_and_non_mapping() -> None:
    assert normalize_help_cycle_audit_fields(None) == {}
    assert normalize_help_cycle_audit_fields("not-a-mapping") == {}
    assert normalize_help_cycle_audit_fields(["not", "a", "mapping"]) == {}


def test_normalize_help_cycle_audit_fields_converts_tuples_and_filters_unknown_keys() -> None:
    normalized = normalize_help_cycle_audit_fields(
        {
            "help_cycle_id": "cycle-123",
            "fused_missing_conditions": ("vision_facts.fcs_reset_seen==seen",),
            "vision_fact_summary": {"status": "available"},
            "unknown_key": "should-be-dropped",
        }
    )

    assert normalized == {
        "help_cycle_id": "cycle-123",
        "fused_missing_conditions": ["vision_facts.fcs_reset_seen==seen"],
        "vision_fact_summary": {"status": "available"},
    }


def test_normalize_help_cycle_audit_fields_copies_mutable_values() -> None:
    raw_summary = {"status": "available", "seen_fact_ids": ["fcs_reset_seen"]}
    raw_missing = ["vision_facts.fcs_reset_seen==seen"]

    normalized = normalize_help_cycle_audit_fields(
        {
            "vision_fact_summary": raw_summary,
            "fused_missing_conditions": raw_missing,
        }
    )

    raw_summary["status"] = "mutated"
    raw_missing.append("vars.apu_ready==true")

    assert normalized["vision_fact_summary"] == {
        "status": "available",
        "seen_fact_ids": ["fcs_reset_seen"],
    }
    assert normalized["fused_missing_conditions"] == ["vision_facts.fcs_reset_seen==seen"]
