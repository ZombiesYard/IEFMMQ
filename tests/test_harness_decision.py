from __future__ import annotations

from core.harness_decision import (
    HARNESS_DECISION_REQUIRED_FIELDS,
    build_harness_decision_contract,
    get_harness_decision_schema,
)


def test_harness_decision_schema_exposes_adjudication_fields() -> None:
    schema = get_harness_decision_schema(
        step_ids=["S01", "S08"],
        overlay_targets=["battery_switch", "left_mdi_pb18"],
        error_categories=["OM", "CO"],
        allowed_evidence_refs=["VISION_FACTS.tac_page_visible@frame-1"],
        max_overlay_targets=1,
    )

    assert schema["title"] == "HarnessDecision"
    assert schema["required"] == list(HARNESS_DECISION_REQUIRED_FIELDS)
    assert schema["properties"]["chosen_step_id"]["enum"] == ["S01", "S08"]
    assert schema["properties"]["diagnosis_category"]["enum"] == ["OM", "CO"]
    assert schema["properties"]["proposed_overlay_targets"]["items"]["enum"] == [
        "battery_switch",
        "left_mdi_pb18",
    ]
    assert schema["properties"]["proposed_overlay_targets"]["maxItems"] == 1
    assert schema["properties"]["evidence_refs"]["items"]["enum"] == [
        "VISION_FACTS.tac_page_visible@frame-1"
    ]
    assert schema["properties"]["validator_repair_expected"]["type"] == "boolean"


def test_harness_decision_schema_disables_overlay_targets_when_requested() -> None:
    schema = get_harness_decision_schema(
        step_ids=["S08"],
        overlay_targets=["left_mdi_pb18"],
        error_categories=["CO"],
        allowed_evidence_refs=[],
        max_overlay_targets=0,
    )

    assert schema["properties"]["proposed_overlay_targets"]["maxItems"] == 0
    assert schema["properties"]["evidence_refs"]["maxItems"] == 0


def test_harness_decision_contract_keeps_help_response_mapping_stable() -> None:
    contract = build_harness_decision_contract(
        step_ids=["S01", "S08"],
        overlay_targets=["battery_switch", "left_mdi_pb18"],
        error_categories=["OM", "CO"],
    )

    assert contract["decision_schema"]["required"] == list(HARNESS_DECISION_REQUIRED_FIELDS)
    assert contract["maps_to_public_help_response"] == {
        "diagnosis.step_id": "chosen_step_id",
        "diagnosis.error_category": "diagnosis_category",
        "next.step_id": "chosen_step_id unless next_action_intent selects a later allowed step",
        "overlay.targets": "proposed_overlay_targets",
        "overlay.evidence[].ref": "evidence_refs",
        "explanations": "conflict_resolution plus next_action_intent",
    }
    assert contract["public_output_schema"] == "TutorResponse via existing HelpResponse mapping"
