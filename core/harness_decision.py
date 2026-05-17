"""
Internal decision contract for text-only help-cycle adjudication.

The model still returns the existing HelpResponse shape. This contract is the
intermediate reasoning schema the prompt asks the model to apply before mapping
the decision back to the stable public response.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


HARNESS_DECISION_REQUIRED_FIELDS: tuple[str, ...] = (
    "chosen_step_id",
    "diagnosis_category",
    "rejected_candidates",
    "conflict_resolution",
    "next_action_intent",
    "proposed_overlay_targets",
    "evidence_refs",
    "uncertainty_status",
    "validator_repair_expected",
)


def _string_enum(values: Sequence[str], label: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    if not out:
        raise ValueError(f"{label} must contain at least one string")
    return out


def _optional_string_enum(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def get_harness_decision_schema(
    *,
    step_ids: Sequence[str],
    overlay_targets: Sequence[str],
    error_categories: Sequence[str],
    allowed_evidence_refs: Sequence[str] | None = None,
    max_overlay_targets: int | None = None,
) -> dict[str, Any]:
    step_enum = _string_enum(step_ids, "step_ids")
    target_enum = _string_enum(overlay_targets, "overlay_targets")
    category_enum = _string_enum(error_categories, "error_categories")
    evidence_ref_enum = _optional_string_enum(allowed_evidence_refs)
    effective_max_targets = None
    if isinstance(max_overlay_targets, int) and not isinstance(max_overlay_targets, bool):
        effective_max_targets = max(0, max_overlay_targets)
    evidence_ref_items: dict[str, Any] = {"type": "string", "minLength": 1}
    if evidence_ref_enum:
        evidence_ref_items["enum"] = evidence_ref_enum
    evidence_refs_schema: dict[str, Any] = {
        "type": "array",
        "items": evidence_ref_items,
    }
    if allowed_evidence_refs is not None and not evidence_ref_enum:
        evidence_refs_schema["maxItems"] = 0
    proposed_overlay_targets_schema: dict[str, Any] = {
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "string", "enum": target_enum},
    }
    if effective_max_targets is not None:
        proposed_overlay_targets_schema["maxItems"] = effective_max_targets
    return {
        "title": "HarnessDecision",
        "type": "object",
        "additionalProperties": False,
        "required": list(HARNESS_DECISION_REQUIRED_FIELDS),
        "properties": {
            "chosen_step_id": {"type": "string", "enum": step_enum},
            "diagnosis_category": {"type": "string", "enum": category_enum},
            "rejected_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["step_id", "source", "reason"],
                    "properties": {
                        "step_id": {"type": "string", "enum": step_enum},
                        "source": {"type": "string", "minLength": 1},
                        "reason": {"type": "string", "minLength": 1},
                        "evidence_refs": evidence_refs_schema,
                    },
                },
            },
            "conflict_resolution": {"type": "string", "minLength": 1},
            "next_action_intent": {"type": "string", "minLength": 1},
            "proposed_overlay_targets": proposed_overlay_targets_schema,
            "evidence_refs": evidence_refs_schema,
            "uncertainty_status": {
                "type": "string",
                "enum": ["certain", "partial", "uncertain", "conflict"],
            },
            "validator_repair_expected": {"type": "boolean"},
        },
    }


def build_harness_decision_contract(
    *,
    step_ids: Sequence[str],
    overlay_targets: Sequence[str],
    error_categories: Sequence[str],
    allowed_evidence_refs: Sequence[str] | None = None,
    max_overlay_targets: int | None = None,
    compact: bool = False,
    minimal: bool = False,
) -> dict[str, Any]:
    decision_schema = get_harness_decision_schema(
        step_ids=step_ids,
        overlay_targets=overlay_targets,
        error_categories=error_categories,
        allowed_evidence_refs=allowed_evidence_refs,
        max_overlay_targets=max_overlay_targets,
    )
    if compact:
        if minimal:
            return {
                "decision_schema": {
                    "title": decision_schema["title"],
                    "type": decision_schema["type"],
                    "additionalProperties": decision_schema["additionalProperties"],
                    "required": list(HARNESS_DECISION_REQUIRED_FIELDS),
                },
                "task": "adjudicate candidates; map to HelpResponse",
                "public_output_schema": "TutorResponse via existing HelpResponse mapping",
            }
        properties = {
            "chosen_step_id": {"type": "string"},
            "diagnosis_category": {"type": "string"},
            "rejected_candidates": {"type": "array"},
            "conflict_resolution": {"type": "string"},
            "next_action_intent": {"type": "string"},
            "proposed_overlay_targets": decision_schema["properties"]["proposed_overlay_targets"],
            "evidence_refs": decision_schema["properties"]["evidence_refs"],
            "uncertainty_status": decision_schema["properties"]["uncertainty_status"],
            "validator_repair_expected": {"type": "boolean"},
        }
        decision_schema = {
            "title": decision_schema["title"],
            "type": decision_schema["type"],
            "additionalProperties": decision_schema["additionalProperties"],
            "required": list(HARNESS_DECISION_REQUIRED_FIELDS),
            "properties": properties,
        }
        return {
            "decision_schema": decision_schema,
            "task": "adjudicate candidates, explain conflicts, map decision to HelpResponse",
            "public_output_schema": "TutorResponse via existing HelpResponse mapping",
        }
    return {
        "decision_schema": decision_schema,
        "task": (
            "Compare the harness packet evidence and candidates; choose one "
            "step, reject weaker candidates with evidence refs, explain "
            "conflicts, choose next action intent, and map the decision to "
            "the stable HelpResponse JSON."
        ),
        "input_packet_fields": [
            "evidence_packet",
            "step_candidates",
            "step_specs",
            "gates",
            "recent_actions",
            "allowed_evidence_refs",
        ],
        "maps_to_public_help_response": {
            "diagnosis.step_id": "chosen_step_id",
            "diagnosis.error_category": "diagnosis_category",
            "next.step_id": "chosen_step_id unless next_action_intent selects a later allowed step",
            "overlay.targets": "proposed_overlay_targets",
            "overlay.evidence[].ref": "evidence_refs",
            "explanations": "conflict_resolution plus next_action_intent",
        },
        "public_output_schema": "TutorResponse via existing HelpResponse mapping",
    }


__all__ = [
    "HARNESS_DECISION_REQUIRED_FIELDS",
    "build_harness_decision_contract",
    "get_harness_decision_schema",
]
