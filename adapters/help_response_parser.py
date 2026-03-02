"""
Shared parsing helpers for model-generated HelpResponse payloads.

Pipeline:
raw model text -> code fence strip -> JSON object extract -> json.loads -> schema validate.
"""

from __future__ import annotations

from typing import Any, Mapping

from adapters.json_extract import JsonExtractionResult, extract_first_json, parse_first_json
from core.llm_schema import validate_help_response

_EVIDENCE_TYPES = {"var", "gate", "rag", "delta"}
_EVIDENCE_PREFIX_TO_TYPE: tuple[tuple[str, str], ...] = (
    ("VARS.", "var"),
    ("GATES.", "gate"),
    ("RAG_SNIPPETS.", "rag"),
    ("RECENT_UI_TARGETS.", "delta"),
    ("DELTA_KEYS.", "delta"),
)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return stripped


def json_extract(raw_text: str) -> str:
    return extract_first_json(raw_text).json_text


def parse_help_response(raw_text: str) -> dict[str, Any]:
    obj, _, _ = parse_help_response_with_diagnostics(raw_text)
    return obj


def parse_help_response_with_meta(raw_text: str) -> tuple[dict[str, Any], JsonExtractionResult]:
    obj, extraction, _ = parse_help_response_with_diagnostics(raw_text)
    return obj, extraction


def parse_help_response_with_diagnostics(
    raw_text: str,
) -> tuple[dict[str, Any], JsonExtractionResult, dict[str, Any]]:
    obj, extraction = parse_first_json(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("HelpResponse must be a JSON object")

    repair_meta = _repair_help_response_overlay_evidence(obj)
    validate_help_response(obj)
    return obj, extraction, repair_meta


def _infer_evidence_type_from_ref(ref: str) -> str | None:
    for prefix, evidence_type in _EVIDENCE_PREFIX_TO_TYPE:
        if ref.startswith(prefix):
            return evidence_type
    return None


def _repair_help_response_overlay_evidence(help_obj: dict[str, Any]) -> dict[str, Any]:
    overlay = help_obj.get("overlay")
    if not isinstance(overlay, Mapping):
        return {
            "repair_applied": False,
            "repaired_evidence_types": 0,
            "dropped_unrepairable_evidence": 0,
            "details": [],
        }

    evidence_raw = overlay.get("evidence")
    if not isinstance(evidence_raw, list):
        return {
            "repair_applied": False,
            "repaired_evidence_types": 0,
            "dropped_unrepairable_evidence": 0,
            "details": [],
        }

    repaired_count = 0
    dropped_count = 0
    details: list[dict[str, Any]] = []
    rewritten_evidence: list[Any] = []

    for idx, item in enumerate(evidence_raw):
        if not isinstance(item, Mapping):
            rewritten_evidence.append(item)
            continue

        evidence_type = item.get("type")
        if isinstance(evidence_type, str) and evidence_type in _EVIDENCE_TYPES:
            rewritten_evidence.append(item)
            continue

        ref = item.get("ref")
        if not isinstance(ref, str) or not ref:
            dropped_count += 1
            details.append({"index": idx, "action": "dropped", "reason": "missing_or_invalid_ref"})
            continue

        mapped_type = _infer_evidence_type_from_ref(ref)
        if mapped_type is None:
            dropped_count += 1
            details.append({"index": idx, "action": "dropped", "reason": f"unmapped_ref:{ref}"})
            continue

        repaired_item = dict(item)
        repaired_item["type"] = mapped_type
        rewritten_evidence.append(repaired_item)
        repaired_count += 1
        details.append(
            {
                "index": idx,
                "action": "retyped",
                "from_type": evidence_type,
                "to_type": mapped_type,
                "ref": ref,
            }
        )

    repair_applied = repaired_count > 0 or dropped_count > 0
    if repair_applied:
        overlay_mut = dict(overlay)
        overlay_mut["evidence"] = rewritten_evidence
        help_obj["overlay"] = overlay_mut

    return {
        "repair_applied": repair_applied,
        "repaired_evidence_types": repaired_count,
        "dropped_unrepairable_evidence": dropped_count,
        "details": details,
    }


__all__ = [
    "json_extract",
    "parse_help_response",
    "parse_help_response_with_diagnostics",
    "parse_help_response_with_meta",
    "strip_code_fence",
]
