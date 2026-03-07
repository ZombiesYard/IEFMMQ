"""
Help-cycle failure classification helpers.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


JSON_EXTRACT_FAIL = "json_extract_fail"
SCHEMA_FAIL = "schema_fail"
EVIDENCE_FAIL = "evidence_fail"
ALLOWLIST_FAIL = "allowlist_fail"
MODEL_HTTP_FAIL = "model_http_fail"

HELP_FAILURE_CODES = (
    JSON_EXTRACT_FAIL,
    SCHEMA_FAIL,
    EVIDENCE_FAIL,
    ALLOWLIST_FAIL,
    MODEL_HTTP_FAIL,
)

_EVIDENCE_REASON_PREFIXES = (
    "missing_target_evidence",
    "invalid_target_evidence_refs",
    "missing_overlay_evidence",
    "no_verifiable_evidence_refs",
    "invalid_overlay_evidence_item",
    "evidence_type_ref_mismatch",
    "unknown_evidence_ref",
    "evidence_target_not_in_overlay_targets",
)

_ALLOWLIST_REASON_PREFIXES = (
    "overlay_target_not_in_request_allowlist",
    "overlay_target_not_in_allowlist",
    "target_not_in_request_allowlist",
    "target_not_in_runtime_allowlist",
)


def annotate_exception(exc: Exception, *, code: str, stage: str) -> Exception:
    setattr(exc, "help_failure_code", code)
    setattr(exc, "help_failure_stage", stage)
    return exc


def exception_failure_code(exc: BaseException) -> str | None:
    code = getattr(exc, "help_failure_code", None)
    if isinstance(code, str) and code in HELP_FAILURE_CODES:
        return code
    return None


def exception_failure_stage(exc: BaseException) -> str | None:
    stage = getattr(exc, "help_failure_stage", None)
    if isinstance(stage, str) and stage:
        return stage
    return None


def classify_mapping_failure(mapping_meta: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(mapping_meta, Mapping):
        return []

    reasons: list[str] = []
    overlay_rejected_reasons = mapping_meta.get("overlay_rejected_reasons")
    if isinstance(overlay_rejected_reasons, list):
        reasons.extend(item for item in overlay_rejected_reasons if isinstance(item, str) and item)

    mapping_errors = mapping_meta.get("mapping_errors")
    if isinstance(mapping_errors, list):
        reasons.extend(item for item in mapping_errors if isinstance(item, str) and item)

    if mapping_meta.get("rejected_targets_by_request_allowlist"):
        reasons.append("overlay_target_not_in_request_allowlist")

    codes: list[str] = []
    for reason in reasons:
        if _matches_any_prefix(reason, _ALLOWLIST_REASON_PREFIXES):
            _append_unique(codes, ALLOWLIST_FAIL)
        elif _matches_any_prefix(reason, _EVIDENCE_REASON_PREFIXES):
            _append_unique(codes, EVIDENCE_FAIL)

    if mapping_meta.get("overlay_rejected") is True and EVIDENCE_FAIL not in codes:
        codes.append(EVIDENCE_FAIL)
    return codes


def merge_failure_codes(existing: Sequence[Any] | None, *new_codes: str | None) -> list[str]:
    merged: list[str] = []
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        for item in existing:
            if isinstance(item, str) and item in HELP_FAILURE_CODES:
                _append_unique(merged, item)
    for code in new_codes:
        if isinstance(code, str) and code in HELP_FAILURE_CODES:
            _append_unique(merged, code)
    return merged


def merge_failure_metadata(
    metadata: Mapping[str, Any] | None,
    *new_codes: str | None,
    stage: str | None = None,
) -> dict[str, Any]:
    merged = dict(metadata) if isinstance(metadata, Mapping) else {}
    failure_codes = merge_failure_codes(merged.get("failure_codes"), *new_codes)
    if failure_codes:
        merged["failure_codes"] = failure_codes
        merged["failure_code"] = failure_codes[0]
    if isinstance(stage, str) and stage and "failure_stage" not in merged:
        merged["failure_stage"] = stage
    return merged


def overlay_rejection_payload(
    *,
    response_metadata: Mapping[str, Any] | None,
    response_mapping: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(response_mapping, Mapping):
        return None

    reasons = response_mapping.get("overlay_rejected_reasons")
    mapping_errors = response_mapping.get("mapping_errors")
    rejected_targets = response_mapping.get("rejected_targets")
    rejected_by_request_allowlist = response_mapping.get("rejected_targets_by_request_allowlist")

    has_overlay_rejection = bool(response_mapping.get("overlay_rejected"))
    has_allowlist_rejection = isinstance(rejected_by_request_allowlist, list) and bool(rejected_by_request_allowlist)
    if not has_overlay_rejection and not has_allowlist_rejection:
        return None

    payload: dict[str, Any] = {
        "failure_codes": classify_mapping_failure(response_mapping),
        "overlay_rejected": has_overlay_rejection,
        "reasons": list(reasons) if isinstance(reasons, list) else [],
        "mapping_errors": list(mapping_errors) if isinstance(mapping_errors, list) else [],
        "rejected_targets": list(rejected_targets) if isinstance(rejected_targets, list) else [],
        "rejected_targets_by_request_allowlist": (
            list(rejected_by_request_allowlist) if isinstance(rejected_by_request_allowlist, list) else []
        ),
    }
    if isinstance(response_metadata, Mapping):
        for key in ("failure_code", "failure_codes", "provider", "model"):
            value = response_metadata.get(key)
            if value is not None:
                payload[key] = value
    if not payload.get("failure_code") and payload["failure_codes"]:
        payload["failure_code"] = payload["failure_codes"][0]
    return payload


def _matches_any_prefix(reason: str, prefixes: Sequence[str]) -> bool:
    return any(reason == prefix or reason.startswith(prefix + ":") for prefix in prefixes)


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


__all__ = [
    "ALLOWLIST_FAIL",
    "EVIDENCE_FAIL",
    "HELP_FAILURE_CODES",
    "JSON_EXTRACT_FAIL",
    "MODEL_HTTP_FAIL",
    "SCHEMA_FAIL",
    "annotate_exception",
    "classify_mapping_failure",
    "exception_failure_code",
    "exception_failure_stage",
    "merge_failure_codes",
    "merge_failure_metadata",
    "overlay_rejection_payload",
]
