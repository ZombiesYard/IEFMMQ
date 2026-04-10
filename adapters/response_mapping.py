"""
Map LLM HelpResponse payloads into v1 TutorResponse with safe overlay actions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from adapters.evidence_refs import EVIDENCE_TYPE_PREFIXES, collect_evidence_refs_from_context
from core.overlay import OverlayPlanner
from core.step_signal_metadata import compute_requires_visual_confirmation, normalize_observability_status
from core.types import TutorRequest, TutorResponse

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_ui_map_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "ui_map.yaml"


def _resolve_ui_map_path(ui_map_path: str | Path | None) -> Path:
    path = Path(ui_map_path) if ui_map_path else _default_ui_map_path()
    return path.resolve()


@lru_cache(maxsize=8)
def _get_overlay_planner(ui_map_path: str) -> OverlayPlanner:
    return OverlayPlanner(ui_map_path)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _append_mapping_error(metadata: dict[str, Any], reason: str) -> None:
    errors = metadata.get("mapping_errors")
    if not isinstance(errors, list):
        errors = []
        metadata["mapping_errors"] = errors
    errors.append(reason)
    if "mapping_error" not in metadata:
        metadata["mapping_error"] = reason


def _collect_allowed_evidence_refs(request: TutorRequest | None) -> set[str]:
    if request is None or not isinstance(request.context, Mapping):
        return set()
    return collect_evidence_refs_from_context(request.context)


def _extract_observability_metadata(
    request: TutorRequest | None,
) -> tuple[str | None, bool, list[str]]:
    if request is None or not isinstance(request.context, Mapping):
        return None, False, []
    hint = request.context.get("deterministic_step_hint")
    if not isinstance(hint, Mapping):
        return None, False, []

    observability = normalize_observability_status(
        hint.get("observability_status") if "observability_status" in hint else hint.get("observability")
    )
    requirements_raw = hint.get("step_evidence_requirements")
    if requirements_raw is None:
        requirements_raw = hint.get("evidence_requirements")
    requirements = [item for item in requirements_raw if isinstance(item, str) and item] if isinstance(requirements_raw, list) else []

    requires_visual_confirmation_raw = hint.get("requires_visual_confirmation")
    if isinstance(requires_visual_confirmation_raw, bool):
        requires_visual_confirmation = requires_visual_confirmation_raw
    else:
        requires_visual_confirmation = compute_requires_visual_confirmation(observability, requirements)
    return observability, requires_visual_confirmation, requirements


def _visual_confirmation_note(observability_status: str | None, *, lang: str) -> str:
    if lang == "en":
        if observability_status == "partial":
            return "Visual confirmation required: only partial observable evidence is available for this step."
        return "Visual confirmation required: no direct telemetry evidence is available for this step."
    if observability_status == "partial":
        return "待视觉确认：当前步骤只有部分可观测证据，请按高亮执行后进行目视确认。"
    return "待视觉确认：当前步骤缺少直接遥测证据，请按高亮执行后进行目视确认。"


def _append_unique_explanation(explanations: list[str], note: str) -> list[str]:
    if note in explanations:
        return explanations
    return [*explanations, note]


def _annotate_response_metadata(
    *,
    help_obj: Mapping[str, Any] | None,
    request: TutorRequest | None,
    explanations: list[str],
    metadata: dict[str, Any],
    lang: str,
) -> list[str]:
    if isinstance(help_obj, Mapping):
        diagnosis = help_obj.get("diagnosis")
        next_step = help_obj.get("next")
        if isinstance(diagnosis, Mapping):
            metadata["diagnosis"] = dict(diagnosis)
        if isinstance(next_step, Mapping):
            metadata["next"] = dict(next_step)

    observability_status, requires_visual_confirmation, _requirements = _extract_observability_metadata(request)

    if observability_status is not None:
        metadata["observability_status"] = observability_status
    metadata["requires_visual_confirmation"] = bool(requires_visual_confirmation)

    if requires_visual_confirmation:
        metadata["evidence_strength"] = "limited"

    if requires_visual_confirmation:
        explanations = _append_unique_explanation(
            explanations,
            _visual_confirmation_note(observability_status, lang=lang),
        )
    return explanations


def _validate_overlay_evidence(
    *,
    help_obj: Mapping[str, Any] | None,
    targets: list[str],
    request: TutorRequest | None,
) -> tuple[bool, list[str], dict[str, set[str]], int]:
    if not targets:
        return True, [], {}, 0

    allowed_refs = _collect_allowed_evidence_refs(request)
    allowed_ref_count = len(allowed_refs)

    overlay = help_obj.get("overlay") if isinstance(help_obj, Mapping) else None
    evidence_raw = overlay.get("evidence") if isinstance(overlay, Mapping) else None
    if not isinstance(evidence_raw, list):
        return False, ["missing_overlay_evidence"], {}, allowed_ref_count

    if not allowed_refs:
        return False, ["no_verifiable_evidence_refs"], {}, allowed_ref_count

    targets_set = set(targets)
    refs_by_target: dict[str, set[str]] = {target: set() for target in targets}
    reasons: list[str] = []

    invalid_items = 0
    unexpected_targets: set[str] = set()
    unknown_refs: set[str] = set()
    type_ref_mismatch_targets: set[str] = set()

    for item in evidence_raw:
        if not isinstance(item, Mapping):
            invalid_items += 1
            continue

        target = item.get("target")
        evidence_type = item.get("type")
        ref = item.get("ref")
        if not isinstance(target, str) or not target:
            invalid_items += 1
            continue
        if target not in targets_set:
            unexpected_targets.add(target)
            continue
        if not isinstance(evidence_type, str) or evidence_type not in EVIDENCE_TYPE_PREFIXES:
            invalid_items += 1
            continue
        if not isinstance(ref, str) or not ref:
            invalid_items += 1
            continue

        prefixes = EVIDENCE_TYPE_PREFIXES[evidence_type]
        if not any(ref.startswith(prefix) for prefix in prefixes):
            type_ref_mismatch_targets.add(target)
            continue

        if ref not in allowed_refs:
            unknown_refs.add(ref)
            continue

        refs_by_target[target].add(ref)

    if invalid_items:
        reasons.append("invalid_overlay_evidence_item")
    if unexpected_targets:
        reasons.append("evidence_target_not_in_overlay_targets")
    if type_ref_mismatch_targets:
        reasons.append("evidence_type_ref_mismatch:" + ",".join(sorted(type_ref_mismatch_targets)))
    if unknown_refs:
        reasons.append("unknown_evidence_ref:" + ",".join(sorted(unknown_refs)))

    # Keep this invariant explicit for auditability: every requested target must
    # end up with at least one validated evidence ref, even when other item-level
    # reasons (invalid/mismatch/unknown) are also present.
    missing_targets = sorted(target for target, refs in refs_by_target.items() if not refs)
    if missing_targets:
        reasons.append("missing_target_evidence:" + ",".join(missing_targets))

    return len(reasons) == 0, reasons, refs_by_target, allowed_ref_count


def _overlay_error_code(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "ui_map_not_found"
    if isinstance(exc, KeyError):
        return "target_not_mapped"
    if isinstance(exc, yaml.YAMLError):
        return "ui_map_invalid"
    if isinstance(exc, ValueError):
        return "ui_map_invalid"
    return "overlay_mapping_failed"


def map_help_response_to_tutor_response(
    help_obj: Mapping[str, Any] | None,
    *,
    request: TutorRequest | None = None,
    status: str = "ok",
    max_overlay_targets: int = 1,
    ui_map_path: str | Path | None = None,
    overlay_intent: str = "highlight",
    lang: str = "zh",
) -> TutorResponse:
    metadata: dict[str, Any] = {}

    if max_overlay_targets < 0:
        max_overlay_targets = 0

    allowed_intents = {"highlight", "clear", "pulse"}
    effective_overlay_intent = overlay_intent
    if overlay_intent not in allowed_intents:
        effective_overlay_intent = "highlight"
        _append_mapping_error(metadata, f"invalid_overlay_intent:{overlay_intent}")

    explanations_raw = help_obj.get("explanations") if isinstance(help_obj, Mapping) else None
    explanations = [item for item in explanations_raw if isinstance(item, str) and item] if isinstance(explanations_raw, list) else []
    explanations = _annotate_response_metadata(
        help_obj=help_obj,
        request=request,
        explanations=explanations,
        metadata=metadata,
        lang=lang,
    )
    message = explanations[0] if explanations else None

    raw_targets: list[str] = []
    if isinstance(help_obj, Mapping):
        overlay = help_obj.get("overlay")
        if isinstance(overlay, Mapping):
            targets = overlay.get("targets")
            if isinstance(targets, list):
                raw_targets = [item for item in targets if isinstance(item, str) and item]

    deduped_targets = _dedupe_preserve_order(raw_targets)

    rejected_targets: list[str] = []
    actions: list[dict[str, Any]] = []
    if max_overlay_targets == 0:
        if deduped_targets:
            metadata["dropped_targets"] = list(deduped_targets)
        effective_status = status
        if status not in {"ok", "pending", "error"}:
            effective_status = "error"
            _append_mapping_error(metadata, f"invalid_status:{status}")
        return TutorResponse(
            status=effective_status,
            in_reply_to=request.request_id if request else None,
            message=message,
            actions=actions,
            explanations=explanations,
            metadata=metadata,
        )

    evidence_ok, evidence_reasons, refs_by_target, allowed_ref_count = _validate_overlay_evidence(
        help_obj=help_obj,
        targets=deduped_targets,
        request=request,
    )
    if deduped_targets:
        metadata["allowed_evidence_ref_count"] = allowed_ref_count

    if not deduped_targets:
        effective_status = status
        if status not in {"ok", "pending", "error"}:
            effective_status = "error"
            _append_mapping_error(metadata, f"invalid_status:{status}")
        return TutorResponse(
            status=effective_status,
            in_reply_to=request.request_id if request else None,
            message=message,
            actions=actions,
            explanations=explanations,
            metadata=metadata,
        )

    if not evidence_ok:
        rejected_targets.extend(deduped_targets)
        metadata["overlay_rejected"] = True
        metadata["overlay_rejected_reasons"] = evidence_reasons
        for reason in evidence_reasons:
            _append_mapping_error(metadata, reason)
        metadata["rejected_targets"] = rejected_targets

        effective_status = status
        if status not in {"ok", "pending", "error"}:
            effective_status = "error"
            _append_mapping_error(metadata, f"invalid_status:{status}")
        return TutorResponse(
            status=effective_status,
            in_reply_to=request.request_id if request else None,
            message=message,
            actions=[],
            explanations=explanations,
            metadata=metadata,
        )

    planner: OverlayPlanner | None = None
    planner_error: dict[str, str] | None = None
    try:
        resolved_ui_map = _resolve_ui_map_path(ui_map_path)
        planner = _get_overlay_planner(str(resolved_ui_map))
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        planner_error = {
            "error_type": type(exc).__name__,
            "error_code": _overlay_error_code(exc),
        }

    if planner is None:
        rejected_targets.extend(deduped_targets)
        if planner_error:
            metadata["overlay_mapping_error"] = planner_error
    else:
        overlay_failures: list[dict[str, str]] = []
        dropped_for_limit: list[str] = []
        for idx, target in enumerate(deduped_targets):
            if len(actions) >= max_overlay_targets:
                dropped_for_limit.extend(deduped_targets[idx:])
                break
            try:
                action = planner.plan(target, intent=effective_overlay_intent).to_action()
                action["evidence_required"] = True
                action["evidence_refs"] = sorted(refs_by_target.get(target, set()))
                actions.append(action)
            except (KeyError, ValueError) as exc:
                rejected_targets.append(target)
                overlay_failures.append(
                    {
                        "target": target,
                        "error_type": type(exc).__name__,
                        "error_code": _overlay_error_code(exc),
                    }
                )
        if dropped_for_limit:
            metadata["dropped_targets"] = dropped_for_limit
        if overlay_failures:
            metadata["overlay_mapping_failures"] = overlay_failures

    if rejected_targets:
        metadata["rejected_targets"] = rejected_targets

    effective_status = status
    if status not in {"ok", "pending", "error"}:
        effective_status = "error"
        _append_mapping_error(metadata, f"invalid_status:{status}")

    return TutorResponse(
        status=effective_status,
        in_reply_to=request.request_id if request else None,
        message=message,
        actions=actions,
        explanations=explanations,
        metadata=metadata,
    )


__all__ = ["map_help_response_to_tutor_response"]
