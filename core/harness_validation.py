"""
Validator and action planner for harness decisions.

This module stays in core on purpose: it only consumes the stable
StepHarnessSpec contract plus plain decision/context data, then returns a
small plan that adapters can map to their transport-specific actions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.step_harness import StepHarnessSpec


@dataclass(frozen=True)
class HarnessActionPlan:
    step_id: str | None
    overlay_step_id: str | None
    targets: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    guidance: str | None
    text_only: bool
    validator_rejected: bool
    repair_applied: bool
    rejected_model_step_id: str | None
    final_action_plan_source: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class HarnessCompletionAdvance:
    step_id: str
    fact_id: str
    next_step_id: str
    source: str


@dataclass(frozen=True)
class HarnessActionHintFactRule:
    step_id: str
    fact_id: str


@dataclass(frozen=True)
class HarnessTextGuidanceRule:
    step_id: str
    target: str
    guidance: str
    source: str = "validator_text_only_guidance"


def _strings(raw: Sequence[str] | set[str] | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple, set)):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _hint_targets(action_hint: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(action_hint, Mapping):
        return ()
    raw_targets = action_hint.get("targets")
    targets = _strings(raw_targets if isinstance(raw_targets, (list, tuple, set)) else None)
    if targets:
        return targets
    target = action_hint.get("target")
    return (target,) if isinstance(target, str) and target else ()


def _seen_or_fresh(
    fact_id: str,
    *,
    vision_seen_fact_ids: Sequence[str] | None,
    vision_fresh_fact_ids: Sequence[str] | None,
) -> bool:
    return fact_id in set(_strings(vision_seen_fact_ids)) or fact_id in set(_strings(vision_fresh_fact_ids))


def _completion_advance_for_seen_fact(
    step_id: str | None,
    *,
    completion_advancements: Sequence[HarnessCompletionAdvance] | None,
    vision_seen_fact_ids: Sequence[str] | None,
    vision_fresh_fact_ids: Sequence[str] | None,
) -> HarnessCompletionAdvance | None:
    if not isinstance(step_id, str) or not step_id:
        return None
    for rule in completion_advancements or ():
        if rule.step_id != step_id:
            continue
        if _seen_or_fresh(
            rule.fact_id,
            vision_seen_fact_ids=vision_seen_fact_ids,
            vision_fresh_fact_ids=vision_fresh_fact_ids,
        ):
            return rule
    return None


def _text_guidance_for_targets(
    step_id: str | None,
    targets: Sequence[str],
    *,
    text_guidance_rules: Sequence[HarnessTextGuidanceRule] | None,
) -> HarnessTextGuidanceRule | None:
    if not isinstance(step_id, str) or not step_id:
        return None
    target_set = set(_strings(targets))
    for rule in text_guidance_rules or ():
        if rule.step_id == step_id and (rule.target == "*" or rule.target in target_set):
            return rule
    return None


def _hint_allowed_by_fact_rule(
    step_id: str | None,
    *,
    action_hint_fact_rules: Sequence[HarnessActionHintFactRule] | None,
    vision_seen_fact_ids: Sequence[str] | None,
    vision_fresh_fact_ids: Sequence[str] | None,
) -> bool:
    if not isinstance(step_id, str) or not step_id:
        return False
    for rule in action_hint_fact_rules or ():
        if rule.step_id != step_id:
            continue
        if _seen_or_fresh(
            rule.fact_id,
            vision_seen_fact_ids=vision_seen_fact_ids,
            vision_fresh_fact_ids=vision_fresh_fact_ids,
        ):
            return True
    return False


def _filter_targets(
    targets: Sequence[str],
    *,
    spec: StepHarnessSpec | None,
    runtime_overlay_targets: Sequence[str] | None,
    request_overlay_targets: Sequence[str] | None,
    max_overlay_targets: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    allowed_by_spec = set(spec.allowed_overlay_targets) if spec is not None else set()
    runtime_allowset = set(_strings(runtime_overlay_targets))
    request_allowset = set(_strings(request_overlay_targets))
    limit = max(0, int(max_overlay_targets))
    out: list[str] = []
    reasons: list[str] = []
    for target in _strings(list(targets)):
        if spec is not None and target not in allowed_by_spec:
            reasons.append(f"target_not_allowed:{target}")
            continue
        if runtime_allowset and target not in runtime_allowset:
            reasons.append(f"target_not_in_runtime_allowlist:{target}")
            continue
        if request_allowset and target not in request_allowset:
            reasons.append(f"target_not_in_request_allowlist:{target}")
            continue
        if target in out:
            continue
        if len(out) >= limit:
            reasons.append(f"target_dropped_by_max_overlay_targets:{target}")
            continue
        out.append(target)
    return tuple(out), tuple(reasons)


def _valid_evidence_refs(
    evidence_refs: Sequence[str] | None,
    allowed_evidence_refs: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    refs = _strings(evidence_refs)
    allowed = set(_strings(allowed_evidence_refs))
    if not allowed:
        return refs, ()
    valid = tuple(ref for ref in refs if ref in allowed)
    invalid = tuple(ref for ref in refs if ref not in allowed)
    return valid, tuple(f"unknown_evidence_ref:{ref}" for ref in invalid)


def plan_harness_action(
    *,
    step_specs: Mapping[str, StepHarnessSpec],
    inferred_step_id: str | None,
    model_step_id: str | None = None,
    overlay_step_id: str | None = None,
    proposed_overlay_targets: Sequence[str] | None = None,
    candidate_step_ids: Sequence[str] | None = None,
    runtime_overlay_targets: Sequence[str] | None = None,
    request_overlay_targets: Sequence[str] | None = None,
    allowed_evidence_refs: Sequence[str] | None = None,
    evidence_refs: Sequence[str] | None = None,
    max_overlay_targets: int = 1,
    vision_seen_fact_ids: Sequence[str] | None = None,
    vision_fresh_fact_ids: Sequence[str] | None = None,
    action_hint: Mapping[str, Any] | None = None,
    completion_advancements: Sequence[HarnessCompletionAdvance] | None = None,
    action_hint_step_ids: Sequence[str] | None = None,
    action_hint_fact_rules: Sequence[HarnessActionHintFactRule] | None = None,
    text_guidance_rules: Sequence[HarnessTextGuidanceRule] | None = None,
) -> HarnessActionPlan:
    reasons: list[str] = []
    rejected_model_step_id: str | None = None
    selected_step_id = inferred_step_id if isinstance(inferred_step_id, str) and inferred_step_id else None
    if isinstance(model_step_id, str) and model_step_id and selected_step_id is None:
        selected_step_id = model_step_id

    selected_overlay_step_id = (
        overlay_step_id if isinstance(overlay_step_id, str) and overlay_step_id else selected_step_id
    )
    source = "model"

    completion_advance = _completion_advance_for_seen_fact(
        selected_step_id,
        completion_advancements=completion_advancements,
        vision_seen_fact_ids=vision_seen_fact_ids,
        vision_fresh_fact_ids=vision_fresh_fact_ids,
    )
    if completion_advance is not None:
        if isinstance(model_step_id, str) and model_step_id != completion_advance.next_step_id:
            rejected_model_step_id = model_step_id
        selected_step_id = completion_advance.next_step_id
        selected_overlay_step_id = completion_advance.next_step_id
        source = completion_advance.source
        reasons.append(f"known_completion_fact:{completion_advance.fact_id}")

    candidate_set = set(_strings(candidate_step_ids))
    if isinstance(model_step_id, str) and model_step_id:
        if candidate_set and model_step_id not in candidate_set:
            rejected_model_step_id = model_step_id
            reasons.append(f"model_step_not_candidate:{model_step_id}")
        elif selected_step_id is not None and model_step_id != selected_step_id:
            rejected_model_step_id = model_step_id
            reasons.append(f"model_step_mismatch:{model_step_id}")

    spec = step_specs.get(selected_overlay_step_id or "") if selected_overlay_step_id else None
    proposed_targets = _strings(proposed_overlay_targets)
    text_guidance = _text_guidance_for_targets(
        selected_step_id,
        proposed_targets,
        text_guidance_rules=text_guidance_rules,
    )
    if text_guidance is not None:
        return HarnessActionPlan(
            step_id=selected_step_id,
            overlay_step_id=selected_overlay_step_id,
            targets=(),
            evidence_refs=(),
            guidance=text_guidance.guidance,
            text_only=True,
            validator_rejected=True,
            repair_applied=True,
            rejected_model_step_id=rejected_model_step_id,
            final_action_plan_source=text_guidance.source,
            reasons=tuple((*reasons, f"text_only_guidance:{text_guidance.target}")),
        )

    hinted_targets = _hint_targets(action_hint)
    use_hint = False
    action_hint_step_set = set(_strings(action_hint_step_ids))
    if hinted_targets:
        if selected_step_id in action_hint_step_set:
            use_hint = True
        if _hint_allowed_by_fact_rule(
            selected_step_id,
            action_hint_fact_rules=action_hint_fact_rules,
            vision_seen_fact_ids=vision_seen_fact_ids,
            vision_fresh_fact_ids=vision_fresh_fact_ids,
        ):
            use_hint = True

    if completion_advance is not None:
        base_targets = spec.allowed_overlay_targets if spec is not None else ()
    elif use_hint:
        base_targets = hinted_targets
        source = "validator_action_hint"
    else:
        base_targets = proposed_targets

    targets, target_reasons = _filter_targets(
        base_targets,
        spec=spec,
        runtime_overlay_targets=runtime_overlay_targets,
        request_overlay_targets=request_overlay_targets,
        max_overlay_targets=max_overlay_targets,
    )
    reasons.extend(target_reasons)

    if not targets and spec is not None and spec.allowed_overlay_targets and max_overlay_targets > 0:
        repaired_targets, repair_reasons = _filter_targets(
            spec.allowed_overlay_targets,
            spec=spec,
            runtime_overlay_targets=runtime_overlay_targets,
            request_overlay_targets=request_overlay_targets,
            max_overlay_targets=max_overlay_targets,
        )
        if repaired_targets:
            targets = repaired_targets
            source = "validator_repair" if source == "model" else source
            reasons.extend(repair_reasons)

    valid_refs, ref_reasons = _valid_evidence_refs(evidence_refs, allowed_evidence_refs)
    reasons.extend(ref_reasons)

    proposed_set = set(proposed_targets)
    target_set = set(targets)
    repaired = (
        source != "model"
        or bool(target_reasons)
        or bool(ref_reasons)
        or (bool(targets) and proposed_set != target_set)
    )
    rejected = bool(reasons) or rejected_model_step_id is not None or repaired
    if source == "model" and repaired:
        source = "validator_repair"

    guidance = None
    if use_hint and isinstance(action_hint, Mapping):
        hint_reason = action_hint.get("reason")
        if isinstance(hint_reason, str) and hint_reason:
            guidance = hint_reason

    return HarnessActionPlan(
        step_id=selected_step_id,
        overlay_step_id=selected_overlay_step_id,
        targets=targets,
        evidence_refs=valid_refs,
        guidance=guidance,
        text_only=False,
        validator_rejected=rejected,
        repair_applied=repaired,
        rejected_model_step_id=rejected_model_step_id,
        final_action_plan_source=source,
        reasons=tuple(reasons),
    )


__all__ = [
    "HarnessActionHintFactRule",
    "HarnessActionPlan",
    "HarnessCompletionAdvance",
    "HarnessTextGuidanceRule",
    "plan_harness_action",
]
