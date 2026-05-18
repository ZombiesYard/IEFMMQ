"""
Validator and action planner for harness decisions.

This module stays in core on purpose: it only consumes the stable
StepHarnessSpec contract plus plain decision/context data, then returns a
small plan that adapters can map to their transport-specific actions.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.step_harness import StepHarnessSpec


_VARS_PREDICATE_RE = re.compile(
    r"^\s*(?:payload\.)?vars\.([A-Za-z0-9_]+)\s*(==|!=|>=|<=|>|<|\bin\b)\s*(.+?)\s*$"
)


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
    rejected_model_targets: tuple[str, ...]
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
    not_seen_fact_id: str | None = None
    source: str = "validator_action_hint"


@dataclass(frozen=True)
class HarnessTextGuidanceRule:
    step_id: str
    target: str
    guidance: str
    source: str = "validator_text_only_guidance"


@dataclass(frozen=True)
class EvidenceConsistencyResult:
    accepted: bool
    validator_rejected: bool
    repair_applied: bool
    rejected_missing_conditions: tuple[str, ...]
    final_action_plan_source: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class _StateActionPlan:
    targets: tuple[str, ...]
    guidance: str | None
    text_only: bool
    source: str
    reasons: tuple[str, ...] = ()


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


def _telemetry_digest_has_true_var(evidence_packet: Any, var_name: str) -> bool:
    digest = getattr(evidence_packet, "telemetry_window_digest", None)
    if digest is None:
        return False
    stable_true = getattr(digest, "stable_true_vars", ())
    if var_name in set(_strings(stable_true if isinstance(stable_true, (list, tuple, set)) else None)):
        return True
    return False


def _parse_predicate_literal(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_predicate_value(value: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return value == expected
    if op == "!=":
        return value != expected
    if op in {">=", "<=", ">", "<"}:
        if not (_is_number(value) and _is_number(expected)):
            return False
        if op == ">=":
            return value >= expected
        if op == "<=":
            return value <= expected
        if op == ">":
            return value > expected
        return value < expected
    if op == "in":
        if isinstance(expected, (list, tuple)) and len(expected) == 2 and all(_is_number(item) for item in expected):
            low, high = expected
            return _is_number(value) and low <= value <= high
        if isinstance(expected, (list, tuple, set, frozenset)):
            return value in expected
    return False


def _predicate_satisfied_by_latest_evidence(
    predicate: str,
    *,
    latest_vars: Mapping[str, Any],
    evidence_packet: Any = None,
) -> bool:
    matched = _VARS_PREDICATE_RE.match(predicate)
    if matched is None:
        return False
    var_name = matched.group(1)
    op = matched.group(2).strip()
    expected = _parse_predicate_literal(matched.group(3))
    latest_value = latest_vars.get(var_name)
    if var_name in latest_vars:
        return _compare_predicate_value(latest_value, op, expected)
    if not (op == "==" and expected is True):
        return False
    return _telemetry_digest_has_true_var(evidence_packet, var_name)


def _completion_gate_satisfied(evidence_packet: Any, step_id: str | None) -> bool:
    if not isinstance(step_id, str) or not step_id:
        return False
    gate_evidence = getattr(evidence_packet, "gate_evidence", None)
    gate_id = f"{step_id}.completion"
    for gate in getattr(gate_evidence, "satisfied_gates", ()):
        if not isinstance(gate, Mapping):
            continue
        if gate.get("gate_id") != gate_id:
            continue
        if gate.get("reason_code") == "no_rules":
            return False
        status = gate.get("status")
        if status == "satisfied":
            return True
        if status == "allowed" or gate.get("allowed") is True:
            return True
    satisfied_ids = getattr(gate_evidence, "satisfied_gate_ids", ())
    if gate_id in set(_strings(satisfied_ids if isinstance(satisfied_ids, (list, tuple, set)) else None)):
        return True
    return False


def validate_final_evidence_consistency(
    *,
    accepted_step_id: str | None,
    accepted_overlay_targets: Sequence[str] | None,
    accepted_missing_conditions: Sequence[str] | None,
    latest_vars: Mapping[str, Any] | None,
    evidence_packet: Any = None,
) -> EvidenceConsistencyResult:
    del accepted_overlay_targets
    vars_map = latest_vars if isinstance(latest_vars, Mapping) else {}
    missing_conditions = _strings(
        accepted_missing_conditions
        if isinstance(accepted_missing_conditions, (list, tuple, set))
        else None
    )
    rejected_missing = tuple(
        condition
        for condition in missing_conditions
        if _predicate_satisfied_by_latest_evidence(
            condition,
            latest_vars=vars_map,
            evidence_packet=evidence_packet,
        )
    )

    reasons = [
        f"missing_condition_satisfied_by_latest_telemetry:{condition}"
        for condition in rejected_missing
    ]
    if _completion_gate_satisfied(evidence_packet, accepted_step_id):
        reasons.append(f"completion_gate_already_satisfied:{accepted_step_id}")

    rejected = bool(reasons)
    return EvidenceConsistencyResult(
        accepted=not rejected,
        validator_rejected=rejected,
        repair_applied=rejected,
        rejected_missing_conditions=rejected_missing,
        final_action_plan_source="final_evidence_consistency_validator" if rejected else "model",
        reasons=tuple(reasons),
    )


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


def _not_seen(
    fact_id: str,
    *,
    vision_not_seen_fact_ids: Sequence[str] | None,
) -> bool:
    return fact_id in set(_strings(vision_not_seen_fact_ids))


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
    vision_not_seen_fact_ids: Sequence[str] | None,
) -> str | None:
    if not isinstance(step_id, str) or not step_id:
        return None
    for rule in action_hint_fact_rules or ():
        if rule.step_id != step_id:
            continue
        seen_requirement_met = _seen_or_fresh(
            rule.fact_id,
            vision_seen_fact_ids=vision_seen_fact_ids,
            vision_fresh_fact_ids=vision_fresh_fact_ids,
        )
        not_seen_requirement_met = (
            True
            if rule.not_seen_fact_id is None
            else _not_seen(
                rule.not_seen_fact_id,
                vision_not_seen_fact_ids=vision_not_seen_fact_ids,
            )
        )
        if seen_requirement_met and not_seen_requirement_met:
            return rule.source
    return None


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


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    return int(number) if number is not None else None


def _latest_vars(latest_vars: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return latest_vars if isinstance(latest_vars, Mapping) else {}


def _normalize_ufc_scratchpad_text(vars_map: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "ufc_scratchpad_string_1_display",
        "ufc_scratchpad_string_2_display",
        "ufc_scratchpad_number_display",
    ):
        value = vars_map.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "".join(parts).replace("。", ".").upper()


def _s09_comm1_frequency_complete(vars_map: Mapping[str, Any]) -> bool:
    if vars_map.get("comm1_freq_134_000") is True:
        return True
    value = _coerce_int(vars_map.get("comm1_freq_value"))
    return value == 13400


def _s09_state_action_plan(
    vars_map: Mapping[str, Any],
    *,
    spec: StepHarnessSpec | None,
    recent_action_targets: Sequence[str] | None,
) -> _StateActionPlan | None:
    allowed = set(spec.allowed_overlay_targets) if spec is not None else set()
    if not allowed or _s09_comm1_frequency_complete(vars_map):
        return None

    scratchpad_text = _normalize_ufc_scratchpad_text(vars_map)
    compact = scratchpad_text.replace(" ", "")
    payload = compact[3:] if compact.startswith("1--") else compact
    recent = set(_strings(recent_action_targets))

    def _target(name: str, guidance: str) -> _StateActionPlan | None:
        if name not in allowed:
            return None
        return _StateActionPlan(
            targets=(name,),
            guidance=guidance,
            text_only=False,
            source="state_action_planner",
            reasons=(f"s09_state_target:{name}",),
        )

    if payload.endswith("134.000"):
        return _target("ufc_ent_button", "The UFC scratchpad shows 134.000; press ENT to commit the COMM1 preset.")
    if payload.endswith("13.400") or payload.endswith("1.340") or payload.endswith(".134") or payload.endswith("1.34"):
        return _target("ufc_key_0", "COMM1 preset entry is partway through 134.000; press 0 next.")
    if payload.endswith(".13"):
        return _target("ufc_key_4", "COMM1 preset entry shows 13; press 4 next.")
    if payload.endswith(".1"):
        return _target("ufc_key_3", "COMM1 preset entry shows 1; press 3 next.")
    if payload.endswith("305.000") or payload.endswith("305000"):
        return _target("ufc_key_1", "COMM1 preset 1 is open with the old 305.000 value; press 1 next.")
    if vars_map.get("ufc_comm1_pull_pressed") is True or "ufc_comm1_channel_selector_pull" in recent:
        return _target("ufc_key_1", "COMM1 preset entry is open; start typing 134.000 with key 1.")
    return _target("ufc_comm1_channel_selector_pull", "Pull the UFC COMM1 channel selector before entering 134.000.")


def _changed_var(evidence_packet: Any, var_name: str) -> Mapping[str, Any] | None:
    digest = getattr(evidence_packet, "telemetry_window_digest", None)
    for item in getattr(digest, "changed_vars", ()):
        if isinstance(item, Mapping) and item.get("var") == var_name:
            return item
    return None


def _probe_motion_state(
    step_id: str | None,
    vars_map: Mapping[str, Any],
    *,
    evidence_packet: Any = None,
) -> str | None:
    probe_value = _coerce_float(vars_map.get("ext_refuel_probe_value"))
    switch_value = _coerce_int(vars_map.get("probe_switch_value"))
    changed = _changed_var(evidence_packet, "ext_refuel_probe_value")
    first_value = _coerce_float(changed.get("first_value")) if isinstance(changed, Mapping) else None
    last_value = _coerce_float(changed.get("last_value")) if isinstance(changed, Mapping) else None
    is_extending = first_value is not None and last_value is not None and last_value > first_value
    is_retracting = first_value is not None and last_value is not None and last_value < first_value

    if step_id == "S20":
        if probe_value is not None and 0 < probe_value < 60000 and is_extending:
            return "s20_extending"
    if step_id == "S21":
        near_retract_threshold = probe_value is not None and 5000 < probe_value <= 7000
        if probe_value is not None and probe_value > 5000 and (
            is_retracting or (switch_value == 1 and near_retract_threshold)
        ):
            return "s21_retracting"
    return None


def _state_action_plan(
    *,
    step_id: str | None,
    spec: StepHarnessSpec | None,
    latest_vars: Mapping[str, Any] | None,
    evidence_packet: Any = None,
    recent_action_targets: Sequence[str] | None = None,
    vision_seen_fact_ids: Sequence[str] | None = None,
    vision_fresh_fact_ids: Sequence[str] | None = None,
    vision_not_seen_fact_ids: Sequence[str] | None = None,
) -> _StateActionPlan | None:
    vars_map = _latest_vars(latest_vars)
    if step_id == "S09":
        return _s09_state_action_plan(
            vars_map,
            spec=spec,
            recent_action_targets=recent_action_targets,
        )

    motion_state = _probe_motion_state(step_id, vars_map, evidence_packet=evidence_packet)
    if motion_state == "s20_extending":
        return _StateActionPlan(
            targets=(),
            guidance="The refueling probe is extending. Wait until it is fully extended before continuing.",
            text_only=True,
            source="state_action_planner_wait",
            reasons=("probe_motion:s20_extending",),
        )
    if motion_state == "s21_retracting":
        return _StateActionPlan(
            targets=(),
            guidance="The refueling probe is retracting. Wait until it is fully stowed before continuing.",
            text_only=True,
            source="state_action_planner_wait",
            reasons=("probe_motion:s21_retracting",),
        )

    seen_or_fresh = set(_strings(vision_seen_fact_ids)) | set(_strings(vision_fresh_fact_ids))
    not_seen = set(_strings(vision_not_seen_fact_ids))
    allowed = set(spec.allowed_overlay_targets) if spec is not None else set()
    if step_id == "S18" and "bit_root_page_visible" in seen_or_fresh and "fcsmc_page_visible" in not_seen:
        if "right_mdi_pb5" in allowed:
            return _StateActionPlan(
                targets=("right_mdi_pb5",),
                guidance="BIT root is visible; press Right DDI PB5/FCS-MC to enter the FCS-MC BIT page.",
                text_only=False,
                source="state_action_planner",
                reasons=("s18_bit_root_to_pb5",),
            )
    if step_id == "S19":
        if "fcsmc_in_test_visible" in seen_or_fresh:
            return _StateActionPlan(
                targets=(),
                guidance="The FCS BIT is already running. Release the switch and wait for the final GO result.",
                text_only=True,
                source="state_action_planner_wait",
                reasons=("s19_in_test_wait",),
            )
        if "fcsmc_intermediate_result_visible" in seen_or_fresh:
            targets = tuple(target for target in ("fcs_bit_switch", "right_mdi_pb5") if target in allowed)
            if targets:
                return _StateActionPlan(
                    targets=targets,
                    guidance="Hold the FCS BIT switch up while pressing Right DDI PB5 to start the BIT.",
                    text_only=False,
                    source="state_action_planner",
                    reasons=("s19_intermediate_start_bit",),
                )
    return None


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
    vision_not_seen_fact_ids: Sequence[str] | None = None,
    action_hint: Mapping[str, Any] | None = None,
    completion_advancements: Sequence[HarnessCompletionAdvance] | None = None,
    action_hint_step_ids: Sequence[str] | None = None,
    action_hint_fact_rules: Sequence[HarnessActionHintFactRule] | None = None,
    text_guidance_rules: Sequence[HarnessTextGuidanceRule] | None = None,
    latest_vars: Mapping[str, Any] | None = None,
    evidence_packet: Any = None,
    recent_action_targets: Sequence[str] | None = None,
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
    state_plan = _state_action_plan(
        step_id=selected_step_id,
        spec=spec,
        latest_vars=latest_vars,
        evidence_packet=evidence_packet,
        recent_action_targets=recent_action_targets,
        vision_seen_fact_ids=vision_seen_fact_ids,
        vision_fresh_fact_ids=vision_fresh_fact_ids,
        vision_not_seen_fact_ids=vision_not_seen_fact_ids,
    )
    if state_plan is not None and state_plan.text_only:
        return HarnessActionPlan(
            step_id=selected_step_id,
            overlay_step_id=selected_overlay_step_id,
            targets=(),
            evidence_refs=(),
            guidance=state_plan.guidance,
            text_only=True,
            validator_rejected=True,
            repair_applied=True,
            rejected_model_step_id=rejected_model_step_id,
            rejected_model_targets=proposed_targets,
            final_action_plan_source=state_plan.source,
            reasons=tuple((*reasons, *state_plan.reasons)),
        )

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
            rejected_model_targets=(),
            final_action_plan_source=text_guidance.source,
            reasons=tuple((*reasons, f"text_only_guidance:{text_guidance.target}")),
        )

    hinted_targets = _hint_targets(action_hint)
    use_hint = False
    action_hint_step_set = set(_strings(action_hint_step_ids))
    if hinted_targets:
        if selected_step_id in action_hint_step_set:
            use_hint = True
        hint_rule_source = _hint_allowed_by_fact_rule(
            selected_step_id,
            action_hint_fact_rules=action_hint_fact_rules,
            vision_seen_fact_ids=vision_seen_fact_ids,
            vision_fresh_fact_ids=vision_fresh_fact_ids,
            vision_not_seen_fact_ids=vision_not_seen_fact_ids,
        )
        if hint_rule_source is not None:
            use_hint = True
            source = hint_rule_source

    if state_plan is not None and len(state_plan.targets) > 1 and hinted_targets != state_plan.targets:
        if use_hint and hinted_targets:
            reasons.append(
                "action_hint_incomplete_for_state_plan:" + ",".join(hinted_targets)
            )
        use_hint = False

    if completion_advance is not None:
        base_targets = spec.allowed_overlay_targets if spec is not None else ()
    elif use_hint:
        base_targets = hinted_targets
        if source == "model":
            source = "validator_action_hint"
    elif state_plan is not None:
        base_targets = state_plan.targets
        source = state_plan.source
        reasons.extend(state_plan.reasons)
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
    rejected_model_targets: tuple[str, ...] = ()
    if use_hint and proposed_targets:
        hinted_target_set = set(hinted_targets)
        rejected_model_targets = tuple(
            target for target in proposed_targets
            if target not in hinted_target_set
        )
        reasons.extend(f"action_hint_target_mismatch:{target}" for target in rejected_model_targets)
    elif state_plan is not None and proposed_targets:
        planned_target_set = set(state_plan.targets)
        rejected_model_targets = tuple(
            target for target in proposed_targets
            if target not in planned_target_set
        )
        reasons.extend(f"state_action_target_mismatch:{target}" for target in rejected_model_targets)

    if state_plan is not None and not use_hint and tuple(targets) != state_plan.targets:
        return HarnessActionPlan(
            step_id=selected_step_id,
            overlay_step_id=selected_overlay_step_id,
            targets=(),
            evidence_refs=(),
            guidance=state_plan.guidance,
            text_only=True,
            validator_rejected=True,
            repair_applied=True,
            rejected_model_step_id=rejected_model_step_id,
            rejected_model_targets=rejected_model_targets or proposed_targets,
            final_action_plan_source=state_plan.source,
            reasons=tuple(reasons),
        )

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
    if (
        not use_hint
        and state_plan is not None
        and isinstance(state_plan.guidance, str)
        and state_plan.guidance
    ):
        guidance = state_plan.guidance

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
        rejected_model_targets=rejected_model_targets,
        final_action_plan_source=source,
        reasons=tuple(reasons),
    )


__all__ = [
    "EvidenceConsistencyResult",
    "HarnessActionHintFactRule",
    "HarnessActionPlan",
    "HarnessCompletionAdvance",
    "HarnessTextGuidanceRule",
    "plan_harness_action",
    "validate_final_evidence_consistency",
]
