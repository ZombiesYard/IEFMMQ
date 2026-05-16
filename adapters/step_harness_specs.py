"""
Pack-derived StepHarnessSpec loader.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from adapters.pack_gates import load_pack_gate_config
from adapters.step_inference import format_gate_rule_condition, load_pack_steps
from core.step_harness import RecoveryActionPolicy, SignalQualityRequirement, StepHarnessSpec
from core.step_signal_metadata import (
    STEP_EVIDENCE_REQUIREMENT_VALUES,
    compute_requires_visual_confirmation,
    normalize_observability_status,
)
from core.vision_facts import VisionFactsConfigError, load_vision_facts_config


_DEFAULT_LATCH_SEMANTICS = "none"


def load_step_harness_specs(
    pack_path: str | Path | None = None,
    *,
    scenario_profile: str | None = None,
) -> dict[str, StepHarnessSpec]:
    path = Path(pack_path) if pack_path else None
    steps = load_pack_steps(path)
    gate_config = load_pack_gate_config(path, scenario_profile=scenario_profile)
    precondition_gates = gate_config.get("precondition_gates", {})
    completion_gates = gate_config.get("completion_gates", {})
    vision_config = _load_vision_config(path)

    specs: dict[str, StepHarnessSpec] = {}
    for step_idx, step in enumerate(steps):
        step_id = _extract_step_id(step)
        if step_id is None or step_id in specs:
            continue
        specs[step_id] = _build_step_harness_spec(
            step_id,
            step_idx=step_idx,
            step=step,
            pack_path=path,
            precondition_gates=precondition_gates,
            completion_gates=completion_gates,
            vision_config=vision_config,
        )
    return specs


def step_signal_profiles_from_specs(
    specs: Mapping[str, StepHarnessSpec],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for step_id, spec in specs.items():
        profile: dict[str, Any] = {
            "observability": spec.observability_status,
            "observability_status": spec.observability_status,
            "ui_targets": list(spec.declared_ui_targets),
            "overlay_enabled": bool(spec.allowed_overlay_targets),
            "requires_visual_confirmation": spec.requires_visual_confirmation,
            "step_harness_spec": spec,
        }
        if spec.required_observables:
            profile["evidence_requirements"] = list(spec.required_observables)
        profiles[step_id] = profile
    return profiles


def step_fallback_profiles_from_specs(
    specs: Mapping[str, StepHarnessSpec],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for step_id, spec in specs.items():
        profiles[step_id] = {
            "ui_targets": list(spec.declared_ui_targets),
            "gate_var_refs": list(spec.telemetry_facts),
            "overlay_enabled": bool(spec.allowed_overlay_targets),
            "step_harness_spec": spec,
        }
    return profiles


def _build_step_harness_spec(
    step_id: str,
    *,
    step_idx: int,
    step: Mapping[str, Any],
    pack_path: Path | None,
    precondition_gates: Mapping[str, Any],
    completion_gates: Mapping[str, Any],
    vision_config: Mapping[str, Any],
) -> StepHarnessSpec:
    harness = step.get("harness")
    harness_map = harness if isinstance(harness, Mapping) else {}
    required_observables = _evidence_requirements(
        step.get("evidence_requirements"),
        step_idx=step_idx,
        pack_path=pack_path,
    )
    optional_observables = _dedupe_non_empty_strings(harness_map.get("optional_observables"))
    raw_targets = _ui_targets(step.get("ui_targets"), step_idx=step_idx, pack_path=pack_path)
    overlay_enabled = _overlay_enabled(step.get("overlay_enabled"), step_idx=step_idx, pack_path=pack_path)
    allowed_overlay_targets = raw_targets if overlay_enabled else ()
    forbidden_overlay_targets = _dedupe_non_empty_strings(harness_map.get("forbidden_overlay_targets"))
    if not overlay_enabled:
        forbidden_overlay_targets = _dedupe((*forbidden_overlay_targets, *raw_targets))

    telemetry_facts = _collect_gate_var_refs(
        step_id,
        precondition_gates=precondition_gates,
        completion_gates=completion_gates,
    )
    vision_facts, completion_vision_facts = _collect_step_vision_fact_refs(step_id, vision_config)
    recent_action_facts = (
        tuple(f"RECENT_ACTIONS.{target}" for target in raw_targets)
        if "delta" in required_observables
        else ()
    )
    completion_predicates = _completion_predicates(
        step_id,
        completion_gates=completion_gates,
        vision_facts=completion_vision_facts,
    )

    observability = _observability_status(step.get("observability"), step_idx=step_idx, pack_path=pack_path)
    requires_visual_confirmation_raw = step.get("requires_visual_confirmation")
    requires_visual_confirmation = (
        bool(requires_visual_confirmation_raw)
        if isinstance(requires_visual_confirmation_raw, bool)
        else compute_requires_visual_confirmation(observability, required_observables)
    )
    signal_quality = _signal_quality_requirement(
        vision_facts=vision_facts,
        vision_config=vision_config,
        harness_map=harness_map,
    )
    latch_semantics = _latch_semantics(
        vision_facts=vision_facts,
        vision_config=vision_config,
        harness_map=harness_map,
    )
    recovery_policy = _recovery_policy(
        requires_visual_confirmation=requires_visual_confirmation,
        recovery_targets=raw_targets,
        harness_map=harness_map,
    )
    return StepHarnessSpec(
        step_id=step_id,
        required_observables=required_observables,
        optional_observables=optional_observables,
        telemetry_facts=telemetry_facts,
        vision_facts=vision_facts,
        recent_action_facts=recent_action_facts,
        completion_predicates=completion_predicates,
        latch_semantics=latch_semantics,
        allowed_overlay_targets=allowed_overlay_targets,
        forbidden_overlay_targets=forbidden_overlay_targets,
        recovery_policy=recovery_policy,
        observability_status=observability,
        signal_quality=signal_quality,
        requires_visual_confirmation=requires_visual_confirmation,
        declared_ui_targets=raw_targets,
    )


def _load_vision_config(pack_path: Path | None) -> dict[str, Any]:
    try:
        return load_vision_facts_config(pack_path=pack_path) if pack_path is not None else load_vision_facts_config()
    except (FileNotFoundError, OSError, ValueError, VisionFactsConfigError):
        return {"facts_by_id": {}, "step_bindings": {}}


def _extract_step_id(step: Mapping[str, Any]) -> str | None:
    for key in ("id", "step_id"):
        raw = step.get(key)
        if isinstance(raw, str) and raw:
            return raw
    return None


def _pack_path_text(pack_path: Path | None) -> str:
    return str(pack_path) if pack_path is not None else "<default pack>"


def _observability_status(raw: Any, *, step_idx: int, pack_path: Path | None) -> str:
    if raw is None:
        return "observable"
    observability = normalize_observability_status(raw)
    if observability is None:
        allowed = ", ".join(sorted(("observable", "partial", "unobservable")))
        raise ValueError(
            f"pack.steps[{step_idx}].observability must be one of {{{allowed}}}: {_pack_path_text(pack_path)}"
        )
    return observability


def _evidence_requirements(raw: Any, *, step_idx: int, pack_path: Path | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"pack.steps[{step_idx}].evidence_requirements must be a list: {_pack_path_text(pack_path)}"
        )
    out: list[str] = []
    for req_idx, item in enumerate(raw):
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"pack.steps[{step_idx}].evidence_requirements[{req_idx}] must be non-empty string: "
                f"{_pack_path_text(pack_path)}"
            )
        if item not in STEP_EVIDENCE_REQUIREMENT_VALUES:
            allowed = ", ".join(sorted(STEP_EVIDENCE_REQUIREMENT_VALUES))
            raise ValueError(
                f"pack.steps[{step_idx}].evidence_requirements[{req_idx}] must be one of "
                f"{{{allowed}}}: {_pack_path_text(pack_path)}"
            )
        if item in out:
            continue
        out.append(item)
    return tuple(out)


def _ui_targets(raw: Any, *, step_idx: int, pack_path: Path | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"pack.steps[{step_idx}].ui_targets must be a list: {_pack_path_text(pack_path)}")
    out: list[str] = []
    for target_idx, target in enumerate(raw):
        if not isinstance(target, str) or not target:
            raise ValueError(
                f"pack.steps[{step_idx}].ui_targets[{target_idx}] must be non-empty string: "
                f"{_pack_path_text(pack_path)}"
            )
        if target in out:
            continue
        out.append(target)
    return tuple(out)


def _overlay_enabled(raw: Any, *, step_idx: int, pack_path: Path | None) -> bool:
    if raw is None:
        return True
    if not isinstance(raw, bool):
        raise ValueError(f"pack.steps[{step_idx}].overlay_enabled must be a bool: {_pack_path_text(pack_path)}")
    return raw


def _dedupe_non_empty_strings(raw: Any) -> tuple[str, ...]:
    return _dedupe(_iter_strings(raw))


def _iter_strings(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if not isinstance(raw, Iterable) or isinstance(raw, (bytes, Mapping)):
        return ()
    return tuple(item for item in raw if isinstance(item, str) and item)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def _collect_gate_var_refs(
    step_id: str,
    *,
    precondition_gates: Mapping[str, Any],
    completion_gates: Mapping[str, Any],
) -> tuple[str, ...]:
    refs: list[str] = []
    for gate_map in (precondition_gates, completion_gates):
        for rule in _coerce_rules(gate_map.get(step_id)):
            ref = _normalize_rule_var_ref(rule.get("var"))
            if ref is None:
                continue
            refs.append(ref)
    return _dedupe(refs)


def _coerce_rules(raw: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(raw, Mapping):
        return (raw,)
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _normalize_rule_var_ref(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    value = raw
    if value.startswith("payload.vars."):
        value = value[len("payload.vars.") :]
    elif value.startswith("vars."):
        value = value[len("vars.") :]
    elif "." in value:
        return None
    if not value:
        return None
    return f"VARS.{value}"


def _collect_step_vision_fact_refs(
    step_id: str,
    vision_config: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    binding = vision_config.get("step_bindings", {}).get(step_id, {})
    binding_fact_ids: tuple[str, ...] = ()
    if isinstance(binding, Mapping):
        binding_fact_ids = _dedupe((*_iter_strings(binding.get("all_of")), *_iter_strings(binding.get("any_of"))))

    support_fact_ids: list[str] = list(binding_fact_ids)
    facts_by_id = vision_config.get("facts_by_id")
    if isinstance(facts_by_id, Mapping):
        for fact_id, fact in facts_by_id.items():
            if not isinstance(fact_id, str) or not isinstance(fact, Mapping):
                continue
            steps = fact.get("steps")
            if not isinstance(steps, (list, tuple)) or step_id not in steps:
                continue
            support_fact_ids.append(fact_id)

    all_fact_ids = _dedupe(support_fact_ids)
    completion_fact_ids = _dedupe(binding_fact_ids)
    return (
        tuple(f"VISION_FACTS.{fact_id}" for fact_id in all_fact_ids),
        tuple(f"VISION_FACTS.{fact_id}" for fact_id in completion_fact_ids),
    )


def _completion_predicates(
    step_id: str,
    *,
    completion_gates: Mapping[str, Any],
    vision_facts: tuple[str, ...],
) -> tuple[str, ...]:
    predicates: list[str] = []
    completion_rules = _coerce_rules(completion_gates.get(step_id))
    if completion_rules:
        predicates.append(f"GATES.{step_id}.completion")
        for rule in completion_rules:
            condition = format_gate_rule_condition(rule)
            if condition:
                predicates.append(condition)
    predicates.extend(f"{fact_ref}==seen" for fact_ref in vision_facts)
    return _dedupe(predicates)


def _signal_quality_requirement(
    *,
    vision_facts: tuple[str, ...],
    vision_config: Mapping[str, Any],
    harness_map: Mapping[str, Any],
) -> SignalQualityRequirement:
    min_confidence = _optional_float(harness_map.get("min_confidence"))
    freshness_ms = _optional_int(harness_map.get("freshness_ms"))
    if freshness_ms is None:
        freshness_ms = _freshness_from_vision_facts(vision_facts, vision_config)
    return SignalQualityRequirement(min_confidence=min_confidence, freshness_ms=freshness_ms)


def _freshness_from_vision_facts(
    vision_facts: tuple[str, ...],
    vision_config: Mapping[str, Any],
) -> int | None:
    facts_by_id = vision_config.get("facts_by_id", {})
    if not isinstance(facts_by_id, Mapping):
        return None
    values: list[int] = []
    for fact_ref in vision_facts:
        fact_id = fact_ref.removeprefix("VISION_FACTS.")
        fact = facts_by_id.get(fact_id)
        if not isinstance(fact, Mapping):
            continue
        expires_after_ms = fact.get("expires_after_ms")
        if isinstance(expires_after_ms, int) and not isinstance(expires_after_ms, bool):
            values.append(expires_after_ms)
    if not values:
        return None
    return max(values)


def _latch_semantics(
    *,
    vision_facts: tuple[str, ...],
    vision_config: Mapping[str, Any],
    harness_map: Mapping[str, Any],
) -> str:
    raw = harness_map.get("latch_semantics")
    if isinstance(raw, str) and raw:
        return raw
    facts_by_id = vision_config.get("facts_by_id", {})
    if isinstance(facts_by_id, Mapping):
        for fact_ref in vision_facts:
            fact_id = fact_ref.removeprefix("VISION_FACTS.")
            fact = facts_by_id.get(fact_id)
            if isinstance(fact, Mapping) and fact.get("sticky") is True:
                return "sticky_visual_completion"
    return _DEFAULT_LATCH_SEMANTICS


def _recovery_policy(
    *,
    requires_visual_confirmation: bool,
    recovery_targets: tuple[str, ...],
    harness_map: Mapping[str, Any],
) -> RecoveryActionPolicy:
    raw = harness_map.get("recovery_policy")
    if isinstance(raw, Mapping):
        kind = raw.get("kind")
        if isinstance(kind, str) and kind:
            return RecoveryActionPolicy(
                kind=kind,
                allowed_targets=_dedupe_non_empty_strings(raw.get("allowed_targets")),
                fallback_message=raw.get("fallback_message") if isinstance(raw.get("fallback_message"), str) else None,
            )
    if requires_visual_confirmation:
        return RecoveryActionPolicy(kind="visual_confirmation", allowed_targets=recovery_targets)
    if recovery_targets:
        return RecoveryActionPolicy(kind="retry_allowed_targets", allowed_targets=recovery_targets)
    return RecoveryActionPolicy(kind="manual_guidance")


def _optional_float(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _optional_int(raw: Any) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw
    return None


__all__ = [
    "load_step_harness_specs",
    "step_fallback_profiles_from_specs",
    "step_signal_profiles_from_specs",
]
