"""
Deterministic startup step inference for model fallback and prompt hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from adapters.pack_gates import evaluate_pack_gates, load_pack_gate_config
from core.step_signal_metadata import STEP_OBSERVABILITY_VALUES
from core.step_registry import StepRegistryError, default_step_registry_path, load_step_registry_dicts

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PACK_PATH = _REPO_ROOT / "packs" / "fa18c_startup" / "pack.yaml"
_MAX_MISSING_CONDITIONS = 8
_MAX_RECENT_UI_TARGETS = 24
_UNKNOWN_TEXT_VALUES = frozenset({"unknown", "unk", "missing", "n/a", "na"})


@dataclass(frozen=True)
class _StepProfile:
    step_id: str
    observability: str | None
    ui_targets: tuple[str, ...]


@dataclass(frozen=True)
class StepInferenceResult:
    inferred_step_id: str | None
    missing_conditions: tuple[str, ...]


def _path_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (int(stat.st_mtime_ns), int(stat.st_size))


def load_pack_steps(pack_path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Load procedure steps from canonical step registry when available.
    Falls back to pack.yaml for compatibility.

    Returns an empty list when file/content is invalid so fallback remains safe.
    """
    path = Path(pack_path) if pack_path else _DEFAULT_PACK_PATH
    try:
        registry_path = default_step_registry_path(path)
    except StepRegistryError:
        registry_path = None
    pack_steps_for_merge: tuple[dict[str, Any], ...] = ()
    pack_path_resolved = path.resolve()
    pack_signature = _path_signature(pack_path_resolved)
    if pack_signature is not None:
        pack_steps_for_merge = _load_pack_steps_cached(
            str(pack_path_resolved),
            pack_signature[0],
            pack_signature[1],
        )

    if registry_path is not None:
        registry_path_resolved = registry_path.resolve()
        registry_signature = _path_signature(registry_path_resolved)
        if registry_signature is not None:
            cached_registry_steps = _load_registry_steps_cached(
                str(registry_path_resolved),
                registry_signature[0],
                registry_signature[1],
            )
            if cached_registry_steps:
                registry_steps = [_clone_mapping(step) for step in cached_registry_steps]
                if pack_steps_for_merge:
                    _merge_pack_step_metadata(registry_steps, pack_steps_for_merge)
                return registry_steps

    if pack_signature is None:
        return []
    return [_clone_mapping(step) for step in pack_steps_for_merge]


@lru_cache(maxsize=8)
def _load_registry_steps_cached(
    resolved_registry_path: str,
    registry_mtime_ns: int,
    registry_size_bytes: int,
) -> tuple[dict[str, Any], ...]:
    del registry_mtime_ns, registry_size_bytes  # cache-key components only
    try:
        entries = load_step_registry_dicts(Path(resolved_registry_path), expected_count=25)
    except (StepRegistryError, OSError, ValueError):
        return ()
    return tuple(dict(step) for step in entries)


@lru_cache(maxsize=8)
def _load_pack_steps_cached(
    resolved_pack_path: str,
    pack_mtime_ns: int,
    pack_size_bytes: int,
) -> tuple[dict[str, Any], ...]:
    del pack_mtime_ns, pack_size_bytes  # cache-key components only
    path = Path(resolved_pack_path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return ()
    if not isinstance(data, Mapping):
        return ()
    steps = data.get("steps")
    if not isinstance(steps, list):
        return ()
    out: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, Mapping):
            out.append(dict(step))
    return tuple(out)


def _clone_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        out[key] = _clone_value(value)
    return out


def _clone_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _clone_mapping(value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return [_clone_value(item) for item in value]
    return value


def _merge_pack_step_metadata(
    registry_steps: list[dict[str, Any]],
    pack_steps: Sequence[Mapping[str, Any]],
) -> None:
    pack_by_id: dict[str, Mapping[str, Any]] = {}
    for step in pack_steps:
        step_id = _extract_step_id(step)
        if step_id and step_id not in pack_by_id:
            pack_by_id[step_id] = step
    for step in registry_steps:
        step_id = _extract_step_id(step)
        if not step_id:
            continue
        pack_step = pack_by_id.get(step_id)
        if not isinstance(pack_step, Mapping):
            continue
        for field in ("observability", "evidence_requirements", "ui_targets", "requires_visual_confirmation"):
            if field in step:
                continue
            value = pack_step.get(field)
            step[field] = _clone_value(value)


def normalize_recent_ui_targets(raw: Any, *, max_items: int = _MAX_RECENT_UI_TARGETS) -> list[str]:
    candidates: list[Any] = []
    if isinstance(raw, list):
        candidates = list(raw)
    elif isinstance(raw, tuple):
        candidates = list(raw)
    elif isinstance(raw, Mapping):
        value = raw.get("recent_buttons")
        if isinstance(value, (list, tuple)):
            candidates = list(value)
        elif isinstance(value, str):
            candidates = [value]

    out: list[str] = []
    seen: set[str] = set()
    cap = max(0, int(max_items))
    for item in candidates:
        if not isinstance(item, str) or not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= cap:
            break
    return out


def extract_recent_ui_targets(
    context: Mapping[str, Any] | None,
    *,
    max_items: int = _MAX_RECENT_UI_TARGETS,
) -> list[str]:
    if not isinstance(context, Mapping):
        return []

    direct = normalize_recent_ui_targets(context.get("recent_ui_targets"), max_items=max_items)
    if direct:
        return direct

    signal = normalize_recent_ui_targets(context.get("recent_actions"), max_items=max_items)
    if signal:
        return signal

    raw_recent_actions = context.get("recent_actions")
    if isinstance(raw_recent_actions, list):
        extracted: list[str] = []
        for item in raw_recent_actions:
            if not isinstance(item, Mapping):
                continue
            for key in ("ui_target", "mapped_ui_target", "target"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    extracted.append(value)
            ui_targets = item.get("ui_targets")
            if isinstance(ui_targets, (list, tuple)):
                extracted.extend(t for t in ui_targets if isinstance(t, str) and t)
        signal = normalize_recent_ui_targets(extracted, max_items=max_items)
        if signal:
            return signal

    raw_recent_deltas = context.get("recent_deltas")
    if isinstance(raw_recent_deltas, list):
        extracted = []
        for item in raw_recent_deltas:
            if not isinstance(item, Mapping):
                continue
            for key in ("ui_target", "mapped_ui_target", "target"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    extracted.append(value)
        return normalize_recent_ui_targets(extracted, max_items=max_items)
    return []


def infer_step_id(
    pack_steps: Sequence[Mapping[str, Any]],
    vars_map: Mapping[str, Any] | None,
    recent_ui_targets: Sequence[str] | None,
    *,
    gates: Mapping[str, Any] | None = None,
    precondition_gates: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    completion_gates: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    scenario_profile: str | None = None,
    pack_path: str | Path | None = None,
) -> StepInferenceResult:
    """
    Infer likely blocked step for deterministic help fallback.
    The inference is fully pack-driven:
    - scan ordered steps from pack
    - evaluate precondition/completion gates
    - apply observability-aware hold behavior for partially/unknown steps
    """
    step_profiles = _ordered_step_profiles(pack_steps)
    if not step_profiles:
        return StepInferenceResult(inferred_step_id=None, missing_conditions=())

    vars_safe = vars_map if isinstance(vars_map, Mapping) else {}
    recent = normalize_recent_ui_targets(recent_ui_targets or ())
    recent_set = set(recent)

    effective_pack_path = Path(pack_path).expanduser().resolve() if pack_path else _DEFAULT_PACK_PATH
    vars_source_missing = _extract_source_missing_vars(vars_safe)
    pre_map, comp_map = _resolve_gate_maps(
        precondition_gates=precondition_gates,
        completion_gates=completion_gates,
        scenario_profile=scenario_profile,
        pack_path=effective_pack_path,
    )
    gate_statuses = _resolve_gate_statuses(
        vars_map=vars_safe,
        gates=gates,
        precondition_gates=pre_map,
        completion_gates=comp_map,
    )

    soft_candidate: tuple[str, tuple[str, ...]] | None = None
    for idx, profile in enumerate(step_profiles):
        step_id = profile.step_id
        pre_gate = gate_statuses.get(f"{step_id}.precondition")
        comp_gate = gate_statuses.get(f"{step_id}.completion")
        completion_rules = comp_map.get(step_id, ())
        pre_reason_code = pre_gate.get("reason_code") if isinstance(pre_gate, Mapping) else None
        comp_reason_code = comp_gate.get("reason_code") if isinstance(comp_gate, Mapping) else None
        pre_failed_rule = _pick_failed_rule(pre_map.get(step_id, ()), pre_reason_code)
        comp_failed_rule = _pick_failed_rule(completion_rules, comp_reason_code)

        pre_missing = _missing_conditions_from_gate(
            step_id=step_id,
            gate_type="precondition",
            gate_info=pre_gate,
            failed_rule=pre_failed_rule,
        )
        if _is_gate_blocked(pre_gate):
            if _is_soft_block_from_rule(pre_failed_rule, vars_safe, vars_source_missing):
                if soft_candidate is None:
                    soft_candidate = (step_id, pre_missing)
                if _should_hold_by_observability(profile, completion_rules):
                    if _has_progression_evidence(
                        step_profiles,
                        idx,
                        gate_statuses,
                        comp_map,
                        recent_set,
                        vars_safe,
                        vars_source_missing,
                    ):
                        continue
                    return _result(step_id, ())
            else:
                return _result(step_id, pre_missing)
            continue

        comp_missing = _missing_conditions_from_gate(
            step_id=step_id,
            gate_type="completion",
            gate_info=comp_gate,
            failed_rule=comp_failed_rule,
        )
        if _is_gate_blocked(comp_gate):
            if _is_soft_block_from_rule(comp_failed_rule, vars_safe, vars_source_missing):
                if soft_candidate is None:
                    soft_candidate = (step_id, comp_missing)
            else:
                return _result(step_id, comp_missing)
            continue

        if _should_hold_by_observability(profile, completion_rules):
            if _has_progression_evidence(
                step_profiles,
                idx,
                gate_statuses,
                comp_map,
                recent_set,
                vars_safe,
                vars_source_missing,
            ):
                continue
            return _result(step_id, ())

    if soft_candidate is not None:
        return _result(soft_candidate[0], soft_candidate[1])
    return _result(step_profiles[-1].step_id, ())


def _result(
    inferred_step_id: str | None,
    missing_conditions: Sequence[str],
) -> StepInferenceResult:
    filtered: list[str] = []
    for item in missing_conditions:
        if not isinstance(item, str) or not item:
            continue
        filtered.append(item)
        if len(filtered) >= _MAX_MISSING_CONDITIONS:
            break
    return StepInferenceResult(inferred_step_id=inferred_step_id, missing_conditions=tuple(filtered))


def _ordered_step_profiles(pack_steps: Sequence[Mapping[str, Any]]) -> list[_StepProfile]:
    out: list[_StepProfile] = []
    seen: set[str] = set()
    for step in pack_steps:
        if not isinstance(step, Mapping):
            continue
        step_id = _extract_step_id(step)
        if not step_id or step_id in seen:
            continue
        seen.add(step_id)
        observability_raw = step.get("observability")
        observability = (
            observability_raw
            if isinstance(observability_raw, str) and observability_raw in STEP_OBSERVABILITY_VALUES
            else None
        )
        ui_targets_raw = step.get("ui_targets")
        ui_targets: list[str] = []
        if isinstance(ui_targets_raw, (list, tuple)):
            for item in ui_targets_raw:
                if isinstance(item, str) and item:
                    ui_targets.append(item)
        out.append(_StepProfile(step_id=step_id, observability=observability, ui_targets=tuple(ui_targets)))
    return out


def _extract_step_id(step: Mapping[str, Any]) -> str | None:
    candidates = (step.get("id"), step.get("step_id"))
    for raw in candidates:
        if isinstance(raw, str) and raw:
            return raw
    return None


def _resolve_gate_maps(
    *,
    precondition_gates: Mapping[str, Iterable[Mapping[str, Any]]] | None,
    completion_gates: Mapping[str, Iterable[Mapping[str, Any]]] | None,
    scenario_profile: str | None,
    pack_path: Path,
) -> tuple[dict[str, tuple[dict[str, Any], ...]], dict[str, tuple[dict[str, Any], ...]]]:
    if precondition_gates is None or completion_gates is None:
        loaded = load_pack_gate_config(pack_path, scenario_profile=scenario_profile)
        if precondition_gates is None:
            precondition_gates = loaded.get("precondition_gates", {})
        if completion_gates is None:
            completion_gates = loaded.get("completion_gates", {})
    return _coerce_gate_rules_map(precondition_gates), _coerce_gate_rules_map(completion_gates)


def _coerce_gate_rules_map(
    raw: Mapping[str, Iterable[Mapping[str, Any]]] | None,
) -> dict[str, tuple[dict[str, Any], ...]]:
    out: dict[str, tuple[dict[str, Any], ...]] = {}
    if not isinstance(raw, Mapping):
        return out
    for step_id, rules_raw in raw.items():
        if not isinstance(step_id, str) or not step_id:
            continue
        normalized_rules: list[dict[str, Any]] = []
        if isinstance(rules_raw, Mapping):
            normalized_rules.append(_clone_mapping(rules_raw))
        elif isinstance(rules_raw, Iterable) and not isinstance(rules_raw, (str, bytes)):
            for item in rules_raw:
                if isinstance(item, Mapping):
                    normalized_rules.append(_clone_mapping(item))
        out[step_id] = tuple(normalized_rules)
    return out


def _resolve_gate_statuses(
    *,
    vars_map: Mapping[str, Any],
    gates: Mapping[str, Any] | None,
    precondition_gates: Mapping[str, Iterable[Mapping[str, Any]]],
    completion_gates: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    provided: dict[str, dict[str, Any]] = {}
    if isinstance(gates, Mapping):
        for gate_id, gate_info in gates.items():
            if not isinstance(gate_id, str) or not isinstance(gate_info, Mapping):
                continue
            parts = gate_id.split(".")
            if len(parts) != 2 or parts[1] not in {"precondition", "completion"}:
                continue
            provided[gate_id] = _clone_mapping(gate_info)

        expected_ids = {
            f"{step_id}.precondition"
            for step_id in precondition_gates.keys()
            if isinstance(step_id, str) and step_id
        } | {
            f"{step_id}.completion"
            for step_id in completion_gates.keys()
            if isinstance(step_id, str) and step_id
        }
        if provided and expected_ids and expected_ids.issubset(provided.keys()):
            return provided

    obs = {
        "observation_id": "deterministic-step-inference",
        "timestamp": "1970-01-01T00:00:00+00:00",
        "source": "inference",
        "payload": {"vars": dict(vars_map)},
        "vars": dict(vars_map),
        "version": "v1",
    }
    evaluated = evaluate_pack_gates(
        observations=[obs],
        precondition_gates=precondition_gates,
        completion_gates=completion_gates,
    )
    if provided:
        evaluated.update(provided)
    return evaluated


def _is_gate_blocked(gate_info: Mapping[str, Any] | None) -> bool:
    if not isinstance(gate_info, Mapping):
        return False
    return gate_info.get("status") == "blocked"


def _missing_conditions_from_gate(
    *,
    step_id: str,
    gate_type: str,
    gate_info: Mapping[str, Any] | None,
    failed_rule: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if not _is_gate_blocked(gate_info):
        return ()
    reason_code = gate_info.get("reason_code") if isinstance(gate_info, Mapping) else None
    reason = gate_info.get("reason") if isinstance(gate_info, Mapping) else None

    missing: list[str] = []
    if failed_rule is not None:
        condition = _rule_to_condition_text(failed_rule)
        if condition:
            missing.append(condition)

    if not missing and isinstance(reason_code, str) and reason_code and reason_code not in {"ok", "no_rules"}:
        missing.append(f"gates.{step_id}.{gate_type}.{reason_code}")
    if not missing and isinstance(reason, str) and reason:
        missing.append(reason)
    return tuple(missing)


def _pick_failed_rule(
    gate_rules: Sequence[Mapping[str, Any]],
    reason_code: Any,
) -> Mapping[str, Any] | None:
    if not gate_rules:
        return None
    if isinstance(reason_code, str) and reason_code:
        for rule in gate_rules:
            code = rule.get("reason_code")
            if isinstance(code, str) and code == reason_code:
                return rule
    return gate_rules[0]


def _rule_to_condition_text(rule: Mapping[str, Any]) -> str | None:
    op = rule.get("op")
    if not isinstance(op, str) or not op:
        return None
    if op == "flag_true":
        var = _normalize_var_path(rule.get("var"))
        return f"{var}==true" if var else None
    if op == "var_gte":
        var = _normalize_var_path(rule.get("var"))
        value = _format_rule_number(rule.get("value"))
        if var is None or value is None:
            return None
        return f"{var}>={value}"
    if op == "arg_in_range":
        var = _normalize_var_path(rule.get("var"))
        min_value = _format_rule_number(rule.get("min"))
        max_value = _format_rule_number(rule.get("max"))
        if var is None or min_value is None or max_value is None:
            return None
        return f"{var} in [{min_value},{max_value}]"
    if op == "time_since":
        tag = rule.get("tag")
        at_least = _format_rule_number(rule.get("at_least"))
        if not isinstance(tag, str) or not tag or at_least is None:
            return None
        return f"time_since({tag})>={at_least}"
    return None


def _normalize_var_path(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("payload.vars."):
        return "vars." + raw[len("payload.vars.") :]
    if raw.startswith("payload.") and raw.count(".") >= 2:
        return raw[len("payload.") :]
    if "." not in raw:
        return f"vars.{raw}"
    return raw


def _format_rule_number(raw: Any) -> str | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, float):
        if raw.is_integer():
            return str(int(raw))
        return str(raw)
    return None


def _should_hold_by_observability(
    profile: _StepProfile,
    completion_rules: Sequence[Mapping[str, Any]],
) -> bool:
    if profile.observability not in {"partially", "unknown"}:
        return False
    return len(completion_rules) == 0


def _has_progression_evidence(
    step_profiles: Sequence[_StepProfile],
    idx: int,
    gate_statuses: Mapping[str, Mapping[str, Any]],
    completion_gates: Mapping[str, Sequence[Mapping[str, Any]]],
    recent_set: set[str],
    vars_map: Mapping[str, Any],
    vars_source_missing: set[str],
) -> bool:
    current = step_profiles[idx]
    if current.ui_targets:
        earlier_targets = {target for step in step_profiles[:idx] for target in step.ui_targets}
        for target in current.ui_targets:
            if target in recent_set and target not in earlier_targets:
                return True
    for later in step_profiles[idx + 1 :]:
        completion_gate = gate_statuses.get(f"{later.step_id}.completion")
        reason_code = completion_gate.get("reason_code") if isinstance(completion_gate, Mapping) else None
        failed_rule = _pick_failed_rule(completion_gates.get(later.step_id, ()), reason_code)
        if _is_strong_gate_signal(
            completion_gate,
            failed_rule=failed_rule,
            vars_map=vars_map,
            vars_source_missing=vars_source_missing,
        ):
            return True
    return False


def _is_strong_gate_signal(
    gate_info: Mapping[str, Any] | None,
    *,
    failed_rule: Mapping[str, Any] | None,
    vars_map: Mapping[str, Any],
    vars_source_missing: set[str],
) -> bool:
    if not isinstance(gate_info, Mapping):
        return False
    status = gate_info.get("status")
    reason_code = gate_info.get("reason_code")
    if status == "allowed":
        return reason_code == "ok"
    if status != "blocked":
        return False
    if isinstance(reason_code, str) and reason_code == "no_rules":
        return False
    return not _is_soft_block_from_rule(failed_rule, vars_map, vars_source_missing)


def _is_soft_block_from_rule(
    failed_rule: Mapping[str, Any] | None,
    vars_map: Mapping[str, Any],
    vars_source_missing: set[str],
) -> bool:
    if not isinstance(failed_rule, Mapping):
        return False
    op = failed_rule.get("op")
    var_raw = failed_rule.get("var")
    value, missing = _read_rule_var(vars_map, var_raw, vars_source_missing)
    if missing:
        return True
    if _is_unknown_text(value):
        return True
    if op == "flag_true":
        return _coerce_bool(value) is None
    if op in {"var_gte", "arg_in_range"}:
        return _coerce_number(value) is None
    return False


def _read_rule_var(
    vars_map: Mapping[str, Any],
    var_raw: Any,
    vars_source_missing: set[str],
) -> tuple[Any, bool]:
    if not isinstance(var_raw, str) or not var_raw:
        return None, True
    key = _extract_var_key_from_path(var_raw)
    if key is not None and key in vars_source_missing:
        return None, True
    if key is not None and key in vars_map:
        return vars_map.get(key), False
    if key is not None:
        return None, True
    value = _read_by_path(vars_map, var_raw)
    return value, value is None


def _extract_var_key_from_path(path: str) -> str | None:
    if path.startswith("payload.vars."):
        key = path[len("payload.vars.") :]
        return key if key and "." not in key else None
    if path.startswith("vars."):
        key = path[len("vars.") :]
        return key if key and "." not in key else None
    if "." not in path:
        return path
    return None


def _extract_source_missing_vars(vars_map: Mapping[str, Any]) -> set[str]:
    raw = vars_map.get("vars_source_missing")
    if not isinstance(raw, list):
        return set()
    out: set[str] = set()
    for item in raw:
        if isinstance(item, str) and item:
            out.add(item)
    return out


def _read_by_path(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue
        return None
    return current


def _is_unknown_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _UNKNOWN_TEXT_VALUES


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


__all__ = [
    "StepInferenceResult",
    "extract_recent_ui_targets",
    "infer_step_id",
    "load_pack_steps",
    "normalize_recent_ui_targets",
]
