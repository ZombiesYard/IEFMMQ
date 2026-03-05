"""
Gating rule engine for procedure preconditions and completion conditions.

Supported DSL operations (v1):
- var_gte:      {"op": "var_gte", "var": "path.to.value", "value": number}
- arg_in_range: {"op": "arg_in_range", "var": "path.to.value", "min": a, "max": b}
- flag_true:    {"op": "flag_true", "var": "path.to.flag"}
- time_since:   {"op": "time_since", "tag": "string", "at_least": seconds}

Input observations are dicts (matching Observation.to_dict()) and can be
evaluated using dot-delimited paths into the payload or top-level fields.
Stable vars are preferred when available (vars.<key> or bare <key>).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_UNKNOWN_TEXT_VALUES = frozenset({"unknown", "unk", "missing", "n/a", "na"})


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _get_by_path(data: Dict[str, Any], path: str) -> Optional[Any]:
    parts = path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _get_var(data: Dict[str, Any], path: str) -> Optional[Any]:
    # Prefer stable vars when a bare key is used.
    if "." not in path:
        val = _get_by_path(data, f"payload.vars.{path}")
        if val is not None:
            return val
        val = _get_by_path(data, f"vars.{path}")
        if val is not None:
            return val
    # Direct vars.<key> lookup (payload.vars or top-level vars).
    if path.startswith("vars."):
        val = _get_by_path(data, f"payload.{path}")
        if val is not None:
            return val
        val = _get_by_path(data, path)
        if val is not None:
            return val
    return _get_by_path(data, path)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_unknown_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _UNKNOWN_TEXT_VALUES
    return False


def _coerce_flag_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if _is_number(value):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _iter_vars_source_missing(data: Dict[str, Any]) -> List[str]:
    candidates: list[Any] = []
    payload_vars = _get_by_path(data, "payload.vars")
    if isinstance(payload_vars, dict):
        candidates.append(payload_vars.get("vars_source_missing"))
    top_level_vars = data.get("vars")
    if isinstance(top_level_vars, dict):
        candidates.append(top_level_vars.get("vars_source_missing"))

    out: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, str) or not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def _var_key_from_path(path: str) -> Optional[str]:
    if path.startswith("payload.vars."):
        suffix = path[len("payload.vars.") :]
        return suffix if suffix and "." not in suffix else None
    if path.startswith("vars."):
        suffix = path[len("vars.") :]
        return suffix if suffix and "." not in suffix else None
    if "." not in path:
        return path
    return None


def _is_var_source_missing(data: Dict[str, Any], path: str) -> bool:
    var_key = _var_key_from_path(path)
    if var_key is None:
        return False
    return var_key in _iter_vars_source_missing(data)


def _missing_keys(rule: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    return [key for key in keys if key not in rule]


@dataclass
class RuleResult:
    allowed: bool
    reason: Optional[str] = None


class GatingEngine:
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules

    def evaluate(self, observations: Iterable[Dict[str, Any]]) -> RuleResult:
        result, _ = self.evaluate_with_failure_index(observations)
        return result

    def evaluate_with_failure_index(
        self,
        observations: Iterable[Dict[str, Any]],
    ) -> Tuple[RuleResult, Optional[int]]:
        obs_list = list(observations)
        return self.evaluate_with_failure_index_from_history(obs_list)

    def evaluate_with_failure_index_from_history(
        self,
        observations_history: Sequence[Dict[str, Any]],
    ) -> Tuple[RuleResult, Optional[int]]:
        history = observations_history if isinstance(observations_history, list) else list(observations_history)
        if not history:
            return RuleResult(False, "no observations"), None
        latest = history[-1]
        for idx, rule in enumerate(self.rules):
            ok, why = self._eval_rule(rule, latest, history)
            if not ok:
                return RuleResult(False, why), idx
        return RuleResult(True, None), None

    def _eval_rule(
        self, rule: Dict[str, Any], latest: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        op = rule.get("op")
        if op == "var_gte":
            missing = _missing_keys(rule, ("var", "value"))
            if missing:
                return False, f"rule var_gte missing keys: {missing}"
            val = _get_var(latest, rule["var"])
            if _is_var_source_missing(latest, rule["var"]) or _is_unknown_value(val):
                return False, f"{rule['var']} unknown"
            if val is None:
                return False, f"{rule['var']} missing"
            if not _is_number(val) or not _is_number(rule["value"]):
                return False, f"{rule['var']} not numeric"
            if val < rule["value"]:
                return False, f"{rule['var']} ({val}) < {rule['value']}"
            return True, None
        if op == "arg_in_range":
            missing = _missing_keys(rule, ("var", "min", "max"))
            if missing:
                return False, f"rule arg_in_range missing keys: {missing}"
            val = _get_var(latest, rule["var"])
            if _is_var_source_missing(latest, rule["var"]) or _is_unknown_value(val):
                return False, f"{rule['var']} unknown"
            if val is None:
                return False, f"{rule['var']} missing"
            if not _is_number(val) or not _is_number(rule["min"]) or not _is_number(rule["max"]):
                return False, f"{rule['var']} not numeric"
            if val < rule["min"] or val > rule["max"]:
                return False, f"{rule['var']} ({val}) not in [{rule['min']},{rule['max']}]"
            return True, None
        if op == "flag_true":
            missing = _missing_keys(rule, ("var",))
            if missing:
                return False, f"rule flag_true missing keys: {missing}"
            val = _get_var(latest, rule["var"])
            if _is_var_source_missing(latest, rule["var"]) or _is_unknown_value(val):
                return False, f"{rule['var']} unknown"
            if val is None:
                return False, f"{rule['var']} missing"
            bool_val = _coerce_flag_bool(val)
            if bool_val is None:
                return False, f"{rule['var']} not boolean"
            if not bool_val:
                return False, f"{rule['var']} not true"
            return True, None
        if op == "time_since":
            missing = _missing_keys(rule, ("tag", "at_least"))
            if missing:
                return False, f"rule time_since missing keys: {missing}"
            tag = rule["tag"]
            at_least = rule["at_least"]
            if not _is_number(at_least):
                return False, "time_since at_least not numeric"
            last_ts = None
            for obs in reversed(history[:-1]):
                tags = obs.get("tags") or []
                if tag in tags:
                    last_ts = _parse_time(obs["timestamp"])
                    break
            if last_ts is None:
                return False, f"tag {tag} never seen"
            now_ts = _parse_time(latest["timestamp"])
            delta = (now_ts - last_ts).total_seconds()
            if delta < at_least:
                return False, f"time_since {tag} {delta:.1f}s<{at_least}"
            return True, None
        return False, f"unknown op {op}"

