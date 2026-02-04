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
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
        obs_list = list(observations)
        if not obs_list:
            return RuleResult(False, "no observations")
        latest = obs_list[-1]
        for rule in self.rules:
            ok, why = self._eval_rule(rule, latest, obs_list)
            if not ok:
                return RuleResult(False, why)
        return RuleResult(True, None)

    def _eval_rule(
        self, rule: Dict[str, Any], latest: Dict[str, Any], history: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        op = rule.get("op")
        if op == "var_gte":
            missing = _missing_keys(rule, ("var", "value"))
            if missing:
                return False, f"rule var_gte missing keys: {missing}"
            val = _get_var(latest, rule["var"])
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
            if val is None:
                return False, f"{rule['var']} missing"
            if not bool(val):
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

