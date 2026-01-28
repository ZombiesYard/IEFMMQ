"""
Gating rule engine for procedure preconditions and completion conditions.

Supported DSL operations (v1):
- var_gte:      {"op": "var_gte", "var": "path.to.value", "value": number}
- arg_in_range: {"op": "arg_in_range", "var": "path.to.value", "min": a, "max": b}
- flag_true:    {"op": "flag_true", "var": "path.to.flag"}
- time_since:   {"op": "time_since", "tag": "string", "at_least": seconds}

Input observations are dicts (matching Observation.to_dict()) and can be
evaluated using dot-delimited paths into the payload or top-level fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _get_var(data: Dict[str, Any], path: str) -> Optional[Any]:
    parts = path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


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
            val = _get_var(latest, rule["var"])
            if val is None or val < rule["value"]:
                return False, f"{rule['var']}<{rule['value']}"
            return True, None
        if op == "arg_in_range":
            val = _get_var(latest, rule["var"])
            if val is None or val < rule["min"] or val > rule["max"]:
                return False, f"{rule['var']} not in [{rule['min']},{rule['max']}]"
            return True, None
        if op == "flag_true":
            val = _get_var(latest, rule["var"])
            if not bool(val):
                return False, f"{rule['var']} not true"
            return True, None
        if op == "time_since":
            tag = rule["tag"]
            at_least = rule["at_least"]
            last_ts = None
            for obs in reversed(history):
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

