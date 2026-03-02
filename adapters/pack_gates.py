"""
Pack-level precondition/completion gates for deterministic help context.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from core.gating import GatingEngine

_GATE_FIELD_NAMES = ("precondition_gates", "completion_gates")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_pack_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "pack.yaml"


def _step_sort_key(step_id: str) -> tuple[int, str]:
    if step_id.startswith("S") and step_id[1:].isdigit():
        return int(step_id[1:]), step_id
    return 10_000, step_id


def _normalize_rules_map(raw: Any, *, field_name: str, pack_path: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping in pack: {pack_path}")

    out: dict[str, tuple[dict[str, Any], ...]] = {}
    for step_id, rules_raw in raw.items():
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"{field_name} step id must be non-empty string in pack: {pack_path}")
        if rules_raw is None:
            out[step_id] = ()
            continue
        if not isinstance(rules_raw, list):
            raise ValueError(f"{field_name}.{step_id} must be a list in pack: {pack_path}")

        normalized_rules: list[dict[str, Any]] = []
        for idx, rule_raw in enumerate(rules_raw):
            if not isinstance(rule_raw, Mapping):
                raise ValueError(f"{field_name}.{step_id}[{idx}] must be a mapping in pack: {pack_path}")
            rule = dict(rule_raw)
            op = rule.get("op")
            if not isinstance(op, str) or not op:
                raise ValueError(f"{field_name}.{step_id}[{idx}].op must be non-empty string in pack: {pack_path}")
            normalized_rules.append(rule)
        out[step_id] = tuple(normalized_rules)
    return out


@lru_cache(maxsize=8)
def _load_pack_gate_config_cached(resolved_pack_path: str) -> dict[str, dict[str, tuple[dict[str, Any], ...]]]:
    pack_path = Path(resolved_pack_path)
    data = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"pack must be a YAML mapping: {pack_path}")

    out: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {}
    for field_name in _GATE_FIELD_NAMES:
        out[field_name] = _normalize_rules_map(data.get(field_name), field_name=field_name, pack_path=pack_path)
    return out


def load_pack_gate_config(pack_path: str | Path | None = None) -> dict[str, dict[str, tuple[dict[str, Any], ...]]]:
    path = Path(pack_path) if pack_path else _default_pack_path()
    resolved = path.resolve()
    loaded = _load_pack_gate_config_cached(str(resolved))
    return {
        field_name: {
            step_id: tuple(dict(rule) for rule in rules)
            for step_id, rules in loaded[field_name].items()
        }
        for field_name in _GATE_FIELD_NAMES
    }


def _default_reason_code(step_id: str, gate_type: str, rule: Mapping[str, Any], index: int) -> str:
    op = rule.get("op")
    op_text = str(op) if isinstance(op, str) and op else "rule"
    key = rule.get("var")
    if not isinstance(key, str) or not key:
        key = rule.get("tag")
    key_text = str(key).replace(".", "_") if isinstance(key, str) and key else f"rule_{index + 1}"
    return f"{step_id.lower()}_{gate_type}_{op_text}_{key_text}"


def _evaluate_gate_rules(
    *,
    step_id: str,
    gate_type: str,
    rules: Iterable[Mapping[str, Any]],
    observations: list[dict[str, Any]],
) -> tuple[bool, str, str | None]:
    normalized_rules = [dict(rule) for rule in rules]
    if not normalized_rules:
        return True, "no_rules", None

    for idx, rule in enumerate(normalized_rules):
        result = GatingEngine([rule]).evaluate(observations)
        if result.allowed:
            continue
        reason_code_raw = rule.get("reason_code")
        reason_code = (
            reason_code_raw.strip()
            if isinstance(reason_code_raw, str) and reason_code_raw.strip()
            else _default_reason_code(step_id, gate_type, rule, idx)
        )
        reason_raw = rule.get("reason")
        reason = reason_raw.strip() if isinstance(reason_raw, str) and reason_raw.strip() else result.reason
        return False, reason_code, reason

    return True, "ok", None


def _coerce_rules_iterable(raw: Any) -> tuple[Mapping[str, Any], ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        return (raw,)
    if isinstance(raw, (str, bytes)):
        return ()
    if not isinstance(raw, Iterable):
        return ()

    out: list[Mapping[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            out.append(item)
    return tuple(out)


def evaluate_pack_gates(
    *,
    observations: Iterable[Mapping[str, Any]],
    precondition_gates: Mapping[str, Iterable[Mapping[str, Any]]],
    completion_gates: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    obs_list = [dict(item) for item in observations if isinstance(item, Mapping)]
    if not obs_list:
        return {}

    step_ids = sorted(
        {step_id for step_id in precondition_gates.keys() if isinstance(step_id, str)}
        | {step_id for step_id in completion_gates.keys() if isinstance(step_id, str)},
        key=_step_sort_key,
    )

    out: dict[str, dict[str, Any]] = {}
    for step_id in step_ids:
        for gate_type, gate_map in (
            ("precondition", precondition_gates),
            ("completion", completion_gates),
        ):
            rules = gate_map.get(step_id, ())
            allowed, reason_code, reason = _evaluate_gate_rules(
                step_id=step_id,
                gate_type=gate_type,
                rules=_coerce_rules_iterable(rules),
                observations=obs_list,
            )
            gate_id = f"{step_id}.{gate_type}"
            out[gate_id] = {
                "step_id": step_id,
                "gate_type": gate_type,
                "status": "allowed" if allowed else "blocked",
                "allowed": allowed,
                "reason_code": reason_code,
                "reason": reason,
            }
    return out


__all__ = ["evaluate_pack_gates", "load_pack_gate_config"]
