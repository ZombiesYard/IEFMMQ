import os
from collections import deque
from collections.abc import Iterable as IterableABC
from pathlib import Path
import time
from typing import Any, Mapping

import pytest
import yaml

from adapters.pack_gates import load_pack_gate_config
from adapters.step_inference import (
    StepInferenceResult,
    extract_recent_ui_targets,
    format_gate_rule_condition,
    infer_step_id,
    load_pack_steps,
)


BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
_SYNTHETIC_SUPPORTED_OPS = frozenset({"flag_true", "var_gte", "arg_in_range"})


def _step_ids(pack_steps: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for step in pack_steps:
        step_id = step.get("id")
        if isinstance(step_id, str) and step_id:
            out.append(step_id)
    return out


def _step_index(pack_steps: list[dict[str, Any]]) -> dict[str, int]:
    return {step_id: idx for idx, step_id in enumerate(_step_ids(pack_steps))}


def _step_by_id(pack_steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for step in pack_steps:
        step_id = step.get("id")
        if isinstance(step_id, str) and step_id and step_id not in out:
            out[step_id] = step
    return out


def _extract_var_key(rule: Mapping[str, Any]) -> str | None:
    raw = rule.get("var")
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("payload.vars."):
        suffix = raw[len("payload.vars.") :]
        return suffix if suffix else None
    if raw.startswith("vars."):
        suffix = raw[len("vars.") :]
        return suffix if suffix else None
    if "." in raw:
        return None
    return raw


def _rule_pass_value(rule: Mapping[str, Any]) -> Any:
    op = rule.get("op")
    if op == "flag_true":
        return True
    if op == "var_gte":
        value = rule.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 1
        return value + 5
    if op == "arg_in_range":
        min_value = rule.get("min")
        max_value = rule.get("max")
        if (
            isinstance(min_value, bool)
            or isinstance(max_value, bool)
            or not isinstance(min_value, (int, float))
            or not isinstance(max_value, (int, float))
        ):
            return 0
        midpoint = (float(min_value) + float(max_value)) / 2.0
        if midpoint.is_integer():
            return int(midpoint)
        return midpoint
    return True


def _rule_fail_value(rule: Mapping[str, Any]) -> Any:
    op = rule.get("op")
    if op == "flag_true":
        return False
    if op == "var_gte":
        value = rule.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        return value - 1
    if op == "arg_in_range":
        min_value = rule.get("min")
        if isinstance(min_value, bool) or not isinstance(min_value, (int, float)):
            return -1
        return min_value - 1
    return False


def _is_supported_rule_for_synthetic_case(rule: Mapping[str, Any]) -> bool:
    op = rule.get("op")
    return isinstance(op, str) and op in _SYNTHETIC_SUPPORTED_OPS


def _filter_supported_gate_map(pack_gates: Mapping[str, Any]) -> dict[str, dict[str, tuple[dict[str, Any], ...]]]:
    out: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {
        "precondition_gates": {},
        "completion_gates": {},
    }
    for gate_type in ("precondition_gates", "completion_gates"):
        gate_map = pack_gates.get(gate_type, {})
        if not isinstance(gate_map, Mapping):
            out[gate_type] = {}
            continue
        filtered_map: dict[str, tuple[dict[str, Any], ...]] = {}
        for step_id, rules in gate_map.items():
            if not isinstance(step_id, str) or not step_id:
                continue
            if isinstance(rules, Mapping):
                rules_iter = [rules]
            elif isinstance(rules, (list, tuple)):
                rules_iter = list(rules)
            elif isinstance(rules, IterableABC) and not isinstance(rules, (str, bytes)):
                rules_iter = list(rules)
            else:
                continue
            filtered_map[step_id] = tuple(
                dict(rule)
                for rule in rules_iter
                if isinstance(rule, Mapping) and _is_supported_rule_for_synthetic_case(rule)
            )
        out[gate_type] = filtered_map
    return out


def _build_baseline_vars(pack_gates: Mapping[str, Any]) -> dict[str, Any]:
    vars_map: dict[str, Any] = {}
    for gate_type in ("precondition_gates", "completion_gates"):
        gate_map = pack_gates.get(gate_type, {})
        if not isinstance(gate_map, Mapping):
            continue
        for rules in gate_map.values():
            for rule in rules:
                key = _extract_var_key(rule)
                if not key:
                    continue
                vars_map[key] = _rule_pass_value(rule)
    return vars_map


def _step_observability(step_id: str, step_meta: Mapping[str, Mapping[str, Any]]) -> str | None:
    step = step_meta.get(step_id, {})
    value = step.get("observability")
    return value if isinstance(value, str) else None


def _first_ui_target(step_id: str, step_meta: Mapping[str, Mapping[str, Any]]) -> str | None:
    step = step_meta.get(step_id, {})
    targets = step.get("ui_targets")
    if not isinstance(targets, list):
        return None
    for item in targets:
        if isinstance(item, str) and item:
            return item
    return None


def _is_observability_hold_step(
    step_id: str,
    *,
    pack_gates: Mapping[str, Any],
    step_meta: Mapping[str, Mapping[str, Any]],
) -> bool:
    obs = _step_observability(step_id, step_meta)
    completion_map = pack_gates.get("completion_gates", {})
    comp_rules = completion_map.get(step_id, ()) if isinstance(completion_map, Mapping) else ()
    return obs in {"partially", "unknown"} and len(comp_rules) == 0


def _completion_var_keys_after(
    step_id: str,
    *,
    step_ids: list[str],
    step_index: Mapping[str, int],
    pack_gates: Mapping[str, Any],
) -> set[str]:
    idx = step_index[step_id]
    out: set[str] = set()
    completion_map = pack_gates.get("completion_gates", {})
    if not isinstance(completion_map, Mapping):
        return out
    for later_id in step_ids[idx + 1 :]:
        for rule in completion_map.get(later_id, ()):
            key = _extract_var_key(rule)
            if key:
                out.add(key)
    return out


@pytest.fixture(scope="session")
def pack_ctx() -> dict[str, Any]:
    pack_steps = load_pack_steps(PACK_PATH)
    pack_gates = load_pack_gate_config(PACK_PATH)
    synthetic_pack_gates = _filter_supported_gate_map(pack_gates)
    step_ids = _step_ids(pack_steps)
    return {
        "pack_steps": pack_steps,
        "pack_gates": synthetic_pack_gates,
        "step_ids": step_ids,
        "step_index": _step_index(pack_steps),
        "step_meta": _step_by_id(pack_steps),
        "baseline_vars": _build_baseline_vars(synthetic_pack_gates),
    }


def _build_step_blocking_scenario(step_id: str, ctx: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    step_ids: list[str] = ctx["step_ids"]
    step_index: Mapping[str, int] = ctx["step_index"]
    step_meta: Mapping[str, Mapping[str, Any]] = ctx["step_meta"]
    pack_gates: Mapping[str, Any] = ctx["pack_gates"]
    vars_map = dict(ctx["baseline_vars"])
    recent_ui_targets: list[str] = []

    for prev_step_id in step_ids[: step_index[step_id]]:
        if not _is_observability_hold_step(prev_step_id, pack_gates=pack_gates, step_meta=step_meta):
            continue
        target = _first_ui_target(prev_step_id, step_meta)
        if target:
            recent_ui_targets.append(target)

    completion_map = pack_gates.get("completion_gates", {})
    completion_rules = completion_map.get(step_id, ()) if isinstance(completion_map, Mapping) else ()
    if completion_rules:
        failing_rule = completion_rules[0]
        key = _extract_var_key(failing_rule)
        if key:
            vars_map[key] = _rule_fail_value(failing_rule)
        return vars_map, recent_ui_targets, format_gate_rule_condition(failing_rule)

    if _is_observability_hold_step(step_id, pack_gates=pack_gates, step_meta=step_meta):
        hidden_keys = sorted(
            _completion_var_keys_after(
                step_id,
                step_ids=step_ids,
                step_index=step_index,
                pack_gates=pack_gates,
            )
        )
        if hidden_keys:
            vars_map["vars_source_missing"] = hidden_keys
    target = _first_ui_target(step_id, step_meta)
    if target:
        recent_ui_targets = [item for item in recent_ui_targets if item != target]
    return vars_map, recent_ui_targets, None


def test_infer_step_pack_gate_driven_blocking_scenarios_cover_all_pack_steps(pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = pack_ctx["pack_gates"]
    step_ids: list[str] = pack_ctx["step_ids"]

    for step_id in step_ids:
        vars_map, recent_ui_targets, expected_condition = _build_step_blocking_scenario(step_id, pack_ctx)

        result = infer_step_id(
            pack_steps,
            vars_map,
            recent_ui_targets,
            precondition_gates=pack_gates["precondition_gates"],
            completion_gates=pack_gates["completion_gates"],
            pack_path=PACK_PATH,
        )

        assert result.inferred_step_id == step_id
        if expected_condition is None:
            assert result.missing_conditions == ()
        else:
            assert expected_condition in result.missing_conditions


def test_infer_step_mvp_progresses_to_s03_when_apu_not_ready(pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = pack_ctx["pack_gates"]
    result = infer_step_id(
        pack_steps,
        {"power_available": True, "apu_on": True, "apu_ready": False},
        ["apu_switch"],
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=PACK_PATH,
    )

    assert result.inferred_step_id == "S03"
    assert "vars.apu_ready==true" in result.missing_conditions


def test_infer_step_accepts_iterable_recent_ui_targets(pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = pack_ctx["pack_gates"]
    result = infer_step_id(
        pack_steps,
        {"power_available": True, "apu_on": True, "apu_ready": False},
        deque(["apu_switch"]),
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=PACK_PATH,
    )

    assert result.inferred_step_id == "S03"
    assert "vars.apu_ready==true" in result.missing_conditions


def test_infer_step_mvp_progresses_to_s07_without_missing_conditions(pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = pack_ctx["pack_gates"]
    result = infer_step_id(
        pack_steps,
        {
            "power_available": True,
            "apu_ready": True,
            "engine_crank_right": True,
            "rpm_r": 65,
            "bleed_air_norm": True,
        },
        ["bleed_air_knob", "eng_crank_switch"],
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=PACK_PATH,
    )

    assert result.inferred_step_id == "S07"
    assert result.missing_conditions == ()


def test_infer_step_is_robust_on_invalid_inputs(pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = pack_ctx["pack_gates"]
    step_ids: list[str] = pack_ctx["step_ids"]
    result = infer_step_id(
        pack_steps,
        vars_map={"power_available": "unknown"},
        recent_ui_targets=["apu_switch", "", 1],  # type: ignore[list-item]
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=PACK_PATH,
    )
    assert isinstance(result, StepInferenceResult)
    assert result.inferred_step_id in step_ids


def test_extract_recent_ui_targets_prefers_direct_recent_ui_targets() -> None:
    context = {
        "recent_ui_targets": ["eng_crank_switch", "eng_crank_switch", "apu_switch"],
        "recent_actions": {"recent_buttons": ["fire_test_switch"]},
    }
    assert extract_recent_ui_targets(context) == ["eng_crank_switch", "apu_switch"]


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _bump_mtime(path: Path) -> None:
    now = time.time()
    current = path.stat().st_mtime
    bumped = max(current + 2.0, now + 2.0)
    os.utime(path, (bumped, bumped))


def _registry_payload(first_short_explanation: str) -> dict:
    steps = []
    for i in range(1, 26):
        sid = f"S{i:02d}"
        short = first_short_explanation if i == 1 else f"step-{sid}"
        steps.append(
            {
                "step_id": sid,
                "phase": "P1",
                "official_description": f"Official {sid}",
                "short_explanation": short,
                "source_chunk_refs": [f"doc/chunk:{i}-{i}"],
            }
        )
    return {"schema_version": "v1", "steps": steps}


def test_load_pack_steps_cache_invalidates_when_registry_changes(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    registry_path = tmp_path / "step_registry.yaml"

    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "metadata": {"step_registry_path": "step_registry.yaml"},
            "steps": [{"id": "S01", "marker": "fallback"}],
        },
    )
    _write_yaml(registry_path, _registry_payload("first"))

    first = load_pack_steps(pack_path)
    assert first[0]["short_explanation"] == "first"

    _write_yaml(registry_path, _registry_payload("updated"))
    _bump_mtime(registry_path)

    second = load_pack_steps(pack_path)
    assert second[0]["short_explanation"] == "updated"


def test_load_pack_steps_cache_invalidates_when_pack_fallback_changes(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "steps": [{"id": "S01", "marker": "first"}],
        },
    )

    first = load_pack_steps(pack_path)
    assert first[0]["marker"] == "first"

    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "steps": [{"id": "S01", "marker": "updated"}],
        },
    )
    _bump_mtime(pack_path)

    second = load_pack_steps(pack_path)
    assert second[0]["marker"] == "updated"


def test_load_pack_steps_falls_back_when_pack_metadata_registry_path_is_invalid(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "metadata": {"step_registry_path": 123},
            "steps": [{"id": "S01", "marker": "from_pack"}],
        },
    )

    loaded = load_pack_steps(pack_path)
    assert loaded[0]["id"] == "S01"
    assert loaded[0]["marker"] == "from_pack"


def test_load_pack_steps_does_not_inject_none_when_pack_step_field_absent(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    registry_path = tmp_path / "step_registry.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "metadata": {"step_registry_path": "step_registry.yaml"},
            "steps": [{"id": "S01"}],
        },
    )
    _write_yaml(registry_path, _registry_payload("first"))

    loaded = load_pack_steps(pack_path)
    first = loaded[0]
    assert first["id"] == "S01"
    assert "ui_targets" not in first
    assert "observability" not in first


def test_infer_step_preserves_caller_precondition_map_when_completion_is_missing() -> None:
    pack_steps = [{"id": "S01"}, {"id": "S02"}]
    precondition_gates = {
        "S01": (
            {
                "op": "flag_true",
                "var": "vars.custom_ready",
                "reason_code": "s01_requires_custom_ready",
            },
        ),
        "S02": (),
    }
    vars_map = {
        "custom_ready": False,
        "battery_on": True,
        "l_gen_on": True,
        "r_gen_on": True,
    }

    result = infer_step_id(
        pack_steps,
        vars_map,
        [],
        precondition_gates=precondition_gates,
        completion_gates=None,
        pack_path=PACK_PATH,
    )

    assert result.inferred_step_id == "S01"
    assert "vars.custom_ready==true" in result.missing_conditions
