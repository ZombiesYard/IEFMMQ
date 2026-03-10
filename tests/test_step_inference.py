import os
from collections import deque
from collections.abc import Iterable as IterableABC
from pathlib import Path
import time
from typing import Any, Mapping

import pytest
import yaml

import adapters.step_inference as step_inference_module
from adapters.pack_gates import load_pack_gate_config
from adapters.step_inference import (
    StepInferenceResult,
    extract_recent_ui_targets,
    format_gate_rule_condition,
    infer_step_id,
    load_pack_steps,
    normalize_recent_ui_targets,
)


BASE_DIR = Path(__file__).resolve().parent.parent
REAL_PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
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
    return obs in {"partial", "unobservable"} and len(comp_rules) == 0


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


def _write_synthetic_inference_pack(pack_path: Path) -> None:
    _write_yaml(
        pack_path,
        {
            "pack_id": "synthetic_inference_pack",
            "steps": [
                {
                    "id": "S01",
                    "observability": "observable",
                    "ui_targets": ["battery_switch"],
                },
                {
                    "id": "S02",
                    "observability": "observable",
                    "ui_targets": ["apu_switch"],
                },
                {
                    "id": "S03",
                    "observability": "partial",
                    "ui_targets": ["eng_crank_switch"],
                },
                {
                    "id": "S04",
                    "observability": "observable",
                    "ui_targets": ["throttle_right"],
                },
            ],
            "precondition_gates": {
                "S01": [{"op": "flag_true", "var": "vars.power_available"}],
                "S02": [{"op": "flag_true", "var": "vars.apu_on"}],
                "S03": [{"op": "flag_true", "var": "vars.engine_crank_right"}],
                "S04": [{"op": "flag_true", "var": "vars.bleed_air_norm"}],
            },
            "completion_gates": {
                "S01": [{"op": "flag_true", "var": "vars.battery_on"}],
                "S02": [{"op": "var_gte", "var": "vars.apu_ready_pct", "value": 90}],
                "S03": [],
                "S04": [{"op": "arg_in_range", "var": "vars.rpm_r", "min": 60, "max": 70}],
            },
        },
    )


def _build_pack_ctx(pack_path: Path) -> dict[str, Any]:
    pack_steps = load_pack_steps(pack_path)
    pack_gates = load_pack_gate_config(pack_path)
    synthetic_pack_gates = _filter_supported_gate_map(pack_gates)
    step_ids = _step_ids(pack_steps)
    return {
        "pack_path": pack_path,
        "pack_steps": pack_steps,
        "pack_gates": synthetic_pack_gates,
        "step_ids": step_ids,
        "step_index": _step_index(pack_steps),
        "step_meta": _step_by_id(pack_steps),
        "baseline_vars": _build_baseline_vars(synthetic_pack_gates),
    }


@pytest.fixture()
def synthetic_pack_ctx(tmp_path: Path) -> dict[str, Any]:
    pack_path = tmp_path / "synthetic_inference_pack.yaml"
    _write_synthetic_inference_pack(pack_path)
    return _build_pack_ctx(pack_path)


@pytest.fixture(scope="session")
def real_pack_ctx() -> dict[str, Any]:
    return _build_pack_ctx(REAL_PACK_PATH)


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


def test_infer_step_pack_gate_driven_blocking_scenarios_cover_all_pack_steps(
    synthetic_pack_ctx: Mapping[str, Any],
) -> None:
    pack_steps: list[dict[str, Any]] = synthetic_pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = synthetic_pack_ctx["pack_gates"]
    step_ids: list[str] = synthetic_pack_ctx["step_ids"]
    pack_path: Path = synthetic_pack_ctx["pack_path"]

    for step_id in step_ids:
        vars_map, recent_ui_targets, expected_condition = _build_step_blocking_scenario(step_id, synthetic_pack_ctx)

        result = infer_step_id(
            pack_steps,
            vars_map,
            recent_ui_targets,
            precondition_gates=pack_gates["precondition_gates"],
            completion_gates=pack_gates["completion_gates"],
            pack_path=pack_path,
        )

        assert result.inferred_step_id == step_id
        if expected_condition is None:
            assert result.missing_conditions == ()
        else:
            assert expected_condition in result.missing_conditions


def test_infer_step_mvp_progresses_to_s03_when_apu_not_ready(real_pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = real_pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = real_pack_ctx["pack_gates"]
    result = infer_step_id(
        pack_steps,
        {"power_available": True, "apu_on": True, "apu_ready": False},
        ["apu_switch"],
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=REAL_PACK_PATH,
    )

    assert result.inferred_step_id == "S03"
    assert "vars.apu_ready==true" in result.missing_conditions


def test_infer_step_accepts_iterable_recent_ui_targets(synthetic_pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = synthetic_pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = synthetic_pack_ctx["pack_gates"]
    pack_path: Path = synthetic_pack_ctx["pack_path"]
    result = infer_step_id(
        pack_steps,
        {
            "power_available": True,
            "battery_on": False,
            "apu_on": True,
            "apu_ready_pct": 100,
            "engine_crank_right": True,
            "bleed_air_norm": True,
            "rpm_r": 65,
        },
        deque(["battery_switch"]),
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=pack_path,
    )

    assert result.inferred_step_id == "S01"
    assert "vars.battery_on==true" in result.missing_conditions


def test_infer_step_accepts_mapping_recent_ui_targets(synthetic_pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = synthetic_pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = synthetic_pack_ctx["pack_gates"]
    pack_path: Path = synthetic_pack_ctx["pack_path"]
    result = infer_step_id(
        pack_steps,
        {
            "power_available": True,
            "battery_on": False,
            "apu_on": True,
            "apu_ready_pct": 100,
            "engine_crank_right": True,
            "bleed_air_norm": True,
            "rpm_r": 65,
        },
        {"recent_buttons": ["battery_switch"]},
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=pack_path,
    )

    assert result.inferred_step_id == "S01"
    assert "vars.battery_on==true" in result.missing_conditions


def test_infer_step_mvp_progresses_to_s07_without_missing_conditions(real_pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = real_pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = real_pack_ctx["pack_gates"]
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
        pack_path=REAL_PACK_PATH,
    )

    assert result.inferred_step_id == "S07"
    assert result.missing_conditions == ()


def test_infer_step_holds_s08_until_visual_page_facts_are_seen(real_pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = real_pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = real_pack_ctx["pack_gates"]
    _vars_map, recent_ui_targets, _ = _build_step_blocking_scenario("S08", real_pack_ctx)
    vars_map = dict(real_pack_ctx["baseline_vars"])

    blocked = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets,
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=REAL_PACK_PATH,
    )
    advanced = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets,
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=REAL_PACK_PATH,
        vision_facts=[
            {"fact_id": "fcs_page_visible", "state": "seen"},
            {"fact_id": "bit_page_visible", "state": "seen"},
        ],
    )

    assert blocked.inferred_step_id == "S08"
    assert "vision_facts.fcs_page_visible==seen" in blocked.missing_conditions
    assert advanced.inferred_step_id != "S08"


def test_infer_step_uses_sticky_fcs_reset_fact_to_advance_past_s15(real_pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = real_pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = real_pack_ctx["pack_gates"]
    _vars_map, recent_ui_targets, _ = _build_step_blocking_scenario("S15", real_pack_ctx)
    vars_map = dict(real_pack_ctx["baseline_vars"])
    prior_vision_facts = [
        {"fact_id": "fcs_page_visible", "state": "seen"},
        {"fact_id": "bit_page_visible", "state": "seen"},
    ]

    blocked = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets,
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=REAL_PACK_PATH,
        vision_facts=prior_vision_facts,
    )
    advanced = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets,
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=REAL_PACK_PATH,
        vision_facts=[*prior_vision_facts, {"fact_id": "fcs_reset_seen", "state": "seen"}],
    )

    assert blocked.inferred_step_id == "S15"
    assert "vision_facts.fcs_reset_seen==seen" in blocked.missing_conditions
    assert advanced.inferred_step_id != "S15"


def test_infer_step_uses_sticky_fcs_bit_facts_to_advance_past_s18(real_pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = real_pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = real_pack_ctx["pack_gates"]
    _vars_map, recent_ui_targets, _ = _build_step_blocking_scenario("S18", real_pack_ctx)
    vars_map = dict(real_pack_ctx["baseline_vars"])
    prior_vision_facts = [
        {"fact_id": "fcs_page_visible", "state": "seen"},
        {"fact_id": "bit_page_visible", "state": "seen"},
        {"fact_id": "fcs_reset_seen", "state": "seen"},
        {"fact_id": "takeoff_trim_seen", "state": "seen"},
    ]

    blocked = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets,
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=REAL_PACK_PATH,
        vision_facts=prior_vision_facts,
    )
    advanced = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets,
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=REAL_PACK_PATH,
        vision_facts=[
            *prior_vision_facts,
            {"fact_id": "fcs_bit_interaction_seen", "state": "seen"},
            {"fact_id": "fcs_bit_result_visible", "state": "seen"},
        ],
    )

    assert blocked.inferred_step_id == "S18"
    assert "vision_facts.fcs_bit_interaction_seen==seen" in blocked.missing_conditions
    assert advanced.inferred_step_id != "S18"


def test_infer_step_uses_pack_specific_vision_fact_bindings(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    vision_facts_path = tmp_path / "configs" / "custom_vision_facts.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "custom_pack",
            "metadata": {"vision_facts_path": "configs/custom_vision_facts.yaml"},
            "steps": [
                {
                    "id": "S08",
                    "observability": "observable",
                    "ui_targets": ["custom_target"],
                }
            ],
        },
    )
    _write_yaml(
        vision_facts_path,
        {
            "schema_version": "v1",
            "layout_id": "custom_layout",
            "facts": [
                {
                    "fact_id": "fcs_page_visible",
                    "sticky": False,
                    "expires_after_ms": 2000,
                }
            ],
            "step_bindings": {},
        },
    )

    result = infer_step_id(
        load_pack_steps(pack_path),
        vars_map={},
        recent_ui_targets=[],
        pack_path=pack_path,
    )

    assert result.inferred_step_id == "S08"
    assert result.missing_conditions == ()


def test_infer_step_is_robust_on_invalid_inputs(synthetic_pack_ctx: Mapping[str, Any]) -> None:
    pack_steps: list[dict[str, Any]] = synthetic_pack_ctx["pack_steps"]
    pack_gates: Mapping[str, Any] = synthetic_pack_ctx["pack_gates"]
    step_ids: list[str] = synthetic_pack_ctx["step_ids"]
    pack_path: Path = synthetic_pack_ctx["pack_path"]
    result = infer_step_id(
        pack_steps,
        vars_map={"power_available": "unknown"},
        recent_ui_targets=["apu_switch", "", 1],  # type: ignore[list-item]
        precondition_gates=pack_gates["precondition_gates"],
        completion_gates=pack_gates["completion_gates"],
        pack_path=pack_path,
    )
    assert isinstance(result, StepInferenceResult)
    assert result.inferred_step_id in step_ids


def test_infer_step_reads_nested_payload_vars_before_marking_soft_block() -> None:
    pack_steps = [
        {"id": "S01", "observability": "partial", "ui_targets": ["s01_btn"]},
        {"id": "S02", "observability": "observable", "ui_targets": ["s02_btn"]},
    ]
    precondition_gates = {
        "S01": (
            {
                "op": "flag_true",
                "var": "payload.vars.power_available",
                "reason_code": "s01_requires_power",
            },
        ),
        "S02": (),
    }
    completion_gates = {
        "S01": (),
        "S02": (
            {
                "op": "var_gte",
                "var": "vars.after_step_metric",
                "value": 1,
                "reason_code": "s02_requires_progress_metric",
            },
        ),
    }

    result = infer_step_id(
        pack_steps,
        vars_map={
            "payload": {"vars": {"power_available": False}},
            "after_step_metric": 0,
        },
        recent_ui_targets=["s01_btn"],
        precondition_gates=precondition_gates,
        completion_gates=completion_gates,
    )

    assert result.inferred_step_id == "S01"
    assert "vars.power_available==true" in result.missing_conditions


def test_infer_step_prefers_nested_payload_vars_shape_when_present() -> None:
    pack_steps = [{"id": "S01"}, {"id": "S02"}]
    precondition_gates = {
        "S01": ({"op": "flag_true", "var": "payload.vars.power_available"},),
        "S02": ({"op": "flag_true", "var": "vars.s2_ready"},),
    }
    completion_gates = {"S01": (), "S02": ()}

    result = infer_step_id(
        pack_steps,
        vars_map={
            "payload": {"vars": {"power_available": True}},
            "power_available": False,
            "s2_ready": False,
        },
        recent_ui_targets=[],
        precondition_gates=precondition_gates,
        completion_gates=completion_gates,
    )

    assert result.inferred_step_id == "S02"
    assert "vars.s2_ready==true" in result.missing_conditions


def test_infer_step_qualified_var_path_prioritizes_nested_value_over_conflicting_top_level_key() -> None:
    pack_steps = [
        {"id": "S01", "observability": "partial", "ui_targets": ["s01_btn"]},
        {"id": "S02", "observability": "observable", "ui_targets": ["s02_btn"]},
    ]
    precondition_gates = {
        "S01": (
            {
                "op": "flag_true",
                "var": "payload.vars.power_available",
                "reason_code": "s01_requires_payload_power",
            },
        ),
        "S02": (),
    }
    completion_gates = {"S01": (), "S02": ()}

    result = infer_step_id(
        pack_steps,
        vars_map={
            "payload": {"vars": {"power_available": "unknown"}},
            "power_available": False,
            "s2_ready": False,
        },
        recent_ui_targets=[],
        precondition_gates=precondition_gates,
        completion_gates=completion_gates,
    )

    assert result.inferred_step_id == "S01"
    assert result.missing_conditions == ()


def test_infer_step_uses_scenario_profile_specific_gate_overrides(tmp_path: Path) -> None:
    pack_path = tmp_path / "profile_override_pack.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "profile_override_pack",
            "steps": [{"id": "S01"}, {"id": "S02"}],
            "precondition_gates": {
                "S01": [{"op": "flag_true", "var": "vars.airfield_ready"}],
                "S02": [{"op": "flag_true", "var": "vars.s2_ready"}],
            },
            "completion_gates": {"S01": [], "S02": []},
            "profile_overrides": {
                "carrier": {
                    "precondition_gates": {
                        "S01": [{"op": "flag_true", "var": "vars.carrier_ready"}],
                    }
                }
            },
        },
    )
    pack_steps = load_pack_steps(pack_path)
    vars_map = {
        "airfield_ready": False,
        "carrier_ready": True,
        "s2_ready": False,
    }

    airfield = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets=[],
        scenario_profile="airfield",
        pack_path=pack_path,
    )
    carrier = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets=[],
        scenario_profile="carrier",
        pack_path=pack_path,
    )

    assert airfield.inferred_step_id == "S01"
    assert carrier.inferred_step_id == "S02"


def test_infer_step_falls_back_to_default_profile_when_scenario_profile_invalid(tmp_path: Path) -> None:
    pack_path = tmp_path / "profile_invalid_fallback_pack.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "profile_invalid_fallback_pack",
            "steps": [{"id": "S01"}, {"id": "S02"}],
            "precondition_gates": {
                "S01": [{"op": "flag_true", "var": "vars.airfield_ready"}],
                "S02": [{"op": "flag_true", "var": "vars.s2_ready"}],
            },
            "completion_gates": {"S01": [], "S02": []},
            "profile_overrides": {
                "carrier": {
                    "precondition_gates": {
                        "S01": [{"op": "flag_true", "var": "vars.carrier_ready"}],
                    }
                }
            },
        },
    )
    pack_steps = load_pack_steps(pack_path)
    vars_map = {
        "airfield_ready": False,
        "carrier_ready": True,
        "s2_ready": False,
    }

    invalid = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets=[],
        scenario_profile="invalid-profile",
        pack_path=pack_path,
    )
    airfield = infer_step_id(
        pack_steps,
        vars_map,
        recent_ui_targets=[],
        scenario_profile="airfield",
        pack_path=pack_path,
    )

    assert invalid.inferred_step_id == "S01"
    assert invalid == airfield


def test_extract_recent_ui_targets_prefers_direct_recent_ui_targets() -> None:
    context = {
        "recent_ui_targets": ["eng_crank_switch", "eng_crank_switch", "apu_switch"],
        "recent_actions": {"recent_buttons": ["fire_test_switch"]},
    }
    assert extract_recent_ui_targets(context) == ["eng_crank_switch", "apu_switch"]


def test_normalize_recent_ui_targets_does_not_overconsume_iterables() -> None:
    def _guarded_targets():
        for idx in range(10):
            if idx >= 3:
                raise AssertionError("iterator consumed beyond cap")
            yield f"btn_{idx}"

    result = normalize_recent_ui_targets(_guarded_targets(), max_items=3)
    assert result == ["btn_0", "btn_1", "btn_2"]


def test_normalize_recent_ui_targets_accepts_single_string() -> None:
    assert normalize_recent_ui_targets("battery_switch") == ["battery_switch"]


def test_infer_step_normalizes_default_scenario_profile_for_gate_map_cache(tmp_path: Path) -> None:
    pack_path = tmp_path / "cache_profile_pack.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "cache_profile_pack",
            "steps": [{"id": "S01"}, {"id": "S02"}],
            "precondition_gates": {
                "S01": [{"op": "flag_true", "var": "vars.s01_ready"}],
                "S02": [{"op": "flag_true", "var": "vars.s02_ready"}],
            },
            "completion_gates": {"S01": [], "S02": []},
        },
    )
    pack_steps = load_pack_steps(pack_path)

    step_inference_module._load_coerced_pack_gate_maps_cached.cache_clear()
    try:
        infer_step_id(
            pack_steps,
            {"s01_ready": False, "s02_ready": False},
            recent_ui_targets=[],
            scenario_profile=None,
            pack_path=pack_path,
        )
        infer_step_id(
            pack_steps,
            {"s01_ready": False, "s02_ready": False},
            recent_ui_targets=[],
            scenario_profile="airfield",
            pack_path=pack_path,
        )
        info = step_inference_module._load_coerced_pack_gate_maps_cached.cache_info()
    finally:
        step_inference_module._load_coerced_pack_gate_maps_cached.cache_clear()

    assert info.currsize == 1
    assert info.hits >= 1


def test_infer_step_keeps_top_level_vars_when_payload_vars_present() -> None:
    pack_steps = [{"id": "S01"}, {"id": "S02"}]
    precondition_gates = {
        "S01": ({"op": "flag_true", "var": "payload.vars.power_available"},),
        "S02": ({"op": "var_gte", "var": "vars.after_step_metric", "value": 1},),
    }
    completion_gates = {"S01": (), "S02": ()}

    result = infer_step_id(
        pack_steps,
        vars_map={
            "payload": {"vars": {"power_available": True}},
            "after_step_metric": 2,
        },
        recent_ui_targets=[],
        precondition_gates=precondition_gates,
        completion_gates=completion_gates,
    )

    assert result.inferred_step_id == "S02"
    assert result.missing_conditions == ()


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


def test_load_pack_steps_does_not_inject_none_when_pack_step_field_is_null(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    registry_path = tmp_path / "step_registry.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "tmp_pack",
            "metadata": {"step_registry_path": "step_registry.yaml"},
            "steps": [
                {
                    "id": "S01",
                    "ui_targets": None,
                    "observability": None,
                }
            ],
        },
    )
    _write_yaml(registry_path, _registry_payload("first"))

    loaded = load_pack_steps(pack_path)
    first = loaded[0]
    assert first["id"] == "S01"
    assert "ui_targets" not in first
    assert "observability" not in first


def test_infer_step_preserves_caller_precondition_map_when_completion_is_missing(tmp_path: Path) -> None:
    pack_path = tmp_path / "fallback_pack.yaml"
    _write_yaml(
        pack_path,
        {
            "pack_id": "fallback_pack",
            "steps": [{"id": "S01"}, {"id": "S02"}],
            "completion_gates": {
                "S01": [{"op": "flag_true", "var": "vars.battery_on"}],
                "S02": [{"op": "flag_true", "var": "vars.r_gen_on"}],
            },
        },
    )
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
        pack_path=pack_path,
    )

    assert result.inferred_step_id == "S01"
    assert "vars.custom_ready==true" in result.missing_conditions
