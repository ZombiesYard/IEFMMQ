import os
from pathlib import Path
import time
from typing import Any, Mapping

import pytest
import yaml

from adapters.pack_gates import load_pack_gate_config
from adapters.step_inference import StepInferenceResult, extract_recent_ui_targets, infer_step_id, load_pack_steps


BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
STEP_IDS = [f"S{i:02d}" for i in range(1, 26)]
PACK_STEPS = load_pack_steps(PACK_PATH)
PACK_GATES = load_pack_gate_config(PACK_PATH)


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


def _rule_condition_text(rule: Mapping[str, Any]) -> str | None:
    op = rule.get("op")
    key = _extract_var_key(rule)
    if not isinstance(op, str) or not key:
        return None
    if op == "flag_true":
        return f"vars.{key}==true"
    if op == "var_gte":
        value = rule.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return f"vars.{key}>={value}"
    if op == "arg_in_range":
        min_value = rule.get("min")
        max_value = rule.get("max")
        if (
            isinstance(min_value, bool)
            or isinstance(max_value, bool)
            or not isinstance(min_value, (int, float))
            or not isinstance(max_value, (int, float))
        ):
            return None
        if isinstance(min_value, float) and min_value.is_integer():
            min_value = int(min_value)
        if isinstance(max_value, float) and max_value.is_integer():
            max_value = int(max_value)
        return f"vars.{key} in [{min_value},{max_value}]"
    return None


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


def _build_baseline_vars() -> dict[str, Any]:
    vars_map: dict[str, Any] = {}
    for gate_type in ("precondition_gates", "completion_gates"):
        gate_map = PACK_GATES[gate_type]
        for rules in gate_map.values():
            for rule in rules:
                key = _extract_var_key(rule)
                if not key:
                    continue
                vars_map[key] = _rule_pass_value(rule)
    return vars_map


BASELINE_VARS = _build_baseline_vars()
STEP_INDEX = _step_index(PACK_STEPS)
STEP_META = _step_by_id(PACK_STEPS)


def _step_observability(step_id: str) -> str | None:
    step = STEP_META.get(step_id, {})
    value = step.get("observability")
    return value if isinstance(value, str) else None


def _first_ui_target(step_id: str) -> str | None:
    step = STEP_META.get(step_id, {})
    targets = step.get("ui_targets")
    if not isinstance(targets, list):
        return None
    for item in targets:
        if isinstance(item, str) and item:
            return item
    return None


def _is_observability_hold_step(step_id: str) -> bool:
    obs = _step_observability(step_id)
    comp_rules = PACK_GATES["completion_gates"].get(step_id, ())
    return obs in {"partially", "unknown"} and len(comp_rules) == 0


def _completion_var_keys_after(step_id: str) -> set[str]:
    idx = STEP_INDEX[step_id]
    out: set[str] = set()
    for later_id in STEP_IDS[idx + 1 :]:
        for rule in PACK_GATES["completion_gates"].get(later_id, ()):
            key = _extract_var_key(rule)
            if key:
                out.add(key)
    return out


def _build_step_blocking_scenario(step_id: str) -> tuple[dict[str, Any], list[str], str | None]:
    vars_map = dict(BASELINE_VARS)
    recent_ui_targets: list[str] = []

    for prev_step_id in STEP_IDS[: STEP_INDEX[step_id]]:
        if not _is_observability_hold_step(prev_step_id):
            continue
        target = _first_ui_target(prev_step_id)
        if target:
            recent_ui_targets.append(target)

    completion_rules = PACK_GATES["completion_gates"].get(step_id, ())
    if completion_rules:
        failing_rule = completion_rules[0]
        key = _extract_var_key(failing_rule)
        if key:
            vars_map[key] = _rule_fail_value(failing_rule)
        return vars_map, recent_ui_targets, _rule_condition_text(failing_rule)

    if _is_observability_hold_step(step_id):
        hidden_keys = sorted(_completion_var_keys_after(step_id))
        if hidden_keys:
            vars_map["vars_source_missing"] = hidden_keys
    target = _first_ui_target(step_id)
    if target:
        recent_ui_targets = [item for item in recent_ui_targets if item != target]
    return vars_map, recent_ui_targets, None


@pytest.mark.parametrize("step_id", STEP_IDS)
def test_infer_step_pack_gate_driven_blocking_scenarios_cover_s01_to_s25(step_id: str) -> None:
    vars_map, recent_ui_targets, expected_condition = _build_step_blocking_scenario(step_id)

    result = infer_step_id(
        PACK_STEPS,
        vars_map,
        recent_ui_targets,
        precondition_gates=PACK_GATES["precondition_gates"],
        completion_gates=PACK_GATES["completion_gates"],
        pack_path=PACK_PATH,
    )

    assert result.inferred_step_id == step_id
    if expected_condition is None:
        assert result.missing_conditions == ()
    else:
        assert expected_condition in result.missing_conditions


def test_infer_step_mvp_progresses_to_s03_when_apu_not_ready() -> None:
    result = infer_step_id(
        PACK_STEPS,
        {"power_available": True, "apu_on": True, "apu_ready": False},
        ["apu_switch"],
        precondition_gates=PACK_GATES["precondition_gates"],
        completion_gates=PACK_GATES["completion_gates"],
        pack_path=PACK_PATH,
    )

    assert result.inferred_step_id == "S03"
    assert "vars.apu_ready==true" in result.missing_conditions


def test_infer_step_mvp_progresses_to_s07_without_missing_conditions() -> None:
    result = infer_step_id(
        PACK_STEPS,
        {
            "power_available": True,
            "apu_ready": True,
            "engine_crank_right": True,
            "rpm_r": 65,
            "bleed_air_norm": True,
        },
        ["bleed_air_knob", "eng_crank_switch"],
        precondition_gates=PACK_GATES["precondition_gates"],
        completion_gates=PACK_GATES["completion_gates"],
        pack_path=PACK_PATH,
    )

    assert result.inferred_step_id == "S07"
    assert result.missing_conditions == ()


def test_infer_step_is_robust_on_invalid_inputs() -> None:
    result = infer_step_id(
        PACK_STEPS,
        vars_map={"power_available": "unknown"},
        recent_ui_targets=["apu_switch", "", 1],  # type: ignore[list-item]
        precondition_gates=PACK_GATES["precondition_gates"],
        completion_gates=PACK_GATES["completion_gates"],
        pack_path=PACK_PATH,
    )
    assert isinstance(result, StepInferenceResult)
    assert isinstance(result.inferred_step_id, str)


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
