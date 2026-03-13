from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from adapters.dcs_bios.bios_ui_map import BiosUiMapper
from adapters.event_store.telemetry_writer import TelemetryWriter
from adapters.pack_gates import (
    DEFAULT_SCENARIO_PROFILE,
    SUPPORTED_SCENARIO_PROFILES,
    load_pack_gate_config,
    normalize_scenario_profile,
)
from adapters.step_inference import infer_step_id, load_pack_steps
from core.step_signal_metadata import normalize_observability_status
from core.vars import VarResolver

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PACK_PATH = _REPO_ROOT / "packs" / "fa18c_startup" / "pack.yaml"
_DEFAULT_TELEMETRY_MAP_PATH = _REPO_ROOT / "packs" / "fa18c_startup" / "telemetry_map.yaml"
_DEFAULT_BIOS_TO_UI_PATH = _REPO_ROOT / "packs" / "fa18c_startup" / "bios_to_ui.yaml"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "artifacts" / "regression" / "coldstart_state_matrix"
_AIRCRAFT = "FA-18C_hornet"


class StateMatrixBuildError(ValueError):
    pass


Setter = Callable[[dict[str, Any], Any], None]


@dataclass(frozen=True)
class _VarBinding:
    primary_bios_key: str
    setter: Setter


@dataclass(frozen=True)
class _StepProfile:
    step_id: str
    observability: str | None
    ui_targets: tuple[str, ...]


@dataclass(frozen=True)
class _CaseSpec:
    case_id: str
    step_id: str
    state_kind: str
    scenario_profile: str
    expected_inferred_step_id: str | None
    expected_missing_conditions: tuple[str, ...]
    recent_ui_targets: tuple[str, ...]
    replay_input: str
    frame_count: int


def _case_dict(case: _CaseSpec) -> dict[str, Any]:
    payload = asdict(case)
    payload["expected_missing_conditions"] = list(case.expected_missing_conditions)
    payload["recent_ui_targets"] = list(case.recent_ui_targets)
    return payload


def _bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return bool(value)


def _set_enum(key: str, *, true_value: Any, false_value: Any) -> _VarBinding:
    def _setter(bios: dict[str, Any], desired: Any) -> None:
        bios[key] = true_value if _bool_like(desired) else false_value

    return _VarBinding(primary_bios_key=key, setter=_setter)


def _set_numeric(key: str) -> _VarBinding:
    def _setter(bios: dict[str, Any], desired: Any) -> None:
        bios[key] = desired

    return _VarBinding(primary_bios_key=key, setter=_setter)


def _set_threshold_numeric(key: str, *, pass_value: int, fail_value: int) -> _VarBinding:
    def _setter(bios: dict[str, Any], desired: Any) -> None:
        if isinstance(desired, bool):
            bios[key] = pass_value if desired else fail_value
            return
        bios[key] = desired

    return _VarBinding(primary_bios_key=key, setter=_setter)


def _set_power_available(bios: dict[str, Any], desired: Any) -> None:
    if _bool_like(desired):
        bios["BATTERY_SW"] = 2
        bios["L_GEN_SW"] = 1
        bios["R_GEN_SW"] = 1
        return
    bios["BATTERY_SW"] = 1
    bios["L_GEN_SW"] = 0
    bios["R_GEN_SW"] = 0


def _set_right_engine_nominal_start_params(bios: dict[str, Any], desired: Any) -> None:
    if _bool_like(desired):
        bios["IFEI_RPM_R"] = 65
        bios["IFEI_TEMP_R"] = 300
        bios["IFEI_FF_R"] = 600
        bios["IFEI_OIL_PRESS_R"] = 60
        bios["EXT_NOZZLE_POS_R"] = 51118
        return
    bios["IFEI_RPM_R"] = 55
    bios["IFEI_TEMP_R"] = 100
    bios["IFEI_FF_R"] = 200
    bios["IFEI_OIL_PRESS_R"] = 20
    bios["EXT_NOZZLE_POS_R"] = 10000


def _set_apu_start_support_complete(bios: dict[str, Any], desired: Any) -> None:
    if _bool_like(desired):
        bios["APU_READY_LT"] = 1
        return
    bios["APU_READY_LT"] = 0
    bios["ENGINE_CRANK_SW"] = 1
    bios["IFEI_RPM_R"] = 0


def _set_obogs_ready(bios: dict[str, Any], desired: Any) -> None:
    if _bool_like(desired):
        bios["OBOGS_SW"] = 1
        bios["OXY_FLOW"] = 1
        return
    bios["OBOGS_SW"] = 0
    bios["OXY_FLOW"] = 0


def _set_flap_auto(bios: dict[str, Any], desired: Any) -> None:
    bios["FLAP_SW"] = 0 if _bool_like(desired) else 1


def _set_parking_brake_released(bios: dict[str, Any], desired: Any) -> None:
    bios["EMERGENCY_PARKING_BRAKE_PULL"] = 0 if _bool_like(desired) else 1


def _set_bingo_fuel_set(bios: dict[str, Any], desired: Any) -> None:
    bios["IFEI_BINGO"] = "1000" if _bool_like(desired) else "     "


def _set_standby_attitude_uncaged(bios: dict[str, Any], desired: Any) -> None:
    bios["SAI_ATT_WARNING_FLAG"] = 0 if _bool_like(desired) else 1


_VAR_BINDINGS: dict[str, _VarBinding] = {
    "apu_on": _set_enum("APU_CONTROL_SW", true_value=1, false_value=0),
    "apu_ready": _set_enum("APU_READY_LT", true_value=1, false_value=0),
    "apu_start_support_complete": _VarBinding(
        primary_bios_key="APU_READY_LT",
        setter=_set_apu_start_support_complete,
    ),
    "battery_on": _set_enum("BATTERY_SW", true_value=2, false_value=1),
    "bleed_air_norm": _set_enum("BLEED_AIR_KNOB", true_value=2, false_value=0),
    "bleed_air_cycle_complete": _set_enum("BLEED_AIR_KNOB", true_value=2, false_value=0),
    "bingo_fuel_set": _VarBinding(primary_bios_key="IFEI_BINGO", setter=_set_bingo_fuel_set),
    "engine_crank_left": _set_enum("ENGINE_CRANK_SW", true_value=0, false_value=1),
    "engine_crank_right": _set_enum("ENGINE_CRANK_SW", true_value=2, false_value=1),
    "engine_crank_right_complete": _set_enum("ENGINE_CRANK_SW", true_value=2, false_value=1),
    "flap_auto": _VarBinding(primary_bios_key="FLAP_SW", setter=_set_flap_auto),
    "fire_test_complete": _set_enum("FIRE_TEST_SW", true_value=1, false_value=0),
    "hud_on": _set_enum("HUD_SYM_BRT", true_value=1, false_value=0),
    "ins_mode": _set_numeric("INS_SW"),
    "l_gen_on": _set_enum("L_GEN_SW", true_value=1, false_value=0),
    "left_ddi_on": _set_enum("LEFT_DDI_BRT_CTL", true_value=1, false_value=0),
    "mpcd_on": _set_enum("AMPCD_BRT_CTL", true_value=1, false_value=0),
    "obogs_ready": _VarBinding(primary_bios_key="OBOGS_SW", setter=_set_obogs_ready),
    "parking_brake_released": _VarBinding(
        primary_bios_key="EMERGENCY_PARKING_BRAKE_PULL",
        setter=_set_parking_brake_released,
    ),
    "power_available": _VarBinding(primary_bios_key="BATTERY_SW", setter=_set_power_available),
    "r_gen_on": _set_enum("R_GEN_SW", true_value=1, false_value=0),
    "radar_altimeter_bug_value": _set_numeric("RADALT_MIN_HEIGHT_PTR"),
    "radar_on": _set_enum("RADAR_SW", true_value=2, false_value=0),
    "right_ddi_on": _set_enum("RIGHT_DDI_BRT_CTL", true_value=1, false_value=0),
    "right_engine_nominal_start_params": _VarBinding(
        primary_bios_key="IFEI_RPM_R",
        setter=_set_right_engine_nominal_start_params,
    ),
    "rpm_l": _set_numeric("IFEI_RPM_L"),
    "rpm_l_gte_60": _set_threshold_numeric("IFEI_RPM_L", pass_value=65, fail_value=59),
    "rpm_r": _set_numeric("IFEI_RPM_R"),
    "rpm_r_gte_60": _set_threshold_numeric("IFEI_RPM_R", pass_value=65, fail_value=59),
    "standby_attitude_uncaged": _VarBinding(
        primary_bios_key="SAI_ATT_WARNING_FLAG",
        setter=_set_standby_attitude_uncaged,
    ),
    "attitude_source_auto": _set_enum("HUD_ATT_SW", true_value=1, false_value=0),
    "throttle_l_not_off": _set_enum("INT_THROTTLE_LEFT", true_value=1, false_value=0),
    "throttle_r_not_off": _set_enum("INT_THROTTLE_RIGHT", true_value=1, false_value=0),
    "throttle_r_idle_complete": _set_enum("INT_THROTTLE_RIGHT", true_value=1, false_value=0),
    "takeoff_trim_set": _set_enum("TO_TRIM_BTN", true_value=1, false_value=0),
}


def _step_sort_key(step_id: str) -> tuple[int, str]:
    if step_id.startswith("S") and step_id[1:].isdigit():
        return int(step_id[1:]), step_id
    return (10_000, step_id)


def _extract_var_key(rule: Mapping[str, Any]) -> str | None:
    raw = rule.get("var")
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("payload.vars."):
        key = raw[len("payload.vars.") :]
        return key if key and "." not in key else None
    if raw.startswith("vars."):
        key = raw[len("vars.") :]
        return key if key and "." not in key else None
    if "." in raw:
        return None
    return raw


def _pass_value(rule: Mapping[str, Any]) -> Any:
    op = rule.get("op")
    if op == "flag_true":
        return True
    if op == "var_gte":
        value = rule.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 1
        return int(value) if float(value).is_integer() else float(value)
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
        return int(midpoint) if midpoint.is_integer() else midpoint
    raise StateMatrixBuildError(f"unsupported gate op for pass synthesis: {op!r}")


def _fail_value(rule: Mapping[str, Any]) -> Any:
    op = rule.get("op")
    if op == "flag_true":
        return False
    if op == "var_gte":
        value = rule.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        failed = float(value) - 1.0
        return int(failed) if failed.is_integer() else failed
    if op == "arg_in_range":
        min_value = rule.get("min")
        if isinstance(min_value, bool) or not isinstance(min_value, (int, float)):
            return -1
        failed = float(min_value) - 1.0
        return int(failed) if failed.is_integer() else failed
    raise StateMatrixBuildError(f"unsupported gate op for fail synthesis: {op!r}")


def _apply_rule_state(bios: dict[str, Any], rule: Mapping[str, Any], *, pass_state: bool) -> str:
    key = _extract_var_key(rule)
    if key is None:
        raise StateMatrixBuildError(f"unsupported gate var path: {rule.get('var')!r}")
    binding = _VAR_BINDINGS.get(key)
    if binding is None:
        raise StateMatrixBuildError(f"missing synthetic BIOS binding for gate var {key!r}")
    desired = _pass_value(rule) if pass_state else _fail_value(rule)
    binding.setter(bios, desired)
    return binding.primary_bios_key


def _clone_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in raw.items()}


def _step_profiles_from_pack_steps(pack_steps: Sequence[Mapping[str, Any]]) -> list[_StepProfile]:
    out: list[_StepProfile] = []
    for step in pack_steps:
        if not isinstance(step, Mapping):
            continue
        step_id_raw = step.get("id") or step.get("step_id")
        if not isinstance(step_id_raw, str) or not step_id_raw:
            continue
        targets_raw = step.get("ui_targets")
        ui_targets = tuple(
            item
            for item in (targets_raw or ())
            if isinstance(item, str) and item
        )
        out.append(
            _StepProfile(
                step_id=step_id_raw,
                observability=normalize_observability_status(step.get("observability")),
                ui_targets=ui_targets,
            )
        )
    out.sort(key=lambda item: _step_sort_key(item.step_id))
    return out


def _build_target_reverse_map(mapper: BiosUiMapper) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for bios_key, targets in mapper.rules.items():
        for target in targets:
            out.setdefault(target, []).append(bios_key)
    return {target: tuple(keys) for target, keys in out.items()}


def _is_hold_step(profile: _StepProfile, completion_rules: Sequence[Mapping[str, Any]]) -> bool:
    return profile.observability in {"partial", "unobservable"} and len(completion_rules) == 0


def _primary_target(profile: _StepProfile) -> str | None:
    return profile.ui_targets[0] if profile.ui_targets else None


def _select_step_delta_key(
    profile: _StepProfile,
    *,
    precondition_rules: Sequence[Mapping[str, Any]],
    completion_rules: Sequence[Mapping[str, Any]],
    reverse_target_map: Mapping[str, Sequence[str]],
) -> str | None:
    for rule in completion_rules:
        key = _extract_var_key(rule)
        binding = _VAR_BINDINGS.get(key or "")
        if binding is not None:
            return binding.primary_bios_key
    target = _primary_target(profile)
    if target:
        mapped = reverse_target_map.get(target, ())
        if mapped:
            return mapped[0]
    for rule in precondition_rules:
        key = _extract_var_key(rule)
        binding = _VAR_BINDINGS.get(key or "")
        if binding is not None:
            return binding.primary_bios_key
    return None


def _dedupe_keep_order(items: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _recent_targets_from_frames(frames: Sequence[Mapping[str, Any]], mapper: BiosUiMapper) -> tuple[str, ...]:
    recent: list[str] = []
    seen: set[str] = set()
    for frame in frames[:-1]:
        delta = frame.get("delta")
        if not isinstance(delta, Mapping):
            continue
        for target in mapper.map_delta(delta):
            if target in seen:
                continue
            seen.add(target)
            recent.append(target)
    return tuple(recent)


def _build_profile_cases(
    *,
    pack_path: Path,
    telemetry_map_path: Path,
    bios_to_ui_path: Path,
    scenario_profile: str,
    output_dir: Path,
) -> list[_CaseSpec]:
    raw_pack_steps = load_pack_steps(pack_path)
    pack_steps = _step_profiles_from_pack_steps(raw_pack_steps)
    if not pack_steps or not raw_pack_steps:
        raise StateMatrixBuildError(f"no pack steps loaded from {pack_path}")

    step_ids = [item.step_id for item in pack_steps]
    gate_cfg = load_pack_gate_config(pack_path, scenario_profile=scenario_profile)
    precondition_gates = gate_cfg["precondition_gates"]
    completion_gates = gate_cfg["completion_gates"]
    mapper = BiosUiMapper.from_yaml(bios_to_ui_path)
    reverse_target_map = _build_target_reverse_map(mapper)
    resolver = VarResolver.from_yaml(telemetry_map_path)
    replay_dir = output_dir / "replay_inputs"
    replay_dir.mkdir(parents=True, exist_ok=True)

    state_before: dict[str, dict[str, Any]] = {}
    state_after: dict[str, dict[str, Any]] = {}
    current_state: dict[str, Any] = {}
    for profile in pack_steps:
        before = dict(current_state)
        for rule in precondition_gates.get(profile.step_id, ()):
            _apply_rule_state(before, rule, pass_state=True)
        state_before[profile.step_id] = before

        after = dict(before)
        for rule in completion_gates.get(profile.step_id, ()):
            _apply_rule_state(after, rule, pass_state=True)
        state_after[profile.step_id] = after
        current_state = after

    hold_step_targets: dict[str, str] = {}
    step_delta_keys: dict[str, str] = {}
    for profile in pack_steps:
        delta_key = _select_step_delta_key(
            profile,
            precondition_rules=precondition_gates.get(profile.step_id, ()),
            completion_rules=completion_gates.get(profile.step_id, ()),
            reverse_target_map=reverse_target_map,
        )
        if delta_key is not None:
            step_delta_keys[profile.step_id] = delta_key
        if _is_hold_step(profile, completion_gates.get(profile.step_id, ())):
            target = _primary_target(profile)
            if target is None:
                raise StateMatrixBuildError(f"hold-step {profile.step_id} missing ui_targets")
            if delta_key is None:
                raise StateMatrixBuildError(f"hold-step {profile.step_id} missing replay delta key")
            hold_step_targets[profile.step_id] = target

    cases: list[_CaseSpec] = []
    for idx, profile in enumerate(pack_steps):
        prior_hold_targets = _dedupe_keep_order(
            [hold_step_targets[step_id] for step_id in step_ids[:idx] if step_id in hold_step_targets]
        )
        completion_rules = completion_gates.get(profile.step_id, ())
        current_target = _primary_target(profile)
        for state_kind in ("blocked", "just_completed"):
            bios_state = dict(state_before[profile.step_id])
            if state_kind == "blocked":
                if completion_rules:
                    _apply_rule_state(bios_state, completion_rules[0], pass_state=False)
                recent_ui_targets = prior_hold_targets
            else:
                bios_state = dict(state_after[profile.step_id])
                extra_targets = list(prior_hold_targets)
                if current_target:
                    extra_targets.append(current_target)
                recent_ui_targets = _dedupe_keep_order(extra_targets)
                if idx + 1 < len(step_ids):
                    next_step_id = step_ids[idx + 1]
                    next_profile = pack_steps[idx + 1]
                    next_completion_rules = completion_gates.get(next_step_id, ())
                    next_precondition_rules = precondition_gates.get(next_step_id, ())
                    if not _is_hold_step(next_profile, next_completion_rules):
                        if next_completion_rules:
                            _apply_rule_state(bios_state, next_completion_rules[0], pass_state=False)
                        elif next_precondition_rules:
                            _apply_rule_state(bios_state, next_precondition_rules[0], pass_state=False)

            case_id = f"{scenario_profile}_{profile.step_id.lower()}_{state_kind}"
            replay_filename = f"{case_id}.jsonl"
            replay_path = replay_dir / replay_filename
            _write_replay_case(
                replay_path,
                bios_state=bios_state,
                recent_ui_targets=recent_ui_targets,
                step_delta_keys=step_delta_keys,
                reverse_target_map=reverse_target_map,
            )

            frames = TelemetryWriter.load(replay_path)
            recent_targets_for_inference = _recent_targets_from_frames(frames, mapper)
            final_vars = resolver.resolve(frames[-1])
            result = infer_step_id(
                raw_pack_steps,
                final_vars,
                recent_targets_for_inference,
                precondition_gates=precondition_gates,
                completion_gates=completion_gates,
                pack_path=pack_path,
                scenario_profile=scenario_profile,
            )
            if result.inferred_step_id is None:
                raise StateMatrixBuildError(f"{case_id} inferred None")

            cases.append(
                _CaseSpec(
                    case_id=case_id,
                    step_id=profile.step_id,
                    state_kind=state_kind,
                    scenario_profile=scenario_profile,
                    expected_inferred_step_id=result.inferred_step_id,
                    expected_missing_conditions=result.missing_conditions,
                    recent_ui_targets=recent_targets_for_inference,
                    replay_input=str(replay_path.relative_to(output_dir)),
                    frame_count=len(frames),
                )
            )
    return cases


def _write_replay_case(
    path: Path,
    *,
    bios_state: Mapping[str, Any],
    recent_ui_targets: Sequence[str],
    step_delta_keys: Mapping[str, str],
    reverse_target_map: Mapping[str, Sequence[str]],
) -> None:
    frames: list[dict[str, Any]] = []
    seq = 1
    t_wall = 1.0
    for target in recent_ui_targets:
        delta_key = None
        mapped_keys = reverse_target_map.get(target, ())
        if mapped_keys:
            delta_key = mapped_keys[0]
        if delta_key is None:
            for step_id, candidate in step_delta_keys.items():
                if step_id.lower() in target:
                    delta_key = candidate
                    break
        if delta_key is None:
            raise StateMatrixBuildError(f"missing delta key for recent ui target {target!r}")
        bios_payload = _clone_mapping(bios_state)
        frames.append(
            {
                "schema_version": "v2",
                "seq": seq,
                "t_wall": t_wall,
                "aircraft": _AIRCRAFT,
                "bios": bios_payload,
                "delta": {delta_key: bios_payload.get(delta_key, 1)},
            }
        )
        seq += 1
        t_wall += 0.1

    final_bios_payload = _clone_mapping(bios_state)
    frames.append(
        {
            "schema_version": "v2",
            "seq": seq,
            "t_wall": t_wall,
            "aircraft": _AIRCRAFT,
            "bios": final_bios_payload,
            "delta": {},
        }
    )
    with TelemetryWriter(path) as writer:
        for frame in frames:
            writer.append(frame)


def _normalize_requested_profiles(requested: str | None) -> tuple[str, ...]:
    if requested is None:
        return (DEFAULT_SCENARIO_PROFILE,)
    normalized = requested.strip().lower()
    if normalized == "all":
        return tuple(SUPPORTED_SCENARIO_PROFILES)
    return (normalize_scenario_profile(normalized),)


def build_coldstart_state_matrix_dataset(
    *,
    output_dir: str | Path,
    pack_path: str | Path | None = None,
    telemetry_map_path: str | Path | None = None,
    bios_to_ui_path: str | Path | None = None,
    scenario_profile: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_pack_path = Path(pack_path) if pack_path else _DEFAULT_PACK_PATH
    resolved_telemetry_map_path = Path(telemetry_map_path) if telemetry_map_path else _DEFAULT_TELEMETRY_MAP_PATH
    resolved_bios_to_ui_path = Path(bios_to_ui_path) if bios_to_ui_path else _DEFAULT_BIOS_TO_UI_PATH
    profiles = _normalize_requested_profiles(scenario_profile)

    all_cases: list[dict[str, Any]] = []
    for profile in profiles:
        profile_dir = output_root if len(profiles) == 1 else output_root / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        cases = _build_profile_cases(
            pack_path=resolved_pack_path,
            telemetry_map_path=resolved_telemetry_map_path,
            bios_to_ui_path=resolved_bios_to_ui_path,
            scenario_profile=profile,
            output_dir=profile_dir,
        )
        profile_manifest = {
            "generator": "tools/build_coldstart_state_matrix.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pack_path": str(resolved_pack_path),
            "telemetry_map_path": str(resolved_telemetry_map_path),
            "bios_to_ui_path": str(resolved_bios_to_ui_path),
            "scenario_profile": profile,
            "case_count": len(cases),
            "cases": [_case_dict(case) for case in cases],
        }
        (profile_dir / "matrix.json").write_text(
            json.dumps(profile_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if len(profiles) == 1:
            return profile_manifest

        for case in cases:
            case_dict = _case_dict(case)
            case_dict["replay_input"] = str((profile_dir / case.replay_input).relative_to(output_root))
            all_cases.append(case_dict)

    manifest = {
        "generator": "tools/build_coldstart_state_matrix.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pack_path": str(resolved_pack_path),
        "telemetry_map_path": str(resolved_telemetry_map_path),
        "bios_to_ui_path": str(resolved_bios_to_ui_path),
        "scenario_profile": "all",
        "profiles": list(profiles),
        "case_count": len(all_cases),
        "cases": all_cases,
    }
    (output_root / "matrix.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build representative cold-start state-matrix replay inputs for offline help regression."
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Output directory for matrix.json and replay_inputs/",
    )
    parser.add_argument("--pack", default=str(_DEFAULT_PACK_PATH), help="pack.yaml path")
    parser.add_argument(
        "--telemetry-map",
        default=str(_DEFAULT_TELEMETRY_MAP_PATH),
        help="telemetry_map.yaml path",
    )
    parser.add_argument(
        "--bios-to-ui",
        default=str(_DEFAULT_BIOS_TO_UI_PATH),
        help="bios_to_ui.yaml path",
    )
    parser.add_argument(
        "--scenario-profile",
        default=DEFAULT_SCENARIO_PROFILE,
        help=f"Scenario profile: {', '.join(SUPPORTED_SCENARIO_PROFILES)} or all",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_coldstart_state_matrix_dataset(
        output_dir=args.output_dir,
        pack_path=args.pack,
        telemetry_map_path=args.telemetry_map,
        bios_to_ui_path=args.bios_to_ui,
        scenario_profile=args.scenario_profile,
    )
    print(
        json.dumps(
            {
                "scenario_profile": manifest["scenario_profile"],
                "case_count": manifest["case_count"],
                "output_dir": str(Path(args.output_dir)),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
