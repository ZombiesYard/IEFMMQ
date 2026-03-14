from datetime import datetime, timezone
from pathlib import Path

import pytest

from adapters.pack_gates import evaluate_pack_gates, load_pack_gate_config
from adapters.prompting import build_help_prompt_result


BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"


def _obs_with_vars(**vars_map):
    return {
        "observation_id": "obs-1",
        "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "source": "unit",
        "payload": {"vars": vars_map},
        "version": "v1",
    }


def test_pack_gate_config_contains_s01_to_s25_for_precondition_and_completion() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    pre = cfg["precondition_gates"]
    comp = cfg["completion_gates"]

    for step_id in [f"S{i:02d}" for i in range(1, 26)]:
        assert step_id in pre, f"missing precondition gate config for {step_id}"
        assert step_id in comp, f"missing completion gate config for {step_id}"
        assert isinstance(pre[step_id], tuple)
        assert isinstance(comp[step_id], tuple)


def test_evaluate_pack_gates_reports_blocked_reason_code_for_s12_precondition() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(rpm_l_gte_60=False, throttle_l_not_off=False)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    s12_pre = gates["S12.precondition"]

    assert s12_pre["status"] == "blocked"
    assert s12_pre["allowed"] is False
    assert s12_pre["reason_code"] == "s12_requires_rpm_l_gte_60"
    assert isinstance(s12_pre["reason"], str) and s12_pre["reason"]


def test_evaluate_pack_gates_allows_s12_precondition_when_left_engine_ready() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(rpm_l_gte_60=True, throttle_l_not_off=True)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    s12_pre = gates["S12.precondition"]

    assert s12_pre["status"] == "allowed"
    assert s12_pre["allowed"] is True
    assert s12_pre["reason_code"] == "ok"
    assert s12_pre["reason"] is None


def test_evaluate_pack_gates_reports_blocked_reason_code_for_s13_completion() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(radar_mode_opr=False)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    s13_comp = gates["S13.completion"]

    assert s13_comp["status"] == "blocked"
    assert s13_comp["allowed"] is False
    assert s13_comp["reason_code"] == "s13_requires_radar_mode_opr"
    assert isinstance(s13_comp["reason"], str) and s13_comp["reason"]


def test_evaluate_pack_gates_allows_s13_completion_when_radar_on() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(radar_mode_opr=True)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    s13_comp = gates["S13.completion"]

    assert s13_comp["status"] == "allowed"
    assert s13_comp["allowed"] is True
    assert s13_comp["reason_code"] == "ok"
    assert s13_comp["reason"] is None


def test_evaluate_pack_gates_reports_blocked_reason_code_for_s04_precondition() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(apu_start_support_complete=False)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    s04_pre = gates["S04.precondition"]

    assert s04_pre["status"] == "blocked"
    assert s04_pre["allowed"] is False
    assert s04_pre["reason_code"] == "s04_requires_apu_start_support_complete"
    assert isinstance(s04_pre["reason"], str) and s04_pre["reason"]


def test_evaluate_pack_gates_allows_s03_precondition_without_fire_test_gate() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(power_available=True, fire_test_complete="unknown")],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )

    s03_pre = gates["S03.precondition"]
    assert s03_pre["status"] == "allowed"
    assert s03_pre["allowed"] is True
    assert s03_pre["reason_code"] == "ok"


def test_evaluate_pack_gates_allows_s02_completion_without_rules() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(fire_test_complete=False)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )

    s02_comp = gates["S02.completion"]
    assert s02_comp["status"] == "allowed"
    assert s02_comp["allowed"] is True
    assert s02_comp["reason_code"] == "no_rules"


def test_evaluate_pack_gates_allows_s04_precondition_when_apu_ready_true() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(apu_start_support_complete=True)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    s04_pre = gates["S04.precondition"]

    assert s04_pre["status"] == "allowed"
    assert s04_pre["allowed"] is True
    assert s04_pre["reason_code"] == "ok"
    assert s04_pre["reason"] is None


def test_load_pack_gate_config_rejects_non_mapping_gate_block(tmp_path: Path) -> None:
    bad_pack = tmp_path / "bad_pack.yaml"
    bad_pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "precondition_gates: []\n"
        "completion_gates: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="precondition_gates must be a mapping"):
        load_pack_gate_config(bad_pack)


def test_load_pack_gate_config_rejects_rule_without_op(tmp_path: Path) -> None:
    bad_pack = tmp_path / "bad_pack_missing_op.yaml"
    bad_pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "precondition_gates:\n"
        "  S01:\n"
        "    - var: vars.power_available\n"
        "completion_gates: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"precondition_gates\.S01\[0\]\.op"):
        load_pack_gate_config(bad_pack)


def test_evaluate_pack_gates_accepts_generator_rules_iterable() -> None:
    precondition_gates = {
        "S99": ({"op": "flag_true", "var": "vars.apu_ready"} for _ in [0]),
    }
    completion_gates: dict[str, tuple[dict[str, object], ...]] = {}
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(apu_ready=False)],
        precondition_gates=precondition_gates,
        completion_gates=completion_gates,
    )

    assert gates["S99.precondition"]["status"] == "blocked"
    assert gates["S99.precondition"]["allowed"] is False


def test_evaluate_pack_gates_treats_single_mapping_rule_as_one_rule() -> None:
    precondition_gates = {
        "S98": {"op": "flag_true", "var": "vars.apu_ready"},
    }
    completion_gates: dict[str, tuple[dict[str, object], ...]] = {}
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(apu_ready=False)],
        precondition_gates=precondition_gates,  # type: ignore[arg-type]
        completion_gates=completion_gates,
    )

    assert gates["S98.precondition"]["status"] == "blocked"
    assert gates["S98.precondition"]["allowed"] is False


def test_load_pack_gate_config_applies_carrier_profile_overrides() -> None:
    airfield = load_pack_gate_config(PACK_PATH, scenario_profile="airfield")
    carrier = load_pack_gate_config(PACK_PATH, scenario_profile="carrier")

    assert airfield["completion_gates"]["S12"][0]["reason_code"] == "s12_requires_ins_mode_gnd"
    assert carrier["completion_gates"]["S12"][0]["reason_code"] == "s12_requires_ins_mode_cv"
    assert airfield["completion_gates"]["S23"][0]["reason_code"] == "s23_requires_radalt_bug_airfield_200"
    assert carrier["completion_gates"]["S23"][0]["reason_code"] == "s23_requires_radalt_bug_carrier_40"


def test_load_pack_gate_config_rejects_unknown_scenario_profile() -> None:
    with pytest.raises(ValueError, match="unsupported scenario_profile"):
        load_pack_gate_config(PACK_PATH, scenario_profile="invalid_profile")


def test_evaluate_pack_gates_profile_changes_s12_and_s23_gate_results() -> None:
    airfield_cfg = load_pack_gate_config(PACK_PATH, scenario_profile="airfield")
    carrier_cfg = load_pack_gate_config(PACK_PATH, scenario_profile="carrier")
    obs = _obs_with_vars(
        rpm_l_gte_60=True,
        throttle_l_not_off=True,
        ins_mode=2,
        radar_altimeter_bug_value=200,
    )

    airfield_gates = evaluate_pack_gates(
        observations=[obs],
        precondition_gates=airfield_cfg["precondition_gates"],
        completion_gates=airfield_cfg["completion_gates"],
    )
    carrier_gates = evaluate_pack_gates(
        observations=[obs],
        precondition_gates=carrier_cfg["precondition_gates"],
        completion_gates=carrier_cfg["completion_gates"],
    )

    assert airfield_gates["S12.completion"]["status"] == "allowed"
    assert airfield_gates["S23.completion"]["status"] == "allowed"
    assert carrier_gates["S12.completion"]["status"] == "blocked"
    assert carrier_gates["S12.completion"]["reason_code"] == "s12_requires_ins_mode_cv"
    assert carrier_gates["S23.completion"]["status"] == "blocked"
    assert carrier_gates["S23.completion"]["reason_code"] == "s23_requires_radalt_bug_carrier_40"


def test_evaluate_pack_gates_allows_bios_observable_steps_s14_s16_s20_s21_s24_s25() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[
            _obs_with_vars(
                obogs_ready=True,
                flap_auto=True,
                parking_brake_released=True,
                bingo_fuel_set=True,
                standby_attitude_uncaged=True,
                attitude_source_auto=True,
            )
        ],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )

    for gate_id in ("S14.completion", "S16.completion", "S20.completion", "S21.completion", "S24.completion", "S25.completion"):
        assert gates[gate_id]["status"] == "allowed"
        assert gates[gate_id]["allowed"] is True


def test_evaluate_pack_gates_blocks_bios_observable_steps_when_bios_state_missing() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[
            _obs_with_vars(
                obogs_ready=False,
                flap_auto=False,
                parking_brake_released=False,
                bingo_fuel_set=False,
                standby_attitude_uncaged=False,
                attitude_source_auto=False,
            )
        ],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )

    assert gates["S14.completion"]["reason_code"] == "s14_requires_obogs_ready"
    assert gates["S16.completion"]["reason_code"] == "s16_requires_flap_auto"
    assert gates["S20.completion"]["reason_code"] == "s20_requires_parking_brake_released"
    assert gates["S21.completion"]["reason_code"] == "s21_requires_bingo_fuel_set"
    assert gates["S24.completion"]["reason_code"] == "s24_requires_standby_attitude_uncaged"
    assert gates["S25.completion"]["reason_code"] == "s25_requires_attitude_source_auto"


def test_evaluate_pack_gates_uses_takeoff_trim_button_press_for_s17() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    gates = evaluate_pack_gates(
        observations=[_obs_with_vars(takeoff_trim_set=True)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    assert gates["S17.completion"]["status"] == "allowed"

    blocked = evaluate_pack_gates(
        observations=[_obs_with_vars(takeoff_trim_set=False)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    assert blocked["S17.completion"]["reason_code"] == "s17_requires_takeoff_trim_pressed"


def test_evaluate_pack_gates_allows_s07_completion_without_rules() -> None:
    cfg = load_pack_gate_config(PACK_PATH)
    allowed = evaluate_pack_gates(
        observations=[_obs_with_vars(lights_test_complete=True)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    assert allowed["S07.completion"]["status"] == "allowed"
    assert allowed["S07.completion"]["reason_code"] == "no_rules"

    blocked = evaluate_pack_gates(
        observations=[_obs_with_vars(lights_test_complete=False)],
        precondition_gates=cfg["precondition_gates"],
        completion_gates=cfg["completion_gates"],
    )
    assert blocked["S07.completion"]["status"] == "allowed"
    assert blocked["S07.completion"]["reason_code"] == "no_rules"


def test_scenario_profile_changes_prompt_gate_hints() -> None:
    obs = _obs_with_vars(
        rpm_l_gte_60=True,
        throttle_l_not_off=True,
        ins_mode=0,
        radar_altimeter_bug_value=0,
    )
    airfield_cfg = load_pack_gate_config(PACK_PATH, scenario_profile="airfield")
    carrier_cfg = load_pack_gate_config(PACK_PATH, scenario_profile="carrier")
    airfield_gates = evaluate_pack_gates(
        observations=[obs],
        precondition_gates=airfield_cfg["precondition_gates"],
        completion_gates=airfield_cfg["completion_gates"],
    )
    carrier_gates = evaluate_pack_gates(
        observations=[obs],
        precondition_gates=carrier_cfg["precondition_gates"],
        completion_gates=carrier_cfg["completion_gates"],
    )

    base_context = {
        "candidate_steps": ["S12", "S23"],
        "overlay_target_allowlist": ["ins_mode_knob", "radar_altimeter_bug_knob"],
        "vars": {},
        "recent_deltas": [],
        "recent_actions": [],
        "rag_topk": [],
    }
    airfield_prompt = build_help_prompt_result(
        {
            **base_context,
            "scenario_profile": "airfield",
            "gates": {
                "S12.completion": airfield_gates["S12.completion"],
                "S23.completion": airfield_gates["S23.completion"],
            },
        },
        "en",
    ).prompt
    carrier_prompt = build_help_prompt_result(
        {
            **base_context,
            "scenario_profile": "carrier",
            "gates": {
                "S12.completion": carrier_gates["S12.completion"],
                "S23.completion": carrier_gates["S23.completion"],
            },
        },
        "en",
    ).prompt

    assert '"scenario_profile":"airfield"' in airfield_prompt
    assert '"scenario_profile":"carrier"' in carrier_prompt
    assert "GND for airfield startup" in airfield_prompt
    assert "CV for carrier startup" in carrier_prompt
