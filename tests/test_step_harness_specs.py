from pathlib import Path

import pytest

from adapters.step_harness_specs import load_step_harness_specs
from core.step_harness import StepHarnessSpec


BASE_DIR = Path(__file__).resolve().parent.parent
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"


def test_load_step_harness_specs_exposes_all_fa18c_steps_without_live_dcs_import() -> None:
    specs = load_step_harness_specs(PACK_PATH)

    assert list(specs.keys()) == [f"S{i:02d}" for i in range(1, 34)]
    assert all(isinstance(spec, StepHarnessSpec) for spec in specs.values())


def test_step_harness_spec_derives_early_telemetry_delta_and_gate_contract() -> None:
    specs = load_step_harness_specs(PACK_PATH)

    s01 = specs["S01"]
    assert s01.step_id == "S01"
    assert s01.observability_status == "observable"
    assert s01.required_observables == ("var", "gate", "delta")
    assert s01.telemetry_facts == ("VARS.battery_on", "VARS.l_gen_on", "VARS.r_gen_on")
    assert s01.recent_action_facts == (
        "RECENT_ACTIONS.battery_switch",
        "RECENT_ACTIONS.generator_left_switch",
        "RECENT_ACTIONS.generator_right_switch",
    )
    assert "GATES.S01.completion" in s01.completion_predicates
    assert s01.allowed_overlay_targets == (
        "battery_switch",
        "generator_left_switch",
        "generator_right_switch",
    )
    assert s01.recovery_policy.kind == "retry_allowed_targets"

    s02 = specs["S02"]
    assert s02.required_observables == ("var", "delta", "gate")
    assert s02.telemetry_facts == (
        "VARS.power_available",
        "VARS.fire_test_a_complete",
        "VARS.fire_test_b_complete",
    )
    assert s02.allowed_overlay_targets == ("fire_test_switch",)


def test_step_harness_spec_derives_vlm_contract_for_s18_and_s19() -> None:
    specs = load_step_harness_specs(PACK_PATH)

    s15 = specs["S15"]
    assert s15.vision_facts == ("VISION_FACTS.fcs_page_x_marks_visible",)
    assert s15.requires_visual_confirmation is True
    assert s15.signal_quality.freshness_ms == 2000

    s18 = specs["S18"]
    assert s18.observability_status == "partial"
    assert s18.required_observables == ("delta", "gate", "visual")
    assert s18.vision_facts == (
        "VISION_FACTS.fcsmc_page_visible",
        "VISION_FACTS.bit_root_page_visible",
    )
    assert "VISION_FACTS.fcsmc_page_visible==seen" in s18.completion_predicates
    assert s18.requires_visual_confirmation is True
    assert s18.signal_quality.freshness_ms == 2000
    assert s18.allowed_overlay_targets == ("right_mdi_pb18", "right_mdi_pb5")

    s19 = specs["S19"]
    assert "VISION_FACTS.fcsmc_final_go_result_visible" in s19.vision_facts
    assert "VISION_FACTS.fcsmc_final_go_result_visible==seen" in s19.completion_predicates
    assert s19.latch_semantics == "sticky_visual_completion"
    assert s19.signal_quality.freshness_ms == 600000
    assert s19.recovery_policy.kind == "visual_confirmation"


def test_step_harness_spec_marks_partial_manual_steps_without_forcing_visual_confirmation() -> None:
    specs = load_step_harness_specs(PACK_PATH)

    s17 = specs["S17"]
    assert s17.observability_status == "partial"
    assert s17.requires_visual_confirmation is False
    assert s17.required_observables == ("delta", "gate")
    assert s17.recent_action_facts == ("RECENT_ACTIONS.takeoff_trim_button",)

    s30 = specs["S30"]
    assert s30.observability_status == "partial"
    assert s30.requires_visual_confirmation is False
    assert s30.required_observables == ("delta", "rag", "gate")
    assert s30.allowed_overlay_targets == ("standby_altimeter_pressure_knob",)


def test_step_harness_spec_preserves_s20_to_s27_split_four_down_targets() -> None:
    specs = load_step_harness_specs(PACK_PATH)

    expected_targets = {
        "S20": ("refuel_probe_switch",),
        "S21": ("refuel_probe_switch",),
        "S22": ("launch_bar_switch",),
        "S23": ("launch_bar_switch",),
        "S24": ("arresting_hook_handle",),
        "S25": ("arresting_hook_handle",),
        "S26": ("pitot_heater_switch",),
        "S27": ("flap_switch",),
    }
    for step_id, targets in expected_targets.items():
        spec = specs[step_id]
        assert spec.allowed_overlay_targets == targets
        assert spec.observability_status == "observable"
        assert spec.required_observables == ("var", "gate")
        assert f"GATES.{step_id}.completion" in spec.completion_predicates


def test_step_harness_spec_keeps_raw_ui_targets_when_overlay_is_disabled(tmp_path: Path) -> None:
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    observability: partial\n"
        "    overlay_enabled: false\n"
        "    evidence_requirements: [delta]\n"
        "    ui_targets: [manual_switch]\n"
        "precondition_gates: {S01: []}\n"
        "completion_gates: {S01: []}\n",
        encoding="utf-8",
    )

    spec = load_step_harness_specs(pack)["S01"]

    assert spec.allowed_overlay_targets == ()
    assert spec.forbidden_overlay_targets == ("manual_switch",)
    assert spec.declared_ui_targets == ("manual_switch",)
    assert spec.recent_action_facts == ("RECENT_ACTIONS.manual_switch",)
    assert spec.recovery_policy.allowed_targets == ("manual_switch",)


def test_step_harness_spec_validates_pack_step_metadata(tmp_path: Path) -> None:
    pack = tmp_path / "pack.yaml"
    pack.write_text(
        "pack_id: test\n"
        "version: v1\n"
        "steps:\n"
        "  - id: S01\n"
        "    observability: maybe\n"
        "precondition_gates: {S01: []}\n"
        "completion_gates: {S01: []}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"pack\.steps\[0\]\.observability must be one of"):
        load_step_harness_specs(pack)


def test_step_harness_spec_completion_predicates_include_scenario_overrides() -> None:
    airfield = load_step_harness_specs(PACK_PATH)["S31"]
    carrier = load_step_harness_specs(PACK_PATH, scenario_profile="carrier")["S31"]

    assert "vars.radar_altimeter_bug_value in [180,220]" in airfield.completion_predicates
    assert "vars.radar_altimeter_bug_value in [30,60]" in carrier.completion_predicates
