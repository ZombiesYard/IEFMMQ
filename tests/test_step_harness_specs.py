from pathlib import Path

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

    s18 = specs["S18"]
    assert s18.observability_status == "partial"
    assert s18.required_observables == ("delta", "gate", "visual")
    assert s18.vision_facts == ("VISION_FACTS.fcsmc_page_visible",)
    assert "VISION_FACTS.fcsmc_page_visible==seen" in s18.completion_predicates
    assert s18.requires_visual_confirmation is True
    assert s18.signal_quality.freshness_ms == 2000
    assert s18.allowed_overlay_targets == ("right_mdi_pb18", "right_mdi_pb5")

    s19 = specs["S19"]
    assert s19.vision_facts == ("VISION_FACTS.fcsmc_final_go_result_visible",)
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
        assert spec.completion_predicates == (f"GATES.{step_id}.completion",)
