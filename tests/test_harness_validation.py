from __future__ import annotations

from core.harness_validation import (
    HarnessActionHintFactRule,
    HarnessCompletionAdvance,
    HarnessTextGuidanceRule,
    plan_harness_action,
)
from core.step_harness import RecoveryActionPolicy, SignalQualityRequirement, StepHarnessSpec


def _spec(
    step_id: str,
    targets: tuple[str, ...],
    *,
    overlay_enabled: bool = True,
    recovery_kind: str = "retry_allowed_targets",
) -> StepHarnessSpec:
    return StepHarnessSpec(
        step_id=step_id,
        required_observables=("gate",),
        optional_observables=(),
        telemetry_facts=(f"VARS.{step_id.lower()}_done",),
        vision_facts=(),
        recent_action_facts=tuple(f"RECENT_ACTIONS.{target}" for target in targets),
        completion_predicates=(f"GATES.{step_id}.completion",),
        latch_semantics="none",
        allowed_overlay_targets=targets if overlay_enabled else (),
        forbidden_overlay_targets=() if overlay_enabled else targets,
        recovery_policy=RecoveryActionPolicy(kind=recovery_kind, allowed_targets=targets),
        observability_status="observable",
        signal_quality=SignalQualityRequirement(),
        requires_visual_confirmation=False,
        declared_ui_targets=targets,
        overlay_enabled=overlay_enabled,
    )


def test_plan_harness_action_repairs_invalid_target_to_step_spec_target() -> None:
    plan = plan_harness_action(
        step_specs={"S20": _spec("S20", ("refuel_probe_switch",))},
        inferred_step_id="S20",
        model_step_id="S20",
        proposed_overlay_targets=["launch_bar_switch", "refuel_probe_switch"],
        candidate_step_ids=["S20", "S21"],
        runtime_overlay_targets=["refuel_probe_switch", "launch_bar_switch"],
        request_overlay_targets=["refuel_probe_switch", "launch_bar_switch"],
        allowed_evidence_refs=["GATES.S20.completion"],
        evidence_refs=["GATES.S20.completion"],
        max_overlay_targets=1,
    )

    assert plan.targets == ("refuel_probe_switch",)
    assert plan.validator_rejected is True
    assert plan.repair_applied is True
    assert plan.final_action_plan_source == "validator_repair"
    assert "target_not_allowed:launch_bar_switch" in plan.reasons


def test_plan_harness_action_advances_s19_final_go_to_s20() -> None:
    plan = plan_harness_action(
        step_specs={
            "S19": _spec("S19", ("fcs_bit_switch", "right_mdi_pb5"), recovery_kind="visual_confirmation"),
            "S20": _spec("S20", ("refuel_probe_switch",)),
        },
        inferred_step_id="S19",
        model_step_id="S19",
        proposed_overlay_targets=["fcs_bit_switch"],
        candidate_step_ids=["S19", "S20"],
        runtime_overlay_targets=["fcs_bit_switch", "right_mdi_pb5", "refuel_probe_switch"],
        request_overlay_targets=["fcs_bit_switch", "right_mdi_pb5", "refuel_probe_switch"],
        allowed_evidence_refs=["VISION_FACTS.fcsmc_final_go_result_visible@frame-1"],
        evidence_refs=["VISION_FACTS.fcsmc_final_go_result_visible@frame-1"],
        max_overlay_targets=2,
        vision_seen_fact_ids=["fcsmc_final_go_result_visible"],
        completion_advancements=[
            HarnessCompletionAdvance(
                step_id="S19",
                fact_id="fcsmc_final_go_result_visible",
                next_step_id="S20",
                source="validator_s19_final_go",
            )
        ],
    )

    assert plan.step_id == "S20"
    assert plan.overlay_step_id == "S20"
    assert plan.targets == ("refuel_probe_switch",)
    assert plan.rejected_model_step_id == "S19"
    assert plan.validator_rejected is True
    assert plan.repair_applied is True
    assert plan.final_action_plan_source == "validator_s19_final_go"


def test_plan_harness_action_keeps_s19_intermediate_multi_target_guidance() -> None:
    plan = plan_harness_action(
        step_specs={"S19": _spec("S19", ("fcs_bit_switch", "right_mdi_pb5"), recovery_kind="visual_confirmation")},
        inferred_step_id="S19",
        model_step_id="S19",
        proposed_overlay_targets=["right_mdi_pb5"],
        candidate_step_ids=["S19"],
        runtime_overlay_targets=["fcs_bit_switch", "right_mdi_pb5"],
        request_overlay_targets=["fcs_bit_switch", "right_mdi_pb5"],
        allowed_evidence_refs=["VISION_FACTS.fcsmc_intermediate_result_visible@frame-1"],
        evidence_refs=["VISION_FACTS.fcsmc_intermediate_result_visible@frame-1"],
        max_overlay_targets=2,
        vision_seen_fact_ids=["fcsmc_intermediate_result_visible"],
        action_hint={"targets": ["fcs_bit_switch", "right_mdi_pb5"]},
        action_hint_fact_rules=[
            HarnessActionHintFactRule(step_id="S19", fact_id="fcsmc_intermediate_result_visible")
        ],
    )

    assert plan.step_id == "S19"
    assert plan.targets == ("fcs_bit_switch", "right_mdi_pb5")
    assert plan.repair_applied is True
    assert plan.final_action_plan_source == "validator_action_hint"


def test_plan_harness_action_uses_single_current_target_for_s20_to_s27() -> None:
    plan = plan_harness_action(
        step_specs={
            "S26": _spec("S26", ("pitot_heater_switch",)),
            "S20": _spec("S20", ("refuel_probe_switch",)),
        },
        inferred_step_id="S26",
        model_step_id="S26",
        proposed_overlay_targets=["refuel_probe_switch"],
        candidate_step_ids=["S26", "S27"],
        runtime_overlay_targets=["refuel_probe_switch", "pitot_heater_switch"],
        request_overlay_targets=["refuel_probe_switch", "pitot_heater_switch"],
        allowed_evidence_refs=["GATES.S26.completion"],
        evidence_refs=["GATES.S26.completion"],
        max_overlay_targets=2,
        action_hint={"target": "pitot_heater_switch"},
        action_hint_step_ids=["S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27"],
    )

    assert plan.targets == ("pitot_heater_switch",)
    assert plan.validator_rejected is True
    assert plan.repair_applied is True
    assert plan.final_action_plan_source == "validator_action_hint"


def test_plan_harness_action_returns_text_only_for_unhighlightable_manual_control() -> None:
    plan = plan_harness_action(
        step_specs={
            "S11": _spec(
                "S11",
                ("throttle_quadrant_reference",),
                overlay_enabled=False,
                recovery_kind="manual_guidance",
            )
        },
        inferred_step_id="S11",
        model_step_id="S11",
        proposed_overlay_targets=["throttle_quadrant_reference"],
        candidate_step_ids=["S11"],
        runtime_overlay_targets=["throttle_quadrant_reference"],
        request_overlay_targets=["throttle_quadrant_reference"],
        allowed_evidence_refs=["GATES.S11.completion"],
        evidence_refs=["GATES.S11.completion"],
        max_overlay_targets=1,
        text_guidance_rules=[
            HarnessTextGuidanceRule(
                step_id="S11",
                target="throttle_quadrant_reference",
                guidance="Press Right Alt+Home.",
            )
        ],
    )

    assert plan.text_only is True
    assert plan.targets == ()
    assert "Right Alt+Home" in plan.guidance
    assert plan.validator_rejected is True
    assert plan.repair_applied is True
    assert plan.final_action_plan_source == "validator_text_only_guidance"


def test_plan_harness_action_does_not_reject_model_step_that_already_advanced_to_completion_next() -> None:
    plan = plan_harness_action(
        step_specs={
            "S19": _spec("S19", ("fcs_bit_switch", "right_mdi_pb5"), recovery_kind="visual_confirmation"),
            "S20": _spec("S20", ("refuel_probe_switch",)),
        },
        inferred_step_id="S19",
        model_step_id="S20",
        proposed_overlay_targets=["refuel_probe_switch"],
        candidate_step_ids=["S19", "S20"],
        runtime_overlay_targets=["fcs_bit_switch", "right_mdi_pb5", "refuel_probe_switch"],
        request_overlay_targets=["fcs_bit_switch", "right_mdi_pb5", "refuel_probe_switch"],
        allowed_evidence_refs=["VISION_FACTS.fcsmc_final_go_result_visible@frame-1"],
        evidence_refs=["VISION_FACTS.fcsmc_final_go_result_visible@frame-1"],
        max_overlay_targets=1,
        vision_seen_fact_ids=["fcsmc_final_go_result_visible"],
        completion_advancements=[
            HarnessCompletionAdvance(
                step_id="S19",
                fact_id="fcsmc_final_go_result_visible",
                next_step_id="S20",
                source="validator_s19_final_go",
            )
        ],
    )

    assert plan.step_id == "S20"
    assert plan.rejected_model_step_id is None
    assert "model_step_mismatch:S20" not in plan.reasons


def test_plan_harness_action_rejects_overlay_for_disabled_spec_even_with_unrelated_target() -> None:
    plan = plan_harness_action(
        step_specs={
            "S11": _spec(
                "S11",
                ("throttle_quadrant_reference",),
                overlay_enabled=False,
                recovery_kind="manual_guidance",
            )
        },
        inferred_step_id="S11",
        model_step_id="S11",
        proposed_overlay_targets=["battery_switch"],
        candidate_step_ids=["S11"],
        runtime_overlay_targets=["battery_switch", "throttle_quadrant_reference"],
        request_overlay_targets=["battery_switch", "throttle_quadrant_reference"],
        allowed_evidence_refs=["GATES.S11.completion"],
        evidence_refs=["GATES.S11.completion"],
        max_overlay_targets=1,
    )

    assert plan.targets == ()
    assert plan.validator_rejected is True
    assert "target_not_allowed:battery_switch" in plan.reasons


def test_plan_harness_action_records_rejected_target_when_action_hint_repairs_s08() -> None:
    plan = plan_harness_action(
        step_specs={
            "S08": _spec(
                "S08",
                ("left_mdi_brightness_selector", "left_mdi_pb18", "left_mdi_pb15"),
            )
        },
        inferred_step_id="S08",
        model_step_id="S08",
        proposed_overlay_targets=["left_mdi_brightness_selector"],
        candidate_step_ids=["S08"],
        runtime_overlay_targets=["left_mdi_brightness_selector", "left_mdi_pb18", "left_mdi_pb15"],
        request_overlay_targets=["left_mdi_brightness_selector", "left_mdi_pb18", "left_mdi_pb15"],
        allowed_evidence_refs=["VISION_FACTS.supt_page_visible@frame-1"],
        evidence_refs=["VISION_FACTS.supt_page_visible@frame-1"],
        max_overlay_targets=1,
        vision_seen_fact_ids=["supt_page_visible", "bit_root_page_visible"],
        action_hint={"target": "left_mdi_pb15", "reason": "SUPT is visible; press PB15."},
        action_hint_step_ids=["S08"],
    )

    assert plan.targets == ("left_mdi_pb15",)
    assert plan.repair_applied is True
    assert plan.validator_rejected is True
    assert plan.final_action_plan_source == "validator_action_hint"
    assert plan.rejected_model_targets == ("left_mdi_brightness_selector",)
    assert "action_hint_target_mismatch:left_mdi_brightness_selector" in plan.reasons
