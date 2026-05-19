from __future__ import annotations

from core.harness_validation import (
    validate_final_evidence_consistency,
    HarnessActionHintFactRule,
    HarnessCompletionAdvance,
    HarnessTextGuidanceRule,
    plan_harness_action,
)
from core.evidence_packet import build_evidence_packet
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


def test_final_evidence_consistency_rejects_missing_condition_satisfied_by_telemetry() -> None:
    context = {
        "vars": {"comm1_freq_134_000": True},
        "deterministic_step_hint": {
            "inferred_step_id": "S09",
            "overlay_step_id": "S09",
            "missing_conditions": ["vars.comm1_freq_134_000==true"],
        },
    }
    packet = build_evidence_packet(context)

    result = validate_final_evidence_consistency(
        accepted_step_id="S09",
        accepted_overlay_targets=["ufc_comm1_channel_selector_pull"],
        accepted_missing_conditions=["vars.comm1_freq_134_000==true"],
        latest_vars=context["vars"],
        evidence_packet=packet,
    )

    assert result.accepted is False
    assert result.validator_rejected is True
    assert result.repair_applied is True
    assert result.rejected_missing_conditions == ("vars.comm1_freq_134_000==true",)
    assert "missing_condition_satisfied_by_latest_telemetry:vars.comm1_freq_134_000==true" in result.reasons


def test_final_evidence_consistency_rejects_numeric_missing_condition_satisfied_by_telemetry() -> None:
    context = {
        "vars": {"rpm_r": 26, "ext_refuel_probe_value": 1200},
        "deterministic_step_hint": {
            "inferred_step_id": "S05",
            "overlay_step_id": "S05",
            "missing_conditions": [
                "vars.rpm_r>=25",
                "vars.ext_refuel_probe_value in [0,5000]",
            ],
        },
    }
    packet = build_evidence_packet(context)

    result = validate_final_evidence_consistency(
        accepted_step_id="S05",
        accepted_overlay_targets=[],
        accepted_missing_conditions=[
            "vars.rpm_r>=25",
            "vars.ext_refuel_probe_value in [0,5000]",
        ],
        latest_vars=context["vars"],
        evidence_packet=packet,
    )

    assert result.accepted is False
    assert result.rejected_missing_conditions == (
        "vars.rpm_r>=25",
        "vars.ext_refuel_probe_value in [0,5000]",
    )


def test_final_evidence_consistency_does_not_use_stale_last_seen_true_when_latest_false() -> None:
    context = {
        "vars": {"apu_start_support_complete": False},
        "telemetry_window_frames": [
            {"seq": 1, "t_wall": 1.0, "vars": {"apu_start_support_complete": True}},
            {"seq": 2, "t_wall": 2.0, "vars": {"apu_start_support_complete": False}},
        ],
        "deterministic_step_hint": {
            "inferred_step_id": "S03",
            "overlay_step_id": "S03",
            "missing_conditions": ["vars.apu_start_support_complete==true"],
        },
    }
    packet = build_evidence_packet(context)

    result = validate_final_evidence_consistency(
        accepted_step_id="S03",
        accepted_overlay_targets=["apu_switch"],
        accepted_missing_conditions=["vars.apu_start_support_complete==true"],
        latest_vars=context["vars"],
        evidence_packet=packet,
    )

    assert result.accepted is True
    assert result.rejected_missing_conditions == ()


def test_final_evidence_consistency_rejects_already_satisfied_completion_gate() -> None:
    context = {
        "vars": {},
        "gates": {"S03.completion": {"status": "satisfied", "step_id": "S03", "gate_type": "completion"}},
        "deterministic_step_hint": {
            "inferred_step_id": "S03",
            "overlay_step_id": "S03",
            "missing_conditions": [],
        },
    }
    packet = build_evidence_packet(context)

    result = validate_final_evidence_consistency(
        accepted_step_id="S03",
        accepted_overlay_targets=["apu_switch"],
        accepted_missing_conditions=[],
        latest_vars=context["vars"],
        evidence_packet=packet,
    )

    assert result.accepted is False
    assert result.rejected_missing_conditions == ()
    assert "completion_gate_already_satisfied:S03" in result.reasons


def test_final_evidence_consistency_rejects_allowed_completion_gate_but_ignores_no_rules() -> None:
    packet = build_evidence_packet(
        {
            "vars": {},
            "gates": {
                "S03.completion": {
                    "status": "allowed",
                    "allowed": True,
                    "step_id": "S03",
                    "gate_type": "completion",
                    "reason_code": "ok",
                },
                "S04.completion": {
                    "status": "allowed",
                    "allowed": True,
                    "step_id": "S04",
                    "gate_type": "completion",
                    "reason_code": "no_rules",
                },
            },
        }
    )

    rejected = validate_final_evidence_consistency(
        accepted_step_id="S03",
        accepted_overlay_targets=[],
        accepted_missing_conditions=[],
        latest_vars={},
        evidence_packet=packet,
    )
    accepted = validate_final_evidence_consistency(
        accepted_step_id="S04",
        accepted_overlay_targets=[],
        accepted_missing_conditions=[],
        latest_vars={},
        evidence_packet=packet,
    )

    assert rejected.accepted is False
    assert "completion_gate_already_satisfied:S03" in rejected.reasons
    assert accepted.accepted is True


def test_final_evidence_consistency_rejects_late_allowed_completion_gate_after_detail_window() -> None:
    gates = {
        f"S{idx:02d}.completion": {
            "status": "allowed",
            "allowed": True,
            "step_id": f"S{idx:02d}",
            "gate_type": "completion",
            "reason_code": "ok",
        }
        for idx in range(1, 25)
    }
    packet = build_evidence_packet({"vars": {}, "gates": gates})

    result = validate_final_evidence_consistency(
        accepted_step_id="S18",
        accepted_overlay_targets=[],
        accepted_missing_conditions=[],
        latest_vars={},
        evidence_packet=packet,
    )

    assert result.accepted is False
    assert "completion_gate_already_satisfied:S18" in result.reasons


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


def test_plan_harness_action_aligns_four_down_overlay_step_with_plan_step() -> None:
    plan = plan_harness_action(
        step_specs={
            "S24": _spec("S24", ("arresting_hook_handle",)),
            "S25": _spec("S25", ("arresting_hook_handle",)),
        },
        inferred_step_id="S24",
        model_step_id="S24",
        overlay_step_id="S25",
        proposed_overlay_targets=["arresting_hook_handle"],
        candidate_step_ids=["S24", "S25"],
        runtime_overlay_targets=["arresting_hook_handle"],
        request_overlay_targets=["arresting_hook_handle"],
        allowed_evidence_refs=["GATES.S24.completion", "GATES.S25.completion"],
        evidence_refs=["GATES.S24.completion"],
        max_overlay_targets=1,
        action_hint={"target": "arresting_hook_handle", "reason": "Raise the arresting hook handle."},
        action_hint_step_ids=["S20", "S21", "S22", "S23", "S24", "S25"],
    )

    assert plan.step_id == "S24"
    assert plan.overlay_step_id == "S24"
    assert plan.targets == ("arresting_hook_handle",)
    assert plan.guidance is None
    assert "overlay_step_aligned_to_step_id:S25" in plan.reasons


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
    expected_targets = {
        "S20": "refuel_probe_switch",
        "S21": "refuel_probe_switch",
        "S22": "launch_bar_switch",
        "S23": "launch_bar_switch",
        "S24": "arresting_hook_handle",
        "S25": "arresting_hook_handle",
        "S26": "pitot_heater_switch",
        "S27": "flap_switch",
    }
    wrong_target = "battery_switch"
    step_specs = {
        step_id: _spec(step_id, (target,))
        for step_id, target in expected_targets.items()
    }
    runtime_targets = [wrong_target, *expected_targets.values()]

    for step_id, expected_target in expected_targets.items():
        plan = plan_harness_action(
            step_specs=step_specs,
            inferred_step_id=step_id,
            model_step_id=step_id,
            proposed_overlay_targets=[wrong_target],
            candidate_step_ids=[step_id],
            runtime_overlay_targets=runtime_targets,
            request_overlay_targets=runtime_targets,
            allowed_evidence_refs=[f"GATES.{step_id}.completion"],
            evidence_refs=[f"GATES.{step_id}.completion"],
            max_overlay_targets=2,
            action_hint={"target": expected_target},
            action_hint_step_ids=["S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27"],
        )

        assert plan.targets == (expected_target,)
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


def test_plan_harness_action_repairs_s18_bit_root_to_pb5_when_fcsmc_not_seen() -> None:
    plan = plan_harness_action(
        step_specs={
            "S18": _spec(
                "S18",
                ("right_mdi_pb18", "right_mdi_pb5"),
                recovery_kind="visual_confirmation",
            )
        },
        inferred_step_id="S18",
        model_step_id="S18",
        proposed_overlay_targets=["right_mdi_pb18"],
        candidate_step_ids=["S18"],
        runtime_overlay_targets=["right_mdi_pb18", "right_mdi_pb5"],
        request_overlay_targets=["right_mdi_pb18", "right_mdi_pb5"],
        allowed_evidence_refs=["VISION_FACTS.bit_root_page_visible@frame-1"],
        evidence_refs=["VISION_FACTS.bit_root_page_visible@frame-1"],
        max_overlay_targets=1,
        vision_seen_fact_ids=["bit_root_page_visible"],
        vision_not_seen_fact_ids=["fcsmc_page_visible"],
        action_hint={
            "target": "right_mdi_pb5",
            "reason": "BIT root is visible; press PB5 FCS-MC.",
        },
        action_hint_fact_rules=[
            HarnessActionHintFactRule(
                step_id="S18",
                fact_id="bit_root_page_visible",
                not_seen_fact_id="fcsmc_page_visible",
                source="visual_action_hint_repair",
            )
        ],
    )

    assert plan.targets == ("right_mdi_pb5",)
    assert plan.guidance == "BIT root is visible; press PB5 FCS-MC."
    assert plan.validator_rejected is True
    assert plan.repair_applied is True
    assert plan.final_action_plan_source == "visual_action_hint_repair"
    assert plan.rejected_model_targets == ("right_mdi_pb18",)
    assert "action_hint_target_mismatch:right_mdi_pb18" in plan.reasons


def test_plan_harness_action_keeps_s18_recovery_pb18_when_bit_root_not_visible() -> None:
    plan = plan_harness_action(
        step_specs={
            "S18": _spec(
                "S18",
                ("right_mdi_pb18", "right_mdi_pb5"),
                recovery_kind="visual_confirmation",
            )
        },
        inferred_step_id="S18",
        model_step_id="S18",
        proposed_overlay_targets=["right_mdi_pb18"],
        candidate_step_ids=["S18"],
        runtime_overlay_targets=["right_mdi_pb18", "right_mdi_pb5"],
        request_overlay_targets=["right_mdi_pb18", "right_mdi_pb5"],
        allowed_evidence_refs=["VISION_FACTS.fcsmc_page_visible@frame-1"],
        evidence_refs=["VISION_FACTS.fcsmc_page_visible@frame-1"],
        max_overlay_targets=1,
        vision_not_seen_fact_ids=["bit_root_page_visible"],
        action_hint={"target": "right_mdi_pb5"},
        action_hint_fact_rules=[
            HarnessActionHintFactRule(
                step_id="S18",
                fact_id="bit_root_page_visible",
                not_seen_fact_id="fcsmc_page_visible",
                source="visual_action_hint_repair",
            )
        ],
    )

    assert plan.targets == ("right_mdi_pb18",)
    assert plan.repair_applied is False
    assert plan.validator_rejected is False
    assert plan.final_action_plan_source == "model"


def test_plan_harness_action_uses_s18_visual_state_without_live_hint() -> None:
    plan = plan_harness_action(
        step_specs={
            "S18": _spec(
                "S18",
                ("right_mdi_pb18", "right_mdi_pb5"),
                recovery_kind="visual_confirmation",
            )
        },
        inferred_step_id="S18",
        model_step_id="S18",
        proposed_overlay_targets=["right_mdi_pb18"],
        candidate_step_ids=["S18"],
        runtime_overlay_targets=["right_mdi_pb18", "right_mdi_pb5"],
        request_overlay_targets=["right_mdi_pb18", "right_mdi_pb5"],
        max_overlay_targets=1,
        vision_seen_fact_ids=["bit_root_page_visible"],
        vision_not_seen_fact_ids=["fcsmc_page_visible"],
    )

    assert plan.targets == ("right_mdi_pb5",)
    assert plan.final_action_plan_source == "state_action_planner"
    assert "state_action_target_mismatch:right_mdi_pb18" in plan.reasons


def test_plan_harness_action_clears_unavailable_s18_state_target() -> None:
    plan = plan_harness_action(
        step_specs={
            "S18": _spec(
                "S18",
                ("right_mdi_pb18", "right_mdi_pb5"),
                recovery_kind="visual_confirmation",
            )
        },
        inferred_step_id="S18",
        model_step_id="S18",
        proposed_overlay_targets=["right_mdi_pb18"],
        candidate_step_ids=["S18"],
        runtime_overlay_targets=["right_mdi_pb18"],
        request_overlay_targets=["right_mdi_pb18"],
        max_overlay_targets=1,
        vision_seen_fact_ids=["bit_root_page_visible"],
        vision_not_seen_fact_ids=["fcsmc_page_visible"],
    )

    assert plan.text_only is True
    assert plan.targets == ()
    assert "PB5" in (plan.guidance or "")
    assert "target_not_in_runtime_allowlist:right_mdi_pb5" in plan.reasons


def test_plan_harness_action_uses_s08_all_displays_off_power_targets() -> None:
    specs = {
        "S08": _spec(
            "S08",
            (
                "left_mdi_brightness_selector",
                "right_mdi_brightness_selector",
                "ampcd_off_brightness_knob",
                "hud_symbology_brightness_knob",
                "left_mdi_pb18",
                "left_mdi_pb15",
            ),
        )
    }
    allowed_targets = list(specs["S08"].allowed_overlay_targets)

    plan = plan_harness_action(
        step_specs=specs,
        inferred_step_id="S08",
        model_step_id="S08",
        proposed_overlay_targets=["left_mdi_brightness_selector"],
        candidate_step_ids=["S08"],
        runtime_overlay_targets=allowed_targets,
        request_overlay_targets=allowed_targets,
        max_overlay_targets=4,
        latest_vars={
            "left_ddi_on": False,
            "right_ddi_on": False,
            "mpcd_on": False,
            "hud_on": False,
        },
    )

    assert plan.targets == (
        "left_mdi_brightness_selector",
        "right_mdi_brightness_selector",
        "ampcd_off_brightness_knob",
        "hud_symbology_brightness_knob",
    )
    assert "left_mdi_pb18" not in plan.targets
    assert "left_mdi_pb15" not in plan.targets
    assert plan.final_action_plan_source == "state_action_planner"
    assert "warm" in (plan.guidance or "").lower()


def test_plan_harness_action_keeps_s08_all_displays_off_single_target_when_max_below_four() -> None:
    specs = {
        "S08": _spec(
            "S08",
            (
                "left_mdi_brightness_selector",
                "right_mdi_brightness_selector",
                "ampcd_off_brightness_knob",
                "hud_symbology_brightness_knob",
            ),
        )
    }
    allowed_targets = list(specs["S08"].allowed_overlay_targets)

    plan = plan_harness_action(
        step_specs=specs,
        inferred_step_id="S08",
        model_step_id="S08",
        proposed_overlay_targets=["left_mdi_brightness_selector", "right_mdi_brightness_selector"],
        candidate_step_ids=["S08"],
        runtime_overlay_targets=allowed_targets,
        request_overlay_targets=allowed_targets,
        max_overlay_targets=2,
        latest_vars={
            "left_ddi_on": False,
            "right_ddi_on": False,
            "mpcd_on": False,
            "hud_on": False,
        },
    )

    assert plan.targets == ("left_mdi_brightness_selector",)
    assert plan.final_action_plan_source == "state_action_planner"
    assert "state_action_target_mismatch:right_mdi_brightness_selector" in plan.reasons


def test_plan_harness_action_uses_s09_scratchpad_state_without_live_hint() -> None:
    specs = {
        "S09": _spec(
            "S09",
            (
                "ufc_comm1_channel_selector_pull",
                "ufc_key_1",
                "ufc_key_3",
                "ufc_key_4",
                "ufc_key_0",
                "ufc_ent_button",
            ),
        )
    }
    allowed_targets = list(specs["S09"].allowed_overlay_targets)
    cases = [
        (
            {
                "ufc_scratchpad_string_1_display": "1-",
                "ufc_scratchpad_string_2_display": "-",
                "ufc_scratchpad_number_display": "     .1",
            },
            "ufc_key_3",
        ),
        (
            {
                "ufc_scratchpad_string_1_display": "1-",
                "ufc_scratchpad_string_2_display": "-",
                "ufc_scratchpad_number_display": "    .13",
            },
            "ufc_key_4",
        ),
        (
            {
                "ufc_scratchpad_string_1_display": "1-",
                "ufc_scratchpad_string_2_display": "-",
                "ufc_scratchpad_number_display": "   .134",
            },
            "ufc_key_0",
        ),
        (
            {
                "ufc_scratchpad_string_1_display": "1-",
                "ufc_scratchpad_string_2_display": "-",
                "ufc_scratchpad_number_display": "134.000",
            },
            "ufc_ent_button",
        ),
    ]

    for vars_map, expected_target in cases:
        plan = plan_harness_action(
            step_specs=specs,
            inferred_step_id="S09",
            model_step_id="S09",
            proposed_overlay_targets=["ufc_comm1_channel_selector_pull"],
            candidate_step_ids=["S09"],
            runtime_overlay_targets=allowed_targets,
            request_overlay_targets=allowed_targets,
            max_overlay_targets=1,
            latest_vars={"comm1_freq_134_000": False, **vars_map},
            recent_action_targets=[],
        )

        assert plan.targets == (expected_target,)
        assert plan.final_action_plan_source == "state_action_planner"


def test_plan_harness_action_uses_s09_numeric_sequence_for_initial_entry() -> None:
    specs = {
        "S09": _spec(
            "S09",
            (
                "ufc_comm1_channel_selector_pull",
                "ufc_key_1",
                "ufc_key_3",
                "ufc_key_4",
                "ufc_key_0",
                "ufc_ent_button",
            ),
        )
    }
    allowed_targets = list(specs["S09"].allowed_overlay_targets)
    plan = plan_harness_action(
        step_specs=specs,
        inferred_step_id="S09",
        model_step_id="S09",
        proposed_overlay_targets=["ufc_comm1_channel_selector_pull"],
        candidate_step_ids=["S09"],
        runtime_overlay_targets=allowed_targets,
        request_overlay_targets=allowed_targets,
        max_overlay_targets=4,
        latest_vars={
            "comm1_freq_134_000": False,
            "ufc_comm1_pull_pressed": True,
            "ufc_scratchpad_string_1_display": "1-",
            "ufc_scratchpad_string_2_display": "-",
            "ufc_scratchpad_number_display": "305.000",
        },
        recent_action_targets=[],
    )

    assert plan.targets == ("ufc_key_1", "ufc_key_3", "ufc_key_4", "ufc_key_0")
    assert plan.final_action_plan_source == "state_action_planner"
    assert "134.000" in (plan.guidance or "")
    assert "1-3-4-0-0-0" in (plan.guidance or "")
    assert "ENT" in (plan.guidance or "")


def test_plan_harness_action_requires_s09_comm1_pull_before_numeric_sequence() -> None:
    specs = {
        "S09": _spec(
            "S09",
            (
                "ufc_comm1_channel_selector_pull",
                "ufc_key_1",
                "ufc_key_3",
                "ufc_key_4",
                "ufc_key_0",
                "ufc_ent_button",
            ),
        )
    }
    allowed_targets = list(specs["S09"].allowed_overlay_targets)

    plan = plan_harness_action(
        step_specs=specs,
        inferred_step_id="S09",
        model_step_id="S09",
        proposed_overlay_targets=["ufc_comm1_channel_selector_pull"],
        candidate_step_ids=["S09"],
        runtime_overlay_targets=allowed_targets,
        request_overlay_targets=allowed_targets,
        max_overlay_targets=4,
        latest_vars={
            "comm1_freq_134_000": False,
            "comm1_freq_value": 30500,
            "ufc_comm1_pull_pressed": False,
            "ufc_scratchpad_number_display": "305.000",
            "ufc_scratchpad_string_1_display": "  ",
            "ufc_scratchpad_string_2_display": "  ",
        },
        recent_action_targets=[],
    )

    assert plan.targets == ("ufc_comm1_channel_selector_pull",)
    assert plan.final_action_plan_source == "state_action_planner"
    assert "134.000" in (plan.guidance or "")
    assert "1-3-4-0-0-0" not in (plan.guidance or "")


def test_plan_harness_action_uses_recent_action_for_s09_open_scratchpad() -> None:
    specs = {
        "S09": _spec(
            "S09",
            (
                "ufc_comm1_channel_selector_pull",
                "ufc_key_1",
                "ufc_key_3",
                "ufc_key_4",
                "ufc_key_0",
                "ufc_ent_button",
            ),
        )
    }
    allowed_targets = list(specs["S09"].allowed_overlay_targets)

    plan = plan_harness_action(
        step_specs=specs,
        inferred_step_id="S09",
        model_step_id="S09",
        proposed_overlay_targets=["ufc_comm1_channel_selector_pull"],
        candidate_step_ids=["S09"],
        runtime_overlay_targets=allowed_targets,
        request_overlay_targets=allowed_targets,
        max_overlay_targets=4,
        latest_vars={"comm1_freq_134_000": False},
        recent_action_targets=["ufc_comm1_channel_selector_pull"],
    )

    assert plan.targets == ("ufc_key_1", "ufc_key_3", "ufc_key_4", "ufc_key_0")
    assert plan.final_action_plan_source == "state_action_planner"


def test_plan_harness_action_clears_unavailable_s09_state_target_instead_of_repairing_to_pull() -> None:
    specs = {
        "S09": _spec(
            "S09",
            (
                "ufc_comm1_channel_selector_pull",
                "ufc_key_1",
                "ufc_key_3",
                "ufc_key_4",
                "ufc_key_0",
                "ufc_ent_button",
            ),
        )
    }

    plan = plan_harness_action(
        step_specs=specs,
        inferred_step_id="S09",
        model_step_id="S09",
        proposed_overlay_targets=["ufc_comm1_channel_selector_pull"],
        candidate_step_ids=["S09"],
        runtime_overlay_targets=["ufc_comm1_channel_selector_pull"],
        request_overlay_targets=["ufc_comm1_channel_selector_pull"],
        max_overlay_targets=1,
        latest_vars={
            "comm1_freq_134_000": False,
            "ufc_scratchpad_string_1_display": "1-",
            "ufc_scratchpad_string_2_display": "-",
            "ufc_scratchpad_number_display": "     .1",
        },
    )

    assert plan.text_only is True
    assert plan.targets == ()
    assert "press 3 next" in (plan.guidance or "")
    assert "target_not_in_runtime_allowlist:ufc_key_3" in plan.reasons
    assert plan.final_action_plan_source == "state_action_planner"


def test_plan_harness_action_owns_known_control_interaction_guidance() -> None:
    cases = [
        (
            "S10",
            "eng_crank_switch",
            {"engine_crank_left_complete": False},
            ("LEFT", "left-click"),
            "s10_left_engine_left_click_guidance",
        ),
        (
            "S31",
            "radar_altimeter_bug_knob",
            {"radar_altimeter_bug_set": False},
            ("mouse wheel", "200 ft", "40 ft"),
            "s31_radar_altimeter_mouse_wheel_guidance",
        ),
        (
            "S32",
            "standby_attitude_cage_knob",
            {"standby_attitude_uncaged": False},
            ("mouse wheel", "standby attitude"),
            "s32_standby_attitude_mouse_wheel_guidance",
        ),
    ]

    for step_id, target, vars_map, expected_parts, expected_reason in cases:
        plan = plan_harness_action(
            step_specs={step_id: _spec(step_id, (target,))},
            inferred_step_id=step_id,
            model_step_id=step_id,
            proposed_overlay_targets=[target],
            candidate_step_ids=[step_id],
            runtime_overlay_targets=[target],
            request_overlay_targets=[target],
            max_overlay_targets=1,
            latest_vars=vars_map,
        )

        assert plan.targets == (target,)
        assert plan.final_action_plan_source == "state_action_planner"
        assert expected_reason in plan.reasons
        guidance = plan.guidance or ""
        for expected in expected_parts:
            assert expected in guidance


def test_plan_harness_action_returns_text_only_when_probe_is_already_moving() -> None:
    cases = [
        (
            "S20",
            {"probe_switch_value": 2, "ext_refuel_probe_value": 12000},
            [
                {"seq": 1, "t_wall": 1.0, "vars": {"ext_refuel_probe_value": 8000}},
                {"seq": 2, "t_wall": 2.0, "vars": {"ext_refuel_probe_value": 12000}},
            ],
            "extending",
        ),
        (
            "S21",
            {"probe_switch_value": 1, "ext_refuel_probe_value": 5600},
            [
                {"seq": 1, "t_wall": 1.0, "vars": {"ext_refuel_probe_value": 6200}},
                {"seq": 2, "t_wall": 2.0, "vars": {"ext_refuel_probe_value": 5600}},
            ],
            "retracting",
        ),
    ]

    for step_id, vars_map, frames, expected_word in cases:
        packet = build_evidence_packet({"vars": vars_map, "telemetry_window_frames": frames})
        plan = plan_harness_action(
            step_specs={step_id: _spec(step_id, ("refuel_probe_switch",))},
            inferred_step_id=step_id,
            model_step_id=step_id,
            proposed_overlay_targets=["refuel_probe_switch"],
            candidate_step_ids=[step_id],
            runtime_overlay_targets=["refuel_probe_switch"],
            request_overlay_targets=["refuel_probe_switch"],
            max_overlay_targets=1,
            latest_vars=vars_map,
            evidence_packet=packet,
        )

        assert plan.text_only is True
        assert plan.targets == ()
        assert expected_word in (plan.guidance or "")
        assert plan.final_action_plan_source == "state_action_planner_wait"


def test_plan_harness_action_clears_partial_s19_multi_target_plan() -> None:
    plan = plan_harness_action(
        step_specs={"S19": _spec("S19", ("fcs_bit_switch", "right_mdi_pb5"))},
        inferred_step_id="S19",
        model_step_id="S19",
        proposed_overlay_targets=["right_mdi_pb5"],
        candidate_step_ids=["S19"],
        runtime_overlay_targets=["fcs_bit_switch", "right_mdi_pb5"],
        request_overlay_targets=["fcs_bit_switch", "right_mdi_pb5"],
        max_overlay_targets=1,
        vision_seen_fact_ids=["fcsmc_intermediate_result_visible"],
    )

    assert plan.text_only is True
    assert plan.targets == ()
    assert "PB5" in (plan.guidance or "")
    assert "target_dropped_by_max_overlay_targets:right_mdi_pb5" in plan.reasons


def test_plan_harness_action_expands_incomplete_s19_intermediate_hint_to_atomic_pair() -> None:
    plan = plan_harness_action(
        step_specs={"S19": _spec("S19", ("fcs_bit_switch", "right_mdi_pb5"))},
        inferred_step_id="S19",
        model_step_id="S19",
        proposed_overlay_targets=["fcs_bit_switch"],
        candidate_step_ids=["S19"],
        runtime_overlay_targets=["fcs_bit_switch", "right_mdi_pb5"],
        request_overlay_targets=["fcs_bit_switch", "right_mdi_pb5"],
        max_overlay_targets=2,
        vision_seen_fact_ids=["fcsmc_intermediate_result_visible"],
        action_hint={"target": "fcs_bit_switch"},
        action_hint_fact_rules=[
            HarnessActionHintFactRule(step_id="S19", fact_id="fcsmc_intermediate_result_visible")
        ],
    )

    assert plan.text_only is False
    assert plan.targets == ("fcs_bit_switch", "right_mdi_pb5")
    assert "action_hint_incomplete_for_state_plan:fcs_bit_switch" in plan.reasons
    assert plan.final_action_plan_source == "state_action_planner"
