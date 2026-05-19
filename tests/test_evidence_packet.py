from types import SimpleNamespace

from core.evidence_packet import (
    CONFLICT_LATCH_MISSING_VS_LATER_STAGE_EVIDENCE,
    CONFLICT_RECENT_ACTION_VS_GATE_CONTRADICTION,
    CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS,
    build_step_candidates,
    build_evidence_packet,
)


def test_evidence_packet_marks_bootstrap_telemetry_and_missing_variables() -> None:
    packet = build_evidence_packet(
        {
            "vars": {
                "battery_on": False,
                "power_available": False,
                "vars_source_missing": [f"missing_{idx}" for idx in range(25)],
            },
            "vision": {"observation_seq": 2},
            "telemetry": {"t_wall": 12.5},
        }
    )

    payload = packet.to_dict()

    assert payload["telemetry_evidence"]["source_status"] == "low_confidence_bootstrap"
    assert payload["telemetry_evidence"]["confidence"] == "low"
    assert payload["telemetry_evidence"]["missing_source_count"] == 25
    assert payload["telemetry_evidence"]["missing_source_ids"][:2] == ["missing_0", "missing_1"]
    assert payload["telemetry_evidence"]["freshness"] == {"observation_seq": 2, "t_wall": 12.5}


def test_evidence_packet_keeps_vlm_late_display_anchors_and_sticky_metadata() -> None:
    packet = build_evidence_packet(
        {
            "vision_fact_summary": {
                "status": "available",
                "frame_ids": ["frame-001"],
                "fresh_fact_ids": ["tac_page_visible", "bit_root_page_visible"],
                "seen_fact_ids": ["tac_page_visible", "bit_root_page_visible"],
                "not_seen_fact_ids": ["fcs_page_visible"],
                "uncertain_fact_ids": ["supt_page_visible"],
            },
            "vision_facts": [
                {
                    "fact_id": "fcsmc_final_go_result_visible",
                    "state": "seen",
                    "source_frame_id": "frame-009",
                    "sticky": True,
                    "expires_after_ms": 600000,
                }
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S01",
                "missing_conditions": ["vars.fire_test_a_complete==true"],
            },
        }
    )

    payload = packet.to_dict()

    assert payload["vision_evidence"]["late_display_anchors"] == [
        "bit_root_page_visible",
        "fcsmc_final_go_result_visible",
        "tac_page_visible",
    ]
    assert payload["vision_evidence"]["visual_candidate_steps"] == ["S08", "S09", "S18", "S19", "S20"]
    assert payload["vision_evidence"]["source_frame_ids"] == ["frame-001", "frame-009"]
    assert payload["vision_evidence"]["facts"][0]["sticky"] is True
    assert payload["vision_evidence"]["facts"][0]["expires_after_ms"] == 600000
    assert CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS in payload["conflicts"]
    assert CONFLICT_LATCH_MISSING_VS_LATER_STAGE_EVIDENCE in payload["conflicts"]


def test_evidence_packet_uses_raw_vlm_facts_when_summary_is_missing() -> None:
    packet = build_evidence_packet(
        {
            "vision_facts": [
                {
                    "fact_id": "fcsmc_final_go_result_visible",
                    "state": "seen",
                    "source_frame_id": "frame-raw",
                    "sticky": True,
                    "expires_after_ms": 600000,
                }
            ],
            "deterministic_step_hint": {"inferred_step_id": "S01"},
        }
    )

    payload = packet.to_dict()

    assert payload["vision_evidence"]["source_status"] == "available"
    assert payload["vision_evidence"]["seen_fact_ids"] == ["fcsmc_final_go_result_visible"]
    assert payload["vision_evidence"]["late_display_anchors"] == ["fcsmc_final_go_result_visible"]
    assert payload["vision_evidence"]["visual_candidate_steps"] == ["S18", "S19", "S20"]


def test_evidence_packet_records_gate_and_recent_action_evidence() -> None:
    packet = build_evidence_packet(
        {
            "gates": {
                "S08.completion": {
                    "status": "blocked",
                    "reason_code": "left_mdi_pb18_not_pressed",
                    "reason": "left_mdi_pb18 still required",
                },
                "S08.precondition": {
                    "status": "allowed",
                    "reason_code": "ok",
                    "reason": None,
                },
            },
            "recent_actions": {
                "current_button": "left_mdi_pb18",
                "recent_buttons": ["left_mdi_pb18", "battery_switch"],
            },
            "recent_deltas": [
                {"mapped_ui_target": "left_mdi_pb18", "seq": 41, "t_wall": 20.0},
                {"mapped_ui_target": "battery_switch", "seq": 37, "t_wall": 18.0},
            ],
        }
    )

    payload = packet.to_dict()

    assert payload["gate_evidence"]["blocked_gate_ids"] == ["S08.completion"]
    assert payload["gate_evidence"]["source_status"] == "available"
    assert payload["gate_evidence"]["confidence"] == "high"
    assert payload["gate_evidence"]["freshness"] == {"gate_count": 2}
    assert payload["gate_evidence"]["satisfied_gate_ids"] == ["S08.precondition"]
    assert payload["gate_evidence"]["blocked_gates"][0]["reason_code"] == "left_mdi_pb18_not_pressed"
    assert payload["recent_action_evidence"]["target_ids"] == ["left_mdi_pb18", "battery_switch"]
    assert payload["recent_action_evidence"]["freshness"] == {
        "latest_seq": 41,
        "latest_t_wall": 20.0,
    }
    assert payload["recent_action_evidence"]["actions"][0] == {
        "target_id": "left_mdi_pb18",
        "seq": 41,
        "t_wall": 20.0,
    }
    assert CONFLICT_RECENT_ACTION_VS_GATE_CONTRADICTION in payload["conflicts"]


def test_evidence_packet_can_render_legacy_state_harness_payload() -> None:
    packet = build_evidence_packet(
        {
            "vars": {"battery_on": True},
            "deterministic_step_hint": {"inferred_step_id": "S05", "overlay_step_id": "S05"},
            "recent_actions": {"recent_buttons": ["eng_crank_switch"]},
        }
    )

    legacy = packet.to_state_harness_dict()

    assert legacy["telemetry_evidence"]["vars_source_missing_count"] == 0
    assert legacy["vision_evidence"]["source_status"] == "vision_unavailable"
    assert legacy["recent_action_evidence"]["recent_buttons"] == ["eng_crank_switch"]
    assert legacy["deterministic_candidate"]["step_id"] == "S05"
    assert packet.compact_summary()["telemetry_status"] == "nominal"


def test_step_candidates_outrank_bootstrap_deterministic_when_visual_anchors_conflict() -> None:
    packet = build_evidence_packet(
        {
            "vars": {
                "battery_on": False,
                "power_available": False,
                "vars_source_missing": [f"missing_{idx}" for idx in range(25)],
            },
            "vision": {"observation_seq": 2},
            "vision_fact_summary": {
                "status": "available",
                "fresh_fact_ids": ["tac_page_visible", "bit_root_page_visible"],
                "seen_fact_ids": ["tac_page_visible", "bit_root_page_visible"],
                "not_seen_fact_ids": ["fcs_page_visible"],
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S01",
                "missing_conditions": ["vars.battery_on==true"],
            },
        }
    )

    candidates = [item.to_dict() for item in build_step_candidates(packet)]

    assert candidates[0]["step_id"] == "S08"
    assert candidates[0]["source"] == "visual_anchor"
    assert candidates[0]["supporting_evidence_refs"] == [
        "VISION_FACTS.bit_root_page_visible",
        "VISION_FACTS.tac_page_visible",
    ]
    deterministic = next(item for item in candidates if item["source"] == "deterministic")
    assert deterministic["step_id"] == "S01"
    assert deterministic["role"] == "candidate_not_authoritative"


def test_step_candidates_represent_s19_fcsmc_visual_state_and_final_go_sticky_progression() -> None:
    in_test_packet = build_evidence_packet(
        {
            "vision_fact_summary": {
                "status": "available",
                "fresh_fact_ids": ["fcsmc_page_visible", "fcsmc_in_test_visible"],
                "seen_fact_ids": ["fcsmc_page_visible", "fcsmc_in_test_visible"],
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S18",
                "missing_conditions": ["vision_facts.fcsmc_page_visible==seen"],
            },
        }
    )
    in_test_candidates = [item.to_dict() for item in build_step_candidates(in_test_packet)]

    assert in_test_candidates[0]["step_id"] == "S19"
    assert in_test_candidates[0]["source"] == "visual_anchor"
    assert "VISION_FACTS.fcsmc_in_test_visible" in in_test_candidates[0]["supporting_evidence_refs"]
    assert any(
        item["step_id"] == "S18" and item["source"] == "deterministic"
        for item in in_test_candidates
    )

    final_packet = build_evidence_packet(
        {
            "vision_facts": [
                {
                    "fact_id": "fcsmc_final_go_result_visible",
                    "state": "seen",
                    "sticky": True,
                    "expires_after_ms": 600000,
                }
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S19",
                "missing_conditions": ["vision_facts.fcsmc_final_go_result_visible==seen"],
            },
        }
    )
    final_candidates = [item.to_dict() for item in build_step_candidates(final_packet)]

    assert final_candidates[0]["step_id"] == "S20"
    assert final_candidates[0]["source"] == "sticky_state"
    assert final_candidates[0]["supporting_evidence_refs"] == [
        "VISION_FACTS.fcsmc_final_go_result_visible"
    ]


def test_step_candidates_suppress_sticky_visual_state_when_vision_not_required() -> None:
    packet = build_evidence_packet(
        {
            "vision_fact_summary": {
                "status": "vision_not_required",
                "frame_ids": ["frame-s20"],
                "fresh_fact_ids": [],
                "seen_fact_ids": ["fcsmc_final_go_result_visible"],
                "not_seen_fact_ids": [],
                "uncertain_fact_ids": [],
            },
            "vision_facts": [
                {
                    "fact_id": "fcsmc_final_go_result_visible",
                    "state": "seen",
                    "sticky": True,
                    "expires_after_ms": 600000,
                }
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S20",
                "overlay_step_id": "S20",
                "requires_visual_confirmation": False,
                "missing_conditions": ["vars.ext_refuel_probe_value>=60000"],
            },
        }
    )

    payload = packet.to_dict()
    candidates = [item.to_dict() for item in build_step_candidates(packet)]

    assert payload["vision_evidence"]["source_status"] == "vision_not_required"
    assert payload["vision_evidence"]["late_display_anchors"] == []
    assert payload["vision_evidence"]["visual_candidate_steps"] == []
    assert candidates[0]["step_id"] == "S20"
    assert candidates[0]["source"] == "deterministic"
    assert not any(item["source"] == "sticky_state" for item in candidates)


def test_step_candidates_keep_four_down_deterministic_progression_when_telemetry_only() -> None:
    packet = build_evidence_packet(
        {
            "vars": {"refuel_probe_extended": False},
            "gates": {
                "S20.completion": {
                    "status": "blocked",
                    "reason_code": "refuel_probe_not_extended",
                    "reason": "vars.refuel_probe_extended==true",
                }
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S20",
                "overlay_step_id": "S20",
                "missing_conditions": ["vars.refuel_probe_extended==true"],
            },
        }
    )

    ordered_step_ids = [f"S{idx:02d}" for idx in range(1, 34)]
    candidates = [item.to_dict() for item in build_step_candidates(packet, ordered_step_ids=ordered_step_ids)]

    assert candidates[0]["step_id"] == "S20"
    assert candidates[0]["source"] == "deterministic"
    assert candidates[0]["role"] == "candidate_not_authoritative"
    assert "GATES.S20.completion" in candidates[0]["supporting_evidence_refs"]
    assert not any(item["source"] == "procedure_order" and item["step_id"] == "S01" for item in candidates)


def test_step_candidates_include_recent_action_matches_from_harness_specs() -> None:
    packet = build_evidence_packet(
        {
            "recent_actions": {"recent_buttons": ["left_mdi_pb18"]},
            "recent_deltas": [{"mapped_ui_target": "left_mdi_pb18", "seq": 8}],
        }
    )
    specs = {
        "S08": SimpleNamespace(
            allowed_overlay_targets=("left_mdi_pb18", "left_mdi_pb15"),
            declared_ui_targets=("left_mdi_pb18", "left_mdi_pb15"),
        )
    }

    candidates = [
        item.to_dict()
        for item in build_step_candidates(packet, step_harness_specs=specs, ordered_step_ids=["S08"])
    ]

    assert candidates[0]["step_id"] == "S08"
    assert candidates[0]["source"] == "recent_action"
    assert candidates[0]["supporting_evidence_refs"] == ["RECENT_ACTIONS.left_mdi_pb18"]
    assert candidates[0]["proposed_next_action_target_ids"] == ["left_mdi_pb18"]


def test_telemetry_window_digest_marks_first_frame_bootstrap_refuted_by_later_power() -> None:
    packet = build_evidence_packet(
        {
            "vars": {
                "power_available": True,
                "left_ddi_on": True,
                "battery_on": None,
                "vars_source_missing": ["battery_on"],
            },
            "telemetry_window_frames": [
                {
                    "seq": 1,
                    "t_wall": 10.0,
                    "vars": {"battery_on": False, "power_available": False},
                },
                {
                    "seq": 2,
                    "t_wall": 13.0,
                    "vars": {"power_available": True, "left_ddi_on": True},
                },
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S01",
                "missing_conditions": ["vars.battery_on==true"],
            },
        }
    )

    payload = packet.to_dict()
    digest = payload["telemetry_window_digest"]

    assert digest["window_duration_s"] == 3.0
    assert digest["frame_count"] == 2
    assert digest["latest_seq"] == 2
    assert digest["first_frame_only_values"] == [{"var": "battery_on", "value": False}]
    assert any(
        item["var"] == "power_available" and item["last_value"] is True
        for item in digest["changed_vars"]
    )
    assert "left_ddi_on" in digest["stable_true_vars"]
    assert "battery_on" in digest["unknown_or_missing_vars"]
    assert "battery_on=false only in first frame but later downstream avionics evidence is present" in digest["contradictions"]
    assert "telemetry_window_conflicts_with_single_frame_hint" in packet.to_state_harness_dict()["conflicts"]

    candidates = [item.to_dict() for item in build_step_candidates(packet)]
    telemetry_candidate = next(item for item in candidates if item["source"] == "telemetry_window")
    assert telemetry_candidate["step_id"] == "S08"
    assert telemetry_candidate["refuting_evidence_refs"] == ["TELEMETRY_WINDOW.first_frame_only_values.battery_on"]


def test_telemetry_window_candidate_handles_missing_early_latch_with_later_evidence() -> None:
    packet = build_evidence_packet(
        {
            "vars": {
                "power_available": True,
                "left_ddi_on": True,
                "fire_test_a_complete": None,
                "vars_source_missing": ["fire_test_a_complete"],
            },
            "telemetry_window_frames": [
                {"seq": 5, "t_wall": 20.0, "vars": {"power_available": True}},
                {"seq": 6, "t_wall": 21.5, "vars": {"left_ddi_on": True}},
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S02",
                "missing_conditions": ["vars.fire_test_a_complete==true"],
            },
        }
    )

    payload = packet.to_dict()
    digest = payload["telemetry_window_digest"]

    assert "fire_test_a_complete" in digest["unknown_or_missing_vars"]
    assert "fire_test latch missing while later-stage telemetry evidence is present" in digest["contradictions"]
    assert CONFLICT_LATCH_MISSING_VS_LATER_STAGE_EVIDENCE in payload["conflicts"]

    candidates = [item.to_dict() for item in build_step_candidates(packet)]
    telemetry_candidate = next(item for item in candidates if item["source"] == "telemetry_window")
    assert telemetry_candidate["step_id"] == "S08"
    assert "TELEMETRY_WINDOW.stable_true_vars.left_ddi_on" in telemetry_candidate["supporting_evidence_refs"]


def test_telemetry_window_candidate_exposes_recent_transition_when_gate_still_blocked() -> None:
    packet = build_evidence_packet(
        {
            "vars": {"fcs_bit_switch_up": False},
            "telemetry_window_frames": [
                {"seq": 8, "t_wall": 30.0, "vars": {"fcs_bit_switch_up": False}},
                {"seq": 9, "t_wall": 31.0, "vars": {"fcs_bit_switch_up": True}},
            ],
            "gates": {
                "S19.completion": {
                    "status": "blocked",
                    "step_id": "S19",
                    "reason_code": "fcs_bit_switch_up_missing",
                    "reason": "vars.fcs_bit_switch_up==true",
                }
            },
            "deterministic_step_hint": {
                "inferred_step_id": "S19",
                "missing_conditions": ["vars.fcs_bit_switch_up==true"],
            },
        }
    )

    digest = packet.to_dict()["telemetry_window_digest"]

    assert digest["changed_vars"][0]["var"] == "fcs_bit_switch_up"
    assert digest["changed_vars"][0]["first_value"] is False
    assert digest["changed_vars"][0]["last_value"] is True
    assert "recent telemetry transition conflicts with current blocked gate" in digest["contradictions"]

    candidates = [item.to_dict() for item in build_step_candidates(packet)]
    telemetry_candidate = next(item for item in candidates if item["source"] == "telemetry_window")
    assert telemetry_candidate["step_id"] == "S19"
    assert "TELEMETRY_WINDOW.changed_vars.fcs_bit_switch_up" in telemetry_candidate["supporting_evidence_refs"]


def test_telemetry_window_candidates_cover_four_down_progression_from_telemetry_only() -> None:
    packet = build_evidence_packet(
        {
            "vars": {
                "ext_refuel_probe_value": 65000,
                "launch_bar_switch_value": 1,
                "hook_handle_value": 1,
                "pitot_heat_on": True,
                "flap_auto": False,
            },
            "telemetry_window_frames": [
                {
                    "seq": 20,
                    "t_wall": 40.0,
                    "vars": {
                        "ext_refuel_probe_value": 0,
                        "launch_bar_switch_value": 0,
                        "hook_handle_value": 0,
                        "pitot_heat_on": False,
                    },
                },
                {
                    "seq": 21,
                    "t_wall": 44.0,
                    "vars": {
                        "ext_refuel_probe_value": 65000,
                        "launch_bar_switch_value": 1,
                        "hook_handle_value": 1,
                        "pitot_heat_on": True,
                    },
                },
            ],
        }
    )

    candidates = [item.to_dict() for item in build_step_candidates(packet)]
    telemetry_steps = [item["step_id"] for item in candidates if item["source"] == "telemetry_window"]

    assert telemetry_steps[:4] == ["S21", "S23", "S26", "S27"]
    assert all(item["supporting_evidence_refs"][0].startswith("TELEMETRY_WINDOW.") for item in candidates if item["source"] == "telemetry_window")


def test_telemetry_window_candidates_treat_hook_zero_as_down() -> None:
    packet = build_evidence_packet(
        {
            "vars": {
                "hook_handle_value": 0,
            },
            "telemetry_window_frames": [
                {
                    "seq": 30,
                    "t_wall": 60.0,
                    "vars": {
                        "hook_handle_value": 1,
                    },
                },
                {
                    "seq": 31,
                    "t_wall": 64.0,
                    "vars": {
                        "hook_handle_value": 0,
                    },
                },
            ],
        }
    )

    candidates = [item.to_dict() for item in build_step_candidates(packet)]
    telemetry_steps = [item["step_id"] for item in candidates if item["source"] == "telemetry_window"]

    assert telemetry_steps[0] == "S25"


def test_telemetry_window_candidates_mark_refuel_probe_motion_in_progress() -> None:
    packet = build_evidence_packet(
        {
            "vars": {
                "probe_switch_value": 1,
                "ext_refuel_probe_value": 5606,
                "probe_retracted": False,
            },
            "telemetry_window_frames": [
                {"seq": 9675, "t_wall": 100.0, "vars": {"ext_refuel_probe_value": 6200}},
                {"seq": 9676, "t_wall": 100.1, "vars": {"ext_refuel_probe_value": 5606}},
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S21",
                "missing_conditions": ["vars.ext_refuel_probe_value in [0,5000]"],
            },
        }
    )

    candidates = [item.to_dict() for item in build_step_candidates(packet)]
    telemetry_candidate = next(
        item
        for item in candidates
        if item["source"] == "telemetry_window" and item["step_id"] == "S21"
    )

    assert telemetry_candidate["reason"] == "refueling probe retracting in progress"
    assert telemetry_candidate["proposed_next_action_target_ids"] == []


def test_telemetry_window_candidates_mark_refuel_probe_extension_in_progress() -> None:
    packet = build_evidence_packet(
        {
            "vars": {
                "probe_switch_value": 0,
                "ext_refuel_probe_value": 12000,
                "probe_extended": False,
            },
            "telemetry_window_frames": [
                {"seq": 901, "t_wall": 40.0, "vars": {"ext_refuel_probe_value": 8000}},
                {"seq": 902, "t_wall": 40.1, "vars": {"ext_refuel_probe_value": 12000}},
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S20",
                "missing_conditions": ["vars.ext_refuel_probe_value in [60000,65535]"],
            },
        }
    )

    candidates = [item.to_dict() for item in build_step_candidates(packet)]
    telemetry_candidate = next(
        item
        for item in candidates
        if item["source"] == "telemetry_window" and item["step_id"] == "S20"
    )

    assert telemetry_candidate["reason"] == "refueling probe extending in progress"
    assert telemetry_candidate["proposed_next_action_target_ids"] == []


def test_telemetry_window_candidates_advance_after_refuel_probe_retracted() -> None:
    packet = build_evidence_packet(
        {
            "vars": {
                "probe_switch_value": 1,
                "ext_refuel_probe_value": 4652,
                "probe_retracted": True,
            },
            "telemetry_window_frames": [
                {"seq": 9676, "t_wall": 100.0, "vars": {"ext_refuel_probe_value": 5606}},
                {"seq": 9678, "t_wall": 100.2, "vars": {"ext_refuel_probe_value": 4652}},
            ],
        }
    )

    candidates = [item.to_dict() for item in build_step_candidates(packet)]
    telemetry_steps = [item["step_id"] for item in candidates if item["source"] == "telemetry_window"]

    assert telemetry_steps[0] == "S22"


def test_telemetry_window_sparse_delta_does_not_mark_first_frame_only_value() -> None:
    packet = build_evidence_packet(
        {
            "vars": {"power_available": True, "left_ddi_on": True},
            "telemetry_window_frames": [
                {"seq": 1, "t_wall": 1.0, "delta": {"battery_on": False}},
                {"seq": 2, "t_wall": 2.0, "delta": {"left_ddi_on": True}},
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S01",
                "missing_conditions": ["vars.battery_on==true"],
            },
        }
    )

    digest = packet.to_dict()["telemetry_window_digest"]
    candidates = [item.to_dict() for item in build_step_candidates(packet)]

    assert digest["first_frame_only_values"] == []
    assert "battery_on=false only in first frame but later downstream avionics evidence is present" not in digest["contradictions"]
    assert not any(item["source"] == "telemetry_window" and item["step_id"] == "S08" for item in candidates)


def test_telemetry_window_full_snapshot_none_counts_as_missing_not_stable_true() -> None:
    packet = build_evidence_packet(
        {
            "vars": {"power_available": True, "left_ddi_on": True, "battery_on": None},
            "telemetry_window_frames": [
                {
                    "seq": 1,
                    "t_wall": 1.0,
                    "vars": {"battery_on": False, "power_available": False},
                    "vars_is_full_snapshot": True,
                },
                {
                    "seq": 2,
                    "t_wall": 2.0,
                    "vars": {"battery_on": None, "power_available": True, "left_ddi_on": True},
                    "vars_is_full_snapshot": True,
                },
            ],
            "deterministic_step_hint": {
                "inferred_step_id": "S01",
                "missing_conditions": ["vars.battery_on==true"],
            },
        }
    )

    digest = packet.to_dict()["telemetry_window_digest"]
    candidates = [item.to_dict() for item in build_step_candidates(packet)]

    assert digest["first_frame_only_values"] == [{"var": "battery_on", "value": False}]
    assert "battery_on" not in digest["stable_false_vars"]
    assert any(item["source"] == "telemetry_window" and item["step_id"] == "S08" for item in candidates)


def test_telemetry_window_fcs_bit_release_does_not_generate_s19_candidate() -> None:
    packet = build_evidence_packet(
        {
            "telemetry_window_frames": [
                {"seq": 1, "t_wall": 1.0, "vars": {"fcs_bit_switch_up": True}},
                {"seq": 2, "t_wall": 2.0, "vars": {"fcs_bit_switch_up": False}},
            ],
        }
    )

    candidates = [item.to_dict() for item in build_step_candidates(packet)]

    assert not any(item["source"] == "telemetry_window" and item["step_id"] == "S19" for item in candidates)
