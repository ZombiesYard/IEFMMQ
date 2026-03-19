import json
from pathlib import Path

from adapters.evidence_refs import infer_evidence_type_from_ref
from adapters.prompting import (
    MAX_DELTA_SUMMARY_ITEMS,
    MAX_RECENT_ACTIONS_SIGNAL_ITEMS,
    build_help_prompt,
    build_help_prompt_result,
)
from adapters.recent_actions import build_recent_button_signal
from tests._fakes import _extract_prompt_constraints_json


def _base_context() -> dict:
    return {
        "candidate_steps": ["S02", "S03"],
        "overlay_target_allowlist": ["apu_switch", "battery_switch"],
        "vars": {
            "battery_on": True,
            "apu_on": False,
            "api_key": "sk-should-not-leak",
            "cwd": "C:\\Users\\secret\\project",
        },
        "recent_deltas": [
            {"k": "apu_switch", "from": 0, "to": 1, "action": "toggle"},
            {"k": "battery_switch", "from": 0, "to": 1, "action": "toggle"},
        ],
    }


def test_prompt_defaults_to_registry_backed_step_ids() -> None:
    result = build_help_prompt_result({}, "en", max_prompt_chars=20000, max_prompt_tokens_est=6000)
    payload = _extract_prompt_constraints_json(result.prompt)

    assert payload["allowed_step_ids"][0] == "S01"
    assert payload["allowed_step_ids"][-1] == "S25"
    assert len(payload["allowed_step_ids"]) == 25


def test_prompt_contains_enum_constraints_delta_summary_and_evidence_sources() -> None:
    prompt = build_help_prompt(_base_context(), "en")
    payload = _extract_prompt_constraints_json(prompt)

    assert payload["allowed_step_ids"] == ["S02", "S03"]
    assert payload["allowed_overlay_targets"] == ["apu_switch", "battery_switch"]
    assert payload["allowed_overlay_evidence_types"] == ["var", "gate", "rag", "delta", "visual"]
    assert payload["allowed_error_categories"]
    assert payload["output_example_json"]["diagnosis"]["error_category"] in payload["allowed_error_categories"]
    summary = payload["recent_deltas_summary"]
    assert summary["top_k"] == MAX_DELTA_SUMMARY_ITEMS
    assert len(summary["items"]) == 2
    assert summary["items"][0]["ui_target"] == "apu_switch"
    evidence = payload["EVIDENCE_SOURCES"]
    assert set(evidence.keys()) == {"VARS", "GATES", "RECENT_UI_TARGETS", "RAG_SNIPPETS", "VISION_FACTS"}
    assert evidence["VISION_FACTS"] == []
    assert "RECENT_UI_TARGETS.apu_switch" in payload["allowed_evidence_refs"]
    assert payload["deterministic_step_hint"]["inferred_step_id"] is None
    assert payload["deterministic_step_hint"]["missing_conditions"] == []
    assert payload["vision_fact_summary"]["status"] == "vision_unavailable"
    assert payload["grounding"]["missing"] is True
    assert payload["grounding"]["applied"] is False

    sample_evidence = payload["output_example_json"]["overlay"]["evidence"][0]
    assert sample_evidence["type"] in {"var", "gate", "rag", "delta"}
    assert sample_evidence["ref"] in payload["allowed_evidence_refs"]
    assert len(sample_evidence["quote"]) <= 120
    assert sample_evidence["type"] == infer_evidence_type_from_ref(sample_evidence["ref"])


def test_prompt_exposes_single_target_policy_and_evidence_contract() -> None:
    payload = _extract_prompt_constraints_json(build_help_prompt(_base_context(), "en"))

    policy = payload["overlay_target_policy"]
    assert policy["mode"] == "single_target_preferred"
    assert policy["max_targets"] == 1
    assert policy["empty_overlay_if_uncertain"] is True
    assert policy["candidate_targets_in_priority_order"][0] == "apu_switch"

    contract = payload["overlay_evidence_contract"]
    assert contract["field_order"] == ["target", "type", "ref", "quote", "grounding_confidence"]
    assert contract["quote_max_chars"] == 120
    assert contract["same_target_required"] is True
    assert contract["ref_must_exist_in_allowed_evidence_refs"] is True
    assert contract["type_ref_prefixes"]["var"] == ["VARS."]
    assert contract["type_ref_prefixes"]["gate"] == ["GATES."]
    assert contract["type_ref_prefixes"]["delta"] == ["RECENT_UI_TARGETS."]
    assert "DELTA_KEYS." not in contract["type_ref_prefixes"]["delta"]


def test_prompt_includes_explicit_interaction_policy_and_target_hints() -> None:
    ctx = _base_context()
    ctx["candidate_steps"] = ["S05"]
    ctx["overlay_target_allowlist"] = [
        "eng_crank_switch",
        "throttle_quadrant_reference",
        "battery_switch",
    ]

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    assert payload["interaction_policy"]["two_position"] == (
        "For 2-position switches, say left-click or right-click explicitly."
    )
    assert payload["interaction_policy"]["wheel"] == "For brightness increase, default to mouse-wheel up."
    hints = {item["target"]: item for item in payload["target_interaction_hints"]}
    assert hints["eng_crank_switch"]["instruction"] == "Engine Crank: right-click for R, left-click for L."
    assert hints["eng_crank_switch"]["click_type_by_value"] == {"L": "left", "R": "right"}
    assert hints["throttle_quadrant_reference"]["instruction"] == (
        "Throttle to IDLE: right throttle Right Shift+Home, left throttle Right Alt+Home."
    )
    assert hints["throttle_quadrant_reference"]["click_type"] == "keyboard"
    assert hints["throttle_quadrant_reference"]["hotkey_by_action"] == {
        "right_idle": "Right Shift+Home",
        "left_idle": "Right Alt+Home",
    }
    assert hints["battery_switch"]["instruction"] == "Set BATT switch to ON with a right-click."
    assert hints["battery_switch"]["click_type"] == "right"


def test_prompt_reads_interaction_policy_and_hints_from_ui_map(tmp_path: Path) -> None:
    ui_map_path = tmp_path / "ui_map.yaml"
    ui_map_path.write_text(
        """
version: v1
interaction_policy:
  two_position:
    zh: "两位开关测试中文"
    en: "custom two-position rule"
  multi_position:
    zh: "多位开关测试中文"
    en: "custom multi-position rule"
  buttons:
    zh: "按钮测试中文"
    en: "custom button rule"
  wheel:
    zh: "滚轮测试中文"
    en: "custom wheel rule"
  hotkeys:
    zh: "热键测试中文"
    en: "custom hotkey rule"
cockpit_elements:
  battery_switch:
    description: "Battery"
    dcs_id: "pnt_404"
    aliases: ["Battery"]
    panel_area: "test"
    interaction_hint:
      zh: "自定义电瓶中文"
      en: "custom battery hint"
  apu_switch:
    description: "APU"
    dcs_id: "pnt_375"
    aliases: ["APU"]
    panel_area: "test"
default_overlay:
  style: "highlight"
""".strip(),
        encoding="utf-8",
    )

    ctx = _base_context()
    ctx["ui_map_path"] = str(ui_map_path)
    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    assert payload["interaction_policy"]["two_position"] == "custom two-position rule"
    assert payload["interaction_policy"]["wheel"] == "custom wheel rule"
    hints = {item["target"]: item for item in payload["target_interaction_hints"]}
    assert hints["battery_switch"]["instruction"] == "custom battery hint"
    assert hints["battery_switch"]["click_type"] == "right"
    assert hints["apu_switch"]["instruction"] == "Set APU to ON with a left-click."


def test_prompt_reads_structured_interaction_metadata_from_ui_map(tmp_path: Path) -> None:
    ui_map_path = tmp_path / "ui_map.yaml"
    ui_map_path.write_text(
        """
version: v1
interaction_policy:
  two_position:
    zh: "两位"
    en: "two"
  multi_position:
    zh: "多位"
    en: "multi"
  buttons:
    zh: "按钮"
    en: "buttons"
  wheel:
    zh: "滚轮"
    en: "wheel"
  hotkeys:
    zh: "热键"
    en: "hotkeys"
cockpit_elements:
  eng_crank_switch:
    description: "Crank"
    dcs_id: "pnt_377"
    aliases: ["Crank"]
    panel_area: "test"
    interaction:
      click_type_by_value:
        L: "left"
        R: "right"
    interaction_hint:
      zh: "曲柄"
      en: "custom crank hint"
  throttle_quadrant_reference:
    description: "Throttle"
    dcs_id: "pnt_504"
    aliases: ["Throttle"]
    panel_area: "test"
    interaction:
      click_type: "keyboard"
      hotkey_by_action:
        left_idle: "Alt+Home"
        right_idle: "Shift+Home"
    interaction_hint:
      zh: "油门"
      en: "custom throttle hint"
default_overlay:
  style: "highlight"
""".strip(),
        encoding="utf-8",
    )

    ctx = _base_context()
    ctx["ui_map_path"] = str(ui_map_path)
    ctx["overlay_target_allowlist"] = ["eng_crank_switch", "throttle_quadrant_reference"]
    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    hints = {item["target"]: item for item in payload["target_interaction_hints"]}
    assert hints["eng_crank_switch"]["instruction"] == "custom crank hint"
    assert hints["eng_crank_switch"]["click_type_by_value"] == {"L": "left", "R": "right"}
    assert hints["throttle_quadrant_reference"]["instruction"] == "custom throttle hint"
    assert hints["throttle_quadrant_reference"]["click_type"] == "keyboard"
    assert hints["throttle_quadrant_reference"]["hotkey_by_action"] == {
        "left_idle": "Alt+Home",
        "right_idle": "Shift+Home",
    }


def test_prompt_makes_unknown_and_partial_constraints_explicit() -> None:
    ctx = _base_context()
    ctx["deterministic_step_hint"] = {
        "inferred_step_id": "S15",
        "observability_status": "partial",
        "step_evidence_requirements": ["visual", "gate"],
        "recent_ui_targets": ["left_mdi_pb5"],
    }

    result = build_help_prompt_result(ctx, "en", max_prompt_chars=20000, max_prompt_tokens_est=6000)
    payload = _extract_prompt_constraints_json(result.prompt)

    uncertainty = payload["uncertainty_policy"]
    assert uncertainty["partial"]["requires_confirmation_phrase"] is True
    assert uncertainty["partial"]["allow_diagnosis_from_hint"] is True
    assert uncertainty["partial"]["allow_single_target_only"] is True
    assert uncertainty["partial"]["applies_when"] == "current_observability_status=partial or requires_visual_confirmation=true"
    assert uncertainty["unknown"]["force_empty_overlay"] is True
    assert uncertainty["unknown"]["requires_confirmation_phrase"] is True
    assert uncertainty["unknown"]["applies_when"] == (
        "current_inferred_step_id is null, evidence conflicts, or no verifiable evidence exists"
    )
    assert payload["deterministic_step_hint"]["requires_visual_confirmation"] is True
    assert "vision_fact_summary" in payload
    assert "Return at most one overlay target." in result.prompt
    assert "If uncertainty_policy.partial applies" in result.prompt
    assert "If uncertainty_policy.unknown applies" in result.prompt


def test_prompt_includes_vision_fact_summary_and_visual_overlay_evidence_refs() -> None:
    ctx = _base_context()
    ctx["vision_fact_summary"] = {
        "status": "available",
        "frame_ids": ["1772872444950_000122", "1772872445010_000123"],
        "seen_fact_ids": ["fcs_reset_seen"],
        "uncertain_fact_ids": ["fcs_bit_result_visible"],
        "not_seen_fact_ids": [],
        "summary_text": "seen=fcs_reset_seen; uncertain=fcs_bit_result_visible",
    }
    ctx["vision_facts"] = [
        {
            "fact_id": "fcs_reset_seen",
            "state": "seen",
            "source_frame_id": "1772872445010_000123",
            "confidence": 0.96,
            "evidence_note": "RESET cue visible on the FCS page.",
        }
    ]

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    assert payload["vision_fact_summary"]["status"] == "available"
    assert payload["vision_fact_summary"]["seen_fact_ids"] == ["fcs_reset_seen"]
    assert payload["allowed_overlay_evidence_types"] == ["var", "gate", "rag", "delta", "visual"]
    assert "VISION_FACTS.fcs_reset_seen@1772872445010_000123" in payload["allowed_evidence_refs"]
    assert payload["EVIDENCE_SOURCES"]["VISION_FACTS"] == [
        {
            "ref": "VISION_FACTS.fcs_reset_seen@1772872445010_000123",
            "fact_id": "fcs_reset_seen",
            "state": "seen",
            "source_frame_id": "1772872445010_000123",
            "confidence": 0.96,
            "evidence_note": "RESET cue visible on the FCS page.",
        }
    ]
    assert payload["decision_priority"][:3] == [
        "deterministic_step_hint",
        "gates_summary",
        "vision_fact_summary",
    ]


def test_prompt_requires_exact_visual_fact_refs_and_forbids_alias_names() -> None:
    ctx = _base_context()
    ctx["vision_fact_summary"] = {
        "status": "available",
        "frame_ids": ["1773950407644_000006"],
        "seen_fact_ids": ["bit_page_failure_visible"],
        "uncertain_fact_ids": [],
        "not_seen_fact_ids": [],
        "summary_text": "seen=bit_page_failure_visible",
    }
    ctx["vision_facts"] = [
        {
            "fact_id": "bit_page_failure_visible",
            "state": "seen",
            "source_frame_id": "1773950407644_000006",
            "confidence": 0.99,
            "evidence_note": "BIT FAILURES line is clearly visible on the right DDI.",
        }
    ]

    result = build_help_prompt_result(ctx, "en")

    assert "must exactly match a full entry from allowed_evidence_refs" in result.prompt
    assert "bit_page_failure_visible" in result.prompt
    assert "right_ddi_bit_failures_page_visible" in result.prompt


def test_prompt_prioritizes_missing_condition_vars_when_var_budget_trims() -> None:
    ctx = _base_context()
    ctx["candidate_steps"] = ["S07"]
    ctx["overlay_target_allowlist"] = ["lights_test_button"]
    ctx["vars"] = {f"aaa_{i:02d}": i for i in range(30)}
    ctx["vars"]["lights_test_complete"] = True
    ctx["deterministic_step_hint"] = {
        "inferred_step_id": "S07",
        "missing_conditions": ["vars.lights_test_complete==true"],
        "observability_status": "partial",
        "step_evidence_requirements": ["var", "delta", "gate"],
        "recent_ui_targets": ["lights_test_button"],
    }

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    assert payload["current_vars_selected"]["lights_test_complete"] is True
    assert len(payload["current_vars_selected"]) == 20


def test_prompt_prioritizes_s05_engine_start_vars_when_var_budget_trims() -> None:
    ctx = _base_context()
    ctx["candidate_steps"] = ["S05"]
    ctx["overlay_target_allowlist"] = []
    ctx["vars"] = {f"aaa_{i:02d}": i for i in range(30)}
    ctx["vars"]["rpm_r"] = 28
    ctx["vars"]["throttle_r_idle_complete"] = False
    ctx["deterministic_step_hint"] = {
        "inferred_step_id": "S05",
        "missing_conditions": ["vars.rpm_r>=25", "vars.throttle_r_idle_complete==true"],
        "observability_status": "observable",
        "step_evidence_requirements": ["var", "gate"],
        "recent_ui_targets": [],
    }

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    assert payload["current_vars_selected"]["rpm_r"] == 28
    assert payload["current_vars_selected"]["throttle_r_idle_complete"] is False


def test_prompt_includes_rag_snippets_with_source_fields_and_metadata() -> None:
    ctx = _base_context()
    ctx["rag_topk"] = [
        {
            "doc_id": "fa18c_startup_master",
            "section": "S03",
            "page_or_heading": "S03",
            "snippet_id": "fa18c_startup_master_12",
            "snippet": "APU switch to ON and wait for the green APU READY light.",
        }
    ]
    result = build_help_prompt_result(ctx, "en")
    payload = _extract_prompt_constraints_json(result.prompt)

    rag_block = payload["EVIDENCE_SOURCES"]["RAG_SNIPPETS"]
    assert len(rag_block) == 1
    assert rag_block[0]["id"] == "fa18c_startup_master_12"
    assert rag_block[0]["doc_id"] == "fa18c_startup_master"
    assert rag_block[0]["section"] == "S03"
    assert rag_block[0]["page_or_heading"] == "S03"
    assert "RAG_SNIPPETS.fa18c_startup_master_12" in payload["allowed_evidence_refs"]
    assert payload["grounding"]["applied"] is True
    assert payload["grounding"]["missing"] is False
    assert result.metadata["rag_snippet_ids"] == ["fa18c_startup_master_12"]
    assert result.metadata["grounding_applied"] is True
    assert result.metadata["grounding_missing"] is False


def test_prompt_metadata_marks_grounding_missing_when_context_flagged() -> None:
    ctx = _base_context()
    ctx["grounding_missing"] = True
    ctx["grounding_reason"] = "index_missing"
    ctx["grounding_query"] = "F/A-18C | S03 | apu_switch"
    result = build_help_prompt_result(ctx, "en")
    payload = _extract_prompt_constraints_json(result.prompt)
    assert payload["grounding"]["requested_missing"] is True
    assert payload["grounding"]["missing"] is True
    assert payload["grounding"]["applied"] is False
    assert payload["grounding"]["reason"] == "index_missing"
    assert payload["grounding"]["query"] == "F/A-18C | S03 | apu_switch"
    assert result.metadata["grounding_missing_requested"] is True
    assert result.metadata["grounding_applied"] is False
    assert result.metadata["grounding_missing"] is True


def test_prompt_effective_grounding_marks_missing_when_no_rag_snippets_injected() -> None:
    ctx = _base_context()
    result = build_help_prompt_result(ctx, "en")
    payload = _extract_prompt_constraints_json(result.prompt)
    assert payload["grounding"]["requested_missing"] is False
    assert payload["grounding"]["applied"] is False
    assert payload["grounding"]["missing"] is True
    assert payload["grounding"]["reason"] == "no_rag_snippets"
    assert result.metadata["grounding_missing_requested"] is False
    assert result.metadata["grounding_applied"] is False
    assert result.metadata["grounding_missing"] is True
    assert result.metadata["grounding_reason"] == "no_rag_snippets"


def test_prompt_compact_template_keeps_grounding_metadata_consistent_with_emitted_prompt() -> None:
    ctx = _base_context()
    ctx["rag_topk"] = [
        {
            "doc_id": "manual",
            "section": "S03",
            "page_or_heading": "S03",
            "snippet_id": "manual_s03_1",
            "snippet": "APU switch to ON and wait for APU READY.",
        }
    ]
    ctx["vars"] = {f"v_{i:03d}": "x" * 200 for i in range(120)}
    result = build_help_prompt_result(ctx, "en", max_prompt_chars=560, max_prompt_tokens_est=140)

    assert "compact_template" in result.metadata["trim_reasons"]
    assert result.metadata["grounding_applied"] is True
    assert result.metadata["grounding_missing"] is False
    assert result.metadata["rag_snippet_count"] == 1
    assert result.metadata["rag_snippet_ids"] == ["manual_s03_1"]
    assert result.metadata["evidence_refs_count"] >= 1
    assert "constraints=" in result.prompt
    assert '"grounding":{"applied":true,"missing":false' in result.prompt
    if "hard_truncate" not in result.metadata["trim_reasons"]:
        assert 'Output shape={"diagnosis":{"step_id":"...","error_category":"..."}' in result.prompt


def test_prompt_recomputes_overlay_target_policy_after_overlay_enum_trim() -> None:
    ctx = {
        "candidate_steps": ["S01"],
        "overlay_target_allowlist": ["battery_switch", "apu_switch"],
        "vars": {},
        "recent_deltas": [],
        "recent_actions": {
            "current_button": "apu_switch",
            "recent_buttons": ["apu_switch", "battery_switch"],
        },
    }

    result = build_help_prompt_result(ctx, "en", max_prompt_chars=620, max_prompt_tokens_est=170)

    assert "trimmed_overlay_enum" in result.metadata["trim_reasons"]

    constraints_line = next(
        line for line in result.prompt.splitlines() if line.startswith("constraints=")
    )
    payload = json.loads(constraints_line[len("constraints=") :])

    assert payload["allowed_overlay_targets"] == ["apu_switch"]
    assert payload["overlay_target_policy"]["preferred_target"] == "apu_switch"


def test_prompt_prioritizes_generator_left_switch_from_missing_condition() -> None:
    ctx = {
        "candidate_steps": ["S01"],
        "overlay_target_allowlist": [
            "ampcd_off_brightness_knob",
            "battery_switch",
            "generator_left_switch",
        ],
        "vars": {
            "battery_on": True,
            "l_gen_on": False,
            "r_gen_on": True,
        },
        "recent_deltas": [
            {"ui_target": "ampcd_off_brightness_knob"},
            {"ui_target": "battery_switch"},
        ],
        "deterministic_step_hint": {
            "inferred_step_id": "S01",
            "missing_conditions": ["vars.l_gen_on==true"],
            "recent_ui_targets": [],
        },
    }

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    assert payload["allowed_overlay_targets"][0] == "generator_left_switch"
    assert payload["overlay_target_policy"]["preferred_target"] == "generator_left_switch"
    assert payload["overlay_target_policy"]["candidate_targets_in_priority_order"][0] == "generator_left_switch"


def test_prompt_trim_keeps_high_priority_generator_target_first() -> None:
    ctx = {
        "candidate_steps": ["S01", "S02", "S03", "S04", "S05"],
        "overlay_target_allowlist": [
            "ampcd_off_brightness_knob",
            "apu_switch",
            "battery_switch",
            "generator_left_switch",
            "generator_right_switch",
        ],
        "vars": {f"v_{i:02d}": "x" * 200 for i in range(60)},
        "recent_deltas": [],
        "recent_actions": {"current_button": None, "recent_buttons": []},
        "deterministic_step_hint": {
            "inferred_step_id": "S01",
            "missing_conditions": ["vars.l_gen_on==true"],
            "recent_ui_targets": [],
        },
    }

    result = build_help_prompt_result(ctx, "en", max_prompt_chars=620, max_prompt_tokens_est=170)

    assert "trimmed_overlay_enum" in result.metadata["trim_reasons"]

    constraints_line = next(
        line for line in result.prompt.splitlines() if line.startswith("constraints=")
    )
    payload = json.loads(constraints_line[len("constraints=") :])

    assert payload["allowed_overlay_targets"] == ["generator_left_switch"]
    assert payload["overlay_target_policy"]["preferred_target"] == "generator_left_switch"


def test_prompt_prioritizes_hud_brightness_when_s08_missing_hud_power() -> None:
    ctx = {
        "candidate_steps": ["S08"],
        "overlay_target_allowlist": [
            "left_mdi_brightness_selector",
            "right_mdi_brightness_selector",
            "ampcd_off_brightness_knob",
            "hud_symbology_brightness_knob",
        ],
        "vars": {
            "left_ddi_on": True,
            "right_ddi_on": True,
            "mpcd_on": True,
            "hud_on": False,
        },
        "recent_deltas": [],
        "recent_actions": {"current_button": None, "recent_buttons": []},
        "deterministic_step_hint": {
            "inferred_step_id": "S08",
            "missing_conditions": ["vars.hud_on==true"],
            "recent_ui_targets": [],
            "observability_status": "observable",
            "step_evidence_requirements": ["var", "gate", "delta"],
        },
    }

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    assert payload["overlay_target_policy"]["preferred_target"] == "hud_symbology_brightness_knob"
    assert payload["overlay_target_policy"]["candidate_targets_in_priority_order"][0] == "hud_symbology_brightness_knob"


def test_prompt_prioritizes_visual_action_hint_for_s08_fcs_entry() -> None:
    ctx = {
        "candidate_steps": ["S08"],
        "overlay_target_allowlist": [
            "left_mdi_pb18",
            "left_mdi_pb15",
            "left_mdi_brightness_selector",
        ],
        "vars": {
            "left_ddi_on": True,
            "right_ddi_on": True,
            "mpcd_on": True,
            "hud_on": True,
        },
        "recent_deltas": [],
        "recent_actions": {"current_button": None, "recent_buttons": []},
        "vision_fact_summary": {
            "status": "available",
            "seen_fact_ids": ["bit_page_visible", "left_ddi_fcs_option_visible"],
            "not_seen_fact_ids": ["fcs_page_visible"],
        },
        "vision_facts": [
            {
                "fact_id": "bit_page_visible",
                "state": "seen",
                "source_frame_id": "1772872445010_000123",
                "confidence": 0.98,
                "evidence_note": "BIT page title is visible on the right DDI.",
            },
            {
                "fact_id": "left_ddi_fcs_option_visible",
                "state": "seen",
                "source_frame_id": "1772872445010_000123",
                "confidence": 0.97,
                "evidence_note": "Left DDI menu shows FCS selectable on PB15.",
            },
        ],
        "deterministic_step_hint": {
            "inferred_step_id": "S08",
            "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
            "recent_ui_targets": [],
            "observability_status": "observable",
            "step_evidence_requirements": ["visual", "gate"],
            "visual_action_hint": {
                "target": "left_mdi_pb15",
                "reason": "BIT is already visible and FCS is selectable on PB15.",
            },
        },
    }

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    assert payload["deterministic_step_hint"]["visual_action_hint"]["target"] == "left_mdi_pb15"
    assert payload["overlay_target_policy"]["preferred_target"] == "left_mdi_pb15"
    assert payload["overlay_target_policy"]["candidate_targets_in_priority_order"][0] == "left_mdi_pb15"


def test_prompt_prioritizes_action_hint_for_s09_ufc_entry() -> None:
    ctx = {
        "candidate_steps": ["S09"],
        "overlay_target_allowlist": [
            "ufc_comm1_channel_selector_pull",
            "ufc_key_1",
            "ufc_key_3",
            "ufc_key_4",
            "ufc_key_0",
            "ufc_ent_button",
        ],
        "vars": {
            "comm1_freq_134_000": False,
            "ufc_scratchpad_string_1_display": "1-",
            "ufc_scratchpad_string_2_display": "-",
            "ufc_scratchpad_number_display": "305.000",
        },
        "recent_deltas": [],
        "recent_actions": {"current_button": None, "recent_buttons": []},
        "deterministic_step_hint": {
            "inferred_step_id": "S09",
            "missing_conditions": ["vars.comm1_freq_134_000==true"],
            "recent_ui_targets": [],
            "observability_status": "observable",
            "step_evidence_requirements": ["var", "delta", "rag"],
            "action_hint": {
                "target": "ufc_key_1",
                "reason": "COMM1 preset entry is open; press 1 next.",
            },
        },
    }

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))

    assert payload["deterministic_step_hint"]["action_hint"]["target"] == "ufc_key_1"
    assert payload["overlay_target_policy"]["preferred_target"] == "ufc_key_1"
    assert payload["overlay_target_policy"]["candidate_targets_in_priority_order"][0] == "ufc_key_1"


def test_prompt_explicitly_redirects_completed_s08_help_to_s09_comm1_selector() -> None:
    ctx = {
        "candidate_steps": ["S08", "S09"],
        "overlay_target_allowlist": [
            "ufc_comm1_channel_selector_pull",
            "ufc_key_1",
            "left_mdi_pb15",
        ],
        "vars": {"comm1_freq_134_000": False},
        "recent_deltas": [],
        "recent_actions": {"current_button": None, "recent_buttons": []},
        "deterministic_step_hint": {
            "inferred_step_id": "S08",
            "overlay_step_id": "S09",
            "missing_conditions": [],
            "recent_ui_targets": [],
            "observability_status": "observable",
            "step_evidence_requirements": ["var", "delta", "rag"],
            "action_hint": {
                "target": "ufc_comm1_channel_selector_pull",
                "reason": "S08 is complete; begin S09 by pulling COMM1.",
            },
        },
    }

    prompt = build_help_prompt(ctx, "en")
    payload = _extract_prompt_constraints_json(prompt)

    assert payload["deterministic_step_hint"]["overlay_step_id"] == "S09"
    assert payload["deterministic_step_hint"]["action_hint"]["target"] == "ufc_comm1_channel_selector_pull"
    assert payload["overlay_target_policy"]["preferred_target"] == "ufc_comm1_channel_selector_pull"
    assert payload["overlay_target_policy"]["candidate_targets_in_priority_order"][0] == "ufc_comm1_channel_selector_pull"
    assert "immediately highlight the UFC COMM1 channel selector" in prompt


def test_prompt_trim_keeps_s08_fcs_navigation_target_ahead_of_noisy_recent_actions() -> None:
    ctx = {
        "candidate_steps": ["S08", "S10", "S11", "S12"],
        "overlay_target_allowlist": [
            "left_mdi_brightness_selector",
            "right_mdi_brightness_selector",
            "ampcd_off_brightness_knob",
            "hud_symbology_brightness_knob",
            "left_mdi_pb18",
            "left_mdi_pb15",
            "right_mdi_pb18",
            "right_mdi_pb5",
        ],
        "vars": {f"v_{i:02d}": "x" * 180 for i in range(40)},
        "recent_deltas": [],
        "recent_actions": {
            "current_button": "parking_brake_handle",
            "recent_buttons": [
                "parking_brake_handle",
                "flap_switch",
                "launch_bar_switch",
                "standby_attitude_cage_knob",
                "ifei_down_button",
                "ifei_up_button",
                "hud_symbology_brightness_knob",
                "right_mdi_brightness_selector",
            ],
        },
        "vision": {
            "vision_used": True,
            "frame_ids": ["1773401766789_000003"],
        },
        "deterministic_step_hint": {
            "inferred_step_id": "S08",
            "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
            "recent_ui_targets": [
                "parking_brake_handle",
                "flap_switch",
                "launch_bar_switch",
                "standby_attitude_cage_knob",
                "ifei_down_button",
                "ifei_up_button",
                "hud_symbology_brightness_knob",
                "right_mdi_brightness_selector",
            ],
            "observability_status": "observable",
            "step_evidence_requirements": ["var", "gate", "delta"],
        },
        "vision_fact_summary": {"status": "vision_unavailable"},
    }

    result = build_help_prompt_result(ctx, "zh", max_prompt_chars=860, max_prompt_tokens_est=220)

    assert "trimmed_overlay_enum" in result.metadata["trim_reasons"]
    constraints_line = next(
        line for line in result.prompt.splitlines() if line.startswith("constraints=")
    )
    payload = json.loads(constraints_line[len("constraints=") :])

    assert payload["allowed_overlay_targets"] == ["left_mdi_pb15"]
    assert payload["overlay_target_policy"]["preferred_target"] == "left_mdi_pb15"
    assert payload["multimodal_input"]["attached"] is True


def test_prompt_explicitly_allows_multimodal_guidance_without_vision_fact_refs() -> None:
    ctx = {
        "candidate_steps": ["S08"],
        "overlay_target_allowlist": ["left_mdi_pb15", "left_mdi_pb18"],
        "vars": {},
        "recent_deltas": [],
        "vision": {
            "vision_used": True,
            "frame_ids": ["1773401766789_000003"],
        },
        "vision_fact_summary": {"status": "vision_unavailable"},
        "deterministic_step_hint": {
            "inferred_step_id": "S08",
            "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
            "recent_ui_targets": [],
        },
    }

    result = build_help_prompt_result(ctx, "zh")
    payload = _extract_prompt_constraints_json(result.prompt)

    assert payload["multimodal_input"]["attached"] is True
    assert "可直接依据已附带图像判断 diagnosis/next 与单目标 overlay" in result.prompt


def test_help_prompt_explicitly_distinguishes_fcs_button_from_fcs_page() -> None:
    ctx = {
        "candidate_steps": ["S08"],
        "overlay_target_allowlist": ["left_mdi_pb15", "left_mdi_pb18"],
        "vars": {},
        "recent_deltas": [],
        "vision": {
            "vision_used": True,
            "frame_ids": ["1773401766789_000003"],
        },
        "vision_fact_summary": {
            "status": "available",
            "seen_fact_ids": ["left_ddi_fcs_page_button_visible"],
            "not_seen_fact_ids": ["fcs_page_visible"],
        },
        "deterministic_step_hint": {
            "inferred_step_id": "S08",
            "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
            "recent_ui_targets": [],
            "visual_action_hint": {
                "target": "left_mdi_pb15",
                "reason": "Press PB15 to enter the FCS page.",
            },
        },
    }

    result = build_help_prompt_result(ctx, "zh")

    assert "不要把“左 DDI 看见 FCS 按钮/菜单项”误判成“已经进入 FCS 页面”" in result.prompt
    assert "LEF/TEF/AIL/RUD" in result.prompt
    assert "SV1/SV2" in result.prompt
    assert "大量 X/故障填充" in result.prompt
    assert "先按 PB18 切到 SUPT 页，再找 FCS" in result.prompt


def test_help_prompt_explicitly_stages_s18_root_fcsmc_in_test_and_final_go() -> None:
    ctx = {
        "candidate_steps": ["S18"],
        "overlay_target_allowlist": ["fcs_bit_switch", "right_mdi_pb5"],
        "vars": {"fcs_bit_switch_up": True},
        "recent_deltas": [],
        "vision": {
            "vision_used": True,
            "frame_ids": ["1773420347380_000049"],
        },
        "vision_fact_summary": {"status": "vision_unavailable"},
        "deterministic_step_hint": {
            "inferred_step_id": "S18",
            "overlay_step_id": "S18",
            "observability_status": "partial",
            "requires_visual_confirmation": True,
            "step_evidence_requirements": ["delta", "gate", "visual"],
            "action_hint": {"target": "right_mdi_pb5"},
        },
    }

    zh_result = build_help_prompt_result(ctx, "zh")
    en_result = build_help_prompt_result(ctx, "en")

    assert "若右 DDI 仍是 BIT FAILURES / BIT root 页面，下一步就是按 PB5 进入 FCS-MC" in zh_result.prompt
    assert "若已经进入 FCS-MC 页面但还未开始测试，才是“按住 FCS BIT 开关并按 PB5”这一步" in zh_result.prompt
    assert "若页面已显示 IN TEST、PBIT GO、FCSA/FCSB PBIT GO" in zh_result.prompt
    assert "FCSA/FCSB PBIT GO 不等于最终 GO" in zh_result.prompt
    assert "禁止仅凭 VARS.fcs_bit_switch_up 的 true/false 单独判断 S18 所处页面阶段" in zh_result.prompt

    assert "if the right DDI is still on the BIT FAILURES / BIT root page, the next action is PB5 to enter FCS-MC" in en_result.prompt
    assert "only after the right DDI has entered the FCS-MC page but before the BIT has started" in en_result.prompt
    assert "if the page already shows IN TEST, PBIT GO, FCSA/FCSB PBIT GO" in en_result.prompt
    assert "FCSA/FCSB PBIT GO is not the same as the final GO result" in en_result.prompt
    assert "Never use VARS.fcs_bit_switch_up by itself to decide which S18 page/state the user is on" in en_result.prompt


def test_help_prompt_treats_fcsa_and_fcsb_go_as_final_s18_go_evidence() -> None:
    ctx = {
        "candidate_steps": ["S18"],
        "overlay_target_allowlist": ["right_mdi_pb5"],
        "vars": {},
        "recent_deltas": [],
        "vision": {"vision_used": True, "frame_ids": ["1773420856368_000057"]},
        "vision_fact_summary": {"status": "vision_unavailable"},
        "deterministic_step_hint": {
            "inferred_step_id": "S18",
            "overlay_step_id": "S18",
            "observability_status": "partial",
            "requires_visual_confirmation": True,
            "action_hint": {"target": "right_mdi_pb5"},
        },
    }

    zh_result = build_help_prompt_result(ctx, "zh")
    en_result = build_help_prompt_result(ctx, "en")

    assert "若右 DDI 能同时明确读到 FCSA=GO 与 FCSB=GO，可直接视为最终 GO 已成立" in zh_result.prompt
    assert "If the right DDI clearly shows both FCSA=GO and FCSB=GO at the same time" in en_result.prompt
    assert "clearly reading both FCSA=GO and FCSB=GO is sufficient final-GO evidence" in en_result.prompt


def test_help_prompt_explicitly_distinguishes_fcs_button_from_fcs_page_in_en() -> None:
    ctx = {
        "candidate_steps": ["S08"],
        "overlay_target_allowlist": ["left_mdi_pb15", "left_mdi_pb18"],
        "vars": {},
        "recent_deltas": [],
        "vision": {
            "vision_used": True,
            "frame_ids": ["1773401766789_000003"],
        },
        "vision_fact_summary": {
            "status": "available",
            "seen_fact_ids": ["left_ddi_fcs_page_button_visible"],
            "not_seen_fact_ids": ["fcs_page_visible"],
        },
        "deterministic_step_hint": {
            "inferred_step_id": "S08",
            "missing_conditions": ["vision_facts.fcs_page_visible==seen"],
            "recent_ui_targets": [],
            "visual_action_hint": {
                "target": "left_mdi_pb15",
                "reason": "Press PB15 to enter the FCS page.",
            },
        },
    }

    result = build_help_prompt_result(ctx, "en")

    assert "Do not mistake 'the left DDI shows the FCS button/menu entry'" in result.prompt
    assert "LEF/TEF/AIL/RUD" in result.prompt
    assert "SV1/SV2" in result.prompt
    assert "many X/fault fills" in result.prompt
    assert "press PB18 first to reach the SUPT page, then select FCS" in result.prompt


def test_prompt_omits_page_or_heading_when_non_scalar() -> None:
    ctx = _base_context()
    ctx["rag_topk"] = [
        {
            "doc_id": "manual",
            "section": "S03",
            "page_or_heading": {"unexpected": "mapping"},
            "snippet_id": "manual_s03_1",
            "snippet": "APU switch to ON and wait for APU READY.",
        }
    ]
    result = build_help_prompt_result(ctx, "en")
    payload = _extract_prompt_constraints_json(result.prompt)
    rag_block = payload["EVIDENCE_SOURCES"]["RAG_SNIPPETS"]
    assert len(rag_block) == 1
    assert "page_or_heading" not in rag_block[0]


def test_prompt_omits_page_or_heading_when_float_not_finite() -> None:
    ctx = _base_context()
    ctx["rag_topk"] = [
        {
            "doc_id": "manual",
            "section": "S03",
            "page_or_heading": float("nan"),
            "snippet_id": "manual_s03_1",
            "snippet": "APU switch to ON and wait for APU READY.",
        },
        {
            "doc_id": "manual",
            "section": "S04",
            "page_or_heading": float("inf"),
            "snippet_id": "manual_s04_1",
            "snippet": "Engine crank after APU READY.",
        },
    ]
    result = build_help_prompt_result(ctx, "en")
    payload = _extract_prompt_constraints_json(result.prompt)
    rag_block = payload["EVIDENCE_SOURCES"]["RAG_SNIPPETS"]
    assert len(rag_block) == 2
    assert "page_or_heading" not in rag_block[0]
    assert "page_or_heading" not in rag_block[1]


def test_prompt_sanitizes_non_finite_float_vars_to_strict_json_scalars() -> None:
    ctx = _base_context()
    ctx["vars"]["nan_value"] = float("nan")
    ctx["vars"]["pos_inf"] = float("inf")
    ctx["vars"]["neg_inf"] = float("-inf")
    result = build_help_prompt_result(ctx, "en")
    payload = _extract_prompt_constraints_json(result.prompt)
    selected = payload["current_vars_selected"]
    assert selected["nan_value"] == "NaN"
    assert selected["pos_inf"] == "Infinity"
    assert selected["neg_inf"] == "-Infinity"
    assert ":NaN" not in result.prompt
    assert ":Infinity" not in result.prompt
    assert ":-Infinity" not in result.prompt


def test_prompt_contains_strict_json_output_constraints() -> None:
    prompt = build_help_prompt(_base_context(), "en")
    assert "must output exactly one strict JSON object" in prompt
    assert "no prose, no markdown, no code fences" in prompt
    assert "diagnosis.error_category must be chosen from allowed_error_categories." in prompt
    assert "Each target must have at least one evidence item" in prompt
    assert '"diagnosis":{"step_id":"...","error_category":"..."}' in prompt
    assert (
        '"overlay":{"targets":["..."],"evidence":[{"target":"...","type":"...",'
        '"ref":"...","quote":"...","grounding_confidence":0.0}]}'
    ) in prompt


def test_prompt_lang_switch_zh_and_en() -> None:
    prompt_zh = build_help_prompt(_base_context(), "zh")
    prompt_en = build_help_prompt(_base_context(), "en")

    assert "\u4f60\u662f SimTutor \u52a9\u6559" in prompt_zh
    assert "You are SimTutor tutor assistant" in prompt_en


def test_prompt_does_not_leak_api_key_or_absolute_path() -> None:
    prompt = build_help_prompt(_base_context(), "en")
    assert "sk-should-not-leak" not in prompt
    assert "api_key" not in prompt
    assert "C:\\Users\\secret\\project" not in prompt
    assert "[REDACTED_PATH]" in prompt


def test_delta_summary_top_k_is_capped_to_20() -> None:
    ctx = _base_context()
    many = []
    for i in range(30):
        many.append({"mapped_ui_target": f"target_{i:02d}", "from": 0, "to": 1, "action": "toggle"})
    ctx["recent_deltas"] = many

    result = build_help_prompt_result(
        ctx,
        "en",
        max_prompt_chars=20000,
        max_prompt_tokens_est=6000,
    )
    payload = _extract_prompt_constraints_json(result.prompt)
    summary = payload["recent_deltas_summary"]
    assert summary["top_k"] == MAX_DELTA_SUMMARY_ITEMS
    assert len(summary["items"]) == MAX_DELTA_SUMMARY_ITEMS
    assert summary["total_targets"] == 30


def test_budget_trim_enforced_and_recorded_in_metadata() -> None:
    ctx = _base_context()
    ctx["vars"] = {f"v_{i:03d}": "x" * 200 for i in range(120)}
    ctx["recent_deltas"] = [
        {"mapped_ui_target": f"target_{i:02d}", "from": i, "to": i + 1, "action": "toggle"} for i in range(50)
    ]

    result = build_help_prompt_result(ctx, "en", max_prompt_chars=900, max_prompt_tokens_est=180)

    assert result.metadata["prompt_trimmed"] is True
    assert result.metadata["trim_reasons"]
    assert len(result.prompt) <= 900
    assert result.metadata["prompt_tokens_est"] <= result.metadata["hard_prompt_tokens_est"]
    assert result.metadata["prompt_budget_status"] in {"compacted", "trimmed_to_hard_cap"}
    assert isinstance(result.metadata["allowed_evidence_refs"], list)


def test_budget_over_advisory_does_not_force_trim() -> None:
    ctx = {
        "candidate_steps": ["S01"],
        "overlay_target_allowlist": ["battery_switch", "apu_switch"],
        "vars": {},
        "recent_deltas": [],
    }

    result = build_help_prompt_result(ctx, "en", max_prompt_chars=5000, max_prompt_tokens_est=1300)

    assert result.metadata["prompt_trimmed"] is False
    assert result.metadata["prompt_budget_status"] == "over_advisory"
    assert result.metadata["prompt_chars"] > result.metadata["advisory_prompt_chars"] or (
        result.metadata["prompt_tokens_est"] > result.metadata["advisory_prompt_tokens_est"]
    )
    assert result.metadata["hard_prompt_chars"] >= result.metadata["advisory_prompt_chars"]
    assert result.metadata["hard_prompt_tokens_est"] >= result.metadata["advisory_prompt_tokens_est"]


def test_budget_trim_prioritizes_vars_before_rag_snippets() -> None:
    ctx = _base_context()
    ctx["vars"] = {f"v_{i:03d}": "w" * 120 for i in range(90)}
    ctx["rag_topk"] = [
        {
            "doc_id": "manual",
            "section": "S03",
            "page_or_heading": "S03",
            "snippet_id": f"manual_s03_{i}",
            "snippet": "APU switch ON and wait for READY. " * 12,
        }
        for i in range(5)
    ]

    result = build_help_prompt_result(ctx, "en", max_prompt_chars=900, max_prompt_tokens_est=180)

    reasons = result.metadata["trim_reasons"]
    assert "trimmed_vars" in reasons
    if "trimmed_rag_snippets" in reasons:
        assert reasons.index("trimmed_vars") < reasons.index("trimmed_rag_snippets")


def test_budget_trim_can_drop_last_rag_snippet_before_compact_template() -> None:
    base_ctx = {
        "candidate_steps": ["S01"],
        "overlay_target_allowlist": ["battery_switch"],
        "vars": {"battery_on": False},
        "recent_deltas": [],
    }
    base_result = build_help_prompt_result(
        base_ctx,
        "en",
        max_prompt_chars=20000,
        max_prompt_tokens_est=5000,
    )

    ctx = dict(base_ctx)
    ctx["rag_topk"] = [
        {
            "doc_id": "manual",
            "section": "S01",
            "page_or_heading": "S01",
            "snippet_id": "manual_s01_1",
            "snippet": "Battery ON and generators ON. " * 120,
        }
    ]

    result = build_help_prompt_result(
        ctx,
        "en",
        max_prompt_chars=700,
        max_prompt_tokens_est=160,
    )

    assert "compact_template" in result.metadata["trim_reasons"]
    assert result.metadata["rag_snippet_count"] == 1
    assert result.metadata["rag_snippet_ids"] == ["manual_s01_1"]
    assert result.metadata["grounding_applied"] is True
    assert result.metadata["grounding_missing"] is False


def test_budget_trim_logs_by_default_without_terminal_print(capsys, caplog) -> None:
    ctx = _base_context()
    ctx["vars"] = {f"v_{i:03d}": "y" * 120 for i in range(80)}
    with caplog.at_level("WARNING"):
        build_help_prompt_result(ctx, "zh", max_prompt_chars=700, max_prompt_tokens_est=120)
    out = capsys.readouterr().out
    assert out == ""
    assert "Prompt trimmed to fit budget" in caplog.text


def test_budget_trim_can_print_terminal_when_enabled(monkeypatch, capsys, caplog) -> None:
    ctx = _base_context()
    ctx["vars"] = {f"v_{i:03d}": "z" * 120 for i in range(80)}
    monkeypatch.setenv("SIMTUTOR_PROMPT_TRIM_PRINT", "1")
    with caplog.at_level("WARNING"):
        build_help_prompt_result(ctx, "en", max_prompt_chars=700, max_prompt_tokens_est=120)
    out = capsys.readouterr().out
    assert "[PROMPT] Prompt trimmed to fit budget" in out
    assert "Prompt trimmed to fit budget" in caplog.text


def test_prompt_contains_current_and_recent_button_signal() -> None:
    recent_deltas = [
        {"t_wall": 10.0, "seq": 1, "delta": {"BATTERY_SW": 2}},
        {"t_wall": 11.0, "seq": 2, "delta": {"ENGINE_CRANK_SW": 1}},
    ]
    bios_to_ui = {
        "mappings": {
            "BATTERY_SW": ["battery_switch"],
            "ENGINE_CRANK_SW": ["eng_crank_switch"],
        }
    }

    ctx = _base_context()
    ctx["recent_actions"] = build_recent_button_signal(recent_deltas, bios_to_ui, max_items=8)

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))
    signal = payload["recent_actions_signal"]
    assert signal["current_button"] == "eng_crank_switch"
    assert signal["recent_buttons"] == ["eng_crank_switch", "battery_switch"]


def test_prompt_recent_actions_signal_keeps_all_ui_targets_in_list_style() -> None:
    ctx = _base_context()
    ctx["recent_actions"] = [
        {
            "ui_targets": [
                "ufc_comm1_channel_selector_rotate",
                "ufc_comm1_channel_selector_pull",
            ]
        }
    ]
    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))
    signal = payload["recent_actions_signal"]
    assert signal["current_button"] == "ufc_comm1_channel_selector_rotate"
    assert signal["recent_buttons"] == [
        "ufc_comm1_channel_selector_rotate",
        "ufc_comm1_channel_selector_pull",
    ]


def test_prompt_recent_actions_signal_uses_dedicated_cap() -> None:
    ctx = _base_context()
    ctx["recent_actions"] = {"recent_buttons": [f"btn_{i:02d}" for i in range(30)]}

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))
    signal = payload["recent_actions_signal"]
    assert len(signal["recent_buttons"]) == MAX_RECENT_ACTIONS_SIGNAL_ITEMS
    assert signal["current_button"] == "btn_00"


def test_prompt_contains_deterministic_step_hint_when_provided() -> None:
    ctx = _base_context()
    ctx["deterministic_step_hint"] = {
        "inferred_step_id": "S03",
        "missing_conditions": ["vars.apu_ready==true"],
        "recent_ui_targets": ["apu_switch"],
    }

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))
    hint = payload["deterministic_step_hint"]
    assert hint["inferred_step_id"] == "S03"
    assert hint["missing_conditions"] == ["vars.apu_ready==true"]
    assert hint["recent_ui_targets"] == ["apu_switch"]


def test_prompt_deterministic_step_hint_accepts_tuple_inputs() -> None:
    ctx = _base_context()
    ctx["deterministic_step_hint"] = {
        "inferred_step_id": "S03",
        "missing_conditions": ("vars.apu_ready==true",),
        "recent_ui_targets": ("apu_switch",),
    }

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))
    hint = payload["deterministic_step_hint"]
    assert hint["inferred_step_id"] == "S03"
    assert hint["missing_conditions"] == ["vars.apu_ready==true"]
    assert hint["recent_ui_targets"] == ["apu_switch"]


def test_prompt_deterministic_step_hint_keeps_step_signal_metadata() -> None:
    ctx = _base_context()
    ctx["deterministic_step_hint"] = {
        "inferred_step_id": "S15",
        "missing_conditions": ["check_fcs_page"],
        "recent_ui_targets": ["left_mdi_pb5"],
        "observability_status": "partial",
        "step_evidence_requirements": ["visual", "gate", "visual", "invalid_type"],
    }

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))
    hint = payload["deterministic_step_hint"]
    assert hint["inferred_step_id"] == "S15"
    assert hint["observability"] == "partial"
    assert hint["observability_status"] == "partial"
    assert hint["step_evidence_requirements"] == ["visual", "gate"]
    assert hint["requires_visual_confirmation"] is True
    assert payload["allowed_overlay_evidence_types"] == ["var", "gate", "rag", "delta", "visual"]
