import json

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


def test_prompt_contains_enum_constraints_delta_summary_and_evidence_sources() -> None:
    prompt = build_help_prompt(_base_context(), "en")
    payload = _extract_prompt_constraints_json(prompt)

    assert payload["allowed_step_ids"] == ["S02", "S03"]
    assert payload["allowed_overlay_targets"] == ["apu_switch", "battery_switch"]
    assert payload["allowed_error_categories"]
    assert payload["output_example_json"]["diagnosis"]["error_category"] in payload["allowed_error_categories"]
    summary = payload["recent_deltas_summary"]
    assert summary["top_k"] == MAX_DELTA_SUMMARY_ITEMS
    assert len(summary["items"]) == 2
    assert summary["items"][0]["ui_target"] == "apu_switch"
    evidence = payload["EVIDENCE_SOURCES"]
    assert set(evidence.keys()) == {"VARS", "GATES", "RECENT_UI_TARGETS", "RAG_SNIPPETS"}
    assert "RECENT_UI_TARGETS.apu_switch" in payload["allowed_evidence_refs"]
    assert payload["deterministic_step_hint"]["inferred_step_id"] is None
    assert payload["deterministic_step_hint"]["missing_conditions"] == []
    assert payload["grounding"]["missing"] is True
    assert payload["grounding"]["applied"] is False

    sample_evidence = payload["output_example_json"]["overlay"]["evidence"][0]
    assert sample_evidence["type"] in {"var", "gate", "rag", "delta"}
    assert sample_evidence["ref"] in payload["allowed_evidence_refs"]
    assert len(sample_evidence["quote"]) <= 120


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
    constraints_line = next(
        line for line in result.prompt.splitlines() if line.startswith("constraints=")
    )
    payload = json.loads(constraints_line[len("constraints=") :])

    assert "compact_template" in result.metadata["trim_reasons"]
    assert payload["grounding"]["applied"] is False
    assert payload["grounding"]["missing"] is True
    assert payload["grounding"]["reason"] in {"rag_snippets_not_injected", "no_rag_snippets"}
    assert payload["allowed_evidence_refs"] == []
    assert result.metadata["rag_snippet_count"] == 0
    assert result.metadata["rag_snippet_ids"] == []
    assert result.metadata["allowed_evidence_refs"] == []
    assert result.metadata["evidence_refs_count"] == 0
    assert result.metadata["grounding_applied"] is False
    assert result.metadata["grounding_missing"] is True


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
    assert result.metadata["prompt_tokens_est"] <= 180
    assert isinstance(result.metadata["allowed_evidence_refs"], list)


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
        max_prompt_chars=len(base_result.prompt) + 40,
        max_prompt_tokens_est=5000,
    )

    assert "trimmed_rag_snippets" in result.metadata["trim_reasons"]
    assert "compact_template" not in result.metadata["trim_reasons"]
    assert result.metadata["rag_snippet_count"] == 0
    assert result.metadata["grounding_applied"] is False


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
