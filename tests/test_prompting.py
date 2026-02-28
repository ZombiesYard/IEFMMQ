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

    sample_evidence = payload["output_example_json"]["overlay"]["evidence"][0]
    assert sample_evidence["type"] in {"var", "gate", "rag", "delta"}
    assert sample_evidence["ref"] in payload["allowed_evidence_refs"]
    assert len(sample_evidence["quote"]) <= 120


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

    payload = _extract_prompt_constraints_json(build_help_prompt(ctx, "en"))
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
