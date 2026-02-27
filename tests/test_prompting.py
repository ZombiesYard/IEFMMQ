import json

from adapters.prompting import (
    MAX_DELTA_SUMMARY_ITEMS,
    build_help_prompt,
    build_help_prompt_result,
)


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


def _extract_constraints_json(prompt: str) -> dict:
    marker = "Context and constraints JSON:\n"
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\nOutput must follow this schema shape exactly:")
    return json.loads(prompt[start:end])


def test_prompt_contains_enum_constraints_and_delta_summary() -> None:
    prompt = build_help_prompt(_base_context(), "en")
    payload = _extract_constraints_json(prompt)

    assert payload["allowed_step_ids"] == ["S02", "S03"]
    assert payload["allowed_overlay_targets"] == ["apu_switch", "battery_switch"]
    summary = payload["recent_deltas_summary"]
    assert summary["top_k"] == MAX_DELTA_SUMMARY_ITEMS
    assert len(summary["items"]) == 2
    assert summary["items"][0]["ui_target"] == "apu_switch"


def test_prompt_contains_strict_json_output_constraints() -> None:
    prompt = build_help_prompt(_base_context(), "en")
    assert "must output exactly one strict JSON object" in prompt
    assert "no prose, no markdown, no code fences" in prompt
    assert '"diagnosis":{"step_id":"...","error_category":"..."}' in prompt


def test_prompt_lang_switch_zh_and_en() -> None:
    prompt_zh = build_help_prompt(_base_context(), "zh")
    prompt_en = build_help_prompt(_base_context(), "en")

    assert "你是 SimTutor 助教" in prompt_zh
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

    payload = _extract_constraints_json(build_help_prompt(ctx, "en"))
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


def test_budget_trim_records_terminal_and_log(capsys, caplog) -> None:
    ctx = _base_context()
    ctx["vars"] = {f"v_{i:03d}": "y" * 120 for i in range(80)}
    with caplog.at_level("WARNING"):
        build_help_prompt_result(ctx, "zh", max_prompt_chars=700, max_prompt_tokens_est=120)
    out = capsys.readouterr().out
    assert "[PROMPT] Prompt trimmed to fit budget" in out
    assert "Prompt trimmed to fit budget" in caplog.text
