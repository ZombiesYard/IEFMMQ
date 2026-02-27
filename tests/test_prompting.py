import json

from adapters.prompting import build_help_prompt


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


def test_prompt_contains_enum_constraints_and_recent_actions() -> None:
    prompt = build_help_prompt(_base_context(), "en")
    payload = _extract_constraints_json(prompt)

    assert payload["allowed_step_ids"] == ["S02", "S03"]
    assert payload["allowed_overlay_targets"] == ["apu_switch", "battery_switch"]
    assert len(payload["recent_actions"]) == 2
    assert payload["recent_actions"][0]["ui_target"] == "apu_switch"


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
