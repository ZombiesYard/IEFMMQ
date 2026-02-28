from adapters.openai_compat_model import OpenAICompatModel
from core.llm_schema import validate_help_response
from core.types import Observation
from tests._fakes import (
    FakeClient,
    FakeResponse,
    _help_obj_ok,
    _openai_chat_payload_from_help_obj,
    _request_help,
    extract_prompt_constraints_json,
)


def test_explain_error_success_200_valid_help_response() -> None:
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(_help_obj_ok()), status_code=200)])
    model = OpenAICompatModel(
        model_name="Qwen3-8B-Instruct",
        base_url="http://127.0.0.1:8000",
        timeout_s=15.0,
        api_key="sk-local",
        client=fake,
    )
    obs = Observation(source="mock", procedure_hint="S03")

    res = model.explain_error(obs, _request_help())

    assert res.status == "ok"
    assert res.actions == [{"type": "overlay", "intent": "highlight", "target": "apu_switch"}]
    assert res.metadata["provider"] == "openai_compat"
    assert res.metadata["model"] == "Qwen3-8B-Instruct"
    assert isinstance(res.metadata["latency_ms"], int)
    assert res.metadata["json_repaired"] is False
    assert res.metadata["json_repair_reasons"] == []
    assert res.metadata["evidence_guardrail_applied"] is False
    assert res.metadata["evidence_guardrail_reasons"] == []
    assert isinstance(res.metadata["prompt_budget_used"], int)
    assert res.metadata["prompt_budget_used"] > 0
    assert res.metadata["delta_dropped_count"] == 0
    validate_help_response(res.metadata["help_response"])

    call = fake.calls[0]
    assert call["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert call["json"]["model"] == "Qwen3-8B-Instruct"
    assert call["json"]["temperature"] == 0
    assert call["headers"]["Authorization"] == "Bearer sk-local"
    assert call["timeout"] == 15.0
    prompt_payload = extract_prompt_constraints_json(call["json"]["messages"][1]["content"])
    assert "deterministic_step_hint" in prompt_payload
    assert "inferred_step_id" in prompt_payload["deterministic_step_hint"]


def test_explain_error_http_429_fallback_no_overlay() -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_explain_error_missing_target_evidence_fallback_no_overlay() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["evidence"] = []
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(help_obj), status_code=200)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_explain_error_invalid_evidence_ref_clears_overlay() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["evidence"][0]["ref"] = "UNKNOWN.ref"
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(help_obj), status_code=200)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert res.actions == []
    assert res.explanations[0].startswith("\u9700\u8981\u66f4\u591a\u4fe1\u606f/\u8bf7\u786e\u8ba4")
    assert res.metadata["evidence_guardrail_applied"] is True
    assert "invalid_target_evidence_refs:apu_switch" in res.metadata["evidence_guardrail_reasons"]
    assert res.metadata["help_response"]["overlay"]["targets"] == []


def test_explain_error_http_5xx_fallback_no_overlay() -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=500)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_explain_error_timeout_fallback_no_overlay() -> None:
    fake = FakeClient(to_raise=TimeoutError("request timeout"))
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_explain_error_non_json_output_fallback_no_overlay() -> None:
    payload = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "not-json"},
            }
        ]
    }
    fake = FakeClient(responses=[FakeResponse(payload)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_explain_error_zh_fallback_with_inferred_step_and_missing_conditions() -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake, lang="zh")
    req = _request_help()
    req.context["candidate_steps"] = ["S03", "S04"]
    req.context["vars"] = {"power_available": True, "apu_on": True, "apu_ready": False}
    req.context["recent_ui_targets"] = ["apu_switch"]

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "error"
    assert "\u4f60\u5927\u6982\u7387\u5361\u5728 S03" in (res.message or "")
    assert "\u4e0b\u4e00\u6b65\u8bf7\u5148\u6ee1\u8db3" in (res.message or "")
    hint = res.metadata["deterministic_step_hint"]
    assert hint["inferred_step_id"] == "S03"
    assert hint["missing_conditions"]


def test_explain_error_zh_fallback_with_inferred_step_without_missing_conditions() -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake, lang="zh")
    req = _request_help()
    req.context["vars"] = {
        "power_available": True,
        "apu_ready": True,
        "engine_crank_right": True,
        "rpm_r": 65,
        "bleed_air_norm": True,
    }
    req.context["recent_ui_targets"] = ["bleed_air_knob", "eng_crank_switch"]

    res = model.explain_error(Observation(source="mock"), req)

    assert res.status == "error"
    assert "\u4f60\u5927\u6982\u7387\u5361\u5728 S07" in (res.message or "")
    assert "\u4e0b\u4e00\u6b65\u8bf7\u6309\u8be5\u6b65\u9aa4\u68c0\u67e5\u5e76\u6267\u884c" in (res.message or "")


def test_explain_error_zh_fallback_without_inferred_step(monkeypatch) -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake, lang="zh")
    monkeypatch.setattr("adapters.base_help_model.load_pack_steps", lambda: [])

    res = model.explain_error(Observation(source="mock"), _request_help())

    assert res.status == "error"
    assert res.message == "\u65e0\u6cd5\u751f\u6210\u6a21\u578b\u7b54\u590d\uff0c\u8bf7\u5148\u68c0\u67e5\u5f53\u524d\u6b65\u9aa4\u524d\u7f6e\u6761\u4ef6\u540e\u518d\u89e6\u53d1 Help\u3002"
    assert res.metadata["deterministic_step_hint"]["inferred_step_id"] is None


def test_explain_error_handles_deterministic_inference_exception(monkeypatch) -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake, lang="zh")

    def _raise_inference(*_args, **_kwargs):
        raise RuntimeError("inference boom")

    monkeypatch.setattr(model, "_compute_deterministic_inference", _raise_inference)
    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert "\u4f60\u5927\u6982\u7387\u5361\u5728 S03" in (res.message or "")
    assert res.metadata["deterministic_step_hint"]["inferred_step_id"] == "S03"
    assert res.metadata["deterministic_inference_error"] == "RuntimeError: inference boom"


def test_prompt_is_stable_for_same_input() -> None:
    payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(payload), FakeResponse(payload)])
    model = OpenAICompatModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")
    req = _request_help()

    res1 = model.explain_error(obs, req)
    res2 = model.explain_error(obs, req)

    assert res1.status == "ok"
    assert res2.status == "ok"
    prompt1 = fake.calls[0]["json"]["messages"][1]["content"]
    prompt2 = fake.calls[1]["json"]["messages"][1]["content"]
    assert prompt1 == prompt2


def test_explain_error_prefix_suffix_repair_marks_metadata() -> None:
    help_obj = _help_obj_ok()
    payload = _openai_chat_payload_from_help_obj(help_obj)
    payload["choices"][0]["message"]["content"] = (
        "Here is the result:\n"
        + payload["choices"][0]["message"]["content"]
        + "\nDone."
    )
    fake = FakeClient(responses=[FakeResponse(payload)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert res.metadata["json_repaired"] is True
    assert "dropped_prefix_text" in res.metadata["json_repair_reasons"]
    assert "dropped_suffix_text" in res.metadata["json_repair_reasons"]


def test_explain_error_step_id_not_in_candidate_steps_fallback_no_overlay() -> None:
    help_obj = _help_obj_ok()
    help_obj["next"]["step_id"] = "S01"
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(help_obj), status_code=200)])
    model = OpenAICompatModel(client=fake)
    req = _request_help()
    req.context["candidate_steps"] = ["S02", "S03"]

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_explain_error_overlay_target_not_in_allowlist_fallback_no_overlay() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["targets"] = ["battery_switch"]
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(help_obj), status_code=200)])
    model = OpenAICompatModel(client=fake)
    req = _request_help()
    req.context["overlay_target_allowlist"] = ["apu_switch"]

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_explain_error_metadata_reads_delta_dropped_count_from_context() -> None:
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(_help_obj_ok()), status_code=200)])
    model = OpenAICompatModel(client=fake)
    req = _request_help()
    req.context["delta_summary"] = {"dropped_stats": {"dropped_total": 7, "dropped_by_reason": {"blacklist": 7}}}

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "ok"
    assert res.metadata["delta_dropped_count"] == 7


def test_explain_error_metadata_ignores_bool_delta_dropped_count_direct() -> None:
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(_help_obj_ok()), status_code=200)])
    model = OpenAICompatModel(client=fake)
    req = _request_help()
    req.context["delta_dropped_count"] = True
    req.context["delta_summary"] = {"dropped_stats": {"dropped_total": 3}}

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "ok"
    assert res.metadata["delta_dropped_count"] == 3


def test_explain_error_metadata_ignores_bool_nested_dropped_total() -> None:
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(_help_obj_ok()), status_code=200)])
    model = OpenAICompatModel(client=fake)
    req = _request_help()
    req.context["delta_summary"] = {"dropped_stats": {"dropped_total": True}}

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "ok"
    assert res.metadata["delta_dropped_count"] == 0
