from adapters.ollama_model import OllamaModel
from core.llm_schema import validate_help_response
from core.types import Observation
from tests._fakes import (
    FakeClient,
    FakeResponse,
    _help_obj_ok,
    _ollama_message_payload_from_help_obj,
    _ollama_response_payload_from_help_obj,
    _request_help,
)


def test_explain_error_success_with_valid_help_response() -> None:
    help_obj = _help_obj_ok()
    fake = FakeClient(responses=[FakeResponse(_ollama_message_payload_from_help_obj(help_obj, fenced=True))])
    model = OllamaModel(model_name="qwen3:8b", client=fake)
    obs = Observation(source="mock", procedure_hint="S03")
    req = _request_help()

    res = model.explain_error(obs, req)

    assert res.status == "ok"
    assert res.actions == [{"type": "overlay", "intent": "highlight", "target": "apu_switch"}]
    assert res.metadata["provider"] == "ollama"
    assert res.metadata["model"] == "qwen3:8b"
    assert isinstance(res.metadata["latency_ms"], int)
    assert res.metadata["json_repaired"] is True
    assert "removed_code_fence" in res.metadata["json_repair_reasons"]
    assert res.metadata["evidence_guardrail_applied"] is False
    assert res.metadata["evidence_guardrail_reasons"] == []
    assert "prompt_build" in res.metadata
    assert "max_prompt_chars" in res.metadata["prompt_build"]
    validate_help_response(res.metadata["help_response"])

    call = fake.calls[0]
    assert call["url"].endswith("/api/chat")
    assert call["json"]["model"] == "qwen3:8b"
    assert call["json"]["stream"] is False
    assert call["json"]["options"]["temperature"] == 0


def test_explain_error_bad_output_fallback_no_overlay() -> None:
    fake = FakeClient(responses=[FakeResponse({"message": {"content": "not-json"}})])
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")

    res = model.explain_error(obs, _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "ollama"
    assert res.metadata["model"] == "qwen3:8b"
    assert isinstance(res.metadata["latency_ms"], int)


def test_explain_error_missing_target_evidence_fallback_no_overlay() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["evidence"] = []
    fake = FakeClient(responses=[FakeResponse(_ollama_message_payload_from_help_obj(help_obj))])
    model = OllamaModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "ollama"


def test_explain_error_invalid_evidence_ref_clears_overlay() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["evidence"][0]["ref"] = "UNLISTED.ref"
    fake = FakeClient(responses=[FakeResponse(_ollama_message_payload_from_help_obj(help_obj))])
    model = OllamaModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert res.actions == []
    assert res.explanations[0].startswith("\u9700\u8981\u66f4\u591a\u4fe1\u606f/\u8bf7\u786e\u8ba4")
    assert res.metadata["evidence_guardrail_applied"] is True
    assert "invalid_target_evidence_refs:apu_switch" in res.metadata["evidence_guardrail_reasons"]
    assert res.metadata["help_response"]["overlay"]["targets"] == []


def test_explain_error_no_allowed_evidence_refs_clears_overlay() -> None:
    help_obj = _help_obj_ok()
    fake = FakeClient(responses=[FakeResponse(_ollama_message_payload_from_help_obj(help_obj))])
    model = OllamaModel(client=fake)
    req = _request_help()
    req.context["vars"] = {}
    req.context["recent_deltas"] = []
    req.context["gates"] = {}
    req.context["rag_topk"] = []

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "ok"
    assert res.actions == []
    assert res.explanations[0].startswith("\u9700\u8981\u66f4\u591a\u4fe1\u606f/\u8bf7\u786e\u8ba4")
    assert res.metadata["evidence_guardrail_applied"] is True
    assert "invalid_target_evidence_refs:apu_switch" in res.metadata["evidence_guardrail_reasons"]
    assert res.metadata["help_response"]["overlay"]["targets"] == []


def test_explain_error_context_allowlist_rejects_target_and_fallback() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["targets"] = ["battery_switch"]
    req = _request_help()
    req.context["overlay_target_allowlist"] = ["apu_switch"]
    fake = FakeClient(responses=[FakeResponse(_ollama_message_payload_from_help_obj(help_obj))])
    model = OllamaModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "error"
    assert res.actions == []


def test_prompt_is_stable_for_same_input() -> None:
    help_obj = _help_obj_ok()
    fake = FakeClient(
        responses=[
            FakeResponse(_ollama_message_payload_from_help_obj(help_obj)),
            FakeResponse(_ollama_message_payload_from_help_obj(help_obj)),
        ]
    )
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")
    req = _request_help()

    res1 = model.explain_error(obs, req)
    res2 = model.explain_error(obs, req)

    assert res1.status == "ok"
    assert res2.status == "ok"
    prompt1 = fake.calls[0]["json"]["messages"][1]["content"]
    prompt2 = fake.calls[1]["json"]["messages"][1]["content"]
    assert prompt1 == prompt2


def test_explain_error_http_error_fallback() -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=500)])
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")

    res = model.explain_error(obs, _request_help())

    assert res.status == "error"
    assert res.metadata["provider"] == "ollama"


def test_explain_error_timeout_or_connection_error_fallback() -> None:
    for exc in (TimeoutError("request timeout"), ConnectionError("connection failed")):
        fake = FakeClient(to_raise=exc)
        model = OllamaModel(client=fake)
        obs = Observation(source="mock", procedure_hint="S03")

        res = model.explain_error(obs, _request_help())

        assert res.status == "error"
        assert res.actions == []
        assert res.metadata["provider"] == "ollama"


def test_explain_error_step_id_not_in_candidate_steps_fallback() -> None:
    help_obj = _help_obj_ok()
    help_obj["diagnosis"]["step_id"] = "S02"
    fake = FakeClient(responses=[FakeResponse(_ollama_message_payload_from_help_obj(help_obj))])
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")
    req = _request_help()
    req.context["candidate_steps"] = ["S03"]

    res = model.explain_error(obs, req)

    assert res.status == "error"
    assert res.metadata["provider"] == "ollama"


def test_explain_error_alternate_response_format() -> None:
    help_obj = _help_obj_ok()
    fake = FakeClient(responses=[FakeResponse(_ollama_response_payload_from_help_obj(help_obj))])
    model = OllamaModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert res.actions == [{"type": "overlay", "intent": "highlight", "target": "apu_switch"}]
    assert res.metadata["provider"] == "ollama"
    assert res.metadata["model"] == "qwen3:8b"
    assert res.metadata["json_repaired"] is False
    assert res.metadata["json_repair_reasons"] == []
    validate_help_response(res.metadata["help_response"])
