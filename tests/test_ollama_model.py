from adapters.ollama_model import OllamaModel
from core.llm_schema import validate_help_response
from core.types import Observation
from tests._fakes import (
    FakeClient,
    FakeResponse,
    help_obj_ok,
    ollama_message_payload_from_help_obj,
    ollama_response_payload_from_help_obj,
    request_help,
)


def test_explain_error_success_with_valid_help_response() -> None:
    help_obj = help_obj_ok()
    fake = FakeClient(responses=[FakeResponse(ollama_message_payload_from_help_obj(help_obj, fenced=True))])
    model = OllamaModel(model_name="qwen3.5:35b", client=fake)
    obs = Observation(source="mock", procedure_hint="S03")
    req = request_help()

    res = model.explain_error(obs, req)

    assert res.status == "ok"
    assert res.actions == [{"type": "overlay", "intent": "highlight", "target": "apu_switch"}]
    assert res.metadata["provider"] == "ollama"
    assert res.metadata["model"] == "qwen3.5:35b"
    assert isinstance(res.metadata["latency_ms"], int)
    validate_help_response(res.metadata["help_response"])

    call = fake.calls[0]
    assert call["url"].endswith("/api/chat")
    assert call["json"]["model"] == "qwen3.5:35b"
    assert call["json"]["stream"] is False
    assert call["json"]["options"]["temperature"] == 0


def test_explain_error_bad_output_fallback_no_overlay() -> None:
    fake = FakeClient(responses=[FakeResponse({"message": {"content": "not-json"}})])
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")

    res = model.explain_error(obs, request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "ollama"
    assert res.metadata["model"] == "qwen3.5:35b"
    assert isinstance(res.metadata["latency_ms"], int)


def test_explain_error_context_allowlist_rejects_target_and_fallback() -> None:
    help_obj = help_obj_ok()
    help_obj["overlay"]["targets"] = ["battery_switch"]
    req = request_help()
    req.context["overlay_target_allowlist"] = ["apu_switch"]
    fake = FakeClient(responses=[FakeResponse(ollama_message_payload_from_help_obj(help_obj))])
    model = OllamaModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "error"
    assert res.actions == []


def test_prompt_is_stable_for_same_input() -> None:
    help_obj = help_obj_ok()
    fake = FakeClient(
        responses=[
            FakeResponse(ollama_message_payload_from_help_obj(help_obj)),
            FakeResponse(ollama_message_payload_from_help_obj(help_obj)),
        ]
    )
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")
    req = request_help()

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

    res = model.explain_error(obs, request_help())

    assert res.status == "error"
    assert res.metadata["provider"] == "ollama"


def test_explain_error_step_id_not_in_candidate_steps_fallback() -> None:
    help_obj = help_obj_ok()
    help_obj["diagnosis"]["step_id"] = "S02"
    fake = FakeClient(responses=[FakeResponse(ollama_message_payload_from_help_obj(help_obj))])
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")
    req = request_help()
    req.context["candidate_steps"] = ["S03"]

    res = model.explain_error(obs, req)

    assert res.status == "error"
    assert res.metadata["provider"] == "ollama"


def test_explain_error_alternate_response_format() -> None:
    help_obj = help_obj_ok()
    fake = FakeClient(responses=[FakeResponse(ollama_response_payload_from_help_obj(help_obj))])
    model = OllamaModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request_help())

    assert res.status == "ok"
