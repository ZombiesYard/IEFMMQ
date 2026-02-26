import json
from typing import Any

from adapters.ollama_model import OllamaModel
from core.llm_schema import validate_help_response
from core.types import Observation, TutorRequest


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self._status_code = status_code

    def raise_for_status(self) -> None:
        if self._status_code >= 400:
            raise RuntimeError(f"http {self._status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any], timeout: float) -> _FakeResponse:  # noqa: A002
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if not self._responses:
            raise RuntimeError("no fake response left")
        return self._responses.pop(0)


def _help_obj_ok() -> dict[str, Any]:
    return {
        "diagnosis": {"step_id": "S02", "error_category": "OM"},
        "next": {"step_id": "S03"},
        "overlay": {"targets": ["apu_switch"]},
        "explanations": ["APU 未开启，请先将 APU 开关置于 ON。"],
        "confidence": 0.93,
    }


def _request_help() -> TutorRequest:
    return TutorRequest(
        intent="help",
        message="我卡在启动步骤",
        context={
            "vars": {"battery_on": True, "apu_on": False},
            "recent_deltas": [{"k": "apu_on", "from": 1, "to": 0}],
            "candidate_steps": ["S02", "S03"],
            "overlay_target_allowlist": ["apu_switch", "battery_switch"],
        },
    )


def test_explain_error_success_with_valid_help_response() -> None:
    help_obj = _help_obj_ok()
    fake = _FakeClient(
        [
            _FakeResponse(
                {
                    "message": {
                        "content": "```json\n" + json.dumps(help_obj, ensure_ascii=False) + "\n```",
                    }
                }
            )
        ]
    )
    model = OllamaModel(model_name="qwen3.5:35b", client=fake)
    obs = Observation(source="mock", procedure_hint="S03")
    req = _request_help()

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
    fake = _FakeClient([_FakeResponse({"message": {"content": "not-json"}})])
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")

    res = model.explain_error(obs, _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "ollama"
    assert res.metadata["model"] == "qwen3.5:35b"
    assert isinstance(res.metadata["latency_ms"], int)


def test_explain_error_context_allowlist_rejects_target_and_fallback() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["targets"] = ["battery_switch"]
    req = _request_help()
    req.context["overlay_target_allowlist"] = ["apu_switch"]
    fake = _FakeClient([_FakeResponse({"message": {"content": json.dumps(help_obj, ensure_ascii=False)}})])
    model = OllamaModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "error"
    assert res.actions == []


def test_prompt_is_stable_for_same_input() -> None:
    help_obj = _help_obj_ok()
    fake = _FakeClient(
        [
            _FakeResponse({"message": {"content": json.dumps(help_obj, ensure_ascii=False)}}),
            _FakeResponse({"message": {"content": json.dumps(help_obj, ensure_ascii=False)}}),
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
    fake = _FakeClient([_FakeResponse({}, status_code=500)])
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")

    res = model.explain_error(obs, _request_help())

    assert res.status == "error"
    assert res.metadata["provider"] == "ollama"

def test_explain_error_step_id_not_in_candidate_steps_fallback() -> None:
    help_obj = _help_obj_ok()
    help_obj["diagnosis"]["step_id"] = "S02"
    fake = _FakeClient([_FakeResponse({"message": {"content": json.dumps(help_obj, ensure_ascii=False)}})])
    model = OllamaModel(client=fake)
    obs = Observation(source="mock", procedure_hint="S03")
    req = _request_help()
    req.context["candidate_steps"] = ["S03"]

    res = model.explain_error(obs, req)

    assert res.status == "error"
    assert res.metadata["provider"] == "ollama"

def test_explain_error_alternate_response_format() -> None:
    help_obj = _help_obj_ok()
    fake = _FakeClient([_FakeResponse({"response": json.dumps(help_obj, ensure_ascii=False)})])
    model = OllamaModel(client=fake)
    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())
    assert res.status == "ok"