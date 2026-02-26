import json
from typing import Any

from adapters.openai_compat_model import OpenAICompatModel
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
    def __init__(self, responses: list[_FakeResponse] | None = None, to_raise: Exception | None = None) -> None:
        self._responses = list(responses or [])
        self._to_raise = to_raise
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        json: dict[str, Any],  # noqa: A002
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self._to_raise is not None:
            raise self._to_raise
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


def _chat_payload_from_help_obj(help_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(help_obj, ensure_ascii=False),
                },
            }
        ]
    }


def test_explain_error_success_200_valid_help_response() -> None:
    fake = _FakeClient(responses=[_FakeResponse(_chat_payload_from_help_obj(_help_obj_ok()), status_code=200)])
    model = OpenAICompatModel(
        model_name="Qwen3.5-32B-Instruct",
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
    assert res.metadata["model"] == "Qwen3.5-32B-Instruct"
    assert isinstance(res.metadata["latency_ms"], int)
    validate_help_response(res.metadata["help_response"])

    call = fake.calls[0]
    assert call["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert call["json"]["model"] == "Qwen3.5-32B-Instruct"
    assert call["json"]["temperature"] == 0
    assert call["headers"]["Authorization"] == "Bearer sk-local"
    assert call["timeout"] == 15.0


def test_explain_error_http_429_fallback_no_overlay() -> None:
    fake = _FakeClient(responses=[_FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_explain_error_http_5xx_fallback_no_overlay() -> None:
    fake = _FakeClient(responses=[_FakeResponse({}, status_code=500)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_explain_error_timeout_fallback_no_overlay() -> None:
    fake = _FakeClient(to_raise=TimeoutError("request timeout"))
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
    fake = _FakeClient(responses=[_FakeResponse(payload)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"


def test_prompt_is_stable_for_same_input() -> None:
    payload = _chat_payload_from_help_obj(_help_obj_ok())
    fake = _FakeClient(responses=[_FakeResponse(payload), _FakeResponse(payload)])
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

