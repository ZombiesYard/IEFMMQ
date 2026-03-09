from pathlib import Path

from adapters.openai_compat_model import OpenAICompatModel
from core.help_failure import ALLOWLIST_FAIL, EVIDENCE_FAIL, JSON_EXTRACT_FAIL, MODEL_HTTP_FAIL, SCHEMA_FAIL
from core.llm_schema import validate_help_response
from core.types import Observation
from tests._fakes import (
    FakeClient,
    FakeResponse,
    _extract_prompt_constraints_json,
    _help_obj_ok,
    _openai_chat_payload_from_help_obj,
    _request_help,
)


def _attach_vision_context(
    request,
    *,
    primary_image: Path,
    pre_trigger_image: Path | None = None,
) -> None:
    trigger_frame = {
        "frame_id": "1772872445010_000123",
        "role": "trigger_frame",
        "image_uri": str(primary_image),
        "mime_type": "image/png",
        "sync_status": "matched_future_fallback",
        "sync_delta_ms": 10,
    }
    selected_frames = [trigger_frame]
    vision = {
        "status": "available",
        "vision_used": True,
        "frame_id": trigger_frame["frame_id"],
        "frame_ids": [trigger_frame["frame_id"]],
        "trigger_frame": trigger_frame,
        "selected_frames": selected_frames,
        "sync_status": "matched_future_fallback",
        "sync_delta_ms": 10,
        "frame_stale": False,
        "sync_miss_reason": None,
    }
    if pre_trigger_image is not None:
        pre_trigger_frame = {
            "frame_id": "1772872444950_000122",
            "role": "pre_trigger_frame",
            "image_uri": str(pre_trigger_image),
            "mime_type": "image/png",
            "sync_status": "matched_past",
            "sync_delta_ms": -50,
        }
        vision["pre_trigger_frame"] = pre_trigger_frame
        vision["selected_frames"] = [pre_trigger_frame, trigger_frame]
        vision["frame_id"] = pre_trigger_frame["frame_id"]
        vision["frame_ids"] = [pre_trigger_frame["frame_id"], trigger_frame["frame_id"]]
        vision["sync_status"] = "matched_past"
        vision["sync_delta_ms"] = -50
        vision["frame_stale"] = True
    request.context["vision"] = vision


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
    assert len(res.actions) == 1
    assert res.actions[0]["type"] == "overlay"
    assert res.actions[0]["intent"] == "highlight"
    assert res.actions[0]["target"] == "apu_switch"
    assert res.actions[0]["element_id"] == "pnt_375"
    assert res.metadata["provider"] == "openai_compat"
    assert res.metadata["model"] == "Qwen3-8B-Instruct"
    assert res.metadata["generation_mode"] == "model"
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
    assert call["json"]["response_format"]["type"] == "json_schema"
    assert call["json"]["response_format"]["json_schema"]["name"] == "HelpResponse"
    assert call["json"]["response_format"]["json_schema"]["strict"] is True
    assert call["headers"]["Authorization"] == "Bearer sk-local"
    assert call["timeout"] == 15.0
    prompt_payload = _extract_prompt_constraints_json(call["json"]["messages"][1]["content"])
    assert "deterministic_step_hint" in prompt_payload
    assert "inferred_step_id" in prompt_payload["deterministic_step_hint"]
    assert prompt_payload["overlay_target_policy"]["mode"] == "single_target_preferred"
    assert prompt_payload["overlay_target_policy"]["max_targets"] == 1
    assert "uncertainty_policy" in prompt_payload
    assert prompt_payload["decision_priority"][:2] == ["deterministic_step_hint", "gates_summary"]


def test_openai_compat_schema_is_loaded_once_per_model_instance(monkeypatch) -> None:
    schema_calls = {"count": 0}

    def _fake_schema():
        schema_calls["count"] += 1
        return {"type": "object"}

    monkeypatch.setattr("adapters.openai_compat_model.get_help_response_schema", _fake_schema)
    fake = FakeClient(
        responses=[
            FakeResponse({"choices": [{"message": {"content": "{}"}}]}),
            FakeResponse({"choices": [{"message": {"content": "{}"}}]}),
        ]
    )
    model = OpenAICompatModel(client=fake)
    assert schema_calls["count"] == 1

    messages = [{"role": "user", "content": "ping"}]
    assert model._chat(messages) == "{}"
    assert model._chat(messages) == "{}"
    assert schema_calls["count"] == 1


def test_explain_error_records_raw_llm_text_when_enabled() -> None:
    help_obj = _help_obj_ok()
    payload = _openai_chat_payload_from_help_obj(help_obj)
    fake = FakeClient(responses=[FakeResponse(payload, status_code=200)])
    model = OpenAICompatModel(client=fake, log_raw_llm_text=True)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert isinstance(res.metadata.get("raw_llm_text"), str)
    assert res.metadata["raw_llm_text"]
    attempts = res.metadata.get("raw_llm_text_attempts")
    assert isinstance(attempts, list) and len(attempts) == 1
    assert attempts[0] == res.metadata["raw_llm_text"]


def test_explain_error_http_429_fallback_no_overlay() -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"
    assert res.metadata["generation_mode"] == "fallback"
    assert res.metadata["failure_code"] == MODEL_HTTP_FAIL
    assert res.metadata["failure_codes"] == [MODEL_HTTP_FAIL]


def test_explain_error_missing_target_evidence_fallback_no_overlay() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["evidence"] = []
    invalid = FakeResponse(_openai_chat_payload_from_help_obj(help_obj), status_code=200)
    fake = FakeClient(responses=[invalid, FakeResponse(_openai_chat_payload_from_help_obj(help_obj), status_code=200)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"
    assert res.metadata["generation_mode"] == "fallback"
    assert res.metadata["failure_code"] == SCHEMA_FAIL


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
    assert res.metadata["failure_code"] == EVIDENCE_FAIL


def test_explain_error_http_5xx_fallback_no_overlay() -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=500)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"
    assert res.metadata["generation_mode"] == "fallback"
    assert res.metadata["failure_code"] == MODEL_HTTP_FAIL


def test_explain_error_timeout_fallback_no_overlay() -> None:
    fake = FakeClient(to_raise=TimeoutError("request timeout"))
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"
    assert res.metadata["generation_mode"] == "fallback"
    assert res.metadata["failure_code"] == MODEL_HTTP_FAIL


def test_explain_error_non_json_output_fallback_no_overlay() -> None:
    payload = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "not-json"},
            }
        ]
    }
    fake = FakeClient(responses=[FakeResponse(payload), FakeResponse(payload)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"
    assert res.metadata["generation_mode"] == "fallback"
    assert res.metadata["failure_code"] == JSON_EXTRACT_FAIL


def test_explain_error_malformed_openai_response_envelope_is_schema_fail() -> None:
    fake = FakeClient(responses=[FakeResponse({"id": "chatcmpl-1"}), FakeResponse({"id": "chatcmpl-2"})])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["failure_code"] == SCHEMA_FAIL
    assert res.metadata["failure_stage"] == "model_response_envelope"


def test_explain_error_retries_once_after_structured_output_failure_and_recovers() -> None:
    invalid_payload = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "not-json"},
            }
        ]
    }
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(invalid_payload), FakeResponse(valid_payload)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert len(res.actions) == 1
    assert res.metadata["generation_mode"] == "model"
    assert len(fake.calls) == 2
    assert res.metadata["retry_count"] == 1
    assert isinstance(res.metadata["retry_reason"], str) and res.metadata["retry_reason"]


def test_explain_error_retries_once_and_returns_error_when_structured_output_still_invalid() -> None:
    invalid_payload = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "not-json"},
            }
        ]
    }
    fake = FakeClient(responses=[FakeResponse(invalid_payload), FakeResponse(invalid_payload)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert res.actions == []
    assert len(fake.calls) == 2
    assert res.metadata["retry_count"] == 1
    assert isinstance(res.metadata["retry_reason"], str) and "ValueError" in res.metadata["retry_reason"]
    assert res.metadata["failure_code"] == JSON_EXTRACT_FAIL


def test_openai_compat_downgrades_request_when_json_schema_format_is_rejected() -> None:
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    rejected = {
        "error": {
            "message": "Unknown field response_format: json_schema is not supported by this server",
        }
    }
    fake = FakeClient(responses=[FakeResponse(rejected, status_code=400), FakeResponse(valid_payload, status_code=200)])
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert len(fake.calls) == 2
    first_payload = fake.calls[0]["json"]
    second_payload = fake.calls[1]["json"]
    assert "response_format" in first_payload
    assert "response_format" not in second_payload


def test_openai_compat_does_not_retry_on_unrelated_400_error() -> None:
    fake = FakeClient(
        responses=[
            FakeResponse(
                {"error": {"message": "Invalid model name Qwen/DoesNotExist"}},
                status_code=400,
            )
        ]
    )
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "error"
    assert len(fake.calls) == 1


def test_openai_compat_downgrades_when_400_text_indicates_response_format_unsupported() -> None:
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(
        responses=[
            FakeResponse(
                payload=None,
                status_code=400,
                text="Bad Request: response_format extra inputs are not permitted",
                json_error=ValueError("not json"),
            ),
            FakeResponse(valid_payload, status_code=200),
        ]
    )
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert len(fake.calls) == 2
    assert "response_format" in fake.calls[0]["json"]
    assert "response_format" not in fake.calls[1]["json"]


def test_openai_compat_downgrades_when_vllm_reports_unimplemented_schema_keys() -> None:
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(
        responses=[
            FakeResponse(
                {
                    "error": {
                        "message": 'Grammar error: Unimplemented keys: ["uniqueItems"]',
                    }
                },
                status_code=400,
            ),
            FakeResponse(valid_payload, status_code=200),
        ]
    )
    model = OpenAICompatModel(client=fake)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert len(fake.calls) == 2
    assert "response_format" in fake.calls[0]["json"]
    assert "response_format" not in fake.calls[1]["json"]


def test_openai_compat_qwen35_disables_thinking_and_caps_output_tokens() -> None:
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(valid_payload, status_code=200)])
    model = OpenAICompatModel(client=fake, model_name="Qwen/Qwen3.5-9B")

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    request_payload = fake.calls[0]["json"]
    assert request_payload["max_tokens"] == 384
    assert request_payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_compat_qwen35_sends_multimodal_images_when_vision_context_is_available(tmp_path: Path) -> None:
    primary_image = tmp_path / "trigger_frame.png"
    primary_image.write_bytes(b"primary-frame")
    pre_trigger_image = tmp_path / "pre_trigger_frame.png"
    pre_trigger_image.write_bytes(b"pre-trigger-frame")
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(valid_payload, status_code=200)])
    model = OpenAICompatModel(
        client=fake,
        model_name="Qwen/Qwen3.5-27B",
        lang="en",
        enable_multimodal=True,
        allowed_local_image_roots=[tmp_path],
    )
    request = _request_help()
    _attach_vision_context(request, primary_image=primary_image, pre_trigger_image=pre_trigger_image)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request)

    assert res.status == "ok"
    assert res.metadata["multimodal_capability_enabled"] is True
    assert res.metadata["multimodal_input_present"] is True
    assert res.metadata["multimodal_candidate_frame_ids"] == ["1772872444950_000122", "1772872445010_000123"]
    assert res.metadata["multimodal_primary_frame_id"] == "1772872444950_000122"
    assert res.metadata["multimodal_frame_ids"] == ["1772872444950_000122", "1772872445010_000123"]
    assert res.metadata["multimodal_images_built"] is True
    assert res.metadata["multimodal_image_count"] == 2
    assert res.metadata["multimodal_path_attempted"] is True
    assert res.metadata["multimodal_path_success"] is True
    assert res.metadata["multimodal_fallback_to_text"] is False
    request_payload = fake.calls[0]["json"]
    assert request_payload["max_tokens"] == 640
    assert request_payload["chat_template_kwargs"] == {"enable_thinking": False}
    content = request_payload["messages"][1]["content"]
    assert isinstance(content, list)
    assert [item["type"] for item in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Primary visual frame: 1772872444950_000122" in content[2]["text"]
    assert "Reference pre-trigger frame: 1772872445010_000123" in content[2]["text"]


def test_openai_compat_localizes_multimodal_frame_notes_for_zh(tmp_path: Path) -> None:
    primary_image = tmp_path / "trigger_frame.png"
    primary_image.write_bytes(b"primary-frame")
    pre_trigger_image = tmp_path / "pre_trigger_frame.png"
    pre_trigger_image.write_bytes(b"pre-trigger-frame")
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(valid_payload, status_code=200)])
    model = OpenAICompatModel(
        client=fake,
        model_name="Qwen/Qwen3.5-27B",
        lang="zh",
        enable_multimodal=True,
        allowed_local_image_roots=[tmp_path],
    )
    request = _request_help()
    _attach_vision_context(request, primary_image=primary_image, pre_trigger_image=pre_trigger_image)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request)

    assert res.status == "ok"
    content = fake.calls[0]["json"]["messages"][1]["content"]
    assert isinstance(content, list)
    assert "主视觉帧: 1772872444950_000122" in content[2]["text"]
    assert "触发前参考帧: 1772872445010_000123" in content[2]["text"]
    assert "Primary visual frame" not in content[2]["text"]


def test_openai_compat_multimodal_failure_falls_back_to_text_only_and_records_metadata(tmp_path: Path) -> None:
    primary_image = tmp_path / "trigger_frame.png"
    primary_image.write_bytes(b"primary-frame")
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(
        responses=[
            FakeResponse({"error": {"message": "Unknown field image_url"}}, status_code=400),
            FakeResponse(valid_payload, status_code=200),
        ]
    )
    model = OpenAICompatModel(
        client=fake,
        model_name="Qwen/Qwen3.5-27B",
        enable_multimodal=True,
        allowed_local_image_roots=[tmp_path],
    )
    request = _request_help()
    _attach_vision_context(request, primary_image=primary_image)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request)

    assert res.status == "ok"
    assert len(fake.calls) == 2
    assert isinstance(fake.calls[0]["json"]["messages"][1]["content"], list)
    assert isinstance(fake.calls[1]["json"]["messages"][1]["content"], str)
    assert res.metadata["multimodal_input_present"] is True
    assert res.metadata["multimodal_candidate_frame_ids"] == ["1772872445010_000123"]
    assert res.metadata["multimodal_image_count"] == 1
    assert res.metadata["multimodal_images_built"] is True
    assert res.metadata["multimodal_path_attempted"] is True
    assert res.metadata["multimodal_path_success"] is False
    assert res.metadata["multimodal_fallback_to_text"] is True
    assert "Unknown field image_url" in res.metadata["multimodal_failure_reason"]


def test_openai_compat_does_not_fallback_to_text_only_for_non_multimodal_transport_errors(tmp_path: Path) -> None:
    primary_image = tmp_path / "trigger_frame.png"
    primary_image.write_bytes(b"primary-frame")
    fake = FakeClient(to_raise=TimeoutError("request timeout"))
    model = OpenAICompatModel(
        client=fake,
        model_name="Qwen/Qwen3.5-27B",
        enable_multimodal=True,
        allowed_local_image_roots=[tmp_path],
    )
    request = _request_help()
    _attach_vision_context(request, primary_image=primary_image)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request)

    assert res.status == "error"
    assert len(fake.calls) == 1
    assert res.metadata["multimodal_input_present"] is True
    assert res.metadata["multimodal_images_built"] is True
    assert res.metadata["multimodal_path_attempted"] is True
    assert res.metadata["multimodal_path_success"] is False
    assert res.metadata["multimodal_fallback_to_text"] is False
    assert "TimeoutError: request timeout" == res.metadata["multimodal_failure_reason"]
    assert res.metadata["failure_code"] == MODEL_HTTP_FAIL


def test_openai_compat_keeps_multimodal_input_present_when_image_build_fails() -> None:
    request = _request_help()
    request.context["vision"] = {
        "status": "partial",
        "vision_used": True,
        "frame_id": "1772872445010_000123",
        "trigger_frame": {
            "frame_id": "1772872445010_000123",
            "role": "trigger_frame",
        },
    }
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(valid_payload, status_code=200)])
    model = OpenAICompatModel(client=fake, model_name="Qwen/Qwen3.5-27B", enable_multimodal=True)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request)

    assert res.status == "ok"
    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0]["json"]["messages"][1]["content"], str)
    assert res.metadata["multimodal_input_present"] is True
    assert res.metadata["multimodal_candidate_frame_ids"] == ["1772872445010_000123"]
    assert res.metadata["multimodal_images_built"] is False
    assert res.metadata["multimodal_frame_ids"] == []
    assert res.metadata["multimodal_image_count"] == 0
    assert res.metadata["multimodal_path_attempted"] is False
    assert res.metadata["multimodal_fallback_to_text"] is False
    assert "image_uri/source_image_path" in res.metadata["multimodal_failure_reason"]


def test_openai_compat_keeps_text_only_when_multimodal_capability_is_disabled(tmp_path: Path) -> None:
    primary_image = tmp_path / "trigger_frame.png"
    primary_image.write_bytes(b"primary-frame")
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(valid_payload, status_code=200)])
    model = OpenAICompatModel(client=fake, model_name="TextOnly-Model")
    request = _request_help()
    _attach_vision_context(request, primary_image=primary_image)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request)

    assert res.status == "ok"
    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0]["json"]["messages"][1]["content"], str)
    assert res.metadata["multimodal_capability_enabled"] is False
    assert res.metadata["multimodal_input_present"] is True
    assert res.metadata["multimodal_candidate_frame_ids"] == ["1772872445010_000123"]
    assert res.metadata["multimodal_images_built"] is False
    assert res.metadata["multimodal_path_attempted"] is False
    assert res.metadata["multimodal_failure_reason"] is None


def test_openai_compat_retry_after_multimodal_rejection_stays_text_only(tmp_path: Path) -> None:
    primary_image = tmp_path / "trigger_frame.png"
    primary_image.write_bytes(b"primary-frame")
    invalid_payload = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "not-json"},
            }
        ]
    }
    fake = FakeClient(
        responses=[
            FakeResponse({"error": {"message": "Unknown field image_url"}}, status_code=400),
            FakeResponse(invalid_payload, status_code=200),
            FakeResponse(_openai_chat_payload_from_help_obj(_help_obj_ok()), status_code=200),
        ]
    )
    model = OpenAICompatModel(
        client=fake,
        model_name="Qwen/Qwen3.5-27B",
        enable_multimodal=True,
        allowed_local_image_roots=[tmp_path],
    )
    request = _request_help()
    _attach_vision_context(request, primary_image=primary_image)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request)

    assert res.status == "ok"
    assert len(fake.calls) == 3
    assert isinstance(fake.calls[0]["json"]["messages"][1]["content"], list)
    assert isinstance(fake.calls[1]["json"]["messages"][1]["content"], str)
    assert isinstance(fake.calls[2]["json"]["messages"][1]["content"], str)
    assert res.metadata["retry_count"] == 1


def test_openai_compat_rejects_local_image_path_outside_allowed_roots(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_image = outside_dir / "trigger_frame.png"
    outside_image.write_bytes(b"primary-frame")
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(valid_payload, status_code=200)])
    model = OpenAICompatModel(
        client=fake,
        model_name="Qwen/Qwen3.5-27B",
        enable_multimodal=True,
        allowed_local_image_roots=[tmp_path / "allowed"],
    )
    request = _request_help()
    _attach_vision_context(request, primary_image=outside_image)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request)

    assert res.status == "ok"
    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0]["json"]["messages"][1]["content"], str)
    assert res.metadata["multimodal_images_built"] is False
    assert "outside allowed roots" in res.metadata["multimodal_failure_reason"]


def test_openai_compat_rejects_local_image_path_exceeding_size_limit(tmp_path: Path) -> None:
    primary_image = tmp_path / "trigger_frame.png"
    primary_image.write_bytes(b"x" * 32)
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(valid_payload, status_code=200)])
    model = OpenAICompatModel(
        client=fake,
        model_name="Qwen/Qwen3.5-27B",
        enable_multimodal=True,
        allowed_local_image_roots=[tmp_path],
        max_local_image_bytes=8,
    )
    request = _request_help()
    _attach_vision_context(request, primary_image=primary_image)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), request)

    assert res.status == "ok"
    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0]["json"]["messages"][1]["content"], str)
    assert res.metadata["multimodal_images_built"] is False
    assert "exceeds max size" in res.metadata["multimodal_failure_reason"]


def test_openai_compat_text_only_path_is_unchanged_without_vision_context() -> None:
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(responses=[FakeResponse(valid_payload, status_code=200)])
    model = OpenAICompatModel(client=fake, model_name="Qwen/Qwen3.5-27B")

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert res.metadata["multimodal_input_present"] is False
    assert res.metadata["multimodal_candidate_frame_ids"] == []
    assert res.metadata["multimodal_primary_frame_id"] is None
    assert res.metadata["multimodal_frame_ids"] == []
    assert res.metadata["multimodal_images_built"] is False
    assert res.metadata["multimodal_image_count"] == 0
    assert res.metadata["multimodal_path_attempted"] is False
    assert res.metadata["multimodal_path_success"] is False
    request_payload = fake.calls[0]["json"]
    assert request_payload["max_tokens"] == 384
    assert isinstance(request_payload["messages"][1]["content"], str)


def test_openai_compat_respects_explicit_max_tokens_for_qwen35_fallback_retry() -> None:
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(
        responses=[
            FakeResponse(
                {
                    "error": {
                        "message": 'Grammar error: Unimplemented keys: ["uniqueItems"]',
                    }
                },
                status_code=400,
            ),
            FakeResponse(valid_payload, status_code=200),
        ]
    )
    model = OpenAICompatModel(client=fake, model_name="Qwen/Qwen3.5-9B", max_tokens=128)

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["json"]["max_tokens"] == 128
        assert call["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" in fake.calls[0]["json"]
    assert "response_format" not in fake.calls[1]["json"]


def test_openai_compat_retries_without_chat_template_kwargs_when_server_rejects_it() -> None:
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(
        responses=[
            FakeResponse(
                {
                    "error": {
                        "message": "Unknown field: chat_template_kwargs",
                    }
                },
                status_code=400,
            ),
            FakeResponse(valid_payload, status_code=200),
        ]
    )
    model = OpenAICompatModel(client=fake, model_name="Qwen/Qwen3.5-9B")

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert len(fake.calls) == 2
    assert fake.calls[0]["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in fake.calls[1]["json"]
    assert "response_format" in fake.calls[1]["json"]
    assert fake.calls[1]["json"]["max_tokens"] == 384


def test_openai_compat_retries_without_chat_template_kwargs_then_without_json_schema() -> None:
    valid_payload = _openai_chat_payload_from_help_obj(_help_obj_ok())
    fake = FakeClient(
        responses=[
            FakeResponse(
                {
                    "error": {
                        "message": "Unknown field: chat_template_kwargs",
                    }
                },
                status_code=400,
            ),
            FakeResponse(
                {
                    "error": {
                        "message": 'Grammar error: Unimplemented keys: ["uniqueItems"]',
                    }
                },
                status_code=400,
            ),
            FakeResponse(valid_payload, status_code=200),
        ]
    )
    model = OpenAICompatModel(client=fake, model_name="Qwen/Qwen3.5-9B")

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), _request_help())

    assert res.status == "ok"
    assert len(fake.calls) == 3
    assert fake.calls[0]["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in fake.calls[1]["json"]
    assert "response_format" in fake.calls[1]["json"]
    assert "chat_template_kwargs" not in fake.calls[2]["json"]
    assert "response_format" not in fake.calls[2]["json"]


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


def test_explain_error_en_fallback_with_inferred_step_and_missing_conditions() -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake, lang="en")
    req = _request_help()
    req.context["candidate_steps"] = ["S03", "S04"]
    req.context["vars"] = {"power_available": True, "apu_on": True, "apu_ready": False}
    req.context["recent_ui_targets"] = ["apu_switch"]

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "error"
    assert res.message == "You are likely stuck at S03. Please satisfy: vars.apu_ready==true."


def test_explain_error_en_fallback_with_inferred_step_without_missing_conditions() -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake, lang="en")
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
    assert res.message == "You are likely stuck at S07. Please re-check and execute that step."


def test_explain_error_en_fallback_without_inferred_step(monkeypatch) -> None:
    fake = FakeClient(responses=[FakeResponse({}, status_code=429)])
    model = OpenAICompatModel(client=fake, lang="en")
    monkeypatch.setattr("adapters.base_help_model.load_pack_steps", lambda: [])

    res = model.explain_error(Observation(source="mock"), _request_help())

    assert res.status == "error"
    assert res.message == "Unable to generate help response, please check the current system status and try again."


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
    assert res.metadata["generation_mode"] == "repair"
    assert "dropped_prefix_text" in res.metadata["json_repair_reasons"]
    assert "dropped_suffix_text" in res.metadata["json_repair_reasons"]


def test_explain_error_marks_unobservable_step_with_visual_confirmation_metadata() -> None:
    help_obj = _help_obj_ok()
    help_obj["confidence"] = 0.97
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(help_obj), status_code=200)])
    model = OpenAICompatModel(client=fake)
    req = _request_help()
    req.context["deterministic_step_hint"] = {
        "inferred_step_id": "S20",
        "observability_status": "unobservable",
        "requires_visual_confirmation": True,
    }

    res = model.explain_error(Observation(source="mock", procedure_hint="S20"), req)

    assert res.status == "ok"
    assert res.metadata["observability_status"] == "unobservable"
    assert res.metadata["requires_visual_confirmation"] is True
    assert res.metadata["effective_confidence"] < res.metadata["model_confidence"]
    assert res.metadata["confidence_adjustment_reason"] == "observability:unobservable"
    assert res.metadata["evidence_strength"] == "limited"
    assert any("待视觉确认" in item for item in res.explanations)


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
    assert res.metadata["failure_code"] == SCHEMA_FAIL


def test_explain_error_overlay_target_not_in_allowlist_fallback_no_overlay() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["targets"] = ["battery_switch"]
    help_obj["overlay"]["evidence"] = [
        {
            "target": "battery_switch",
            "type": "var",
            "ref": "VARS.battery_on",
            "quote": "Battery state is known.",
            "grounding_confidence": 0.9,
        }
    ]
    fake = FakeClient(responses=[FakeResponse(_openai_chat_payload_from_help_obj(help_obj), status_code=200)])
    model = OpenAICompatModel(client=fake)
    req = _request_help()
    req.context["overlay_target_allowlist"] = ["apu_switch"]

    res = model.explain_error(Observation(source="mock", procedure_hint="S03"), req)

    assert res.status == "error"
    assert res.actions == []
    assert res.metadata["provider"] == "openai_compat"
    assert res.metadata["failure_code"] == ALLOWLIST_FAIL


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
