from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from adapters.vision_fact_extractor import VisionFactExtractor
from core.vision_facts import VISION_FACT_IDS
from tests._fakes import FakeClient, FakeResponse


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (16, 32, 48)).save(path, format="PNG")


def _vision_context(primary: Path, secondary: Path | None = None) -> dict[str, object]:
    trigger_frame = {
        "frame_id": "1772872445010_000123",
        "role": "trigger_frame",
        "image_uri": str(primary),
        "mime_type": "image/png",
    }
    frames = [trigger_frame]
    frame_ids = [trigger_frame["frame_id"]]
    payload: dict[str, object] = {
        "status": "available",
        "frame_id": trigger_frame["frame_id"],
        "frame_ids": frame_ids,
        "trigger_frame": trigger_frame,
        "selected_frames": frames,
    }
    if secondary is not None:
        pre = {
            "frame_id": "1772872444950_000122",
            "role": "pre_trigger_frame",
            "image_uri": str(secondary),
            "mime_type": "image/png",
        }
        payload["pre_trigger_frame"] = pre
        payload["frame_id"] = pre["frame_id"]
        payload["frame_ids"] = [pre["frame_id"], trigger_frame["frame_id"]]
        payload["selected_frames"] = [pre, trigger_frame]
    return payload


def _chat_payload(facts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"facts": facts}, ensure_ascii=False),
                }
            }
        ]
    }


def test_vision_fact_extractor_builds_two_image_request_and_parses_positive_facts(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    secondary = tmp_path / "1772872444950_000122.png"
    _write_png(primary)
    _write_png(secondary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": "fcsmc_in_test_visible",
                            "state": "seen",
                            "evidence_note": "Right DDI FCS-MC page clearly shows IN TEST.",
                        },
                        {
                            "fact_id": "fcsmc_final_go_result_visible",
                            "state": "seen",
                            "evidence_note": "Right DDI FCS-MC page clearly shows MC1 GO, MC2 GO, FCSA GO, FCSB GO.",
                        },
                    ]
                )
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
        lang="en",
    )

    result = extractor.extract(
        _vision_context(primary, secondary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.status == "uncertain"
    assert result.observation is not None
    facts_by_id = {fact.fact_id: fact for fact in result.observation.facts}
    assert facts_by_id["fcsmc_in_test_visible"].state == "seen"
    assert facts_by_id["fcsmc_final_go_result_visible"].state == "seen"
    assert facts_by_id["tac_page_visible"].state == "uncertain"
    assert facts_by_id["fcsmc_in_test_visible"].source_frame_id == "1772872444950_000122"
    call = fake.calls[0]
    content = call["json"]["messages"][1]["content"]
    assert len(content) == 3
    assert [item["type"] for item in content[:2]] == ["image_url", "image_url"]
    assert call["json"]["response_format"]["json_schema"]["name"] == "VisionFactResponse"
    response_schema = call["json"]["response_format"]["json_schema"]["schema"]
    assert "source_frame_id" not in response_schema["properties"]["facts"]["items"]["required"]
    assert "source_frame_id" in response_schema["properties"]["facts"]["items"]["properties"]


def test_vision_fact_extractor_dashscope_qwen35_uses_json_object_and_omits_max_tokens(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    fake = FakeClient(responses=[FakeResponse(_chat_payload([]), status_code=200)])
    extractor = VisionFactExtractor(
        client=fake,
        model_name="qwen3.5-27b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode",
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.status == "uncertain"
    request_payload = fake.calls[0]["json"]
    assert request_payload["response_format"] == {"type": "json_object"}
    assert "max_tokens" not in request_payload
    assert request_payload["enable_thinking"] is False
    assert "chat_template_kwargs" not in request_payload


def test_vision_fact_extractor_records_raw_json_and_prints_model_io(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    raw_content = json.dumps(
        {
            "facts": [
                {
                    "fact_id": "supt_page_visible",
                    "state": "seen",
                    "evidence_note": "Left DDI clearly shows the SUPT page.",
                }
            ]
        },
        ensure_ascii=False,
    )
    fake = FakeClient(responses=[FakeResponse({"choices": [{"message": {"content": raw_content}}]})])
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
        log_raw_llm_text=True,
        print_model_io=True,
    )

    result = extractor.extract(
        {
            **_vision_context(primary),
            "request_id": "req-123",
            "help_cycle_id": "req-123",
        },
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    out = capsys.readouterr().out
    assert "[MODEL_IO][VISION_FACT_PROMPT]" in out
    assert "[MODEL_IO][VISION_FACT_REPLY]" in out
    assert "[session_id=sess-live]" in out
    assert "[request_id=req-123]" in out
    assert "[help_cycle_id=req-123]" in out
    assert "[frame_ids=1772872445010_000123]" in out
    assert raw_content in out
    assert result.metadata["raw_llm_text"] == raw_content
    assert result.observation is not None
    assert result.observation.metadata["raw_llm_text"] == raw_content


def test_vision_fact_extractor_returns_uncertain_when_model_is_unsure(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": "fcsmc_intermediate_result_visible",
                            "state": "uncertain",
                            "evidence_note": "FCS-MC result text is too blurry to classify.",
                        }
                    ]
                )
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.status == "uncertain"
    assert result.observation is not None
    facts_by_id = {fact.fact_id: fact for fact in result.observation.facts}
    assert facts_by_id["fcsmc_intermediate_result_visible"].state == "uncertain"


def test_vision_fact_extractor_defaults_empty_fact_array_to_all_uncertain(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload([])
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.status == "uncertain"
    assert result.observation is not None
    assert all(fact.state == "uncertain" for fact in result.observation.facts)
    assert result.metadata["vision_fact_summary"]["seen_fact_ids"] == []
    assert set(result.metadata["vision_fact_summary"]["uncertain_fact_ids"]) == {
        fact.fact_id for fact in result.observation.facts
    }


def test_vision_fact_extractor_attaches_default_source_frame_id_when_model_omits_it(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": "supt_page_visible",
                            "state": "seen",
                            "evidence_note": "SUPT page label is clearly visible.",
                        }
                    ]
                )
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.observation is not None
    facts_by_id = {fact.fact_id: fact for fact in result.observation.facts}
    assert facts_by_id["supt_page_visible"].source_frame_id == "1772872445010_000123"
    assert result.observation.metadata["coerced_source_frame_fact_ids"] == []


def test_vision_fact_extractor_accepts_legacy_source_frame_id_from_model(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    secondary = tmp_path / "1772872444950_000122.png"
    _write_png(primary)
    _write_png(secondary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": "fcsmc_page_visible",
                            "state": "seen",
                            "source_frame_id": "1772872445010_000123",
                            "evidence_note": "Right DDI clearly shows the FCS-MC page.",
                        }
                    ]
                )
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary, secondary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.observation is not None
    facts_by_id = {fact.fact_id: fact for fact in result.observation.facts}
    assert facts_by_id["fcsmc_page_visible"].source_frame_id == "1772872445010_000123"
    assert result.observation.metadata["coerced_source_frame_fact_ids"] == []


def test_vision_fact_extractor_coerces_invalid_legacy_source_frame_id_to_default_frame(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    secondary = tmp_path / "1772872444950_000122.png"
    _write_png(primary)
    _write_png(secondary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": "fcsmc_page_visible",
                            "state": "seen",
                            "source_frame_id": "hallucinated_frame",
                            "evidence_note": "Right DDI clearly shows the FCS-MC page.",
                        }
                    ]
                )
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary, secondary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.observation is not None
    facts_by_id = {fact.fact_id: fact for fact in result.observation.facts}
    assert facts_by_id["fcsmc_page_visible"].source_frame_id == "1772872444950_000122"
    assert result.observation.metadata["coerced_source_frame_fact_ids"] == ["fcsmc_page_visible"]


def test_vision_fact_extractor_preserves_model_summary_in_metadata(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": "Model summary from the VLM.",
                                        "facts": [
                                            {
                                                "fact_id": "supt_page_visible",
                                                "state": "seen",
                                                "evidence_note": "Left DDI clearly shows the SUPT page.",
                                            }
                                        ],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.observation is not None
    assert result.observation.summary == result.metadata["vision_fact_summary"]["summary_text"]
    assert result.observation.metadata["model_summary"] == "Model summary from the VLM."
    assert result.metadata["model_summary"] == "Model summary from the VLM."


def test_vision_fact_extractor_downgrades_when_multimodal_rejected(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                {"error": {"message": "Unknown field image_url"}},
                status_code=400,
                text='{"error":{"message":"Unknown field image_url"}}',
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.status == "extractor_failed"
    assert result.observation is None
    assert "Unknown field image_url" in str(result.error)


def test_vision_fact_extractor_uses_only_successful_frame_ids_in_prompt_and_metadata(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    missing = tmp_path / "1772872444950_000122.png"
    _write_png(primary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": "fcs_page_visible",
                            "state": "seen",
                            "evidence_note": "FCS page is visible on the surviving frame.",
                        }
                    ]
                )
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary, missing),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.observation is not None
    assert result.observation.frame_ids == ["1772872445010_000123"]
    assert result.metadata["frame_ids"] == ["1772872445010_000123"]
    assert result.metadata["multimodal_failed_frame_ids"] == ["1772872444950_000122"]
    assert "1772872444950_000122" in result.metadata["multimodal_frame_failures"]
    call = fake.calls[0]
    content = call["json"]["messages"][1]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]
    assert '"frame_ids":["1772872445010_000123"]' in content[1]["text"]
    assert "1772872444950_000122" not in content[1]["text"]


def test_vision_fact_extractor_negative_fcs_bit_sample_stays_not_seen(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": fact_id,
                            "state": "not_seen",
                            "evidence_note": f"{fact_id} is confidently not visible in this frame.",
                        }
                        for fact_id in VISION_FACT_IDS
                    ]
                )
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.status == "available"
    assert result.observation is not None
    assert result.metadata["vision_fact_summary"]["status"] == "available"
    facts_by_id = {fact.fact_id: fact for fact in result.observation.facts}
    assert facts_by_id["fcsmc_in_test_visible"].state == "not_seen"
    assert facts_by_id["fcsmc_final_go_result_visible"].state == "not_seen"


def test_vision_fact_extractor_uses_configured_fact_subset_without_keyerror(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    config_path = tmp_path / "vision_facts_subset.yaml"
    _write_png(primary)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "layout_id": "subset_layout",
                "facts": [
                    {
                        "fact_id": "fcs_page_visible",
                        "sticky": False,
                        "expires_after_ms": 2000,
                        "intended_regions": ["left_ddi"],
                        "steps": ["S08"],
                    }
                ],
                "step_bindings": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": "fcs_page_visible",
                            "state": "seen",
                            "evidence_note": "FCS page is visible.",
                        }
                    ]
                )
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
        config_path=str(config_path),
    )

    result = extractor.extract(
        _vision_context(primary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.observation is not None
    assert [fact.fact_id for fact in result.observation.facts] == ["fcs_page_visible"]
    assert result.observation.facts[0].state == "seen"
    call = fake.calls[0]
    response_schema = call["json"]["response_format"]["json_schema"]["schema"]
    assert response_schema["properties"]["facts"]["items"]["properties"]["fact_id"]["enum"] == ["fcs_page_visible"]


def test_vision_fact_extractor_schema_includes_supt_page_visible(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": "supt_page_visible",
                            "state": "seen",
                            "evidence_note": "Left DDI clearly shows the SUPT page.",
                        }
                    ]
                )
            )
        ]
    )
    extractor = VisionFactExtractor(
        client=fake,
        allowed_local_image_roots=[str(tmp_path)],
    )

    result = extractor.extract(
        _vision_context(primary),
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
    )

    assert result.observation is not None
    facts_by_id = {fact.fact_id: fact for fact in result.observation.facts}
    assert facts_by_id["supt_page_visible"].state == "seen"
    call = fake.calls[0]
    response_schema = call["json"]["response_format"]["json_schema"]["schema"]
    assert "supt_page_visible" in response_schema["properties"]["facts"]["items"]["properties"]["fact_id"]["enum"]
