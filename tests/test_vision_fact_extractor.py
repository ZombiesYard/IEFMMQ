from __future__ import annotations

import json
from pathlib import Path

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
                            "fact_id": "fcs_bit_interaction_seen",
                            "state": "seen",
                            "source_frame_id": "1772872445010_000123",
                            "confidence": 0.91,
                            "evidence_note": "Right DDI BIT/FCS page shows active FCS BIT.",
                        },
                        {
                            "fact_id": "fcs_bit_result_visible",
                            "state": "seen",
                            "source_frame_id": "1772872445010_000123",
                            "confidence": 0.88,
                            "evidence_note": "BIT result text is visible on the right DDI.",
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
    assert facts_by_id["fcs_bit_interaction_seen"].state == "seen"
    assert facts_by_id["fcs_bit_result_visible"].state == "seen"
    assert facts_by_id["fcs_reset_seen"].state == "uncertain"
    call = fake.calls[0]
    content = call["json"]["messages"][1]["content"]
    assert len(content) == 3
    assert [item["type"] for item in content[:2]] == ["image_url", "image_url"]
    assert call["json"]["response_format"]["json_schema"]["name"] == "VisionFactResponse"


def test_vision_fact_extractor_returns_uncertain_when_model_is_unsure(tmp_path: Path) -> None:
    primary = tmp_path / "1772872445010_000123.png"
    _write_png(primary)
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    [
                        {
                            "fact_id": "fcs_reset_seen",
                            "state": "uncertain",
                            "source_frame_id": "1772872445010_000123",
                            "confidence": 0.22,
                            "evidence_note": "FCS page is too blurry to confirm reset marks.",
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
    assert facts_by_id["fcs_reset_seen"].state == "uncertain"


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
                            "source_frame_id": "1772872445010_000123",
                            "confidence": 0.84,
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
    assert facts_by_id["fcs_bit_interaction_seen"].state == "not_seen"
    assert facts_by_id["fcs_bit_result_visible"].state == "not_seen"
