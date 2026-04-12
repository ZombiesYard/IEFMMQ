from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tests._fakes import FakeClient, FakeResponse
from tools import generate_vlm_prelabels_en, generate_vlm_prelabels_zh
from tools.generate_vlm_prelabels import CORE_FACT_IDS, build_arg_parser, generate_prelabels


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (20, 30, 40)).save(path, format="PNG")


def _write_capture_index(
    session_dir: Path,
    *,
    frame_id: str,
    raw_image_path: Path,
    artifact_image_path: Path,
    session_id: str = "sess-live",
) -> None:
    path = session_dir / "capture_index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "frame_id": frame_id,
        "capture_wall_ms": 1772872445010,
        "raw_image_path": str(raw_image_path.resolve()),
        "artifact_image_path": str(artifact_image_path.resolve()),
        "capture_reason": "help_start",
        "session_id": session_id,
        "channel": "composite_panel",
        "layout_id": "fa18c_composite_panel_v2",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _chat_payload(content_obj: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content_obj, ensure_ascii=False),
                }
            }
        ]
    }


def _chat_payload_with_usage(
    content_obj: dict[str, object],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> dict[str, object]:
    payload = _chat_payload(content_obj)
    payload["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_details": {
            "text_tokens": max(0, prompt_tokens - 10),
            "image_tokens": 10,
        },
        "completion_tokens_details": {
            "text_tokens": completion_tokens,
        },
    }
    return payload


def test_generate_vlm_prelabels_builds_dashscope_json_object_request_and_prompt_constraints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_dir = tmp_path / "captures" / "sess-live"
    artifact_path = session_dir / "artifacts" / "1772872445010_000123_vlm.png"
    raw_path = session_dir / "raw" / "1772872445010_000123.png"
    _write_png(artifact_path)
    _write_png(raw_path)
    _write_capture_index(
        session_dir,
        frame_id="1772872445010_000123",
        raw_image_path=raw_path,
        artifact_image_path=artifact_path,
    )
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    {
                        "summary": "Left DDI shows FCS page.",
                        "facts": [
                            {
                                "fact_id": "fcs_page_visible",
                                "state": "seen",
                                "evidence_note": "Left DDI clearly shows the FCS page.",
                            }
                        ],
                    }
                )
            )
        ]
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    stats = generate_prelabels(
        session_dir=session_dir,
        client=fake,
    )

    assert stats.succeeded == 1
    request_payload = fake.calls[0]["json"]
    assert request_payload["response_format"] == {"type": "json_object"}
    assert request_payload["enable_thinking"] is False
    assert "max_tokens" not in request_payload
    content = request_payload["messages"][1]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "严禁输出 frame_id" in content[1]["text"]
    assert "严禁输出 source_frame_id" in content[1]["text"]
    assert "严禁输出 confidence" in content[1]["text"]
    assert "left_ddi、ampcd、right_ddi" in content[1]["text"]


def test_generate_vlm_prelabels_normalizes_facts_and_exports_label_studio_tasks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_dir = tmp_path / "captures" / "sess-live"
    artifact_path = session_dir / "artifacts" / "1772872445010_000123_vlm.png"
    raw_path = session_dir / "raw" / "1772872445010_000123.png"
    _write_png(artifact_path)
    _write_png(raw_path)
    _write_capture_index(
        session_dir,
        frame_id="1772872445010_000123",
        raw_image_path=raw_path,
        artifact_image_path=artifact_path,
    )
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    {
                        "summary": "FCS page is visible; INS GO is not visible.",
                        "facts": [
                            {
                                "fact_id": "fcs_page_visible",
                                "state": "seen",
                                "evidence_note": "The left DDI shows the FCS page.",
                            },
                            {
                                "fact_id": "ins_go",
                                "state": "not_seen",
                                "evidence_note": "AMPCD does not show a GO cue.",
                            },
                        ],
                    }
                )
            )
        ]
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    output_dir = session_dir / "prelabels"
    stats = generate_prelabels(
        session_dir=session_dir,
        output_dir=output_dir,
        client=fake,
    )

    assert stats.succeeded == 1
    rows = _read_jsonl(output_dir / "vision_prelabels.jsonl")
    assert len(rows) == 1
    sample = rows[0]
    assert sample["frame_id"] == "1772872445010_000123"
    assert sample["layout_id"] == "fa18c_composite_panel_v2"
    facts_by_id = {item["fact_id"]: item for item in sample["facts"]}
    assert list(facts_by_id) == list(CORE_FACT_IDS)
    assert facts_by_id["fcs_page_visible"]["state"] == "seen"
    assert facts_by_id["ins_go"]["state"] == "not_seen"
    assert facts_by_id["bit_root_page_visible"]["state"] == "uncertain"

    tasks = json.loads((output_dir / "label_studio_tasks.json").read_text(encoding="utf-8"))
    assert len(tasks) == 1
    task = tasks[0]
    assert task["data"]["image"].startswith("data:image/png;base64,")
    assert task["data"]["frame_id"] == "1772872445010_000123"
    assert task["data"]["fcs_page_visible"] == "seen"
    assert task["data"]["bit_root_page_visible"] == "uncertain"
    assert task["data"]["fcs_page_visible_note"] == "The left DDI shows the FCS page."
    assert "source_frame_id" not in task["data"]["ai_prelabel_json"]
    assert "confidence" not in task["data"]["ai_prelabel_json"]


def test_generate_vlm_prelabels_records_token_usage_in_outputs_and_stats(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_dir = tmp_path / "captures" / "sess-live"
    artifact_path = session_dir / "artifacts" / "1772872445010_000123_vlm.png"
    raw_path = session_dir / "raw" / "1772872445010_000123.png"
    _write_png(artifact_path)
    _write_png(raw_path)
    _write_capture_index(
        session_dir,
        frame_id="1772872445010_000123",
        raw_image_path=raw_path,
        artifact_image_path=artifact_path,
    )
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload_with_usage(
                    {
                        "summary": "FCS page visible.",
                        "facts": [
                            {
                                "fact_id": "fcs_page_visible",
                                "state": "seen",
                                "evidence_note": "FCS page visible on left DDI.",
                            }
                        ],
                    },
                    prompt_tokens=321,
                    completion_tokens=87,
                    total_tokens=408,
                )
            )
        ]
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    output_dir = session_dir / "prelabels"
    stats = generate_prelabels(
        session_dir=session_dir,
        output_dir=output_dir,
        client=fake,
    )

    assert stats.prompt_tokens == 321
    assert stats.completion_tokens == 87
    assert stats.total_tokens == 408

    raw_rows = _read_jsonl(output_dir / "raw_model_outputs.jsonl")
    assert len(raw_rows) == 1
    assert raw_rows[0]["usage"]["prompt_tokens"] == 321
    assert raw_rows[0]["usage"]["completion_tokens"] == 87
    assert raw_rows[0]["usage"]["total_tokens"] == 408
    assert raw_rows[0]["usage"]["prompt_tokens_details"]["image_tokens"] == 10


def test_generate_vlm_prelabels_strips_forbidden_model_fields_and_records_warnings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_dir = tmp_path / "captures" / "sess-live"
    artifact_path = session_dir / "artifacts" / "1772872445010_000123_vlm.png"
    raw_path = session_dir / "raw" / "1772872445010_000123.png"
    _write_png(artifact_path)
    _write_png(raw_path)
    _write_capture_index(
        session_dir,
        frame_id="1772872445010_000123",
        raw_image_path=raw_path,
        artifact_image_path=artifact_path,
    )
    fake = FakeClient(
        responses=[
            FakeResponse(
                _chat_payload(
                    {
                        "frame_id": "hallucinated",
                        "summary": "FCS page visible.",
                        "facts": [
                            {
                                "fact_id": "fcs_page_visible",
                                "state": "seen",
                                "evidence_note": "FCS page visible on left DDI.",
                                "source_frame_id": "hallucinated",
                                "confidence": 0.99,
                            }
                        ],
                    }
                )
            )
        ]
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    output_dir = session_dir / "prelabels"
    generate_prelabels(
        session_dir=session_dir,
        output_dir=output_dir,
        client=fake,
        save_raw_response=True,
    )

    raw_rows = _read_jsonl(output_dir / "raw_model_outputs.jsonl")
    assert len(raw_rows) == 1
    warnings = raw_rows[0]["warnings"]
    assert "stripped_top_level_fields:frame_id" in warnings
    assert "stripped_fact_fields:fcs_page_visible:confidence,source_frame_id" in warnings
    assert '"frame_id": "hallucinated"' in raw_rows[0]["response_text"]

    sample = _read_jsonl(output_dir / "vision_prelabels.jsonl")[0]
    fact = next(item for item in sample["facts"] if item["fact_id"] == "fcs_page_visible")
    assert set(fact) == {"fact_id", "state", "evidence_note"}


def test_generate_vlm_prelabels_records_failures_without_stopping_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session_dir = tmp_path / "captures" / "sess-live"
    artifact_1 = session_dir / "artifacts" / "1772872445010_000123_vlm.png"
    raw_1 = session_dir / "raw" / "1772872445010_000123.png"
    artifact_2 = session_dir / "artifacts" / "1772872446010_000124_vlm.png"
    raw_2 = session_dir / "raw" / "1772872446010_000124.png"
    _write_png(artifact_1)
    _write_png(raw_1)
    _write_png(artifact_2)
    _write_png(raw_2)
    _write_capture_index(
        session_dir,
        frame_id="1772872445010_000123",
        raw_image_path=raw_1,
        artifact_image_path=artifact_1,
    )
    _write_capture_index(
        session_dir,
        frame_id="1772872446010_000124",
        raw_image_path=raw_2,
        artifact_image_path=artifact_2,
    )
    fake = FakeClient(
        responses=[
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"summary":"bad","facts":[{"fact_id":"fcs_page_visible","state":"maybe","evidence_note":"bad"}]}'
                            }
                        }
                    ]
                }
            ),
            FakeResponse(
                _chat_payload(
                    {
                        "summary": "BIT root visible.",
                        "facts": [
                            {
                                "fact_id": "bit_root_page_visible",
                                "state": "seen",
                                "evidence_note": "Right DDI shows BIT root page.",
                            }
                        ],
                    }
                )
            ),
        ]
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")

    output_dir = session_dir / "prelabels"
    stats = generate_prelabels(
        session_dir=session_dir,
        output_dir=output_dir,
        client=fake,
    )

    assert stats.total_images == 2
    assert stats.succeeded == 1
    assert stats.failed == 1
    success_rows = _read_jsonl(output_dir / "vision_prelabels.jsonl")
    failure_rows = _read_jsonl(output_dir / "prelabels_failures.jsonl")
    assert len(success_rows) == 1
    assert len(failure_rows) == 1
    assert failure_rows[0]["frame_id"] == "1772872445010_000123"
    assert "unsupported state" in failure_rows[0]["error"]
    assert success_rows[0]["frame_id"] == "1772872446010_000124"


def test_generate_vlm_prelabels_parser_supports_switchable_default_lang() -> None:
    zh_args = build_arg_parser(default_lang="zh").parse_args(["--session-dir", "/tmp/session"])
    en_args = build_arg_parser(default_lang="en").parse_args(["--session-dir", "/tmp/session"])

    assert zh_args.lang == "zh"
    assert en_args.lang == "en"


def test_generate_vlm_prelabels_en_wrapper_uses_english_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(*, default_lang: str, argv=None) -> int:
        captured["default_lang"] = default_lang
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(generate_vlm_prelabels_en, "run_cli", fake_run_cli)

    assert generate_vlm_prelabels_en.main(["--session-dir", "/tmp/session"]) == 0
    assert captured["default_lang"] == "en"
    assert captured["argv"] == ["--session-dir", "/tmp/session"]


def test_generate_vlm_prelabels_zh_wrapper_uses_chinese_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_cli(*, default_lang: str, argv=None) -> int:
        captured["default_lang"] = default_lang
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(generate_vlm_prelabels_zh, "run_cli", fake_run_cli)

    assert generate_vlm_prelabels_zh.main(["--session-dir", "/tmp/session"]) == 0
    assert captured["default_lang"] == "zh"
    assert captured["argv"] == ["--session-dir", "/tmp/session"]
