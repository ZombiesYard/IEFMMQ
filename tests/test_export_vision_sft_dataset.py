from __future__ import annotations

import json
from pathlib import Path

from tools.export_vision_sft_dataset import CORE_FACT_IDS, export_vision_sft_dataset


def _label_result_choice(name: str, value: str) -> dict[str, object]:
    return {
        "from_name": name,
        "to_name": "panel_image",
        "type": "choices",
        "value": {"choices": [value]},
    }


def _label_result_text(name: str, value: str) -> dict[str, object]:
    return {
        "from_name": name,
        "to_name": "panel_image",
        "type": "textarea",
        "value": {"text": [value]},
    }


FACT_STATES = {
    "tac_page_visible": "not_seen",
    "supt_page_visible": "not_seen",
    "fcs_page_visible": "seen",
    "fcs_page_x_marks_visible": "seen",
    "bit_root_page_visible": "not_seen",
    "fcsmc_page_visible": "not_seen",
    "fcsmc_intermediate_result_visible": "uncertain",
    "fcsmc_in_test_visible": "seen",
    "fcsmc_final_go_result_visible": "not_seen",
    "hsi_page_visible": "seen",
    "hsi_map_layer_visible": "seen",
    "ins_grnd_alignment_text_visible": "seen",
    "ins_ok_text_visible": "not_seen",
}


def _fact_results() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for fact_id in CORE_FACT_IDS:
        results.append(_label_result_choice(f"{fact_id}_review", FACT_STATES[fact_id]))
        note = "" if fact_id == "ins_ok_text_visible" else f"human {fact_id} note"
        results.append(_label_result_text(f"{fact_id}_note_review", note))
    return results


def _task(*, idx: int, summary: str = "summary text", fallback_note: str = "fallback note") -> dict[str, object]:
    return {
        "id": idx,
        "data": {
            "image": "data:image/png;base64,ZmFrZQ==",
            "frame_id": f"17728724450{idx:02d}_000123",
            "session_id": "sess-live",
            "artifact_image_path": "/tmp/artifact.png",
            "raw_image_path": "/tmp/raw.png",
            "summary": "ai summary",
            "fcs_page_visible_note": fallback_note,
        },
        "annotations": [
            {
                "result": [
                    _label_result_text("summary_review", summary),
                    *_fact_results(),
                ]
            }
        ],
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_export_vision_sft_dataset_builds_reviewed_and_bilingual_sft_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "project.json"
    input_path.write_text(json.dumps([_task(idx=1)], ensure_ascii=False), encoding="utf-8")

    output_dir = tmp_path / "datasets"
    stats = export_vision_sft_dataset(
        input_path=input_path,
        output_dir=output_dir,
        lang="both",
        overwrite=True,
    )

    assert stats.reviewed_samples == 1
    reviewed_rows = _read_jsonl(output_dir / "reviewed.jsonl")
    assert len(reviewed_rows) == 1
    reviewed = reviewed_rows[0]
    assert reviewed["sample_id"] == "frame:1772872445001_000123"
    assert reviewed["image_data_url"].startswith("data:image/png;base64,")
    assert reviewed["summary"] == "summary text"
    assert reviewed["facts"][0]["fact_id"] == "tac_page_visible"
    assert reviewed["facts"][2]["fact_id"] == "fcs_page_visible"
    assert reviewed["facts"][2]["state"] == "seen"
    assert reviewed["facts"][2]["evidence_note"] == "human fcs_page_visible note"
    assert reviewed["notes_by_fact"]["ins_ok_text_visible"] == ""

    sft_en = _read_jsonl(output_dir / "sft_en.jsonl")
    sft_zh = _read_jsonl(output_dir / "sft_zh.jsonl")
    assert len(sft_en) == 1
    assert len(sft_zh) == 1
    en_messages = sft_en[0]["messages"]
    zh_messages = sft_zh[0]["messages"]
    assert en_messages[0]["content"] == "You are SimTutor visual fact extractor. Reply with JSON only."
    assert zh_messages[0]["content"] == "你是 SimTutor 的视觉事实抽取器。只能输出 JSON。"
    assert en_messages[1]["content"][0]["type"] == "image_url"
    assert "Do NOT output frame_id." in en_messages[1]["content"][1]["text"]
    assert "严禁输出 frame_id。" in zh_messages[1]["content"][1]["text"]

    assistant_payload = json.loads(en_messages[2]["content"])
    assert "summary" in assistant_payload
    assert len(assistant_payload["facts"]) == 13
    assert "source_frame_id" not in en_messages[2]["content"]
    assert "confidence" not in en_messages[2]["content"]
    assert "frame_id" not in en_messages[2]["content"]
    assert "session_id" not in en_messages[2]["content"]
    assert "artifact_image_path" not in en_messages[2]["content"]
    assert "raw_image_path" not in en_messages[2]["content"]

    stats_json = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats_json["fact_state_counts"]["fcs_page_visible"]["seen"] == 1
    assert stats_json["fact_state_counts"]["ins_ok_text_visible"]["not_seen"] == 1
    assert stats_json["missing_notes_by_fact"]["ins_ok_text_visible"] == 1
    assert stats_json["train_samples_by_lang"] == {"en": 1, "zh": 1}


def test_export_vision_sft_dataset_supports_drop_summary_and_note_fallback(tmp_path: Path) -> None:
    task = _task(idx=2, summary="")
    result = task["annotations"][0]["result"]
    result_index = next(
        index
        for index, item in enumerate(result)
        if item["from_name"] == "fcs_page_visible_note_review"
    )
    result[result_index] = _label_result_text("fcs_page_visible_note_review", "")
    input_path = tmp_path / "project.json"
    input_path.write_text(json.dumps([task], ensure_ascii=False), encoding="utf-8")

    output_dir = tmp_path / "datasets"
    stats = export_vision_sft_dataset(
        input_path=input_path,
        output_dir=output_dir,
        lang="en",
        overwrite=True,
        drop_summary=True,
    )

    assert stats.missing_summary_count == 1
    reviewed_rows = _read_jsonl(output_dir / "reviewed.jsonl")
    reviewed = reviewed_rows[0]
    fcs_fact = next(item for item in reviewed["facts"] if item["fact_id"] == "fcs_page_visible")
    assert fcs_fact["evidence_note"] == "fallback note"

    sft_en = _read_jsonl(output_dir / "sft_en.jsonl")
    assistant_payload = json.loads(sft_en[0]["messages"][2]["content"])
    assert "summary" not in assistant_payload
    sft_fcs_fact = next(item for item in assistant_payload["facts"] if item["fact_id"] == "fcs_page_visible")
    assert sft_fcs_fact["evidence_note"] == "fallback note"


def test_export_vision_sft_dataset_skips_invalid_task_when_not_strict(tmp_path: Path) -> None:
    valid = _task(idx=3)
    invalid = _task(idx=4)
    invalid["annotations"] = []
    input_path = tmp_path / "project.json"
    input_path.write_text(json.dumps([valid, invalid], ensure_ascii=False), encoding="utf-8")

    output_dir = tmp_path / "datasets"
    stats = export_vision_sft_dataset(
        input_path=input_path,
        output_dir=output_dir,
        lang="zh",
        overwrite=True,
        strict=False,
    )

    assert stats.total_tasks == 2
    assert stats.reviewed_samples == 1
    assert stats.skipped_tasks == 1
    reviewed_rows = _read_jsonl(output_dir / "reviewed.jsonl")
    assert len(reviewed_rows) == 1


def test_export_vision_sft_dataset_raises_on_invalid_task_in_strict_mode(tmp_path: Path) -> None:
    invalid = _task(idx=5)
    result = invalid["annotations"][0]["result"]
    result_index = next(
        index
        for index, item in enumerate(result)
        if item["from_name"] == "fcs_page_visible_review"
    )
    result[result_index] = _label_result_choice("fcs_page_visible_review", "maybe")
    input_path = tmp_path / "project.json"
    input_path.write_text(json.dumps([invalid], ensure_ascii=False), encoding="utf-8")

    output_dir = tmp_path / "datasets"
    try:
        export_vision_sft_dataset(
            input_path=input_path,
            output_dir=output_dir,
            overwrite=True,
            strict=True,
        )
    except ValueError as exc:
        assert "unsupported review state" in str(exc)
    else:
        raise AssertionError("expected ValueError in strict mode")
