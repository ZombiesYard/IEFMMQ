"""
Export reviewed Label Studio tasks into VLM SFT JSONL datasets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

DEFAULT_OUTPUT_DIR = Path("datasets/vision_sft")
DEFAULT_LABEL_SOURCE = "label_studio_review"
ALLOWED_STATES = frozenset({"seen", "not_seen", "uncertain"})
CORE_FACT_IDS: tuple[str, ...] = (
    "tac_page_visible",
    "supt_page_visible",
    "fcs_page_visible",
    "fcs_page_x_marks_visible",
    "bit_root_page_visible",
    "fcsmc_page_visible",
    "fcsmc_intermediate_result_visible",
    "fcsmc_in_test_visible",
    "fcsmc_final_go_result_visible",
    "hsi_page_visible",
    "hsi_map_layer_visible",
    "ins_grnd_alignment_text_visible",
    "ins_ok_text_visible",
)


@dataclass(frozen=True)
class ReviewedSample:
    sample_id: str
    image_data_url: str
    facts: tuple[dict[str, str], ...]
    summary: str
    notes_by_fact: dict[str, str]
    label_source: str
    schema_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "image_data_url": self.image_data_url,
            "facts": [dict(item) for item in self.facts],
            "summary": self.summary,
            "notes_by_fact": dict(self.notes_by_fact),
            "label_source": self.label_source,
        }


@dataclass(frozen=True)
class ExportStats:
    input_path: str
    total_tasks: int
    reviewed_samples: int
    skipped_tasks: int
    exported_languages: tuple[str, ...]
    train_samples_by_lang: dict[str, int]
    missing_summary_count: int
    missing_notes_by_fact: dict[str, int]
    fact_state_counts: dict[str, dict[str, int]]
    length_stats: dict[str, dict[str, float | int | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "v1",
            "input_path": self.input_path,
            "total_tasks": self.total_tasks,
            "reviewed_samples": self.reviewed_samples,
            "skipped_tasks": self.skipped_tasks,
            "exported_languages": list(self.exported_languages),
            "train_samples_by_lang": dict(self.train_samples_by_lang),
            "missing_summary_count": self.missing_summary_count,
            "missing_notes_by_fact": dict(self.missing_notes_by_fact),
            "fact_state_counts": {
                fact_id: dict(state_counts)
                for fact_id, state_counts in self.fact_state_counts.items()
            },
            "length_stats": {
                key: dict(value)
                for key, value in self.length_stats.items()
            },
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert reviewed Label Studio exports into VLM SFT datasets."
    )
    parser.add_argument("--input", required=True, help="Label Studio export JSON file.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for reviewed and SFT JSONL files.",
    )
    parser.add_argument(
        "--lang",
        choices=("en", "zh", "both"),
        default="both",
        help="Language(s) to export for SFT prompts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--drop-summary",
        action="store_true",
        help="Omit summary from the assistant target payload.",
    )
    parser.add_argument(
        "--include-notes",
        choices=("true", "false"),
        default="true",
        help="Whether to keep evidence_note values in the assistant target payload.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional max reviewed sample count to export.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed tasks instead of skipping them.",
    )
    return parser


def _selected_languages(lang: str) -> tuple[str, ...]:
    if lang == "both":
        return ("en", "zh")
    return (lang,)


def _ensure_overwrite_safe(output_dir: Path, *, overwrite: bool, languages: Sequence[str]) -> None:
    targets = [output_dir / "reviewed.jsonl", output_dir / "stats.json"]
    for lang in languages:
        targets.append(output_dir / f"sft_{lang}.jsonl")
    if overwrite:
        return
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace them: " + ", ".join(existing)
        )


def _read_export(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Label Studio export root must be a JSON array")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"task[{index}] must be a JSON object")
        rows.append(item)
    return rows


def _extract_first_annotation(task: Mapping[str, Any]) -> Mapping[str, Any]:
    annotations = task.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("task missing annotations[0]")
    annotation = annotations[0]
    if not isinstance(annotation, Mapping):
        raise ValueError("annotations[0] must be an object")
    return annotation


def _result_value_map(annotation: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    results = annotation.get("result")
    if not isinstance(results, list):
        raise ValueError("annotation.result must be a list")
    mapped: dict[str, Mapping[str, Any]] = {}
    for item in results:
        if not isinstance(item, Mapping):
            continue
        from_name = item.get("from_name")
        value = item.get("value")
        if isinstance(from_name, str) and from_name and isinstance(value, Mapping):
            mapped[from_name] = value
    return mapped


def _extract_text(value: Mapping[str, Any]) -> str:
    text = value.get("text")
    if not isinstance(text, list) or not text:
        return ""
    first = text[0]
    return str(first).strip() if first is not None else ""


def _extract_choice(value: Mapping[str, Any]) -> str:
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("choice result missing choices[0]")
    first = choices[0]
    if not isinstance(first, str) or first not in ALLOWED_STATES:
        raise ValueError(f"unsupported review state: {first!r}")
    return first


def _sample_id(task: Mapping[str, Any], data: Mapping[str, Any]) -> str:
    frame_id = data.get("frame_id")
    if isinstance(frame_id, str) and frame_id.strip():
        return f"frame:{frame_id.strip()}"
    task_id = task.get("id")
    return f"task:{task_id}" if task_id is not None else "task:unknown"


def _reviewed_sample_from_task(
    task: Mapping[str, Any],
    *,
    include_notes: bool,
) -> ReviewedSample:
    data = task.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("task.data must be an object")
    image_data_url = data.get("image")
    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:image/"):
        raise ValueError("task.data.image must be an embedded data URL")

    annotation = _extract_first_annotation(task)
    values = _result_value_map(annotation)
    summary = _extract_text(values.get("summary_review", {}))

    facts: list[dict[str, str]] = []
    notes_by_fact: dict[str, str] = {}
    for fact_id in CORE_FACT_IDS:
        review_key = f"{fact_id}_review"
        note_key = f"{fact_id}_note_review"
        fallback_note_key = f"{fact_id}_note"

        if review_key not in values:
            raise ValueError(f"missing review field: {review_key}")
        state = _extract_choice(values[review_key])

        note = _extract_text(values.get(note_key, {}))
        if not note:
            fallback_note = data.get(fallback_note_key)
            if isinstance(fallback_note, str):
                note = fallback_note.strip()
        notes_by_fact[fact_id] = note
        facts.append(
            {
                "fact_id": fact_id,
                "state": state,
                "evidence_note": note if include_notes else "",
            }
        )

    return ReviewedSample(
        sample_id=_sample_id(task, data),
        image_data_url=image_data_url,
        facts=tuple(facts),
        summary=summary,
        notes_by_fact=notes_by_fact,
        label_source=DEFAULT_LABEL_SOURCE,
    )


def _system_prompt(lang: str) -> str:
    if lang == "en":
        return "You are SimTutor visual fact extractor. Reply with JSON only."
    return "你是 SimTutor 的视觉事实抽取器。只能输出 JSON。"


def _user_prompt(lang: str) -> str:
    if lang == "en":
        return (
            "You are the SimTutor visual fact extractor for the F/A-18C cold-start dataset.\n"
            "The input is exactly one composite-panel image.\n"
            "Its fixed top-to-bottom regions are: left_ddi, ampcd, right_ddi.\n"
            "\n"
            "Task:\n"
            "Inspect only this image and output visual fact labels for the 13 core facts below.\n"
            "If the image is blurry, obstructed, or the state cannot be confirmed from this image alone, use state='uncertain'.\n"
            "\n"
            "Facts to label:\n"
            "1. tac_page_visible\n"
            "2. supt_page_visible\n"
            "3. fcs_page_visible\n"
            "4. fcs_page_x_marks_visible\n"
            "5. bit_root_page_visible\n"
            "6. fcsmc_page_visible\n"
            "7. fcsmc_intermediate_result_visible\n"
            "8. fcsmc_in_test_visible\n"
            "9. fcsmc_final_go_result_visible\n"
            "10. hsi_page_visible\n"
            "11. hsi_map_layer_visible\n"
            "12. ins_grnd_alignment_text_visible\n"
            "13. ins_ok_text_visible\n"
            "\n"
            "Key decision boundaries:\n"
            "- Page option labels are not pages. Do not mark a page visible just because its name appears as a pushbutton/menu option.\n"
            "- tac_page_visible requires the actual TAC/TAC MENU page; a small TAC/MENU navigation label alone is not enough.\n"
            "- supt_page_visible requires the actual SUPT/SUPT MENU page; a small SUPT option label alone is not enough.\n"
            "- fcs_page_visible is the dedicated flight-control FCS page with readable LEF/TEF/AIL/RUD/STAB/SV1/SV2/CAS grid labels.\n"
            "- fcs_page_x_marks_visible only means literal X/fault fills inside FCS page channel boxes.\n"
            "- bit_root_page_visible is the BIT FAILURES/root page; an FCS-MC entry label is not the FCS-MC subpage.\n"
            "- fcsmc_page_visible requires a readable FCS-MC title/subpage with MC1, MC2, FCSA, and FCSB rows.\n"
            "- PBIT GO is intermediate, not final GO. IN TEST is running, not final GO.\n"
            "- fcsmc_final_go_result_visible requires final GO results for the relevant FCS-MC rows, especially FCSA and FCSB.\n"
            "- hsi_page_visible means an HSI navigation/POS page is visible; it does not imply INS alignment text or INS OK.\n"
            "- hsi_map_layer_visible requires a colored topographic/chart MAP background; black HSI symbology alone is not MAP.\n"
            "- ins_grnd_alignment_text_visible requires a literal GRND alignment block with QUAL or TIME/countdown text.\n"
            "- ins_ok_text_visible requires clearly readable OK near the INS alignment block, usually near QUAL.\n"
            "\n"
            "Output requirements:\n"
            "- Output exactly one JSON object.\n"
            "- The top-level object may contain only: summary, facts.\n"
            "- Each fact object may contain only: fact_id, state, evidence_note.\n"
            "- Do NOT output frame_id.\n"
            "- Do NOT output source_frame_id.\n"
            "- Do NOT output confidence.\n"
            "- Do NOT output expires_after_ms or sticky.\n"
            "- Do NOT output tutor answers, step suggestions, or action suggestions.\n"
            "- Do NOT wrap the JSON in markdown or any extra prose.\n"
            "- Include all 13 facts. When unsure, use state='uncertain'.\n"
            "\n"
            "Return format:\n"
            "{\n"
            '  "summary": "one short sentence",\n'
            '  "facts": [\n'
            '    {"fact_id": "tac_page_visible", "state": "seen", "evidence_note": "short evidence"},\n'
            '    {"fact_id": "fcsmc_final_go_result_visible", "state": "not_seen", "evidence_note": "short evidence"}\n'
            "  ]\n"
            "}\n"
        )
    return (
        "你是 SimTutor 的视觉事实抽取器，负责给 F/A-18C 冷启动数据集做视觉事实标注。\n"
        "输入只有一张组合面板图。\n"
        "这张图内部的固定区域从上到下依次是：left_ddi、ampcd、right_ddi。\n"
        "\n"
        "任务：\n"
        "只根据当前这一张图，为下面 13 个核心视觉 fact 输出标注。\n"
        "如果当前图模糊、遮挡、页面不完整、或者仅凭这一张图无法确认，请使用 state='uncertain'。\n"
        "\n"
        "需要标注的 fact：\n"
        "1. tac_page_visible\n"
        "2. supt_page_visible\n"
        "3. fcs_page_visible\n"
        "4. fcs_page_x_marks_visible\n"
        "5. bit_root_page_visible\n"
        "6. fcsmc_page_visible\n"
        "7. fcsmc_intermediate_result_visible\n"
        "8. fcsmc_in_test_visible\n"
        "9. fcsmc_final_go_result_visible\n"
        "10. hsi_page_visible\n"
        "11. hsi_map_layer_visible\n"
        "12. ins_grnd_alignment_text_visible\n"
        "13. ins_ok_text_visible\n"
        "\n"
        "关键判别边界：\n"
        "- 页面选项标签不等于页面本身。不要因为某个 pushbutton/menu option 写着页面名就标 seen。\n"
        "- tac_page_visible 需要实际 TAC/TAC MENU 页面；单独小的 TAC/MENU 导航标签不够。\n"
        "- supt_page_visible 需要实际 SUPT/SUPT MENU 页面；单独小的 SUPT 选项标签不够。\n"
        "- fcs_page_visible 是专用飞控 FCS 页面，需要能读到 LEF/TEF/AIL/RUD/STAB/SV1/SV2/CAS grid 等标签。\n"
        "- fcs_page_x_marks_visible 只表示 FCS 页面通道格子内的字面 X/fault fills。\n"
        "- bit_root_page_visible 是 BIT FAILURES/root 页面；FCS-MC 入口标签不等于 FCS-MC 子页面。\n"
        "- fcsmc_page_visible 需要能读到 FCS-MC 标题/子页面，并有 MC1、MC2、FCSA、FCSB 等行。\n"
        "- PBIT GO 是中间态，不是 final GO。IN TEST 是运行中，不是 final GO。\n"
        "- fcsmc_final_go_result_visible 必须是相关 FCS-MC 行的最终 GO，尤其 FCSA 和 FCSB。\n"
        "- hsi_page_visible 是 HSI navigation/POS 页面可见；它本身不代表 INS 对准文字或 INS OK 可见。\n"
        "- hsi_map_layer_visible 需要彩色地形/航图 MAP 背景；黑底 HSI 符号本身不算 MAP。\n"
        "- ins_grnd_alignment_text_visible 需要字面 GRND 对准文字块，并伴随 QUAL 或 TIME/countdown。\n"
        "- ins_ok_text_visible 需要在 INS 对准文字块附近清楚读到 OK，通常在 QUAL 附近。\n"
        "\n"
        "输出要求：\n"
        "- 只输出一个 JSON object。\n"
        "- 顶层只允许包含：summary、facts。\n"
        "- facts 数组里的每个对象只允许包含：fact_id、state、evidence_note。\n"
        "- 严禁输出 frame_id。\n"
        "- 严禁输出 source_frame_id。\n"
        "- 严禁输出 confidence。\n"
        "- 严禁输出 expires_after_ms 或 sticky。\n"
        "- 严禁输出 tutor answer、步骤推进建议、操作建议。\n"
        "- 严禁输出 markdown、代码块或 JSON 之外的说明文字。\n"
        "- 13 个 fact 都要给出条目；看不清就用 state='uncertain'。\n"
        "\n"
        "返回格式：\n"
        "{\n"
        '  "summary": "一句短总结",\n'
        '  "facts": [\n'
        '    {"fact_id": "tac_page_visible", "state": "seen", "evidence_note": "简短证据"},\n'
        '    {"fact_id": "fcsmc_final_go_result_visible", "state": "not_seen", "evidence_note": "简短证据"}\n'
        "  ]\n"
        "}\n"
    )


def _assistant_payload(sample: ReviewedSample, *, drop_summary: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "facts": [dict(item) for item in sample.facts],
    }
    if not drop_summary:
        payload["summary"] = sample.summary
    return payload


def _sft_record(sample: ReviewedSample, *, lang: str, drop_summary: bool) -> dict[str, Any]:
    assistant_payload = _assistant_payload(sample, drop_summary=drop_summary)
    return {
        "messages": [
            {"role": "system", "content": _system_prompt(lang)},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": sample.image_data_url}},
                    {"type": "text", "text": _user_prompt(lang)},
                ],
            },
            {
                "role": "assistant",
                "content": json.dumps(assistant_payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _length_stats(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(mean(values), 2),
    }


def export_vision_sft_dataset(
    *,
    input_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    lang: str = "both",
    overwrite: bool = False,
    drop_summary: bool = False,
    include_notes: bool = True,
    max_samples: int = 0,
    strict: bool = False,
) -> ExportStats:
    resolved_input = Path(input_path).expanduser().resolve()
    resolved_output = Path(output_dir).expanduser().resolve()
    languages = _selected_languages(lang)
    resolved_output.mkdir(parents=True, exist_ok=True)
    _ensure_overwrite_safe(resolved_output, overwrite=overwrite, languages=languages)

    tasks = _read_export(resolved_input)
    reviewed_samples: list[ReviewedSample] = []
    skipped_tasks = 0
    for task in tasks:
        try:
            sample = _reviewed_sample_from_task(task, include_notes=include_notes)
        except Exception:
            if strict:
                raise
            skipped_tasks += 1
            continue
        reviewed_samples.append(sample)
        if max_samples > 0 and len(reviewed_samples) >= max_samples:
            break

    reviewed_rows = [sample.to_dict() for sample in reviewed_samples]
    _write_jsonl(resolved_output / "reviewed.jsonl", reviewed_rows)

    train_samples_by_lang: dict[str, int] = {}
    assistant_lengths: dict[str, list[int]] = {}
    for selected_lang in languages:
        rows = [_sft_record(sample, lang=selected_lang, drop_summary=drop_summary) for sample in reviewed_samples]
        _write_jsonl(resolved_output / f"sft_{selected_lang}.jsonl", rows)
        train_samples_by_lang[selected_lang] = len(rows)
        assistant_lengths[selected_lang] = [
            len(row["messages"][2]["content"])
            for row in rows
            if isinstance(row.get("messages"), list) and len(row["messages"]) >= 3
        ]

    fact_state_counts: dict[str, dict[str, int]] = {}
    missing_notes_by_fact: dict[str, int] = {}
    for fact_id in CORE_FACT_IDS:
        state_counts = {"seen": 0, "not_seen": 0, "uncertain": 0}
        missing_notes = 0
        for sample in reviewed_samples:
            fact = next(item for item in sample.facts if item["fact_id"] == fact_id)
            state_counts[fact["state"]] += 1
            if not sample.notes_by_fact.get(fact_id, "").strip():
                missing_notes += 1
        fact_state_counts[fact_id] = state_counts
        missing_notes_by_fact[fact_id] = missing_notes

    missing_summary_count = sum(1 for sample in reviewed_samples if not sample.summary.strip())
    summary_lengths = [len(sample.summary) for sample in reviewed_samples]
    image_lengths = [len(sample.image_data_url) for sample in reviewed_samples]
    length_stats = {
        "summary_chars": _length_stats(summary_lengths),
        "image_data_url_chars": _length_stats(image_lengths),
    }
    for selected_lang in languages:
        length_stats[f"assistant_chars_{selected_lang}"] = _length_stats(assistant_lengths[selected_lang])

    stats = ExportStats(
        input_path=str(resolved_input),
        total_tasks=len(tasks),
        reviewed_samples=len(reviewed_samples),
        skipped_tasks=skipped_tasks,
        exported_languages=tuple(languages),
        train_samples_by_lang=train_samples_by_lang,
        missing_summary_count=missing_summary_count,
        missing_notes_by_fact=missing_notes_by_fact,
        fact_state_counts=fact_state_counts,
        length_stats=length_stats,
    )
    (resolved_output / "stats.json").write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def _parse_include_notes(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"unsupported boolean value: {value!r}")


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        stats = export_vision_sft_dataset(
            input_path=args.input,
            output_dir=args.output_dir,
            lang=args.lang,
            overwrite=args.overwrite,
            drop_summary=args.drop_summary,
            include_notes=_parse_include_notes(args.include_notes),
            max_samples=args.max_samples,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"[EXPORT_VISION_SFT_DATASET] failed: {type(exc).__name__}: {exc}")
        return 1

    print(
        "[EXPORT_VISION_SFT_DATASET] stats="
        + json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True)
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
