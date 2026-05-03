#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


FACT_FIELDS = [
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
]

VALID_STATES = {"seen", "not_seen", "uncertain"}
TO_NAME = "panel_image"
DEFAULT_SCORE = 0.95


def build_choice_result(field: str, value: str) -> Dict[str, Any]:
    return {
        "from_name": f"{field}_review",
        "to_name": TO_NAME,
        "type": "choices",
        "value": {
            "choices": [value]
        }
    }


def build_textarea_result(from_name: str, text: str) -> Dict[str, Any]:
    return {
        "from_name": from_name,
        "to_name": TO_NAME,
        "type": "textarea",
        "value": {
            "text": [text]
        }
    }


def parse_ai_prelabel_json(data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    从 ai_prelabel_json 兜底提取 facts:
    {
        "fcs_page_visible": {"state": "not_seen", "note": "..."},
        ...
    }
    """
    ai_facts: Dict[str, Dict[str, str]] = {}
    raw = data.get("ai_prelabel_json")

    if not isinstance(raw, str) or not raw.strip():
        return ai_facts

    try:
        parsed = json.loads(raw)
    except Exception:
        return ai_facts

    facts = parsed.get("facts", [])
    if not isinstance(facts, list):
        return ai_facts

    for fact in facts:
        if not isinstance(fact, dict):
            continue
        fact_id = fact.get("fact_id")
        state = fact.get("state")
        note = fact.get("evidence_note", "")
        if isinstance(fact_id, str) and fact_id:
            ai_facts[fact_id] = {
                "state": state if isinstance(state, str) else "",
                "note": note if isinstance(note, str) else "",
            }

    return ai_facts


def normalize_path_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("data:image/"):
        return value
    return value.replace("\\", "/")


def convert_task(task: Dict[str, Any], score: float = DEFAULT_SCORE) -> Dict[str, Any]:
    data = task.get("data", {})
    meta = task.get("meta", {})

    if not isinstance(data, dict):
        data = {}

    if not isinstance(meta, dict):
        meta = {}

    # 简单清理路径展示
    for key in ["artifact_image_path", "raw_image_path"]:
        if key in data:
            data[key] = normalize_path_string(data[key])

    ai_facts = parse_ai_prelabel_json(data)

    results: List[Dict[str, Any]] = []

    # 1) summary 也预填进 review 框
    summary = data.get("summary", "")
    if isinstance(summary, str) and summary.strip():
        results.append(build_textarea_result("summary_review", summary.strip()))

    # 2) facts 的 choices + note
    for field in FACT_FIELDS:
        state = data.get(field)
        note = data.get(f"{field}_note", "")

        # 顶层没有时，用 ai_prelabel_json 兜底
        if (state is None or state == "") and field in ai_facts:
            state = ai_facts[field].get("state", "")
        if (not isinstance(note, str) or not note.strip()) and field in ai_facts:
            note = ai_facts[field].get("note", "")

        if isinstance(state, str) and state in VALID_STATES:
            results.append(build_choice_result(field, state))

        if isinstance(note, str) and note.strip():
            results.append(build_textarea_result(f"{field}_note_review", note.strip()))

    prediction = {
        "model_version": "ai_prelabel_v2",
        "score": score,
        "result": results,
    }

    converted = {
        "data": data,
        "predictions": [prediction],
    }

    if meta:
        converted["meta"] = meta

    for passthrough_key in ("id", "inner_id", "created_at", "updated_at"):
        if passthrough_key in task:
            converted[passthrough_key] = task[passthrough_key]

    return converted


def convert_file(input_path: Path, output_path: Path, *, score: float = DEFAULT_SCORE) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError("Input JSON top-level must be a list of tasks.")

    converted_tasks = [convert_task(task, score=score) for task in tasks]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted_tasks, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted_tasks)} tasks")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")


def resolve_paths(
    *,
    input_path: str | None,
    output_path: str | None,
    session_dir: str | None,
    prelabels_dirname: str,
) -> tuple[Path, Path]:
    if input_path:
        resolved_input = Path(input_path).expanduser()
    else:
        base_dir = Path(session_dir).expanduser() if session_dir else Path(__file__).resolve().parent
        resolved_input = base_dir / prelabels_dirname / "label_studio_tasks.json"

    if output_path:
        resolved_output = Path(output_path).expanduser()
    else:
        resolved_output = resolved_input.with_name("label_studio_tasks_with_predictions.json")

    return resolved_input, resolved_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert generated Label Studio task JSON into a task JSON with "
            "AI prelabels stored as Label Studio predictions."
        )
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Capture session directory, e.g. tools/.captures/fa18c-coldstart-run-002.",
    )
    parser.add_argument(
        "--prelabels-dirname",
        default="prelabels",
        help="Prelabels subdirectory under --session-dir. Defaults to prelabels.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Explicit input label_studio_tasks.json path. Overrides --session-dir.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output JSON path. Defaults to label_studio_tasks_with_predictions.json next to input.",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=DEFAULT_SCORE,
        help="Prediction score written to Label Studio predictions.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_json, output_json = resolve_paths(
        input_path=args.input,
        output_path=args.output,
        session_dir=args.session_dir,
        prelabels_dirname=args.prelabels_dirname,
    )
    convert_file(input_json, output_json, score=args.score)


if __name__ == "__main__":
    main()
