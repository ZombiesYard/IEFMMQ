"""
Benchmark Qwen3.5 VLM fact extraction before and after LoRA fine-tuning.
"""

from __future__ import annotations

import argparse
import base64
import re
from dataclasses import dataclass, field
from io import BytesIO
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

CORE_FACT_IDS: tuple[str, ...] = (
    "fcs_page_visible",
    "bit_root_page_visible",
    "bit_page_failure_visible",
    "right_ddi_fcsmc_page_visible",
    "right_ddi_in_test_visible",
    "fcs_bit_result_visible",
    "ins_alignment_page_visible",
    "ins_go",
)

ALLOWED_STATES = ("seen", "not_seen", "uncertain")
CRITICAL_FACT_IDS = (
    "fcs_bit_result_visible",
    "ins_go",
    "right_ddi_in_test_visible",
    "right_ddi_fcsmc_page_visible",
)

CHART_LABELS = {
    "fcs_page_visible": "FCS\npage",
    "bit_root_page_visible": "BIT\nroot",
    "bit_page_failure_visible": "BIT\nfailure",
    "right_ddi_fcsmc_page_visible": "Right DDI\nFCS-MC",
    "right_ddi_in_test_visible": "Right DDI\nIN TEST",
    "fcs_bit_result_visible": "FCS BIT\nresult",
    "ins_alignment_page_visible": "INS\nalignment",
    "ins_go": "INS\nGO",
    "fact_accuracy": "Fact\naccuracy",
    "sample_exact_match": "Sample\nexact match",
}

_THINK_BLOCK_RE = re.compile(r"(?is)^\s*<think(?:\s[^>]*)?>.*?</think>\s*")


def _chart_label(label: str) -> str:
    return CHART_LABELS.get(label, label.replace("_", "\n"))


@dataclass(frozen=True)
class JsonExtraction:
    json_text: str
    json_repaired: bool
    repair_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReviewedBenchmarkSample:
    sample_id: str
    image_data_url: str
    facts: dict[str, str]


@dataclass
class PredictionRecord:
    sample_id: str
    raw_text: str
    facts: dict[str, str]
    json_valid: bool
    schema_valid: bool
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "raw_text": self.raw_text,
            "facts": dict(self.facts),
            "json_valid": self.json_valid,
            "schema_valid": self.schema_valid,
            "error": self.error,
            "warnings": list(self.warnings),
        }


def _system_prompt(lang: str) -> str:
    if lang == "en":
        return "You are SimTutor visual fact extractor. Reply with JSON only."
    return "你是 SimTutor 的视觉事实抽取器。只能输出 JSON。"


def _user_prompt(lang: str) -> str:
    if lang == "en":
        return (
            "You are the SimTutor visual fact extractor for the F/A-18C cold-start dataset.\n"
            "The input is exactly one composite-panel image.\n"
            "Its fixed top-to-bottom regions are: left_ddi, ampcd, right_ddi.\n\n"
            "Task:\n"
            "Inspect only this image and output visual fact labels for the 8 core facts below.\n"
            "If the image is blurry, obstructed, or the state cannot be confirmed from this image alone, use state='uncertain'.\n\n"
            "Facts to label:\n"
            "1. fcs_page_visible\n"
            "2. bit_root_page_visible\n"
            "3. bit_page_failure_visible\n"
            "4. right_ddi_fcsmc_page_visible\n"
            "5. right_ddi_in_test_visible\n"
            "6. fcs_bit_result_visible\n"
            "7. ins_alignment_page_visible\n"
            "8. ins_go\n\n"
            "Key decision boundaries:\n"
            "- fcs_page_visible means a real FCS page is visible. Seeing only an FCS option/button is not enough.\n"
            "- bit_root_page_visible, bit_page_failure_visible, and right_ddi_fcsmc_page_visible must be distinguished.\n"
            "- right_ddi_in_test_visible means the right DDI is clearly in an IN TEST state, not just a generic BIT context.\n"
            "- fcs_bit_result_visible must only be 'seen' when the final FCS BIT result is clearly visible. Intermediate or PBIT GO cues are not enough.\n"
            "- ins_alignment_page_visible and ins_go must be distinguished.\n"
            "- If AMPCD shows a MAP layer or a non-alignment page, do not hallucinate ins_go.\n\n"
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
            "- Include all 8 facts. When unsure, use state='uncertain'.\n\n"
            "Return format:\n"
            "{\n"
            '  "summary": "one short sentence",\n'
            '  "facts": [\n'
            '    {"fact_id": "fcs_page_visible", "state": "seen", "evidence_note": "short evidence"},\n'
            '    {"fact_id": "bit_root_page_visible", "state": "not_seen", "evidence_note": "short evidence"}\n'
            "  ]\n"
            "}\n"
        )
    return (
        "你是 SimTutor 的视觉事实抽取器，负责给 F/A-18C 冷启动数据集做视觉事实标注。\n"
        "输入只有一张组合面板图。\n"
        "这张图内部的固定区域从上到下依次是：left_ddi、ampcd、right_ddi。\n\n"
        "任务：\n"
        "只根据当前这一张图，为下面 8 个核心视觉 fact 输出标注。\n"
        "如果当前图模糊、遮挡、页面不完整、或者仅凭这一张图无法确认，请使用 state='uncertain'。\n\n"
        "需要标注的 fact：\n"
        "1. fcs_page_visible\n"
        "2. bit_root_page_visible\n"
        "3. bit_page_failure_visible\n"
        "4. right_ddi_fcsmc_page_visible\n"
        "5. right_ddi_in_test_visible\n"
        "6. fcs_bit_result_visible\n"
        "7. ins_alignment_page_visible\n"
        "8. ins_go\n\n"
        "关键判别边界：\n"
        "- fcs_page_visible 只有在真正显示 FCS 页面时才算 seen；仅看到 FCS 选项或按钮不算。\n"
        "- bit_root_page_visible、bit_page_failure_visible、right_ddi_fcsmc_page_visible 必须区分开。\n"
        "- right_ddi_in_test_visible 只有在右 DDI 明确处于 IN TEST 状态时才算 seen。\n"
        "- fcs_bit_result_visible 只有在最终 FCS BIT 结果明确可见时才算 seen；中间态或 PBIT GO 不能算。\n"
        "- ins_alignment_page_visible 和 ins_go 必须区分开。\n"
        "- 如果 AMPCD 显示 MAP 图层或其它非对准页面，不要臆测 ins_go=seen。\n\n"
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
        "- 8 个 fact 都要给出条目；看不清就用 state='uncertain'。\n\n"
        "返回格式：\n"
        "{\n"
        '  "summary": "一句短总结",\n'
        '  "facts": [\n'
        '    {"fact_id": "fcs_page_visible", "state": "seen", "evidence_note": "简短证据"},\n'
        '    {"fact_id": "bit_root_page_visible", "state": "not_seen", "evidence_note": "简短证据"}\n'
        "  ]\n"
        "}\n"
    )


def _strip_balanced_code_fence(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    if not (text.startswith("```") and text.endswith("```")):
        return text, False
    lines = text.splitlines()
    if len(lines) >= 2:
        inner = "\n".join(lines[1:-1]).strip()
        return inner, True
    inner = text[3:-3].strip()
    if inner.lower().startswith("json"):
        inner = inner[4:].strip()
    return inner, True


def _strip_leading_think_blocks(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    repaired = False
    while True:
        match = _THINK_BLOCK_RE.match(text)
        if match is None:
            return text, repaired
        text = text[match.end() :].strip()
        repaired = True


def _find_first_json_segment(text: str) -> tuple[int, int]:
    in_string = False
    escaped = False
    stack: list[str] = []
    start = -1
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if start < 0:
            if char == "{":
                start = index
                stack.append("}")
            elif char == "[":
                start = index
                stack.append("]")
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]") and stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return start, index + 1
    raise ValueError("Model output does not contain JSON object/array")


def _parse_first_json(raw_text: str) -> tuple[Any, JsonExtraction]:
    text, fence_removed = _strip_balanced_code_fence(raw_text)
    text, think_removed = _strip_leading_think_blocks(text)
    start, end = _find_first_json_segment(text)
    json_text = text[start:end]
    reasons: list[str] = []
    if fence_removed:
        reasons.append("removed_code_fence")
    if think_removed:
        reasons.append("removed_think_tags")
    if text[:start].strip():
        reasons.append("dropped_prefix_text")
    if text[end:].strip():
        reasons.append("dropped_suffix_text")
    return json.loads(json_text), JsonExtraction(
        json_text=json_text,
        json_repaired=bool(reasons),
        repair_reasons=tuple(reasons),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark base vs LoRA Qwen3.5 VLM visual fact extraction."
    )
    parser.add_argument("--reviewed-jsonl", required=True, help="Reviewed JSONL from export_vision_sft_dataset.py.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--adapter", default="", help="Optional LoRA adapter path. If set, run LoRA benchmark too.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--benchmark-kind", default="contaminated_dev_set")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model files only from the local Hugging Face cache.",
    )
    parser.add_argument("--skip-base", action="store_true", help="Only run LoRA if --adapter is set.")
    parser.add_argument("--skip-lora", action="store_true", help="Only run base model.")
    return parser


def _load_reviewed(path: str | Path, *, max_samples: int = 0) -> list[ReviewedBenchmarkSample]:
    rows: list[ReviewedBenchmarkSample] = []
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{resolved}:{line_number} must be a JSON object")
            sample_id = str(payload.get("sample_id") or f"row:{line_number}")
            image_data_url = payload.get("image_data_url")
            if not isinstance(image_data_url, str) or not image_data_url.startswith("data:image/"):
                raise ValueError(f"{resolved}:{line_number} missing image_data_url")
            facts_raw = payload.get("facts")
            if not isinstance(facts_raw, list):
                raise ValueError(f"{resolved}:{line_number} missing facts list")
            facts: dict[str, str] = {}
            for item in facts_raw:
                if not isinstance(item, Mapping):
                    continue
                fact_id = item.get("fact_id")
                state = item.get("state")
                if isinstance(fact_id, str) and fact_id in CORE_FACT_IDS and isinstance(state, str):
                    facts[fact_id] = state
            for fact_id in CORE_FACT_IDS:
                if facts.get(fact_id) not in ALLOWED_STATES:
                    raise ValueError(f"{resolved}:{line_number} missing valid state for {fact_id}")
            rows.append(
                ReviewedBenchmarkSample(
                    sample_id=sample_id,
                    image_data_url=image_data_url,
                    facts=facts,
                )
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _image_from_data_url(data_url: str) -> Image.Image:
    if "base64," not in data_url:
        raise ValueError("image data URL must contain base64 data")
    _, encoded = data_url.split("base64,", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _messages_for_sample(sample: ReviewedBenchmarkSample, *, lang: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": _system_prompt(lang)},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample.image_data_url},
                {"type": "text", "text": _user_prompt(lang)},
            ],
        },
    ]


def _model_generate(
    *,
    model: Any,
    processor: Any,
    sample: ReviewedBenchmarkSample,
    lang: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    messages = _messages_for_sample(sample, lang=lang)
    try:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    image = _image_from_data_url(sample.image_data_url)
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    )
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = inputs.to(device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def _release_cuda_memory() -> None:
    try:
        import gc
        import torch
    except Exception:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(
    *,
    base_model: str,
    adapter: str | None,
    max_seq_length: int,
    seed: int,
    local_files_only: bool,
):
    from unsloth import FastVisionModel
    from peft import PeftModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        random_state=seed,
        local_files_only=local_files_only,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    FastVisionModel.for_inference(model)
    return model, processor


def _normalize_prediction(raw_text: str, *, sample_id: str) -> PredictionRecord:
    missing_facts = {fact_id: "uncertain" for fact_id in CORE_FACT_IDS}
    warnings: list[str] = []
    try:
        parsed, extraction = _parse_first_json(raw_text)
    except Exception as exc:
        return PredictionRecord(
            sample_id=sample_id,
            raw_text=raw_text,
            facts=missing_facts,
            json_valid=False,
            schema_valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    if extraction.json_repaired:
        warnings.extend(extraction.repair_reasons)

    schema_valid = True
    if not isinstance(parsed, Mapping):
        return PredictionRecord(
            sample_id=sample_id,
            raw_text=raw_text,
            facts=missing_facts,
            json_valid=True,
            schema_valid=False,
            error="parsed JSON is not an object",
            warnings=warnings,
        )

    top_keys = set(parsed.keys())
    if top_keys - {"summary", "facts"}:
        schema_valid = False
        warnings.append("unknown_top_level_keys:" + ",".join(sorted(top_keys - {"summary", "facts"})))

    facts_payload = parsed.get("facts")
    if not isinstance(facts_payload, list):
        return PredictionRecord(
            sample_id=sample_id,
            raw_text=raw_text,
            facts=missing_facts,
            json_valid=True,
            schema_valid=False,
            error="facts must be a list",
            warnings=warnings,
        )

    facts = dict(missing_facts)
    seen_fact_ids: set[str] = set()
    for index, item in enumerate(facts_payload):
        if not isinstance(item, Mapping):
            schema_valid = False
            warnings.append(f"fact[{index}]_not_object")
            continue
        item_keys = set(item.keys())
        allowed_item_keys = {"fact_id", "state", "evidence_note"}
        if item_keys - allowed_item_keys:
            schema_valid = False
            warnings.append(f"fact[{index}]_unknown_keys:" + ",".join(sorted(item_keys - allowed_item_keys)))
        fact_id = item.get("fact_id")
        state = item.get("state")
        if not isinstance(fact_id, str) or fact_id not in CORE_FACT_IDS:
            schema_valid = False
            warnings.append(f"fact[{index}]_unknown_fact_id")
            continue
        if fact_id in seen_fact_ids:
            schema_valid = False
            warnings.append(f"duplicate_fact_id:{fact_id}")
        seen_fact_ids.add(fact_id)
        if not isinstance(state, str) or state not in ALLOWED_STATES:
            schema_valid = False
            warnings.append(f"invalid_state:{fact_id}")
            continue
        facts[fact_id] = state

    missing = [fact_id for fact_id in CORE_FACT_IDS if fact_id not in seen_fact_ids]
    if missing:
        schema_valid = False
        warnings.append("missing_facts:" + ",".join(missing))

    return PredictionRecord(
        sample_id=sample_id,
        raw_text=raw_text,
        facts=facts,
        json_valid=True,
        schema_valid=schema_valid,
        warnings=warnings,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def compute_metrics(
    *,
    samples: Sequence[ReviewedBenchmarkSample],
    predictions: Mapping[str, PredictionRecord],
    model_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_samples = len(samples)
    total_facts = total_samples * len(CORE_FACT_IDS)
    correct_facts = 0
    exact_matches = 0
    errors: list[dict[str, Any]] = []

    fact_state_counts: dict[str, dict[str, dict[str, int]]] = {}
    fact_scores: dict[str, dict[str, Any]] = {}

    for fact_id in CORE_FACT_IDS:
        matrix = {
            gold: {pred: 0 for pred in ALLOWED_STATES}
            for gold in ALLOWED_STATES
        }
        fact_state_counts[fact_id] = matrix

    json_valid_count = 0
    schema_valid_count = 0
    critical_false_positives = 0
    critical_false_positives_by_fact = {fact_id: 0 for fact_id in CRITICAL_FACT_IDS}

    for sample in samples:
        prediction = predictions[sample.sample_id]
        json_valid_count += int(prediction.json_valid)
        schema_valid_count += int(prediction.schema_valid)
        sample_correct = True
        for fact_id in CORE_FACT_IDS:
            gold = sample.facts[fact_id]
            pred = prediction.facts.get(fact_id, "uncertain")
            fact_state_counts[fact_id][gold][pred] += 1
            if pred == gold:
                correct_facts += 1
            else:
                sample_correct = False
                error = {
                    "model": model_label,
                    "sample_id": sample.sample_id,
                    "fact_id": fact_id,
                    "gold": gold,
                    "predicted": pred,
                    "raw_text": prediction.raw_text,
                    "schema_valid": prediction.schema_valid,
                    "json_valid": prediction.json_valid,
                    "warnings": prediction.warnings,
                    "error": prediction.error,
                }
                errors.append(error)
                if fact_id in CRITICAL_FACT_IDS and pred == "seen" and gold != "seen":
                    critical_false_positives += 1
                    critical_false_positives_by_fact[fact_id] += 1
        if sample_correct:
            exact_matches += 1

    macro_f1_values: list[float] = []
    seen_f1_values: list[float] = []
    for fact_id in CORE_FACT_IDS:
        matrix = fact_state_counts[fact_id]
        state_scores: dict[str, dict[str, float | int]] = {}
        for state in ALLOWED_STATES:
            tp = matrix[state][state]
            fp = sum(matrix[gold][state] for gold in ALLOWED_STATES if gold != state)
            fn = sum(matrix[state][pred] for pred in ALLOWED_STATES if pred != state)
            state_scores[state] = _prf(tp, fp, fn)
        fact_macro_f1 = sum(float(state_scores[state]["f1"]) for state in ALLOWED_STATES) / len(ALLOWED_STATES)
        fact_seen_f1 = float(state_scores["seen"]["f1"])
        macro_f1_values.append(fact_macro_f1)
        seen_f1_values.append(fact_seen_f1)
        fact_scores[fact_id] = {
            "state_scores": state_scores,
            "macro_f1": round(fact_macro_f1, 6),
            "seen_f1": round(fact_seen_f1, 6),
            "confusion_matrix": matrix,
            "accuracy": round(
                _safe_div(sum(matrix[state][state] for state in ALLOWED_STATES), sum(sum(row.values()) for row in matrix.values())),
                6,
            ),
        }

    metrics = {
        "schema_version": "v1",
        "model_label": model_label,
        "sample_count": total_samples,
        "fact_count": total_facts,
        "json_valid_rate": round(_safe_div(json_valid_count, total_samples), 6),
        "schema_valid_rate": round(_safe_div(schema_valid_count, total_samples), 6),
        "fact_accuracy": round(_safe_div(correct_facts, total_facts), 6),
        "sample_exact_match": round(_safe_div(exact_matches, total_samples), 6),
        "macro_f1": round(sum(macro_f1_values) / len(macro_f1_values), 6),
        "seen_f1": round(sum(seen_f1_values) / len(seen_f1_values), 6),
        "critical_false_positive_count": critical_false_positives,
        "critical_false_positives_by_fact": critical_false_positives_by_fact,
        "fact_scores": fact_scores,
    }
    return metrics, errors


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_fact_scores_csv(path: Path, metrics_by_model: Mapping[str, Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "fact_id",
                "accuracy",
                "macro_f1",
                "seen_precision",
                "seen_recall",
                "seen_f1",
                "seen_false_positives",
                "seen_false_negatives",
            ],
        )
        writer.writeheader()
        for model_label, metrics in metrics_by_model.items():
            for fact_id, score in metrics["fact_scores"].items():
                seen = score["state_scores"]["seen"]
                writer.writerow(
                    {
                        "model": model_label,
                        "fact_id": fact_id,
                        "accuracy": score["accuracy"],
                        "macro_f1": score["macro_f1"],
                        "seen_precision": seen["precision"],
                        "seen_recall": seen["recall"],
                        "seen_f1": seen["f1"],
                        "seen_false_positives": seen["fp"],
                        "seen_false_negatives": seen["fn"],
                    }
                )


def _make_charts(output_dir: Path, metrics_by_model: Mapping[str, Mapping[str, Any]]) -> list[str]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    labels = list(metrics_by_model.keys())
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _make_pil_charts(chart_dir, metrics_by_model)

    x = list(range(len(labels)))

    def save_bar(path: Path, title: str, ylabel: str, values: Sequence[float]) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, ha="center")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", pad=8)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))

    save_bar(
        chart_dir / "overall_accuracy.png",
        "Overall Fact Accuracy",
        "accuracy",
        [float(metrics_by_model[label]["fact_accuracy"]) for label in labels],
    )
    save_bar(
        chart_dir / "sample_exact_match.png",
        "Sample Exact Match",
        "exact match",
        [float(metrics_by_model[label]["sample_exact_match"]) for label in labels],
    )

    width = 0.35 if len(labels) == 2 else 0.8 / max(1, len(labels))
    fact_x = list(range(len(CORE_FACT_IDS)))
    for metric_name, file_name, title in [
        ("macro_f1", "fact_f1_by_model.png", "Fact Macro F1 by Model"),
        ("seen_f1", "seen_f1_by_fact.png", "Seen F1 by Fact"),
    ]:
        fig, ax = plt.subplots(figsize=(13, 5.6))
        for idx, label in enumerate(labels):
            offset = (idx - (len(labels) - 1) / 2) * width
            values = [
                float(metrics_by_model[label]["fact_scores"][fact_id][metric_name])
                for fact_id in CORE_FACT_IDS
            ]
            ax.bar([pos + offset for pos in fact_x], values, width=width, label=label)
        ax.set_xticks(fact_x)
        ax.set_xticklabels(
            [_chart_label(fact_id) for fact_id in CORE_FACT_IDS],
            rotation=0,
            ha="center",
            multialignment="center",
        )
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.tick_params(axis="x", pad=10)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)
        path = chart_dir / file_name
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    critical_x = list(range(len(CRITICAL_FACT_IDS)))
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        values = [
            int(metrics_by_model[label]["critical_false_positives_by_fact"].get(fact_id, 0))
            for fact_id in CRITICAL_FACT_IDS
        ]
        ax.bar([pos + offset for pos in critical_x], values, width=width, label=label)
    ax.set_xticks(critical_x)
    ax.set_xticklabels(
        [_chart_label(fact_id) for fact_id in CRITICAL_FACT_IDS],
        rotation=0,
        ha="center",
        multialignment="center",
    )
    ax.set_title("Critical False Positives")
    ax.set_ylabel("count")
    ax.legend()
    ax.tick_params(axis="x", pad=10)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    path = chart_dir / "critical_false_positives.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    for model_label, metrics in metrics_by_model.items():
        for fact_id in CORE_FACT_IDS:
            matrix = metrics["fact_scores"][fact_id]["confusion_matrix"]
            values = [[int(matrix[gold][pred]) for pred in ALLOWED_STATES] for gold in ALLOWED_STATES]
            plt.figure(figsize=(5, 4))
            plt.imshow(values, cmap="Blues")
            plt.title(f"{fact_id} {model_label}")
            plt.xticks(range(len(ALLOWED_STATES)), ALLOWED_STATES, rotation=25, ha="right")
            plt.yticks(range(len(ALLOWED_STATES)), ALLOWED_STATES)
            plt.xlabel("predicted")
            plt.ylabel("gold")
            for row_index, row in enumerate(values):
                for col_index, value in enumerate(row):
                    plt.text(col_index, row_index, str(value), ha="center", va="center")
            plt.tight_layout()
            path = chart_dir / f"confusion_{fact_id}_{model_label}.png"
            plt.savefig(path)
            plt.close()
            generated.append(str(path))

    return generated


def _make_pil_charts(chart_dir: Path, metrics_by_model: Mapping[str, Mapping[str, Any]]) -> list[str]:
    from PIL import ImageDraw

    generated: list[str] = []
    labels = list(metrics_by_model.keys())
    palette = ["#2f6f9f", "#c95f2f", "#5f8f3f", "#8f4f9f"]

    def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str = "#202020") -> None:
        draw.text(xy, text, fill=fill)

    def _wrap_label(label: str, *, max_line_chars: int = 12) -> list[str]:
        chart_label = _chart_label(label)
        if "\n" in chart_label:
            return chart_label.splitlines()
        words = chart_label.split("_")
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current}_{word}"
            if len(candidate) <= max_line_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [label]

    def _save_grouped_bars(
        path: Path,
        *,
        title: str,
        categories: Sequence[str],
        values_by_label: Mapping[str, Sequence[float]],
        max_value: float,
    ) -> None:
        width = max(1100, 140 * len(categories) + 220)
        height = 660
        margin_left = 90
        margin_right = 40
        margin_top = 55
        margin_bottom = 205
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        _draw_text(draw, (margin_left, 20), title)
        draw.line((margin_left, margin_top, margin_left, margin_top + plot_height), fill="#303030")
        draw.line((margin_left, margin_top + plot_height, margin_left + plot_width, margin_top + plot_height), fill="#303030")
        for tick in range(0, 6):
            value = max_value * tick / 5
            y = margin_top + plot_height - int(plot_height * tick / 5)
            draw.line((margin_left - 4, y, margin_left + plot_width, y), fill="#e8e8e8")
            _draw_text(draw, (8, y - 7), f"{value:.2f}")

        category_width = plot_width / max(1, len(categories))
        bar_width = max(8, int(category_width * 0.7 / max(1, len(labels))))
        for category_index, category in enumerate(categories):
            group_left = margin_left + category_width * category_index + category_width * 0.15
            for label_index, label in enumerate(labels):
                raw_value = float(values_by_label[label][category_index])
                bar_height = int(plot_height * (raw_value / max_value)) if max_value else 0
                left = int(group_left + label_index * bar_width)
                right = left + bar_width - 2
                top = margin_top + plot_height - bar_height
                bottom = margin_top + plot_height
                draw.rectangle((left, top, right, bottom), fill=palette[label_index % len(palette)])
                _draw_text(draw, (left, max(margin_top, top - 15)), f"{raw_value:.2f}" if max_value <= 1 else str(int(raw_value)))
            label_x = int(margin_left + category_width * (category_index + 0.5))
            label_y = margin_top + plot_height + 12
            for line_index, line in enumerate(_wrap_label(category)):
                text_width = draw.textlength(line)
                _draw_text(draw, (int(label_x - text_width / 2), label_y + line_index * 14), line)

        legend_x = margin_left
        legend_y = height - 35
        for label_index, label in enumerate(labels):
            color = palette[label_index % len(palette)]
            x = legend_x + label_index * 110
            draw.rectangle((x, legend_y, x + 14, legend_y + 14), fill=color)
            _draw_text(draw, (x + 20, legend_y), label)
        image.save(path)
        generated.append(str(path))

    _save_grouped_bars(
        chart_dir / "overall_accuracy.png",
        title="Overall Accuracy",
        categories=["fact_accuracy", "sample_exact_match"],
        values_by_label={
            label: [
                float(metrics_by_model[label]["fact_accuracy"]),
                float(metrics_by_model[label]["sample_exact_match"]),
            ]
            for label in labels
        },
        max_value=1.0,
    )
    _save_grouped_bars(
        chart_dir / "fact_f1_by_model.png",
        title="Fact Macro F1 by Model",
        categories=list(CORE_FACT_IDS),
        values_by_label={
            label: [
                float(metrics_by_model[label]["fact_scores"][fact_id]["macro_f1"])
                for fact_id in CORE_FACT_IDS
            ]
            for label in labels
        },
        max_value=1.0,
    )
    _save_grouped_bars(
        chart_dir / "seen_f1_by_fact.png",
        title="Seen F1 by Fact",
        categories=list(CORE_FACT_IDS),
        values_by_label={
            label: [
                float(metrics_by_model[label]["fact_scores"][fact_id]["seen_f1"])
                for fact_id in CORE_FACT_IDS
            ]
            for label in labels
        },
        max_value=1.0,
    )
    max_critical = max(
        1,
        *[
            int(metrics_by_model[label]["critical_false_positives_by_fact"].get(fact_id, 0))
            for label in labels
            for fact_id in CRITICAL_FACT_IDS
        ],
    )
    _save_grouped_bars(
        chart_dir / "critical_false_positives.png",
        title="Critical False Positives",
        categories=list(CRITICAL_FACT_IDS),
        values_by_label={
            label: [
                float(metrics_by_model[label]["critical_false_positives_by_fact"].get(fact_id, 0))
                for fact_id in CRITICAL_FACT_IDS
            ]
            for label in labels
        },
        max_value=float(max_critical),
    )

    for model_label, metrics in metrics_by_model.items():
        for fact_id in CORE_FACT_IDS:
            matrix = metrics["fact_scores"][fact_id]["confusion_matrix"]
            max_cell = max(1, *[int(matrix[gold][pred]) for gold in ALLOWED_STATES for pred in ALLOWED_STATES])
            cell = 92
            width = 430
            height = 390
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)
            _draw_text(draw, (25, 18), f"{fact_id} {model_label}")
            _draw_text(draw, (160, 45), "predicted")
            _draw_text(draw, (12, 188), "gold")
            for col_index, pred in enumerate(ALLOWED_STATES):
                _draw_text(draw, (120 + col_index * cell, 70), pred)
            for row_index, gold in enumerate(ALLOWED_STATES):
                _draw_text(draw, (20, 105 + row_index * cell), gold)
                for col_index, pred in enumerate(ALLOWED_STATES):
                    value = int(matrix[gold][pred])
                    intensity = int(245 - 170 * (value / max_cell))
                    color = (intensity, intensity + 5 if intensity < 250 else 250, 255)
                    left = 120 + col_index * cell
                    top = 100 + row_index * cell
                    draw.rectangle((left, top, left + cell - 5, top + cell - 5), fill=color, outline="#404040")
                    _draw_text(draw, (left + 36, top + 35), str(value))
            path = chart_dir / f"confusion_{fact_id}_{model_label}.png"
            image.save(path)
            generated.append(str(path))

    return generated


def _write_report(
    path: Path,
    *,
    reviewed_jsonl: str,
    base_model: str,
    adapter: str,
    benchmark_kind: str,
    metrics_by_model: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        "# Qwen3.5 VLM Fact Benchmark",
        "",
        f"- benchmark_kind: `{benchmark_kind}`",
        f"- dataset: `{reviewed_jsonl}`",
        f"- base_model: `{base_model}`",
        f"- adapter: `{adapter or 'none'}`",
        "",
        "## Overall Metrics",
        "",
        "| model | json_valid_rate | schema_valid_rate | fact_accuracy | macro_f1 | seen_f1 | sample_exact_match | critical_false_positive_count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_label, metrics in metrics_by_model.items():
        lines.append(
            "| {model} | {json_valid_rate} | {schema_valid_rate} | {fact_accuracy} | {macro_f1} | {seen_f1} | {sample_exact_match} | {critical_false_positive_count} |".format(
                model=model_label,
                **metrics,
            )
        )
    lines.extend(["", "## Fact Scores", ""])
    for model_label, metrics in metrics_by_model.items():
        lines.extend([
            f"### {model_label}",
            "",
            "| fact_id | accuracy | macro_f1 | seen_precision | seen_recall | seen_f1 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for fact_id in CORE_FACT_IDS:
            score = metrics["fact_scores"][fact_id]
            seen = score["state_scores"]["seen"]
            lines.append(
                f"| {fact_id} | {score['accuracy']} | {score['macro_f1']} | {seen['precision']} | {seen['recall']} | {seen['f1']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _compare_metrics(metrics_by_model: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if "base" not in metrics_by_model or "lora" not in metrics_by_model:
        return {"schema_version": "v1", "has_comparison": False}
    base = metrics_by_model["base"]
    lora = metrics_by_model["lora"]
    return {
        "schema_version": "v1",
        "has_comparison": True,
        "fact_accuracy_delta": round(float(lora["fact_accuracy"]) - float(base["fact_accuracy"]), 6),
        "macro_f1_delta": round(float(lora["macro_f1"]) - float(base["macro_f1"]), 6),
        "seen_f1_delta": round(float(lora["seen_f1"]) - float(base["seen_f1"]), 6),
        "sample_exact_match_delta": round(
            float(lora["sample_exact_match"]) - float(base["sample_exact_match"]),
            6,
        ),
        "critical_false_positive_delta": int(lora["critical_false_positive_count"])
        - int(base["critical_false_positive_count"]),
        "recommend_simtutor_chain_validation": (
            float(lora["json_valid_rate"]) >= 0.99
            and float(lora["fact_accuracy"]) > float(base["fact_accuracy"])
            and float(lora["seen_f1"]) > float(base["seen_f1"])
            and int(lora["critical_false_positive_count"]) <= int(base["critical_false_positive_count"])
        ),
    }


def _run_predictions(
    *,
    model_label: str,
    base_model: str,
    adapter: str | None,
    samples: Sequence[ReviewedBenchmarkSample],
    lang: str,
    max_seq_length: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    local_files_only: bool,
) -> list[PredictionRecord]:
    model, processor = _load_model(
        base_model=base_model,
        adapter=adapter,
        max_seq_length=max_seq_length,
        seed=seed,
        local_files_only=local_files_only,
    )
    records: list[PredictionRecord] = []
    for index, sample in enumerate(samples, start=1):
        try:
            raw_text = _model_generate(
                model=model,
                processor=processor,
                sample=sample,
                lang=lang,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            record = _normalize_prediction(raw_text, sample_id=sample.sample_id)
        except Exception as exc:
            record = PredictionRecord(
                sample_id=sample.sample_id,
                raw_text="",
                facts={fact_id: "uncertain" for fact_id in CORE_FACT_IDS},
                json_valid=False,
                schema_valid=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(record)
        print(
            "[BENCHMARK_QWEN35_VLM_FACTS] item="
            + json.dumps(
                {
                    "model": model_label,
                    "index": index,
                    "total": len(samples),
                    "sample_id": sample.sample_id,
                    "json_valid": record.json_valid,
                    "schema_valid": record.schema_valid,
                    "error": record.error,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return records


def benchmark_qwen35_vlm_facts(
    *,
    reviewed_jsonl: str | Path,
    base_model: str,
    adapter: str,
    output_dir: str | Path,
    lang: str,
    benchmark_kind: str,
    max_samples: int,
    max_seq_length: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    local_files_only: bool,
    skip_base: bool,
    skip_lora: bool,
) -> dict[str, Any]:
    resolved_reviewed = Path(reviewed_jsonl).expanduser().resolve()
    resolved_output = Path(output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    samples = _load_reviewed(resolved_reviewed, max_samples=max_samples)
    if not samples:
        raise ValueError("No benchmark samples loaded")

    predictions_by_model: dict[str, list[PredictionRecord]] = {}
    metrics_by_model: dict[str, dict[str, Any]] = {}
    errors_by_model: dict[str, list[dict[str, Any]]] = {}

    if not skip_base:
        base_records = _run_predictions(
            model_label="base",
            base_model=base_model,
            adapter=None,
            samples=samples,
            lang=lang,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            local_files_only=local_files_only,
        )
        predictions_by_model["base"] = base_records
        base_metrics, base_errors = compute_metrics(
            samples=samples,
            predictions={item.sample_id: item for item in base_records},
            model_label="base",
        )
        metrics_by_model["base"] = base_metrics
        errors_by_model["base"] = base_errors
        _release_cuda_memory()

    if adapter and not skip_lora:
        lora_records = _run_predictions(
            model_label="lora",
            base_model=base_model,
            adapter=adapter,
            samples=samples,
            lang=lang,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            local_files_only=local_files_only,
        )
        predictions_by_model["lora"] = lora_records
        lora_metrics, lora_errors = compute_metrics(
            samples=samples,
            predictions={item.sample_id: item for item in lora_records},
            model_label="lora",
        )
        metrics_by_model["lora"] = lora_metrics
        errors_by_model["lora"] = lora_errors
        _release_cuda_memory()

    for model_label, records in predictions_by_model.items():
        _write_jsonl(resolved_output / f"predictions_{model_label}.jsonl", [item.to_dict() for item in records])
        _write_json(resolved_output / f"metrics_{model_label}.json", metrics_by_model[model_label])
        _write_jsonl(resolved_output / f"errors_{model_label}.jsonl", errors_by_model[model_label])

    comparison = {
        **_compare_metrics(metrics_by_model),
        "benchmark_kind": benchmark_kind,
        "reviewed_jsonl": str(resolved_reviewed),
        "base_model": base_model,
        "adapter": adapter,
        "sample_count": len(samples),
        "lang": lang,
    }
    _write_json(resolved_output / "comparison.json", comparison)
    _write_fact_scores_csv(resolved_output / "fact_scores.csv", metrics_by_model)
    charts = _make_charts(resolved_output, metrics_by_model)
    _write_report(
        resolved_output / "report.md",
        reviewed_jsonl=str(resolved_reviewed),
        base_model=base_model,
        adapter=adapter,
        benchmark_kind=benchmark_kind,
        metrics_by_model=metrics_by_model,
    )
    summary = {
        "output_dir": str(resolved_output),
        "comparison": comparison,
        "charts": charts,
        "models": sorted(metrics_by_model.keys()),
    }
    _write_json(resolved_output / "benchmark_summary.json", summary)
    return summary


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = benchmark_qwen35_vlm_facts(
        reviewed_jsonl=args.reviewed_jsonl,
        base_model=args.base_model,
        adapter=args.adapter,
        output_dir=args.output_dir,
        lang=args.lang,
        benchmark_kind=args.benchmark_kind,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        local_files_only=args.local_files_only,
        skip_base=args.skip_base,
        skip_lora=args.skip_lora,
    )
    print(
        "[BENCHMARK_QWEN35_VLM_FACTS] summary="
        + json.dumps(summary, ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
