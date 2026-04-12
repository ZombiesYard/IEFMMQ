"""
Generate single-image VLM prelabels for composite-panel artifacts and export
Label Studio review tasks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from adapters.json_extract import parse_first_json
from adapters.openai_compat_multimodal import (
    copy_messages_for_payload,
    extract_response_error_text,
    frame_to_data_url,
    normalize_allowed_local_image_roots,
)
from adapters.vision_frames import DEFAULT_ARTIFACT_SUFFIX

DEFAULT_MODEL_NAME = "qwen3.5-397b-a17b"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_OUTPUT_DIRNAME = "prelabels"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_LOCAL_IMAGE_BYTES = 8 * 1024 * 1024

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
ALLOWED_STATES: frozenset[str] = frozenset({"seen", "not_seen", "uncertain"})
ALLOWED_RESPONSE_FIELDS: frozenset[str] = frozenset({"summary", "facts"})
ALLOWED_FACT_FIELDS: frozenset[str] = frozenset({"fact_id", "state", "evidence_note"})
FORBIDDEN_MODEL_FIELDS: frozenset[str] = frozenset(
    {
        "frame_id",
        "source_frame_id",
        "confidence",
        "expires_after_ms",
        "sticky",
    }
)


@dataclass(frozen=True)
class CaptureIndexEntry:
    frame_id: str
    session_id: str
    raw_image_path: Path
    artifact_image_path: Path
    layout_id: str
    channel: str
    capture_wall_ms: int | None
    capture_reason: str | None


@dataclass(frozen=True)
class NormalizedFact:
    fact_id: str
    state: str
    evidence_note: str

    def to_dict(self) -> dict[str, str]:
        return {
            "fact_id": self.fact_id,
            "state": self.state,
            "evidence_note": self.evidence_note,
        }


@dataclass(frozen=True)
class ParsedModelResponse:
    summary: str
    facts: tuple[NormalizedFact, ...]
    warnings: tuple[str, ...]
    json_repaired: bool
    repair_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactPrelabelResult:
    sample: dict[str, Any]
    task: dict[str, Any]
    raw_output_record: dict[str, Any]
    usage: dict[str, Any]


@dataclass(frozen=True)
class PrelabelRunStats:
    total_images: int
    succeeded: int
    failed: int
    output_dir: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "output_dir": self.output_dir,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ChatCompletionResult:
    raw_text: str
    usage: dict[str, Any]


def build_arg_parser(*, default_lang: str = "zh") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate DashScope VLM prelabels for composite-panel artifacts."
    )
    parser.add_argument(
        "--session-dir",
        required=True,
        help="Capture session directory that contains artifacts/ and capture_index.jsonl.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Artifact image directory. Defaults to <session-dir>/artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <session-dir>/prelabels.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="DashScope VLM model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Environment variable containing the API key.")
    parser.add_argument("--lang", choices=("zh", "en"), default=default_lang, help="Prompt language.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional max image count to process.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in the output directory.",
    )
    parser.add_argument(
        "--print-model-io",
        action="store_true",
        help="Print prompt and raw model reply blocks for each image.",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="Persist the raw model text in raw_model_outputs.jsonl.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="HTTP timeout in seconds.",
    )
    return parser


def _default_input_dir(session_dir: Path) -> Path:
    return session_dir / "artifacts"


def _default_output_dir(session_dir: Path) -> Path:
    return session_dir / DEFAULT_OUTPUT_DIRNAME


def _artifact_frame_id(artifact_path: Path) -> str:
    name = artifact_path.name
    if name.endswith(DEFAULT_ARTIFACT_SUFFIX):
        return name[: -len(DEFAULT_ARTIFACT_SUFFIX)]
    return artifact_path.stem


def _chat_completions_url(base_url: str) -> str:
    normalized = str(base_url).rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _is_qwen35_model(model_name: str) -> bool:
    return "qwen3.5" in str(model_name).lower()


def _is_dashscope_compatible(base_url: str) -> bool:
    normalized = str(base_url).lower()
    return (
        "dashscope.aliyuncs.com/compatible-mode" in normalized
        or "dashscope-intl.aliyuncs.com/compatible-mode" in normalized
        or "dashscope-us.aliyuncs.com/compatible-mode" in normalized
    )


def _build_prompt(*, lang: str) -> str:
    if lang == "en":
        return (
            "You are the SimTutor visual pre-annotator for the F/A-18C cold-start dataset.\n"
            "The input is exactly one composite-panel image.\n"
            "Its fixed top-to-bottom regions are: left_ddi, ampcd, right_ddi.\n"
            "\n"
            "Task:\n"
            "Inspect only this image and output initial visual fact labels for the 8 core facts below.\n"
            "If the image is blurry, obstructed, or the state cannot be confirmed from this image alone, use state='uncertain'.\n"
            "\n"
            "Facts to label:\n"
            "1. fcs_page_visible\n"
            "2. bit_root_page_visible\n"
            "3. bit_page_failure_visible\n"
            "4. right_ddi_fcsmc_page_visible\n"
            "5. right_ddi_in_test_visible\n"
            "6. fcs_bit_result_visible\n"
            "7. ins_alignment_page_visible\n"
            "8. ins_go\n"
            "\n"
            "Key decision boundaries:\n"
            "- fcs_page_visible means a real FCS page is visible. Seeing only an FCS option/button is not enough.\n"
            "- bit_root_page_visible, bit_page_failure_visible, and right_ddi_fcsmc_page_visible must be distinguished.\n"
            "- right_ddi_in_test_visible means the right DDI is clearly in an IN TEST state, not just a generic BIT context.\n"
            "- fcs_bit_result_visible must only be 'seen' when the final FCS BIT result is clearly visible. Intermediate or PBIT GO cues are not enough.\n"
            "- ins_alignment_page_visible and ins_go must be distinguished.\n"
            "- If AMPCD shows a MAP layer or a non-alignment page, do not hallucinate ins_go.\n"
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
            "- Include all 8 facts. When unsure, use state='uncertain'.\n"
            "\n"
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
        "你是 SimTutor 的视觉初标器，负责给 F/A-18C 冷启动数据集做初步视觉标注。\n"
        "输入只有一张组合面板图。\n"
        "这张图内部的固定区域从上到下依次是：left_ddi、ampcd、right_ddi。\n"
        "\n"
        "任务：\n"
        "只根据当前这一张图，为下面 8 个核心视觉 fact 输出初步标注。\n"
        "如果当前图模糊、遮挡、页面不完整、或者仅凭这一张图无法确认，请使用 state='uncertain'。\n"
        "\n"
        "需要标注的 fact：\n"
        "1. fcs_page_visible\n"
        "2. bit_root_page_visible\n"
        "3. bit_page_failure_visible\n"
        "4. right_ddi_fcsmc_page_visible\n"
        "5. right_ddi_in_test_visible\n"
        "6. fcs_bit_result_visible\n"
        "7. ins_alignment_page_visible\n"
        "8. ins_go\n"
        "\n"
        "关键判别边界：\n"
        "- fcs_page_visible 只有在真正显示 FCS 页面时才算 seen；仅看到 FCS 选项或按钮不算。\n"
        "- bit_root_page_visible、bit_page_failure_visible、right_ddi_fcsmc_page_visible 必须区分开。\n"
        "- right_ddi_in_test_visible 只有在右 DDI 明确处于 IN TEST 状态时才算 seen。\n"
        "- fcs_bit_result_visible 只有在最终 FCS BIT 结果明确可见时才算 seen；中间态或 PBIT GO 不能算。\n"
        "- ins_alignment_page_visible 和 ins_go 必须区分开。\n"
        "- 如果 AMPCD 显示 MAP 图层或其它非对准页面，不要臆测 ins_go=seen。\n"
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
        "- 8 个 fact 都要给出条目；看不清就用 state='uncertain'。\n"
        "\n"
        "返回格式：\n"
        "{\n"
        '  "summary": "一句短总结",\n'
        '  "facts": [\n'
        '    {"fact_id": "fcs_page_visible", "state": "seen", "evidence_note": "简短证据"},\n'
        '    {"fact_id": "bit_root_page_visible", "state": "not_seen", "evidence_note": "简短证据"}\n'
        "  ]\n"
        "}\n"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} must be a JSON object")
        rows.append(payload)
    return rows


def _load_capture_index(session_dir: Path) -> dict[str, CaptureIndexEntry]:
    capture_index_path = session_dir / "capture_index.jsonl"
    rows = _read_jsonl(capture_index_path)
    by_frame_id: dict[str, CaptureIndexEntry] = {}
    for row in rows:
        frame_id = row.get("frame_id")
        session_id = row.get("session_id")
        raw_image_path = row.get("raw_image_path")
        artifact_image_path = row.get("artifact_image_path")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError(f"capture index entry missing frame_id: {row!r}")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"capture index entry missing session_id for frame {frame_id}")
        if not isinstance(raw_image_path, str) or not raw_image_path:
            raise ValueError(f"capture index entry missing raw_image_path for frame {frame_id}")
        if not isinstance(artifact_image_path, str) or not artifact_image_path:
            raise ValueError(f"capture index entry missing artifact_image_path for frame {frame_id}")
        by_frame_id[frame_id] = CaptureIndexEntry(
            frame_id=frame_id,
            session_id=session_id,
            raw_image_path=Path(raw_image_path).expanduser().resolve(),
            artifact_image_path=Path(artifact_image_path).expanduser().resolve(),
            layout_id=str(row.get("layout_id") or ""),
            channel=str(row.get("channel") or ""),
            capture_wall_ms=row.get("capture_wall_ms") if isinstance(row.get("capture_wall_ms"), int) else None,
            capture_reason=str(row.get("capture_reason")) if row.get("capture_reason") is not None else None,
        )
    return by_frame_id


def _list_artifact_paths(input_dir: Path, *, max_images: int) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"artifact directory not found: {input_dir}")
    paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    if max_images > 0:
        return paths[:max_images]
    return paths


def _ensure_overwrite_safe(output_dir: Path, *, overwrite: bool) -> None:
    targets = (
        output_dir / "vision_prelabels.jsonl",
        output_dir / "label_studio_tasks.json",
        output_dir / "raw_model_outputs.jsonl",
        output_dir / "prelabels_failures.jsonl",
    )
    if overwrite:
        return
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "output files already exist; pass --overwrite to replace them: " + ", ".join(existing)
        )


def _sample_id(entry: CaptureIndexEntry) -> str:
    return f"{entry.session_id}:{entry.frame_id}:vlm_prelabel"


def _normalized_payload(summary: str, facts: Sequence[NormalizedFact]) -> dict[str, Any]:
    return {
        "summary": summary,
        "facts": [fact.to_dict() for fact in facts],
    }


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _normalize_usage(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _coerce_non_negative_int(payload.get(key))
        if value is not None:
            normalized[key] = value

    for key in ("prompt_tokens_details", "completion_tokens_details"):
        details = payload.get(key)
        if isinstance(details, Mapping):
            normalized[key] = {
                str(detail_key): detail_value
                for detail_key, detail_value in details.items()
            }
    return normalized


def _normalize_model_response(raw_text: str) -> ParsedModelResponse:
    obj, extraction = parse_first_json(raw_text)
    if not isinstance(obj, Mapping):
        raise ValueError("model response must be a JSON object")

    warnings: list[str] = []
    unexpected_top_level = sorted(str(key) for key in obj.keys() if key not in ALLOWED_RESPONSE_FIELDS)
    if unexpected_top_level:
        warnings.append("stripped_top_level_fields:" + ",".join(unexpected_top_level))

    summary = str(obj.get("summary")).strip() if isinstance(obj.get("summary"), str) else ""
    facts_raw = obj.get("facts")
    if not isinstance(facts_raw, list):
        raise ValueError("model response must contain a facts array")

    facts_by_id: dict[str, NormalizedFact] = {}
    for index, item in enumerate(facts_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"facts[{index}] must be a JSON object")

        fact_id = item.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id:
            raise ValueError(f"facts[{index}] is missing fact_id")
        if fact_id not in CORE_FACT_IDS:
            raise ValueError(f"unsupported fact_id returned by model: {fact_id}")
        if fact_id in facts_by_id:
            warnings.append(f"duplicate_fact_id_ignored:{fact_id}")
            continue

        state = item.get("state")
        if not isinstance(state, str) or state not in ALLOWED_STATES:
            raise ValueError(f"unsupported state for {fact_id}: {state!r}")

        evidence_note_raw = item.get("evidence_note")
        evidence_note = str(evidence_note_raw).strip() if evidence_note_raw is not None else ""

        stripped_fields = sorted(str(key) for key in item.keys() if key not in ALLOWED_FACT_FIELDS)
        if stripped_fields:
            warnings.append(f"stripped_fact_fields:{fact_id}:{','.join(stripped_fields)}")

        facts_by_id[fact_id] = NormalizedFact(
            fact_id=fact_id,
            state=state,
            evidence_note=evidence_note,
        )

    normalized_facts: list[NormalizedFact] = []
    for fact_id in CORE_FACT_IDS:
        fact = facts_by_id.get(fact_id)
        if fact is None:
            fact = NormalizedFact(
                fact_id=fact_id,
                state="uncertain",
                evidence_note="Model omitted this fact; defaulted to uncertain.",
            )
            warnings.append(f"defaulted_uncertain:{fact_id}")
        normalized_facts.append(fact)

    return ParsedModelResponse(
        summary=summary,
        facts=tuple(normalized_facts),
        warnings=tuple(warnings),
        json_repaired=bool(extraction.json_repaired),
        repair_reasons=tuple(extraction.repair_reasons),
    )


def _build_label_studio_task(
    *,
    sample: Mapping[str, Any],
    image_data_url: str,
) -> dict[str, Any]:
    facts = sample.get("facts")
    if not isinstance(facts, list):
        raise ValueError("sample facts must be a list")
    data: dict[str, Any] = {
        "image": image_data_url,
        "frame_id": sample.get("frame_id"),
        "session_id": sample.get("session_id"),
        "artifact_image_path": sample.get("artifact_image_path"),
        "raw_image_path": sample.get("raw_image_path"),
        "summary": sample.get("summary") or "",
        "ai_prelabel_json": json.dumps(
            {
                "summary": sample.get("summary") or "",
                "facts": facts,
            },
            ensure_ascii=False,
            indent=2,
        ),
    }
    for item in facts:
        if not isinstance(item, Mapping):
            continue
        fact_id = item.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id:
            continue
        data[fact_id] = item.get("state")
        data[f"{fact_id}_note"] = item.get("evidence_note") or ""

    return {
        "data": data,
        "meta": {
            "sample_id": sample.get("sample_id"),
            "frame_id": sample.get("frame_id"),
            "session_id": sample.get("session_id"),
        },
    }


class DashScopeVlmPrelabeler:
    _DEFAULT_MAX_TOKENS = 640

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        api_key: str | None = None,
        allowed_local_image_roots: Sequence[str | Path] | None = None,
        max_local_image_bytes: int = DEFAULT_MAX_LOCAL_IMAGE_BYTES,
        lang: str = "zh",
        client: object | None = None,
        print_model_io: bool = False,
        save_raw_response: bool = False,
    ) -> None:
        self.model_name = str(model_name).strip()
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.api_key = api_key
        self.allowed_local_image_roots = normalize_allowed_local_image_roots(allowed_local_image_roots)
        self.max_local_image_bytes = int(max_local_image_bytes)
        self.lang = lang if lang in {"zh", "en"} else "zh"
        self.print_model_io = bool(print_model_io)
        self.save_raw_response = bool(save_raw_response)
        self.prompt = _build_prompt(lang=self.lang)
        if client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout_s)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def prelabel_artifact(
        self,
        artifact_path: Path,
        *,
        record: CaptureIndexEntry,
    ) -> ArtifactPrelabelResult:
        image_data_url = self._artifact_to_data_url(artifact_path, frame_id=record.frame_id)
        messages = [
            {
                "role": "system",
                "content": "You are SimTutor visual pre-annotator. Reply with JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]
        if self.print_model_io:
            self._print_model_io_block(
                "VLM_PRELABEL_PROMPT",
                self._render_debug_messages(messages),
                frame_id=record.frame_id,
                session_id=record.session_id,
            )
        chat_result = self._chat(messages)
        raw_text = chat_result.raw_text
        if self.print_model_io:
            self._print_model_io_block(
                "VLM_PRELABEL_REPLY",
                raw_text,
                frame_id=record.frame_id,
                session_id=record.session_id,
            )
        parsed = _normalize_model_response(raw_text)
        payload = _normalized_payload(parsed.summary, parsed.facts)
        sample = {
            "schema_version": "v1",
            "sample_id": _sample_id(record),
            "session_id": record.session_id,
            "frame_id": record.frame_id,
            "artifact_image_path": str(Path(artifact_path).resolve()),
            "raw_image_path": str(record.raw_image_path),
            "layout_id": record.layout_id,
            "channel": record.channel,
            "model": {
                "provider": "dashscope_openai_compat",
                "model_name": self.model_name,
            },
            "summary": parsed.summary,
            "facts": payload["facts"],
        }
        raw_output_record = {
            "sample_id": sample["sample_id"],
            "session_id": record.session_id,
            "frame_id": record.frame_id,
            "artifact_image_path": str(Path(artifact_path).resolve()),
            "raw_image_path": str(record.raw_image_path),
            "model_name": self.model_name,
            "base_url": self.base_url,
            "prompt": self.prompt,
            "warnings": list(parsed.warnings),
            "json_repaired": parsed.json_repaired,
            "repair_reasons": list(parsed.repair_reasons),
            "usage": dict(chat_result.usage),
        }
        if self.save_raw_response:
            raw_output_record["response_text"] = raw_text
        else:
            raw_output_record["response_text_length"] = len(raw_text)
        return ArtifactPrelabelResult(
            sample=sample,
            task=_build_label_studio_task(sample=sample, image_data_url=image_data_url),
            raw_output_record=raw_output_record,
            usage=dict(chat_result.usage),
        )

    def _artifact_to_data_url(self, artifact_path: Path, *, frame_id: str) -> str:
        return frame_to_data_url(
            {
                "frame_id": frame_id,
                "image_uri": str(Path(artifact_path).resolve()),
                "mime_type": "image/png",
            },
            allowed_local_image_roots=self.allowed_local_image_roots,
            max_local_image_bytes=self.max_local_image_bytes,
        )

    def _build_request_payload(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": copy_messages_for_payload(messages),
            "temperature": 0,
            "top_p": 0.1,
            "response_format": {"type": "json_object"},
        }
        if not (_is_dashscope_compatible(self.base_url) and _is_qwen35_model(self.model_name)):
            payload["max_tokens"] = self._DEFAULT_MAX_TOKENS
        if _is_qwen35_model(self.model_name):
            if _is_dashscope_compatible(self.base_url):
                payload["enable_thinking"] = False
            else:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _chat(self, messages: Sequence[Mapping[str, Any]]) -> ChatCompletionResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self._build_request_payload(messages)
        response = self._client.post(
            _chat_completions_url(self.base_url),
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        )
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and status_code >= 400:
            raise RuntimeError(extract_response_error_text(response) or f"http {status_code}")
        body = response.json()
        if not isinstance(body, Mapping):
            raise ValueError("OpenAI-compatible response must be a JSON object")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI-compatible response missing choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError("OpenAI-compatible choice must be an object")
        message = first.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise ValueError("OpenAI-compatible response missing choices[0].message.content")
        return ChatCompletionResult(
            raw_text=str(message["content"]),
            usage=_normalize_usage(body.get("usage")),
        )

    @staticmethod
    def _render_debug_messages(messages: Sequence[Mapping[str, Any]]) -> str:
        rendered: list[str] = []
        for message in messages:
            role = message.get("role")
            role_text = role if isinstance(role, str) and role else "unknown"
            rendered.append(f"[{role_text}]")
            content = message.get("content")
            if isinstance(content, str):
                rendered.append(content)
                continue
            if isinstance(content, list):
                image_count = 0
                for item in content:
                    if not isinstance(item, Mapping):
                        rendered.append(str(item))
                        continue
                    item_type = item.get("type")
                    if item_type == "image_url":
                        image_count += 1
                        continue
                    if item_type == "text" and isinstance(item.get("text"), str):
                        rendered.append(str(item["text"]))
                        continue
                    rendered.append(json.dumps(dict(item), ensure_ascii=False, sort_keys=True))
                if image_count > 0:
                    rendered.append(f"[multimodal_images={image_count}]")
                continue
            rendered.append(str(content))
        return "\n".join(rendered)

    @staticmethod
    def _print_model_io_block(
        kind: str,
        text: str,
        *,
        frame_id: str,
        session_id: str,
    ) -> None:
        header = f"[MODEL_IO][{kind}][session_id={session_id}][frame_id={frame_id}]"
        print(header)
        print(text)
        print(f"{header}[END]")


def generate_prelabels(
    *,
    session_dir: str | Path,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    base_url: str = DEFAULT_BASE_URL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    lang: str = "zh",
    max_images: int = 0,
    overwrite: bool = False,
    print_model_io: bool = False,
    save_raw_response: bool = False,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    client: object | None = None,
) -> PrelabelRunStats:
    resolved_session_dir = Path(session_dir).expanduser().resolve()
    resolved_input_dir = (
        Path(input_dir).expanduser().resolve()
        if input_dir is not None
        else _default_input_dir(resolved_session_dir)
    )
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _default_output_dir(resolved_session_dir)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_overwrite_safe(resolved_output_dir, overwrite=overwrite)

    api_key = os.getenv(api_key_env)
    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(f"environment variable {api_key_env} is not set")

    capture_index = _load_capture_index(resolved_session_dir)
    artifact_paths = _list_artifact_paths(resolved_input_dir, max_images=max_images)
    prelabeler = DashScopeVlmPrelabeler(
        model_name=model_name,
        base_url=base_url,
        timeout_s=timeout_s,
        api_key=api_key,
        allowed_local_image_roots=[resolved_session_dir, resolved_input_dir],
        lang=lang,
        client=client,
        print_model_io=print_model_io,
        save_raw_response=save_raw_response,
    )

    successful_samples: list[dict[str, Any]] = []
    label_studio_tasks: list[dict[str, Any]] = []
    raw_output_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    try:
        for artifact_path in artifact_paths:
            frame_id = _artifact_frame_id(artifact_path)
            record = capture_index.get(frame_id)
            if record is None:
                error_text = "capture_index_entry_missing"
                failures.append(
                    {
                        "frame_id": frame_id,
                        "artifact_image_path": str(artifact_path.resolve()),
                        "error": error_text,
                    }
                )
                _print_item_failure(frame_id=frame_id, session_id=None, error=error_text)
                continue
            try:
                result = prelabeler.prelabel_artifact(artifact_path, record=record)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                failures.append(
                    {
                        "frame_id": frame_id,
                        "session_id": record.session_id,
                        "artifact_image_path": str(artifact_path.resolve()),
                        "raw_image_path": str(record.raw_image_path),
                        "error": error_text,
                    }
                )
                _print_item_failure(frame_id=frame_id, session_id=record.session_id, error=error_text)
                continue
            successful_samples.append(result.sample)
            label_studio_tasks.append(result.task)
            raw_output_records.append(result.raw_output_record)
            total_prompt_tokens += int(result.usage.get("prompt_tokens") or 0)
            total_completion_tokens += int(result.usage.get("completion_tokens") or 0)
            total_tokens += int(result.usage.get("total_tokens") or 0)
            _print_item_success(sample=result.sample, usage=result.usage)
    finally:
        prelabeler.close()

    _write_jsonl(resolved_output_dir / "vision_prelabels.jsonl", successful_samples)
    _write_jsonl(resolved_output_dir / "raw_model_outputs.jsonl", raw_output_records)
    _write_jsonl(resolved_output_dir / "prelabels_failures.jsonl", failures)
    (resolved_output_dir / "label_studio_tasks.json").write_text(
        json.dumps(label_studio_tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return PrelabelRunStats(
        total_images=len(artifact_paths),
        succeeded=len(successful_samples),
        failed=len(failures),
        output_dir=str(resolved_output_dir),
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _print_item_success(*, sample: Mapping[str, Any], usage: Mapping[str, Any]) -> None:
    print(
        "[GENERATE_VLM_PRELABELS] item="
        + json.dumps(
            {
                "status": "ok",
                "session_id": sample.get("session_id"),
                "frame_id": sample.get("frame_id"),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _print_item_failure(*, frame_id: str, session_id: str | None, error: str) -> None:
    print(
        "[GENERATE_VLM_PRELABELS] item="
        + json.dumps(
            {
                "status": "failed",
                "session_id": session_id,
                "frame_id": frame_id,
                "error": error,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def run_cli(*, default_lang: str = "zh", argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser(default_lang=default_lang).parse_args(argv)
    try:
        stats = generate_prelabels(
            session_dir=args.session_dir,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            lang=args.lang,
            max_images=args.max_images,
            overwrite=args.overwrite,
            print_model_io=args.print_model_io,
            save_raw_response=args.save_raw_response,
            timeout_s=args.timeout_s,
        )
    except Exception as exc:
        print(f"[GENERATE_VLM_PRELABELS] failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"[GENERATE_VLM_PRELABELS] stats={json.dumps(stats.to_dict(), ensure_ascii=False, sort_keys=True)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(default_lang="zh", argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
