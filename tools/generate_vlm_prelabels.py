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
LABEL_STUDIO_TO_NAME = "panel_image"
LABEL_STUDIO_PREDICTION_SCORE = 0.95


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
        "--capture-plan-seqs",
        default="",
        help=(
            "Optional comma-separated capture-plan seqs or ranges to process, "
            "for example '1,11,49-51'. Requires capture_plan_progress.jsonl."
        ),
    )
    parser.add_argument(
        "--include-capture-plan-hint",
        action="store_true",
        help=(
            "Append a compact non-authoritative capture-plan hint to each model prompt. "
            "Useful for prelabel review, but the exported training target should still be visual-only."
        ),
    )
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
            "You are a strict visual pre-annotator for F/A-18C cockpit screenshots.\n"
            "Input: one composite image with fixed regions from top to bottom: left_ddi, ampcd, right_ddi.\n"
            "Use only visible evidence in this single image. Do not infer from procedure history.\n"
            "States: seen = clearly visible; not_seen = absent; uncertain = too blurry/obscured/partial to trust.\n"
            "\n"
            "Return all 13 facts:\n"
            "tac_page_visible, supt_page_visible, fcs_page_visible, fcs_page_x_marks_visible,\n"
            "bit_root_page_visible, fcsmc_page_visible, fcsmc_intermediate_result_visible,\n"
            "fcsmc_in_test_visible, fcsmc_final_go_result_visible, hsi_page_visible,\n"
            "hsi_map_layer_visible, ins_grnd_alignment_text_visible, ins_ok_text_visible.\n"
            "\n"
            "Decision boundaries:\n"
            "- Page option labels are not pages. Do not mark a page visible just because its name appears as a pushbutton/menu option.\n"
            "- tac_page_visible requires the actual TAC/TAC MENU page, usually a boxed TAC MENU plus multiple menu items such as STORES, RDR ATTK, HUD, FCS, EW. A small TAC/MENU navigation label, GO/NOGO table, UFC/MSNCDR page, or generic bottom MENU label is not enough.\n"
            "- supt_page_visible requires the actual SUPT/SUPT MENU page. A small SUPT option label alone is not enough. TAC and SUPT pages can contain an FCS option label.\n"
            "- An FCS option/button label on TAC/SUPT is NOT the FCS page.\n"
            "- fcs_page_visible is the dedicated flight-control FCS page with readable channel/control labels such as LEF, TEF, AIL, RUD, STAB, SV1/SV2, and CAS/P/Y/R grid. Do not invent these labels. Do not count pages that only show FLAPS OFF, AIL OFF, CANOPY, FCS option text, INS ATT, INS DEGD, ADV-BIT, circular radar/HUD/SAFE/GUN/FUEL/EPE/DATA symbology, TAC/SUPT menus, or the FCS-MC BIT page.\n"
            "- fcs_page_x_marks_visible is seen only when literal diagonal/cross X fault fills are visible inside FCS page channel boxes. Empty channel boxes, grid lines, numbers, INS ATT/INS DEGD text, caution text, and boxed menu labels are not X marks.\n"
            "- bit_root_page_visible is the BIT FAILURES/root page; seeing an FCS-MC entry label is not FCS-MC page.\n"
            "- fcsmc_page_visible requires a readable FCS-MC title/subpage with rows such as MC1, MC2, FCSA, and FCSB. Do not infer FCS-MC from blank displays, SAFE/GUN/HUD symbology, BIT root summaries, or unreadable transition pages.\n"
            "- FCS-MC result facts require the FCS-MC subpage. A BIT root summary row such as FCS-MC GO is not an FCS-MC subpage result.\n"
            "- fcsmc_intermediate_result_visible means clear PBIT GO/pre-test result text on the FCS-MC subpage. MC1/MC2 GO alone is not enough; IN TEST is not intermediate_result.\n"
            "- IN TEST is running: fcsmc_in_test_visible=seen, final_go=not_seen. If FCSA/FCSB show IN TEST, do not mark final GO even if MC1/MC2 show GO.\n"
            "- fcsmc_final_go_result_visible requires clear final GO results for the relevant FCS-MC rows, especially FCSA and FCSB. Final examples look like MC1 GO, MC2 GO, FCSA GO, FCSB GO with no PBIT or IN TEST on FCSA/FCSB. Do not mark final GO for PBIT GO, IN TEST, MC1/MC2-only GO, or BIT root summary GO.\n"
            "- hsi_page_visible means an HSI navigation/POS page is visible on any display. This alone is not evidence that INS alignment text or INS OK is visible.\n"
            "- hsi_map_layer_visible means a colored topographic/chart MAP background is visible. Black HSI symbology, compass roses, waypoint marks, or text-only HSI pages are not a MAP layer.\n"
            "- ins_grnd_alignment_text_visible requires a literal GRND alignment block with QUAL or TIME/countdown text. Do not infer it from INS ATT/INS DEGD, POS/INS, coordinates, or map-only HSI pages.\n"
            "- ins_ok_text_visible requires clearly readable OK near the INS alignment block, usually near or after QUAL such as 'QUAL 0.5 OK'. On HSI with MAP overlay, inspect the center-lower GRND/QUAL/TIME block; MAP clutter or similar marks are not OK.\n"
            "- If no target page is visible, mark target facts not_seen. If text is unreadable, use uncertain.\n"
            "\n"
            "Output JSON only. Top level may contain only summary and facts. Each fact may contain only fact_id, state, evidence_note.\n"
            "Never output frame_id. Never output source_frame_id. Never output confidence.\n"
            "Never output expires_after_ms, sticky, actions, or tutor advice.\n"
        )
    return (
        "你是 F/A-18C cockpit 截图的严格视觉初标器。\n"
        "输入是一张组合图，固定区域从上到下为：left_ddi、ampcd、right_ddi。\n"
        "只看当前单帧图像，不根据流程历史推断。\n"
        "状态：seen=清楚可见；not_seen=不存在；uncertain=模糊/遮挡/过渡/局部可见而无法可靠判断。\n"
        "\n"
        "必须返回 13 个 facts：\n"
        "tac_page_visible, supt_page_visible, fcs_page_visible, fcs_page_x_marks_visible,\n"
        "bit_root_page_visible, fcsmc_page_visible, fcsmc_intermediate_result_visible,\n"
        "fcsmc_in_test_visible, fcsmc_final_go_result_visible, hsi_page_visible,\n"
        "hsi_map_layer_visible, ins_grnd_alignment_text_visible, ins_ok_text_visible。\n"
        "\n"
        "判别边界：\n"
        "- 页面选项标签不等于页面本身。不要因为某个 pushbutton/menu option 写着页面名，就把该页面标为可见。\n"
        "- tac_page_visible 需要实际 TAC/TAC MENU 页面，通常有 boxed TAC MENU 和 STORES、RDR ATTK、HUD、FCS、EW 等多个菜单项；单独一个小的 TAC/MENU 导航标签、GO/NOGO 表、UFC/MSNCDR 页面或普通底部 MENU 标签不够。\n"
        "- supt_page_visible 需要实际 SUPT/SUPT MENU 页面；单独一个小的 SUPT 选项标签不够。TAC/SUPT 页面上可能有 FCS 选项标签。\n"
        "- TAC/SUPT 上的 FCS 选项/按钮标签不等于 FCS 页面。\n"
        "- fcs_page_visible 是专用飞控 FCS 页面，必须能读到 LEF、TEF、AIL、RUD、STAB、SV1/SV2、CAS/P/Y/R grid 等通道/控制标签；不要凭布局编造这些标签。不要把只显示 FLAPS OFF、AIL OFF、CANOPY、FCS 选项文字、INS ATT、INS DEGD、ADV-BIT、圆形 radar/HUD/SAFE/GUN/FUEL/EPE/DATA 符号、TAC/SUPT 菜单或 FCS-MC BIT 页面当成 FCS 页面。\n"
        "- fcs_page_x_marks_visible 只有 FCS 页面通道格子内有字面斜线/交叉 X fault fills 才算 seen；空通道格、网格线、数字、INS ATT/INS DEGD、故障文字或 boxed menu 标签都不是 X。\n"
        "- bit_root_page_visible 是 BIT FAILURES/root 页面；FCS-MC 入口标签不等于 FCS-MC 页面。\n"
        "- fcsmc_page_visible 需要能读到 FCS-MC 标题/子页面，并有 MC1、MC2、FCSA、FCSB 等行；不要从空白屏、SAFE/GUN/HUD 符号、BIT root 汇总或不可读过渡页推断 FCS-MC。\n"
        "- FCS-MC 结果类 fact 必须来自 FCS-MC 子页面；BIT root 汇总行里的 FCS-MC GO 不算 FCS-MC 子页面结果。\n"
        "- fcsmc_intermediate_result_visible 指 FCS-MC 子页面上清楚的 PBIT GO/预检结果文字；只有 MC1/MC2 GO 不够，IN TEST 也不是 intermediate_result。\n"
        "- IN TEST 是运行中：fcsmc_in_test_visible=seen，final_go=not_seen。如果 FCSA/FCSB 显示 IN TEST，即使 MC1/MC2 是 GO 也不要标 final GO。\n"
        "- fcsmc_final_go_result_visible 必须是相关 FCS-MC 行的最终 GO，尤其 FCSA 和 FCSB；典型最终态是 MC1 GO、MC2 GO、FCSA GO、FCSB GO，且 FCSA/FCSB 没有 PBIT 或 IN TEST。不要把 PBIT GO、IN TEST、只有 MC1/MC2 GO 或 BIT root 汇总 GO 当 final GO。\n"
        "- hsi_page_visible 是任意显示器上可见 HSI navigation/POS 页面；这本身不代表 INS 对准文字或 INS OK 可见。\n"
        "- hsi_map_layer_visible 是 HSI 上有彩色地形/航图 MAP 背景；黑底 HSI 符号、罗盘圈、航点标记或纯文字 HSI 页不算 MAP layer。\n"
        "- ins_grnd_alignment_text_visible 需要字面 GRND 对准文字块，并伴随 QUAL 或 TIME/countdown；不要从 INS ATT/INS DEGD、POS/INS、坐标或只有地图的 HSI 页推断。\n"
        "- ins_ok_text_visible 需要在 INS 对准文字块附近清楚读到 OK，通常在 QUAL 附近或后面，例如 QUAL 0.5 OK；HSI 有 MAP overlay 时也要检查中下部 GRND/QUAL/TIME 文字块，但地图干扰或相似符号不能猜 OK。\n"
        "- 没有目标页时标 not_seen；文字看不清时标 uncertain。\n"
        "\n"
        "只输出 JSON。顶层只允许 summary 和 facts。每个 fact 只允许 fact_id、state、evidence_note。\n"
        "严禁输出 frame_id。严禁输出 source_frame_id。严禁输出 confidence。\n"
        "严禁输出 expires_after_ms、sticky、动作建议或 tutor 建议。\n"
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


def _load_capture_plan_progress(session_dir: Path) -> dict[str, dict[str, Any]]:
    progress_path = session_dir / "capture_plan_progress.jsonl"
    if not progress_path.exists():
        return {}
    rows = _read_jsonl(progress_path)
    by_frame_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") not in (None, "", "captured"):
            continue
        frame_id = row.get("captured_frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        by_frame_id[frame_id] = dict(row)
    return by_frame_id


def _parse_capture_plan_seq_filter(raw_value: str | Sequence[int] | None) -> tuple[int, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        raw_items = [item.strip() for item in raw_value.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip() for item in raw_value if str(item).strip()]
    seqs: list[int] = []
    seen: set[int] = set()
    for raw_item in raw_items:
        if "-" in raw_item:
            start_text, end_text = raw_item.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(f"invalid capture plan seq range: {raw_item!r}") from exc
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"invalid capture plan seq range: {raw_item!r}")
            values = range(start, end + 1)
        else:
            try:
                value = int(raw_item)
            except ValueError as exc:
                raise ValueError(f"invalid capture plan seq: {raw_item!r}") from exc
            if value <= 0:
                raise ValueError(f"invalid capture plan seq: {raw_item!r}")
            values = (value,)
        for seq in values:
            if seq not in seen:
                seen.add(seq)
                seqs.append(seq)
    return tuple(seqs)


def _compact_plan_value(value: Any, *, max_chars: int = 140) -> str:
    text = str(value).strip() if value is not None else ""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_capture_plan_hint(plan_metadata: Mapping[str, Any] | None, *, lang: str) -> str:
    if not isinstance(plan_metadata, Mapping) or not plan_metadata:
        return ""
    fields = {
        "seq": _compact_plan_value(plan_metadata.get("seq"), max_chars=20),
        "category_id": _compact_plan_value(plan_metadata.get("category_id"), max_chars=80),
        "target_display": _compact_plan_value(plan_metadata.get("target_display"), max_chars=40),
        "expected_primary_content": _compact_plan_value(
            plan_metadata.get("expected_primary_content"), max_chars=120
        ),
        "expected_key_facts": _compact_plan_value(plan_metadata.get("expected_key_facts"), max_chars=180),
    }
    fields = {key: value for key, value in fields.items() if value}
    if not fields:
        return ""
    body = "; ".join(f"{key}={value}" for key, value in fields.items())
    if lang == "en":
        return (
            "\nCapture-plan hint for prelabel review only. It may be wrong; visual evidence wins. "
            f"Hint: {body}.\n"
        )
    return (
        "\n采集计划提示，仅用于人工复核前的初标。它可能有错，最终以图像可见证据为准。"
        f"提示：{body}。\n"
    )


def _ordered_artifact_paths(
    *,
    input_dir: Path,
    capture_index: Mapping[str, CaptureIndexEntry],
    progress_by_frame_id: Mapping[str, Mapping[str, Any]],
    max_images: int,
    capture_plan_seqs: Sequence[int] = (),
) -> list[Path]:
    seq_filter = tuple(capture_plan_seqs)
    seq_order = {seq: index for index, seq in enumerate(seq_filter)}
    if progress_by_frame_id:
        ordered_rows: list[tuple[str, Mapping[str, Any]]] = []
        for frame_id, row in progress_by_frame_id.items():
            try:
                seq = int(row.get("seq") or 0)
            except (TypeError, ValueError):
                continue
            if seq_filter and seq not in seq_order:
                continue
            if frame_id in capture_index:
                ordered_rows.append((frame_id, row))
        ordered_rows.sort(
            key=lambda item: seq_order[int(item[1].get("seq") or 0)]
            if seq_filter
            else int(item[1].get("seq") or 0)
        )
        ordered_frame_ids = [frame_id for frame_id, _row in ordered_rows]
        paths = []
        for frame_id in ordered_frame_ids:
            local_path = input_dir / f"{frame_id}{DEFAULT_ARTIFACT_SUFFIX}"
            paths.append(local_path if local_path.exists() else capture_index[frame_id].artifact_image_path)
    else:
        if seq_filter:
            raise ValueError("--capture-plan-seqs requires capture_plan_progress.jsonl")
        paths = _list_artifact_paths(input_dir, max_images=0)
    if max_images > 0:
        return paths[:max_images]
    return paths


def _ensure_overwrite_safe(output_dir: Path, *, overwrite: bool) -> None:
    targets = (
        output_dir / "vision_prelabels.jsonl",
        output_dir / "label_studio_tasks.json",
        output_dir / "label_studio_tasks_with_predictions.json",
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


def _build_label_studio_choice_result(field: str, value: str) -> dict[str, Any]:
    return {
        "from_name": f"{field}_review",
        "to_name": LABEL_STUDIO_TO_NAME,
        "type": "choices",
        "value": {"choices": [value]},
    }


def _build_label_studio_textarea_result(from_name: str, text: str) -> dict[str, Any]:
    return {
        "from_name": from_name,
        "to_name": LABEL_STUDIO_TO_NAME,
        "type": "textarea",
        "value": {"text": [text]},
    }


def _build_label_studio_prediction(sample: Mapping[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    summary = sample.get("summary")
    if isinstance(summary, str) and summary.strip():
        results.append(_build_label_studio_textarea_result("summary_review", summary.strip()))

    facts = sample.get("facts")
    if isinstance(facts, list):
        for item in facts:
            if not isinstance(item, Mapping):
                continue
            fact_id = item.get("fact_id")
            state = item.get("state")
            note = item.get("evidence_note")
            if isinstance(fact_id, str) and isinstance(state, str) and state in ALLOWED_STATES:
                results.append(_build_label_studio_choice_result(fact_id, state))
            if isinstance(fact_id, str) and isinstance(note, str) and note.strip():
                results.append(
                    _build_label_studio_textarea_result(f"{fact_id}_note_review", note.strip())
                )
    return {
        "model_version": str(sample.get("model", {}).get("model_name") or "vlm_prelabel"),
        "score": LABEL_STUDIO_PREDICTION_SCORE,
        "result": results,
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
    capture_plan = sample.get("capture_plan")
    if isinstance(capture_plan, Mapping):
        for key in (
            "seq",
            "total",
            "category_id",
            "category_name",
            "target_display",
            "left_ddi_content",
            "ampcd_content",
            "right_ddi_content",
            "expected_primary_content",
            "expected_key_facts",
            "capture_instruction",
        ):
            if key in capture_plan:
                data[f"capture_plan_{key}"] = capture_plan.get(key)
    for item in facts:
        if not isinstance(item, Mapping):
            continue
        fact_id = item.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id:
            continue
        data[fact_id] = item.get("state")
        data[f"{fact_id}_note"] = item.get("evidence_note") or ""

    task = {
        "data": data,
        "meta": {
            "sample_id": sample.get("sample_id"),
            "frame_id": sample.get("frame_id"),
            "session_id": sample.get("session_id"),
        },
    }
    if isinstance(capture_plan, Mapping):
        task["meta"]["capture_plan_seq"] = capture_plan.get("seq")
        task["meta"]["capture_plan_category_id"] = capture_plan.get("category_id")
    task["predictions"] = [_build_label_studio_prediction(sample)]
    return task


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
        include_capture_plan_hint: bool = False,
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
        self.include_capture_plan_hint = bool(include_capture_plan_hint)
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
        plan_metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactPrelabelResult:
        image_data_url = self._artifact_to_data_url(artifact_path, frame_id=record.frame_id)
        capture_plan_hint = (
            _build_capture_plan_hint(plan_metadata, lang=self.lang)
            if self.include_capture_plan_hint
            else ""
        )
        prompt_text = self.prompt + capture_plan_hint
        messages = [
            {
                "role": "system",
                "content": "You are SimTutor visual pre-annotator. Reply with JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt_text},
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
        if isinstance(plan_metadata, Mapping) and plan_metadata:
            sample["capture_plan"] = {
                str(key): value
                for key, value in plan_metadata.items()
                if key
                in {
                    "seq",
                    "total",
                    "category_id",
                    "category_name",
                    "target_display",
                    "left_ddi_content",
                    "ampcd_content",
                    "right_ddi_content",
                    "expected_primary_content",
                    "expected_key_facts",
                    "capture_instruction",
                }
            }
        raw_output_record = {
            "sample_id": sample["sample_id"],
            "session_id": record.session_id,
            "frame_id": record.frame_id,
            "artifact_image_path": str(Path(artifact_path).resolve()),
            "raw_image_path": str(record.raw_image_path),
            "model_name": self.model_name,
            "base_url": self.base_url,
            "prompt": prompt_text,
            "warnings": list(parsed.warnings),
            "json_repaired": parsed.json_repaired,
            "repair_reasons": list(parsed.repair_reasons),
            "usage": dict(chat_result.usage),
        }
        if "capture_plan" in sample:
            raw_output_record["capture_plan"] = sample["capture_plan"]
        if capture_plan_hint:
            raw_output_record["capture_plan_hint"] = capture_plan_hint.strip()
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
    capture_plan_seqs: str | Sequence[int] | None = None,
    overwrite: bool = False,
    print_model_io: bool = False,
    save_raw_response: bool = False,
    include_capture_plan_hint: bool = False,
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
    progress_by_frame_id = _load_capture_plan_progress(resolved_session_dir)
    selected_capture_plan_seqs = _parse_capture_plan_seq_filter(capture_plan_seqs)
    artifact_paths = _ordered_artifact_paths(
        input_dir=resolved_input_dir,
        capture_index=capture_index,
        progress_by_frame_id=progress_by_frame_id,
        max_images=max_images,
        capture_plan_seqs=selected_capture_plan_seqs,
    )
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
        include_capture_plan_hint=include_capture_plan_hint,
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
                result = prelabeler.prelabel_artifact(
                    artifact_path,
                    record=record,
                    plan_metadata=progress_by_frame_id.get(frame_id),
                )
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
    (resolved_output_dir / "label_studio_tasks_with_predictions.json").write_text(
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
            capture_plan_seqs=args.capture_plan_seqs,
            overwrite=args.overwrite,
            print_model_io=args.print_model_io,
            save_raw_response=args.save_raw_response,
            include_capture_plan_hint=args.include_capture_plan_hint,
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
