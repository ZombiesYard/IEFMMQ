"""
Help prompt builder with strict JSON/output constraints.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from core.llm_schema import get_help_response_schema

_ABS_WIN_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_ABS_POSIX_PATH_RE = re.compile(r"^/")
_SENSITIVE_KEYWORDS = ("api_key", "apikey", "token", "secret", "password", "authorization")
_LOGGER = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 7000
MAX_PROMPT_TOKENS_EST = 1800
MAX_DELTA_SUMMARY_ITEMS = 20
DEFAULT_MAX_VARS_ITEMS = 20


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    metadata: dict[str, Any]


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(word in lowered for word in _SENSITIVE_KEYWORDS)


def _sanitize_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if _ABS_WIN_PATH_RE.match(value) or _ABS_POSIX_PATH_RE.match(value):
        return "[REDACTED_PATH]"
    if "sk-" in value or "api_key" in value.lower() or "token=" in value.lower():
        return "[REDACTED_SECRET]"
    return value


def _sanitize_obj(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key in sorted(value.keys(), key=lambda x: str(x)):
            key_str = str(key)
            if _is_sensitive_key(key_str):
                continue
            sanitized[key_str] = _sanitize_obj(value[key])
        return sanitized
    if isinstance(value, list):
        return [_sanitize_obj(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_obj(v) for v in value]
    return _sanitize_scalar(value)


def _pick_vars(value: Any, max_items: int = DEFAULT_MAX_VARS_ITEMS) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    sanitized = _sanitize_obj(value)
    if not isinstance(sanitized, dict):
        return {}
    out: dict[str, Any] = {}
    for idx, key in enumerate(sorted(sanitized.keys())):
        if idx >= max_items:
            break
        out[key] = sanitized[key]
    return out


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def _build_delta_summary(context: Mapping[str, Any], top_k: int = MAX_DELTA_SUMMARY_ITEMS) -> dict[str, Any]:
    deltas = context.get("recent_deltas")
    if not isinstance(deltas, list):
        return {"top_k": top_k, "total_targets": 0, "items": []}

    buckets: dict[str, dict[str, Any]] = {}
    for item in deltas:
        if not isinstance(item, Mapping):
            continue
        ui_target = (
            item.get("ui_target")
            or item.get("mapped_ui_target")
            or item.get("target")
            or item.get("k")
        )
        ui_target_str = _sanitize_scalar(str(ui_target)) if ui_target is not None else "[UNKNOWN_TARGET]"
        bucket = buckets.get(ui_target_str)
        if bucket is None:
            bucket = {
                "ui_target": ui_target_str,
                "count": 0,
                "last_action": "delta",
                "last_from": None,
                "last_to": None,
            }
            buckets[ui_target_str] = bucket
        bucket["count"] += 1
        bucket["last_action"] = _sanitize_scalar(item.get("action", "delta"))
        bucket["last_from"] = _sanitize_scalar(item.get("from"))
        bucket["last_to"] = _sanitize_scalar(item.get("to"))

    ranked = sorted(
        buckets.values(),
        key=lambda x: (-int(x.get("count", 0)), str(x.get("ui_target", ""))),
    )
    items = ranked[:top_k]
    return {
        "top_k": top_k,
        "total_targets": len(ranked),
        "items": items,
    }


def _normalize_enum_list(values: Any, fallback: list[str]) -> list[str]:
    if not isinstance(values, list) or not values:
        return list(fallback)
    allowed = set(fallback)
    out: list[str] = []
    for v in values:
        if isinstance(v, str) and v and v in allowed:
            out.append(v)
    if out:
        return out
    return list(fallback)


def _compose_prompt(header: str, rules: list[str], payload: dict[str, Any]) -> str:
    rendered_rules = "\n".join(f"- {rule}" for rule in rules)
    return (
        f"{header}\n"
        f"Rules:\n{rendered_rules}\n"
        f"Context and constraints JSON:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
        "Output must follow this schema shape exactly:\n"
        '{"diagnosis":{"step_id":"...","error_category":"..."},'
        '"next":{"step_id":"..."},'
        '"overlay":{"targets":["..."]},'
        '"explanations":["..."],'
        '"confidence":0.0}'
    )


def _record_trim_event(message: str) -> None:
    if os.getenv("SIMTUTOR_PROMPT_TRIM_PRINT", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[PROMPT] {message}")
    _LOGGER.warning(message)


def build_help_prompt_result(
    context: Mapping[str, Any],
    lang: str,
    *,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
    max_prompt_tokens_est: int = MAX_PROMPT_TOKENS_EST,
) -> PromptBuildResult:
    schema = get_help_response_schema()
    step_enum = list(schema["properties"]["next"]["properties"]["step_id"]["enum"])
    target_enum = list(schema["properties"]["overlay"]["properties"]["targets"]["items"]["enum"])
    schema_category_enum = list(schema["properties"]["diagnosis"]["properties"]["error_category"]["enum"])
    category_enum = _normalize_enum_list(context.get("error_category_enum"), schema_category_enum)

    candidate_steps = _normalize_enum_list(context.get("candidate_steps"), step_enum)
    overlay_targets = _normalize_enum_list(context.get("overlay_target_allowlist"), target_enum)
    max_vars = DEFAULT_MAX_VARS_ITEMS
    selected_vars = _pick_vars(context.get("vars"), max_items=max_vars)
    recent_deltas_summary = _build_delta_summary(context, top_k=MAX_DELTA_SUMMARY_ITEMS)

    next_step = candidate_steps[1] if len(candidate_steps) > 1 else candidate_steps[0]
    example_obj = {
        "diagnosis": {"step_id": candidate_steps[0], "error_category": category_enum[0]},
        "next": {"step_id": next_step},
        "overlay": {"targets": [overlay_targets[0]]},
        "explanations": ["Use concise guidance." if lang == "en" else "请给出简洁指导。"],
        "confidence": 0.75,
    }

    payload = {
        "allowed_step_ids": candidate_steps,
        "allowed_overlay_targets": overlay_targets,
        "allowed_error_categories": category_enum,
        "current_vars_selected": selected_vars,
        "recent_deltas_summary": recent_deltas_summary,
        "output_example_json": example_obj,
    }

    if lang == "zh":
        header = (
            "你是 SimTutor 助教。"
            "你必须只输出一个严格 JSON 对象，不得输出任何 JSON 以外的文本、解释、markdown 或代码围栏。"
        )
        rules = [
            "必须从 allowed_step_ids 中选择 diagnosis.step_id 与 next.step_id。",
            "必须从 allowed_overlay_targets 中选择 overlay.targets。",
            "必须从 allowed_error_categories 中选择 diagnosis.error_category。",
            "若不确定，也必须返回合法 JSON，不得输出自然语言段落。",
        ]
    else:
        header = (
            "You are SimTutor tutor assistant. "
            "You must output exactly one strict JSON object and nothing outside JSON "
            "(no prose, no markdown, no code fences)."
        )
        rules = [
            "diagnosis.step_id and next.step_id must be chosen from allowed_step_ids.",
            "overlay.targets must be chosen from allowed_overlay_targets.",
            "diagnosis.error_category must be chosen from allowed_error_categories.",
            "If uncertain, still return valid JSON only.",
        ]

    trim_reasons: list[str] = []

    def _render_and_measure() -> tuple[str, int, int]:
        prompt_text = _compose_prompt(header, rules, payload)
        return prompt_text, len(prompt_text), _estimate_tokens(prompt_text)

    prompt, chars, tokens = _render_and_measure()
    while chars > max_prompt_chars or tokens > max_prompt_tokens_est:
        changed = False
        if recent_deltas_summary["items"]:
            recent_deltas_summary["items"] = recent_deltas_summary["items"][:-1]
            if "trimmed_delta_summary" not in trim_reasons:
                trim_reasons.append("trimmed_delta_summary")
            changed = True
        elif max_vars > 0:
            max_vars = max(0, max_vars - 5)
            selected_vars = _pick_vars(context.get("vars"), max_items=max_vars)
            payload["current_vars_selected"] = selected_vars
            if "trimmed_vars" not in trim_reasons:
                trim_reasons.append("trimmed_vars")
            changed = True
        elif len(candidate_steps) > 1:
            candidate_steps = candidate_steps[:-1]
            payload["allowed_step_ids"] = candidate_steps
            payload["output_example_json"]["diagnosis"]["step_id"] = candidate_steps[0]
            payload["output_example_json"]["next"]["step_id"] = candidate_steps[-1]
            if "trimmed_step_enum" not in trim_reasons:
                trim_reasons.append("trimmed_step_enum")
            changed = True
        elif len(overlay_targets) > 1:
            overlay_targets = overlay_targets[:-1]
            payload["allowed_overlay_targets"] = overlay_targets
            payload["output_example_json"]["overlay"]["targets"] = [overlay_targets[0]]
            if "trimmed_overlay_enum" not in trim_reasons:
                trim_reasons.append("trimmed_overlay_enum")
            changed = True
        if not changed:
            break
        prompt, chars, tokens = _render_and_measure()

    if chars > max_prompt_chars or tokens > max_prompt_tokens_est:
        compact_header = "JSON only. Follow enum constraints strictly."
        if lang == "zh":
            compact_header = "仅输出 JSON；严格遵循枚举约束。"
        compact_payload = {
            "allowed_step_ids": payload["allowed_step_ids"],
            "allowed_overlay_targets": payload["allowed_overlay_targets"],
            "allowed_error_categories": payload["allowed_error_categories"],
            "output_example_json": payload["output_example_json"],
        }
        prompt = (
            f"{compact_header}\n"
            f"constraints={json.dumps(compact_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
            "JSON only."
        )
        chars = len(prompt)
        tokens = _estimate_tokens(prompt)
        if "compact_template" not in trim_reasons:
            trim_reasons.append("compact_template")

    if chars > max_prompt_chars or tokens > max_prompt_tokens_est:
        hard_cap_chars = min(max_prompt_chars, max_prompt_tokens_est * 4)
        prompt = prompt[:hard_cap_chars]
        chars = len(prompt)
        tokens = _estimate_tokens(prompt)
        if "hard_truncate" not in trim_reasons:
            trim_reasons.append("hard_truncate")

    if trim_reasons:
        _record_trim_event(
            "Prompt trimmed to fit budget: "
            f"reasons={trim_reasons}, chars={chars}/{max_prompt_chars}, tokens_est={tokens}/{max_prompt_tokens_est}"
        )

    meta = {
        "max_prompt_chars": max_prompt_chars,
        "max_prompt_tokens_est": max_prompt_tokens_est,
        "prompt_chars": chars,
        "prompt_tokens_est": tokens,
        "prompt_trimmed": bool(trim_reasons),
        "trim_reasons": trim_reasons,
        "delta_summary_top_k": recent_deltas_summary["top_k"],
        "delta_summary_items": len(recent_deltas_summary["items"]),
    }
    return PromptBuildResult(prompt=prompt, metadata=meta)


def build_help_prompt(context: Mapping[str, Any], lang: str) -> str:
    return build_help_prompt_result(context, lang).prompt


__all__ = [
    "MAX_DELTA_SUMMARY_ITEMS",
    "MAX_PROMPT_CHARS",
    "MAX_PROMPT_TOKENS_EST",
    "PromptBuildResult",
    "build_help_prompt",
    "build_help_prompt_result",
]
