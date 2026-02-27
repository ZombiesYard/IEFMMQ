"""
Help prompt builder with strict JSON/output constraints.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from core.llm_schema import get_help_response_schema

_ABS_WIN_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_ABS_POSIX_PATH_RE = re.compile(r"^/")
_SENSITIVE_KEYWORDS = ("api_key", "apikey", "token", "secret", "password", "authorization")


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


def _pick_vars(value: Any, max_items: int = 20) -> dict[str, Any]:
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


def _build_recent_actions(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = context.get("recent_actions")
    if isinstance(actions, list):
        sanitized = _sanitize_obj(actions)
        if isinstance(sanitized, list):
            return [a for a in sanitized if isinstance(a, dict)]

    deltas = context.get("recent_deltas")
    if not isinstance(deltas, list):
        return []

    out: list[dict[str, Any]] = []
    for item in deltas[:10]:
        if not isinstance(item, Mapping):
            continue
        ui_target = (
            item.get("ui_target")
            or item.get("mapped_ui_target")
            or item.get("target")
            or item.get("k")
        )
        out.append(
            {
                "action": item.get("action", "delta"),
                "ui_target": _sanitize_scalar(ui_target),
                "from": _sanitize_scalar(item.get("from")),
                "to": _sanitize_scalar(item.get("to")),
            }
        )
    return out


def _normalize_enum_list(values: Any, fallback: list[str]) -> list[str]:
    if not isinstance(values, list) or not values:
        return list(fallback)
    out: list[str] = []
    for v in values:
        if isinstance(v, str) and v and v in fallback:
            out.append(v)
    if out:
        return out
    return list(fallback)


def build_help_prompt(context: Mapping[str, Any], lang: str) -> str:
    schema = get_help_response_schema()
    step_enum = list(schema["properties"]["next"]["properties"]["step_id"]["enum"])
    target_enum = list(schema["properties"]["overlay"]["properties"]["targets"]["items"]["enum"])
    category_enum = list(schema["properties"]["diagnosis"]["properties"]["error_category"]["enum"])

    candidate_steps = _normalize_enum_list(context.get("candidate_steps"), step_enum)
    overlay_targets = _normalize_enum_list(context.get("overlay_target_allowlist"), target_enum)
    selected_vars = _pick_vars(context.get("vars"))
    recent_actions = _build_recent_actions(context)

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
        "current_vars_selected": selected_vars,
        "recent_actions": recent_actions,
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
            "If uncertain, still return valid JSON only.",
        ]

    return (
        f"{header}\n"
        f"Rules:\n- {rules[0]}\n- {rules[1]}\n- {rules[2]}\n"
        f"Context and constraints JSON:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)}\n"
        "Output must follow this schema shape exactly:\n"
        '{"diagnosis":{"step_id":"...","error_category":"..."},'
        '"next":{"step_id":"..."},'
        '"overlay":{"targets":["..."]},'
        '"explanations":["..."],'
        '"confidence":0.0}'
    )


__all__ = ["build_help_prompt"]

