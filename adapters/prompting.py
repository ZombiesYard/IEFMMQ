"""
Help prompt builder with strict JSON/output constraints.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from adapters.evidence_refs import EVIDENCE_TYPE_PREFIXES, infer_evidence_type_from_ref
from adapters.pack_gates import SUPPORTED_SCENARIO_PROFILES
from core.llm_schema import get_help_response_schema
from core.step_signal_metadata import (
    STEP_EVIDENCE_REQUIREMENT_VALUES,
    compute_requires_visual_confirmation,
    normalize_observability_status,
)

_ABS_WIN_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_ABS_POSIX_PATH_RE = re.compile(r"^/")
_SENSITIVE_KEYWORDS = ("api_key", "apikey", "token", "secret", "password", "authorization")
_LOGGER = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 7000
MAX_PROMPT_TOKENS_EST = 1800
MAX_DELTA_SUMMARY_ITEMS = 20
MAX_RECENT_ACTIONS_SIGNAL_ITEMS = 8
MAX_MISSING_CONDITIONS_SIGNAL_ITEMS = 8
MAX_RECENT_UI_TARGETS_SIGNAL_ITEMS = 8
DEFAULT_MAX_VARS_ITEMS = 20
MAX_RAG_SNIPPETS = 5
MAX_RAG_SNIPPET_CHARS = 220
MAX_PRIORITY_OVERLAY_TARGETS = 8


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    metadata: dict[str, Any]


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(word in lowered for word in _SENSITIVE_KEYWORDS)


def _sanitize_scalar(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if not isinstance(value, str):
        return value
    if _ABS_WIN_PATH_RE.match(value) or _ABS_POSIX_PATH_RE.match(value):
        return "[REDACTED_PATH]"
    if "sk-" in value or "api_key" in value.lower() or "token=" in value.lower():
        return "[REDACTED_SECRET]"
    return value


def _is_json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


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


def _build_gates_summary(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    gates = context.get("gates")
    if not isinstance(gates, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for key in sorted(gates.keys(), key=lambda x: str(x)):
        value = gates[key]
        if isinstance(value, Mapping):
            status = value.get("status", "unknown")
            reason = value.get("reason")
        else:
            status = "allowed" if bool(value) else "blocked"
            reason = None
        out.append(
            {
                "gate_id": _sanitize_scalar(str(key)),
                "status": _sanitize_scalar(status),
                "reason": _sanitize_scalar(reason),
            }
        )
    return out


def _build_rag_snippets(context: Mapping[str, Any], max_items: int = MAX_RAG_SNIPPETS) -> list[dict[str, Any]]:
    rag_topk = context.get("rag_topk")
    if not isinstance(rag_topk, list):
        return []
    out: list[dict[str, Any]] = []
    for item in rag_topk:
        if len(out) >= max_items:
            break
        if isinstance(item, Mapping):
            snippet_id = item.get("snippet_id") or item.get("id") or f"snippet_{len(out)}"
            snippet = str(item.get("snippet", ""))
            doc_id = item.get("doc_id")
            section = item.get("section")
            page_or_heading = item.get("page_or_heading")
            if page_or_heading is None:
                page_or_heading = item.get("page")
            if page_or_heading is None:
                page_or_heading = section
        else:
            snippet_id = f"snippet_{len(out)}"
            snippet = str(item)
            doc_id = None
            section = None
            page_or_heading = None
        normalized: dict[str, Any] = {
            "id": _sanitize_scalar(str(snippet_id)),
            "snippet": _sanitize_scalar(snippet[:MAX_RAG_SNIPPET_CHARS]),
        }
        if isinstance(doc_id, str) and doc_id:
            normalized["doc_id"] = _sanitize_scalar(doc_id)
        if isinstance(section, str) and section:
            normalized["section"] = _sanitize_scalar(section)
        if _is_json_scalar(page_or_heading):
            normalized["page_or_heading"] = _sanitize_scalar(page_or_heading)
        out.append(normalized)
    return out


def _build_recent_actions_signal(context: Mapping[str, Any]) -> dict[str, Any]:
    raw = context.get("recent_actions")
    current_button: str | None = None
    recent_buttons: list[str] = []

    if isinstance(raw, Mapping):
        current_raw = raw.get("current_button")
        if isinstance(current_raw, str) and current_raw:
            current_button = str(_sanitize_scalar(current_raw))
        buttons_raw = raw.get("recent_buttons")
        candidates = buttons_raw if isinstance(buttons_raw, list) else []
    elif isinstance(raw, list):
        candidates = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            target = item.get("ui_target") or item.get("mapped_ui_target") or item.get("target")
            if isinstance(target, str) and target:
                candidates.append(target)
                continue
            targets = item.get("ui_targets")
            if isinstance(targets, list):
                for target_item in targets:
                    if isinstance(target_item, str) and target_item:
                        candidates.append(target_item)
    else:
        candidates = []

    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        normalized = str(_sanitize_scalar(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        recent_buttons.append(normalized)
        if len(recent_buttons) >= MAX_RECENT_ACTIONS_SIGNAL_ITEMS:
            break

    if current_button is None and recent_buttons:
        current_button = recent_buttons[0]
    if current_button is not None and current_button not in recent_buttons:
        recent_buttons.insert(0, current_button)
        if len(recent_buttons) > MAX_RECENT_ACTIONS_SIGNAL_ITEMS:
            recent_buttons = recent_buttons[:MAX_RECENT_ACTIONS_SIGNAL_ITEMS]

    return {
        "current_button": current_button,
        "recent_buttons": recent_buttons,
    }


def _build_overlay_target_priority(
    overlay_targets: list[str],
    recent_actions_signal: Mapping[str, Any],
    deterministic_step_hint: Mapping[str, Any],
    recent_deltas_summary: Mapping[str, Any],
) -> list[str]:
    if not overlay_targets:
        return []

    allowed_targets = set(overlay_targets)
    ranked: list[str] = []
    seen: set[str] = set()

    def _append_if_allowed(value: Any) -> None:
        if not isinstance(value, str) or not value or value not in allowed_targets or value in seen:
            return
        seen.add(value)
        ranked.append(value)

    hint_targets = deterministic_step_hint.get("recent_ui_targets")
    if isinstance(hint_targets, list):
        for item in hint_targets:
            _append_if_allowed(item)

    _append_if_allowed(recent_actions_signal.get("current_button"))

    recent_buttons = recent_actions_signal.get("recent_buttons")
    if isinstance(recent_buttons, list):
        for item in recent_buttons:
            _append_if_allowed(item)

    delta_items = recent_deltas_summary.get("items")
    if isinstance(delta_items, list):
        for item in delta_items:
            if isinstance(item, Mapping):
                _append_if_allowed(item.get("ui_target"))

    for target in overlay_targets:
        _append_if_allowed(target)

    return ranked[:MAX_PRIORITY_OVERLAY_TARGETS]


def _build_overlay_target_policy(priority_targets: list[str]) -> dict[str, Any]:
    return {
        "mode": "single_target_preferred",
        "max_targets": 1,
        "empty_overlay_if_uncertain": True,
        "preferred_target": priority_targets[0] if priority_targets else None,
        "candidate_targets_in_priority_order": list(priority_targets),
    }


def _build_overlay_evidence_contract() -> dict[str, Any]:
    return {
        "field_order": ["target", "type", "ref", "quote", "grounding_confidence"],
        "quote_max_chars": 120,
        "same_target_required": True,
        "ref_must_exist_in_allowed_evidence_refs": True,
        "type_ref_prefixes": {
            evidence_type: list(prefixes)
            for evidence_type, prefixes in sorted(EVIDENCE_TYPE_PREFIXES.items(), key=lambda item: item[0])
        },
    }


def _build_uncertainty_policy(deterministic_step_hint: Mapping[str, Any]) -> dict[str, Any]:
    observability_status = deterministic_step_hint.get("observability_status")
    if not isinstance(observability_status, str) or not observability_status:
        observability_status = None
    inferred_step_id = deterministic_step_hint.get("inferred_step_id")
    if not isinstance(inferred_step_id, str) or not inferred_step_id:
        inferred_step_id = None
    return {
        "current_observability_status": observability_status,
        "current_inferred_step_id": inferred_step_id,
        "requires_visual_confirmation": bool(deterministic_step_hint.get("requires_visual_confirmation")),
        "partial": {
            "applies_when": "observability_status=partial or requires_visual_confirmation=true",
            "allow_diagnosis_from_hint": True,
            "allow_single_target_only": True,
            "prefer_empty_overlay_without_verifiable_evidence": True,
            "requires_confirmation_phrase": True,
        },
        "unknown": {
            "applies_when": "inferred_step_id is null, evidence conflicts, or no verifiable evidence exists",
            "force_empty_overlay": True,
            "requires_confirmation_phrase": True,
        },
    }


def _example_quote_for_evidence_type(evidence_type: str, lang: str) -> str:
    if lang == "zh":
        quotes = {
            "var": "当前变量状态支持这个高亮目标。",
            "gate": "当前 gate 状态支持这个高亮目标。",
            "rag": "检索片段支持这个高亮目标。",
            "delta": "最近变化支持这个高亮目标。",
        }
        return quotes.get(evidence_type, "当前证据支持这个高亮目标。")
    quotes = {
        "var": "Current variable state supports this target.",
        "gate": "Current gate state supports this target.",
        "rag": "Retrieved snippet supports this target.",
        "delta": "Recent delta supports this target.",
    }
    return quotes.get(evidence_type, "Current evidence supports this target.")


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


def _build_deterministic_step_hint(context: Mapping[str, Any]) -> dict[str, Any]:
    raw = context.get("deterministic_step_hint")
    if not isinstance(raw, Mapping):
        return {
            "inferred_step_id": None,
            "missing_conditions": [],
            "recent_ui_targets": [],
            "observability": None,
            "observability_status": None,
            "step_evidence_requirements": [],
            "requires_visual_confirmation": False,
            "scenario_profile": None,
        }

    inferred_raw = raw.get("inferred_step_id")
    inferred_step_id = str(_sanitize_scalar(inferred_raw)) if isinstance(inferred_raw, str) and inferred_raw else None

    missing_raw = raw.get("missing_conditions")
    missing_conditions: list[str] = []
    if isinstance(missing_raw, (list, tuple)):
        seen: set[str] = set()
        for item in missing_raw:
            if not isinstance(item, str) or not item:
                continue
            normalized = str(_sanitize_scalar(item))
            if normalized in seen:
                continue
            seen.add(normalized)
            missing_conditions.append(normalized)
            if len(missing_conditions) >= MAX_MISSING_CONDITIONS_SIGNAL_ITEMS:
                break

    recent_targets = raw.get("recent_ui_targets")
    if isinstance(recent_targets, (list, tuple)):
        recent_ui_targets = []
        seen_targets: set[str] = set()
        for item in recent_targets:
            if not isinstance(item, str) or not item:
                continue
            normalized = str(_sanitize_scalar(item))
            if normalized in seen_targets:
                continue
            seen_targets.add(normalized)
            recent_ui_targets.append(normalized)
            if len(recent_ui_targets) >= MAX_RECENT_UI_TARGETS_SIGNAL_ITEMS:
                break
    else:
        recent_ui_targets = []

    observability = normalize_observability_status(
        raw.get("observability_status") if "observability_status" in raw else raw.get("observability")
    )

    step_evidence_requirements_raw = raw.get("step_evidence_requirements")
    if step_evidence_requirements_raw is None:
        # Backward compatibility with older hint payloads.
        step_evidence_requirements_raw = raw.get("evidence_requirements")

    step_evidence_requirements: list[str] = []
    if isinstance(step_evidence_requirements_raw, (list, tuple)):
        seen_requirements: set[str] = set()
        for item in step_evidence_requirements_raw:
            if not isinstance(item, str) or item not in STEP_EVIDENCE_REQUIREMENT_VALUES:
                continue
            if item in seen_requirements:
                continue
            seen_requirements.add(item)
            step_evidence_requirements.append(item)

    requires_visual_confirmation_raw = raw.get("requires_visual_confirmation")
    if isinstance(requires_visual_confirmation_raw, bool):
        requires_visual_confirmation = requires_visual_confirmation_raw
    else:
        requires_visual_confirmation = compute_requires_visual_confirmation(
            observability,
            step_evidence_requirements,
        )
    scenario_profile_raw = raw.get("scenario_profile")
    scenario_profile = (
        scenario_profile_raw
        if isinstance(scenario_profile_raw, str) and scenario_profile_raw in SUPPORTED_SCENARIO_PROFILES
        else None
    )

    return {
        "inferred_step_id": inferred_step_id,
        "missing_conditions": missing_conditions,
        "recent_ui_targets": recent_ui_targets,
        "observability": observability,
        "observability_status": observability,
        "step_evidence_requirements": step_evidence_requirements,
        "requires_visual_confirmation": requires_visual_confirmation,
        "scenario_profile": scenario_profile,
    }


def _build_evidence_sources(
    selected_vars: Mapping[str, Any],
    gates_summary: list[dict[str, Any]],
    recent_deltas_summary: Mapping[str, Any],
    rag_snippets: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    vars_block: list[dict[str, Any]] = []
    for key in sorted(selected_vars.keys(), key=lambda x: str(x)):
        vars_block.append({"ref": f"VARS.{key}", "value": selected_vars[key]})

    gates_block: list[dict[str, Any]] = []
    for gate in gates_summary:
        gate_id = gate.get("gate_id", "unknown")
        gates_block.append(
            {
                "ref": f"GATES.{gate_id}",
                "status": gate.get("status"),
                "reason": gate.get("reason"),
            }
        )

    recent_block: list[dict[str, Any]] = []
    for item in recent_deltas_summary.get("items", []):
        ui_target = item.get("ui_target", "UNKNOWN")
        recent_block.append(
            {
                "ref": f"RECENT_UI_TARGETS.{ui_target}",
                "ui_target": ui_target,
                "count": item.get("count"),
                "last_action": item.get("last_action"),
                "last_from": item.get("last_from"),
                "last_to": item.get("last_to"),
            }
        )

    rag_block: list[dict[str, Any]] = []
    for item in rag_snippets:
        snippet_id = item.get("id", "snippet")
        rag_entry: dict[str, Any] = {
            "ref": f"RAG_SNIPPETS.{snippet_id}",
            "id": snippet_id,
            "snippet": item.get("snippet"),
        }
        if "doc_id" in item:
            rag_entry["doc_id"] = item.get("doc_id")
        if "section" in item:
            rag_entry["section"] = item.get("section")
        if "page_or_heading" in item:
            rag_entry["page_or_heading"] = item.get("page_or_heading")
        rag_block.append(rag_entry)

    evidence = {
        "VARS": vars_block,
        "GATES": gates_block,
        "RECENT_UI_TARGETS": recent_block,
        "RAG_SNIPPETS": rag_block,
    }
    allowed_refs = [
        entry["ref"]
        for block in (vars_block, gates_block, recent_block, rag_block)
        for entry in block
        if isinstance(entry, Mapping) and isinstance(entry.get("ref"), str)
    ]
    return evidence, allowed_refs


def _build_grounding_payload(
    context: Mapping[str, Any],
    rag_snippets: list[dict[str, Any]],
    *,
    rag_input_count: int,
) -> dict[str, Any]:
    requested_missing = bool(context.get("grounding_missing"))
    requested_reason = context.get("grounding_reason")
    requested_reason_str = _sanitize_scalar(requested_reason) if isinstance(requested_reason, str) else None
    applied = bool(rag_snippets)
    missing_effective = requested_missing or (not applied)
    reason_effective = requested_reason_str
    if missing_effective and reason_effective is None:
        if requested_missing:
            reason_effective = "grounding_unavailable"
        elif rag_input_count > 0:
            reason_effective = "rag_snippets_not_injected"
        else:
            reason_effective = "no_rag_snippets"
    return {
        "requested_missing": requested_missing,
        "missing": missing_effective,
        "applied": applied,
        "reason": reason_effective,
        "query": _sanitize_scalar(context.get("grounding_query")),
    }


def _compose_prompt(header: str, rules: list[str], payload: dict[str, Any]) -> str:
    rendered_rules = "\n".join(f"- {rule}" for rule in rules)
    return (
        f"{header}\n"
        f"Rules:\n{rendered_rules}\n"
        f"Context and constraints JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)}\n"
        "Output must follow this schema shape exactly:\n"
        '{"diagnosis":{"step_id":"...","error_category":"..."},'
        '"next":{"step_id":"..."},'
        '"overlay":{"targets":["..."],"evidence":[{"target":"...","type":"...","ref":"...","quote":"...","grounding_confidence":0.0}]},'
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
    overlay_evidence_type_enum = list(
        schema["properties"]["overlay"]["properties"]["evidence"]["items"]["properties"]["type"]["enum"]
    )
    schema_category_enum = list(schema["properties"]["diagnosis"]["properties"]["error_category"]["enum"])
    category_enum = _normalize_enum_list(context.get("error_category_enum"), schema_category_enum)

    candidate_steps = _normalize_enum_list(context.get("candidate_steps"), step_enum)
    overlay_targets = _normalize_enum_list(context.get("overlay_target_allowlist"), target_enum)
    max_vars = DEFAULT_MAX_VARS_ITEMS
    selected_vars = _pick_vars(context.get("vars"), max_items=max_vars)
    recent_deltas_summary = _build_delta_summary(context, top_k=MAX_DELTA_SUMMARY_ITEMS)
    gates_summary = _build_gates_summary(context)
    rag_snippets = _build_rag_snippets(context, max_items=MAX_RAG_SNIPPETS)
    rag_input_count = len(rag_snippets)
    recent_actions_signal = _build_recent_actions_signal(context)
    deterministic_step_hint = _build_deterministic_step_hint(context)
    overlay_target_priority = _build_overlay_target_priority(
        overlay_targets,
        recent_actions_signal,
        deterministic_step_hint,
        recent_deltas_summary,
    )
    overlay_target_policy = _build_overlay_target_policy(overlay_target_priority)
    overlay_evidence_contract = _build_overlay_evidence_contract()
    uncertainty_policy = _build_uncertainty_policy(deterministic_step_hint)
    scenario_profile_raw = context.get("scenario_profile")
    scenario_profile = (
        scenario_profile_raw
        if isinstance(scenario_profile_raw, str) and scenario_profile_raw in SUPPORTED_SCENARIO_PROFILES
        else None
    )

    payload: dict[str, Any] = {}

    if lang == "zh":
        header = (
            "你是 SimTutor 助教。"
            "你必须只输出一个严格 JSON 对象，不得输出任何 JSON 以外的文本、解释、markdown 或代码围栏。"
        )
        rules = [
            "必须从 allowed_step_ids 中选择 diagnosis.step_id 与 next.step_id。",
            "必须从 allowed_overlay_targets 中选择 overlay.targets。",
            "必须从 allowed_error_categories 中选择 diagnosis.error_category。",
            "overlay.evidence.type 必须从 allowed_overlay_evidence_types 中选择（不得使用 visual）。",
            "最多只返回 1 个 overlay target；必须优先选择 overlay_target_policy.candidate_targets_in_priority_order 中最靠前且证据最强的 target。",
            "只允许引用 EVIDENCE_SOURCES 中出现的 ref。",
            "overlay.evidence 字段顺序必须固定为 target,type,ref,quote,grounding_confidence，且 type 必须与 ref 前缀匹配。",
            "overlay.evidence 每项必须包含 target/type/ref/quote/grounding_confidence，且 quote 最长 120 字符。",
            "deterministic_step_hint.step_evidence_requirements 仅表示步骤证据偏好，不等于 overlay.evidence.type 枚举。",
            "每个 target 至少要有一条 evidence；若证据不足，返回空 targets 和空 evidence，并解释“需要更多信息/请确认XX”。",
            "优先参考 deterministic_step_hint，若证据不冲突，优先沿 inferred_step_id 给出 diagnosis/next。",
            "若 uncertainty_policy.partial 生效：可以沿 deterministic_step_hint 给 diagnosis/next，但 explanation 必须明确要求确认，且 overlay 仍只能返回单目标。",
            "若 uncertainty_policy.unknown 生效：必须返回空 targets 和空 evidence，并要求确认，不得猜测高亮。",
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
            "overlay.evidence.type must be chosen from allowed_overlay_evidence_types (never use \"visual\").",
            "Return at most one overlay target. Pick the highest-confidence target from overlay_target_policy.candidate_targets_in_priority_order.",
            "Only refs that appear in EVIDENCE_SOURCES are allowed.",
            "Emit overlay.evidence fields in this exact order: target, type, ref, quote, grounding_confidence, and type must match the ref prefix.",
            "Each overlay.evidence item must include target/type/ref/quote/grounding_confidence, and quote length must be <= 120 chars.",
            "deterministic_step_hint.step_evidence_requirements describes step-level evidence preference only; it is not the overlay.evidence.type enum.",
            "Each target must have at least one evidence item; if not enough evidence, return empty targets and empty evidence, then explain what to confirm.",
            "Prefer deterministic_step_hint when evidence does not conflict; prioritize inferred_step_id for diagnosis/next.",
            "If uncertainty_policy.partial applies, you may use deterministic_step_hint for diagnosis/next, but the explanation must explicitly ask for confirmation and overlay stays single-target only.",
            "If uncertainty_policy.unknown applies, keep overlay.targets=[] and overlay.evidence=[], then ask for confirmation instead of guessing.",
            "If uncertain, still return valid JSON only.",
        ]

    trim_reasons: list[str] = []

    allowed_refs: list[str] = []

    def _render_and_measure() -> tuple[str, int, int]:
        nonlocal payload, allowed_refs
        evidence_sources, refs = _build_evidence_sources(
            selected_vars=selected_vars,
            gates_summary=gates_summary,
            recent_deltas_summary=recent_deltas_summary,
            rag_snippets=rag_snippets,
        )
        allowed_refs = refs
        next_step = candidate_steps[1] if len(candidate_steps) > 1 else candidate_steps[0]
        example_refs = [allowed_refs[0]] if allowed_refs else []
        example_target = overlay_target_policy["preferred_target"] or (overlay_targets[0] if overlay_targets else None)
        example_targets = [example_target] if example_refs and example_target else []
        example_ref = example_refs[0] if example_refs else None
        example_evidence_type = (
            infer_evidence_type_from_ref(example_ref) if isinstance(example_ref, str) else None
        ) or overlay_evidence_type_enum[0]
        example_overlay_evidence = (
            [
                {
                    "target": example_targets[0],
                    "type": example_evidence_type,
                    "ref": example_ref,
                    "quote": _example_quote_for_evidence_type(example_evidence_type, lang),
                    "grounding_confidence": 0.9,
                }
            ]
            if example_refs and example_targets
            else []
        )
        example_obj = {
            "diagnosis": {"step_id": candidate_steps[0], "error_category": category_enum[0]},
            "next": {"step_id": next_step},
            "overlay": {"targets": example_targets, "evidence": example_overlay_evidence},
            "explanations": ["Use concise guidance." if lang == "en" else "请给出简洁指导。"],
            "confidence": 0.75,
        }
        payload = {
            "allowed_step_ids": candidate_steps,
            "allowed_overlay_targets": overlay_targets,
            "allowed_overlay_evidence_types": overlay_evidence_type_enum,
            "allowed_error_categories": category_enum,
            "decision_priority": [
                "deterministic_step_hint",
                "gates_summary",
                "overlay_target_policy",
                "recent_actions_signal",
                "recent_deltas_summary",
                "current_vars_selected",
                "EVIDENCE_SOURCES.RAG_SNIPPETS",
            ],
            "scenario_profile": scenario_profile,
            "current_vars_selected": selected_vars,
            "gates_summary": gates_summary,
            "recent_deltas_summary": recent_deltas_summary,
            "recent_actions_signal": recent_actions_signal,
            "deterministic_step_hint": deterministic_step_hint,
            "overlay_target_policy": overlay_target_policy,
            "overlay_evidence_contract": overlay_evidence_contract,
            "uncertainty_policy": uncertainty_policy,
            "grounding": _build_grounding_payload(
                context,
                rag_snippets,
                rag_input_count=rag_input_count,
            ),
            "EVIDENCE_SOURCES": evidence_sources,
            "allowed_evidence_refs": allowed_refs,
            "output_example_json": example_obj,
        }
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
            if "trimmed_vars" not in trim_reasons:
                trim_reasons.append("trimmed_vars")
            changed = True
        elif len(candidate_steps) > 1:
            candidate_steps = candidate_steps[:-1]
            if "trimmed_step_enum" not in trim_reasons:
                trim_reasons.append("trimmed_step_enum")
            changed = True
        elif len(overlay_targets) > 1:
            overlay_targets = overlay_targets[:-1]
            if "trimmed_overlay_enum" not in trim_reasons:
                trim_reasons.append("trimmed_overlay_enum")
            changed = True
        elif len(rag_snippets) > 0:
            rag_snippets = rag_snippets[:-1]
            if "trimmed_rag_snippets" not in trim_reasons:
                trim_reasons.append("trimmed_rag_snippets")
            changed = True
        if not changed:
            break
        prompt, chars, tokens = _render_and_measure()
    final_rag_snippets = list(rag_snippets)
    final_allowed_refs = list(allowed_refs)

    if chars > max_prompt_chars or tokens > max_prompt_tokens_est:
        compact_header = "JSON only. Follow enum constraints strictly."
        if lang == "zh":
            compact_header = "仅输出 JSON；严格遵循枚举约束。"
        final_rag_snippets = []
        final_allowed_refs = []
        compact_hint = {
            "inferred_step_id": deterministic_step_hint.get("inferred_step_id"),
            "requires_visual_confirmation": bool(deterministic_step_hint.get("requires_visual_confirmation")),
        }
        compact_grounding_payload = _build_grounding_payload(
            context,
            final_rag_snippets,
            rag_input_count=rag_input_count,
        )
        compact_payload = {
            "allowed_step_ids": candidate_steps[:3],
            "allowed_overlay_targets": overlay_targets,
            "decision_priority": [
                "deterministic_step_hint",
                "gates_summary",
            ],
            "gates_summary": gates_summary,
            "deterministic_step_hint": compact_hint,
            "overlay_target_policy": {
                "mode": "single_target_preferred",
                "preferred_target": overlay_target_policy.get("preferred_target"),
            },
            "grounding": {
                "applied": bool(compact_grounding_payload["applied"]),
                "missing": bool(compact_grounding_payload["missing"]),
                "reason": compact_grounding_payload["reason"],
            },
            "allowed_evidence_refs": final_allowed_refs,
        }
        prompt = (
            f"{compact_header}\n"
            f"constraints={json.dumps(compact_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)}\n"
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

    grounding_payload = _build_grounding_payload(
        context,
        final_rag_snippets,
        rag_input_count=rag_input_count,
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
        "evidence_refs_count": len(final_allowed_refs),
        "allowed_evidence_refs": list(final_allowed_refs),
        "preferred_overlay_target": overlay_target_policy["preferred_target"],
        "rag_snippet_count": len(final_rag_snippets),
        "rag_snippet_ids": [
            str(item.get("id"))
            for item in final_rag_snippets
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        ],
        "grounding_applied": bool(grounding_payload["applied"]),
        "grounding_missing_requested": bool(grounding_payload["requested_missing"]),
        "grounding_missing": bool(grounding_payload["missing"]),
        "grounding_reason": grounding_payload["reason"],
    }
    return PromptBuildResult(prompt=prompt, metadata=meta)


def build_help_prompt(context: Mapping[str, Any], lang: str) -> str:
    return build_help_prompt_result(context, lang).prompt


__all__ = [
    "MAX_DELTA_SUMMARY_ITEMS",
    "MAX_RECENT_ACTIONS_SIGNAL_ITEMS",
    "MAX_PROMPT_CHARS",
    "MAX_PROMPT_TOKENS_EST",
    "PromptBuildResult",
    "build_help_prompt",
    "build_help_prompt_result",
]
