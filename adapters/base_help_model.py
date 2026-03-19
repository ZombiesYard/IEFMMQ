"""
Shared base implementation for HelpResponse-capable model adapters.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from adapters.help_response_parser import parse_help_response_with_diagnostics
from adapters.json_extract import parse_first_json
from adapters.prompting import build_help_prompt_result
from adapters.response_mapping import map_help_response_to_tutor_response
from adapters.step_inference import (
    StepInferenceResult,
    extract_recent_ui_targets,
    infer_step_id,
    load_pack_steps,
    normalize_recent_ui_targets,
)
from core.help_failure import (
    ALLOWLIST_FAIL,
    EVIDENCE_FAIL,
    MODEL_HTTP_FAIL,
    SCHEMA_FAIL,
    annotate_exception,
    exception_failure_code,
    exception_failure_stage,
    merge_failure_metadata,
)
from core.llm_schema import get_help_response_schema, validate_help_response
from core.security import redact_sensitive_text, sanitize_help_response_for_log, sanitize_public_model_text
from core.step_hint import hint_has_hard_blocker
from core.step_signal_metadata import compute_requires_visual_confirmation, normalize_observability_status
from core.types import Observation, TutorRequest, TutorResponse
from core.vars import VarResolver, VarResolverError, _find_missing_refs, _safe_eval
from jsonschema.exceptions import ValidationError
from ports.model_port import ModelPort


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TELEMETRY_MAP_PATH = _REPO_ROOT / "packs" / "fa18c_startup" / "telemetry_map.yaml"
_DERIVED_CONDITION_FALLBACKS: dict[str, tuple[str, ...]] = {
    "vars.apu_start_support_complete==true": (
        "vars.apu_ready==true",
        "vars.engine_crank_right_complete==true",
    ),
}
_VAR_RESOLVER_CACHE: OrderedDict[str, VarResolver] = OrderedDict()
_MAX_VAR_RESOLVER_CACHE = 8


def _load_var_resolver_for_path(path_text: str) -> VarResolver | None:
    cached = _VAR_RESOLVER_CACHE.get(path_text)
    if cached is not None:
        _VAR_RESOLVER_CACHE.move_to_end(path_text)
        return cached
    try:
        resolver = VarResolver.from_yaml(Path(path_text))
    except (FileNotFoundError, OSError, VarResolverError):
        return None
    _VAR_RESOLVER_CACHE[path_text] = resolver
    _VAR_RESOLVER_CACHE.move_to_end(path_text)
    while len(_VAR_RESOLVER_CACHE) > max(1, int(_MAX_VAR_RESOLVER_CACHE)):
        _VAR_RESOLVER_CACHE.popitem(last=False)
    return resolver


def _normalize_telemetry_map_path(raw: Any) -> Path | None:
    if isinstance(raw, Path):
        return raw.expanduser().resolve()
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    return None


def _dedupe_non_empty_strings(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _generation_mode_from_repair_state(
    *,
    json_repaired: bool,
    repair_applied: bool,
) -> str:
    if json_repaired or repair_applied:
        return "repair"
    return "model"


class BaseHelpModel(ModelPort):
    provider: str = "unknown"

    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout_s: float,
        lang: str,
        log_raw_llm_text: bool = False,
        print_model_io: bool = False,
        telemetry_map_path: str | Path | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.lang = lang
        self.log_raw_llm_text = bool(log_raw_llm_text)
        self.print_model_io = bool(print_model_io)
        self.telemetry_map_path = _normalize_telemetry_map_path(telemetry_map_path)

        if client is None:
            try:
                import httpx  # local import keeps unit tests offline with fake client
            except ModuleNotFoundError as exc:
                raise RuntimeError("httpx is required when no client is injected") from exc
            self._client = httpx.Client(timeout=self.timeout_s)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "BaseHelpModel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def plan_next_step(self, observation: Observation, request: TutorRequest | None = None) -> TutorResponse:
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request: TutorRequest | None = None) -> TutorResponse:
        start = perf_counter()
        self._reset_runtime_metadata()
        deterministic_inference = StepInferenceResult(
            inferred_step_id=observation.procedure_hint if isinstance(observation.procedure_hint, str) else None,
            missing_conditions=(),
        )
        recent_ui_targets: list[str] = []
        inference_vars: Mapping[str, Any] = {}
        deterministic_inference_error: str | None = None
        try:
            deterministic_inference, recent_ui_targets, inference_vars = self._compute_deterministic_inference(
                observation,
                request,
            )
        except Exception as exc:
            deterministic_inference_error = f"{type(exc).__name__}: {exc}"
        deterministic_hint = self._serialize_deterministic_hint(
            deterministic_inference,
            recent_ui_targets,
            inference_vars,
        )
        prompt_meta: dict[str, Any] = {}
        delta_dropped_count = self._extract_delta_dropped_count(request)
        prompt_budget_used = 0
        retry_count = 0
        retry_reason: str | None = None
        repair_details = self._empty_repair_details()
        raw_text_attempts: list[str] = []
        try:
            messages, prompt_meta = self._build_messages(
                observation,
                request,
                deterministic_inference=deterministic_inference,
                recent_ui_targets=recent_ui_targets,
                deterministic_hint=deterministic_hint,
            )
            prompt_budget_used = int(prompt_meta.get("prompt_tokens_est") or 0)
            if self.print_model_io:
                self._print_model_io_block(
                    "PROMPT",
                    self._render_debug_messages(messages),
                    request=request,
                    attempt=1,
                )
            raw_text = self._chat_with_failure_classification(messages)
            self._print_model_io_block("REPLY", raw_text, request=request, attempt=1)
            if self.log_raw_llm_text:
                raw_text_attempts.append(raw_text)
            try:
                help_obj, extraction, repair_details = parse_help_response_with_diagnostics(raw_text)
            except Exception as first_parse_exc:
                contextual_repair = self._attempt_contextual_schema_repair(
                    raw_text,
                    request=request,
                    deterministic_hint=deterministic_hint,
                )
                if contextual_repair is not None:
                    help_obj, extraction, repair_details = contextual_repair
                else:
                    retry_count = 1
                    retry_reason = f"{type(first_parse_exc).__name__}: {first_parse_exc}"
                    retry_messages = self._build_retry_messages(messages, first_parse_exc)
                    if self.print_model_io:
                        self._print_model_io_block(
                            "PROMPT",
                            self._render_debug_messages(retry_messages),
                            request=request,
                            attempt=2,
                        )
                    raw_text = self._chat_with_failure_classification(retry_messages)
                    self._print_model_io_block("REPLY", raw_text, request=request, attempt=2)
                    if self.log_raw_llm_text:
                        raw_text_attempts.append(raw_text)
                    try:
                        help_obj, extraction, repair_details = parse_help_response_with_diagnostics(raw_text)
                    except Exception as retry_parse_exc:
                        contextual_repair = self._attempt_contextual_schema_repair(
                            raw_text,
                            request=request,
                            deterministic_hint=deterministic_hint,
                        )
                        if contextual_repair is None:
                            raise retry_parse_exc
                        help_obj, extraction, repair_details = contextual_repair
            self._validate_context_bounds(help_obj, request)
            evidence_guardrail_reasons = self._enforce_evidence_guardrail(help_obj, prompt_meta)
            mapped = map_help_response_to_tutor_response(
                help_obj,
                request=request,
                status="ok",
                lang=self.lang,
            )
            safe_message = sanitize_public_model_text(mapped.message, lang=self.lang)
            safe_explanations = [
                sanitize_public_model_text(item, lang=self.lang) for item in mapped.explanations
            ]
            metadata = dict(mapped.metadata)
            metadata.update(
                {
                    "provider": self.provider,
                    "model": self.model_name,
                    "generation_mode": _generation_mode_from_repair_state(
                        json_repaired=extraction.json_repaired,
                        repair_applied=bool(repair_details.get("repair_applied")),
                    ),
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "help_response": sanitize_help_response_for_log(help_obj, lang=self.lang),
                    "json_repaired": extraction.json_repaired,
                    "json_repair_reasons": list(extraction.repair_reasons),
                    "evidence_guardrail_applied": bool(evidence_guardrail_reasons),
                    "evidence_guardrail_reasons": evidence_guardrail_reasons,
                    "prompt_budget_used": prompt_budget_used,
                    "delta_dropped_count": delta_dropped_count,
                    "prompt_build": prompt_meta,
                    "deterministic_step_hint": deterministic_hint,
                    "deterministic_inference_error": deterministic_inference_error,
                    "retry_count": retry_count,
                    "retry_reason": retry_reason,
                    "repair_applied": bool(repair_details.get("repair_applied")),
                    "repair_details": repair_details,
                    "fallback_overlay_used": False,
                    "fallback_overlay_reason": None,
                }
            )
            metadata.update(self._collect_runtime_metadata())
            if evidence_guardrail_reasons:
                metadata = merge_failure_metadata(
                    metadata,
                    EVIDENCE_FAIL,
                    stage="evidence_guardrail",
                )
            if self.log_raw_llm_text:
                metadata["raw_llm_text"] = raw_text_attempts[-1] if raw_text_attempts else ""
                metadata["raw_llm_text_attempts"] = list(raw_text_attempts)
            return TutorResponse(
                status=mapped.status,
                in_reply_to=mapped.in_reply_to,
                message=safe_message,
                actions=list(mapped.actions),
                explanations=safe_explanations,
                metadata=metadata,
            )
        except Exception as exc:
            fallback_message = self._build_deterministic_fallback_message(
                deterministic_inference,
                inference_vars,
            )
            error_metadata = {
                "provider": self.provider,
                "model": self.model_name,
                "generation_mode": "fallback",
                "latency_ms": int((perf_counter() - start) * 1000),
                "error_type": type(exc).__name__,
                "error": redact_sensitive_text(str(exc)),
                "prompt_budget_used": prompt_budget_used,
                "delta_dropped_count": delta_dropped_count,
                "prompt_build": prompt_meta,
                "deterministic_step_hint": deterministic_hint,
                "deterministic_inference_error": deterministic_inference_error,
                "retry_count": retry_count,
                "retry_reason": retry_reason,
                "repair_applied": bool(repair_details.get("repair_applied")),
                "repair_details": repair_details,
                "fallback_overlay_used": False,
                "fallback_overlay_reason": None,
            }
            error_metadata.update(self._collect_runtime_metadata())
            failure_code = exception_failure_code(exc)
            failure_stage = exception_failure_stage(exc)
            if failure_code is not None:
                error_metadata = merge_failure_metadata(
                    error_metadata,
                    failure_code,
                    stage=failure_stage,
                )
            if self.log_raw_llm_text:
                error_metadata["raw_llm_text"] = raw_text_attempts[-1] if raw_text_attempts else ""
                error_metadata["raw_llm_text_attempts"] = list(raw_text_attempts)
            return TutorResponse(
                status="error",
                in_reply_to=request.request_id if request else None,
                message=fallback_message,
                actions=[],
                metadata=error_metadata,
            )

    def _build_messages(
        self,
        observation: Observation,
        request: TutorRequest | None,
        *,
        deterministic_inference: StepInferenceResult | None = None,
        recent_ui_targets: list[str] | None = None,
        deterministic_hint: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        schema = get_help_response_schema()
        schema_step_ids = schema["properties"]["next"]["properties"]["step_id"]["enum"]
        schema_targets = schema["properties"]["overlay"]["properties"]["targets"]["items"]["enum"]
        schema_categories = schema["properties"]["diagnosis"]["properties"]["error_category"]["enum"]

        context = request.context if request and isinstance(request.context, dict) else {}
        candidate_steps = context.get("candidate_steps")
        if not isinstance(candidate_steps, list) or not candidate_steps:
            candidate_steps = list(schema_step_ids)
        inference = deterministic_inference
        inferred_step_id = None
        if inference is not None:
            inferred_step_id = inference.inferred_step_id
        candidate_steps = self._prioritize_inferred_step(candidate_steps, inferred_step_id)
        allowlist = context.get("overlay_target_allowlist")
        if not isinstance(allowlist, list) or not allowlist:
            allowlist = list(schema_targets)
        scenario_profile = context.get("scenario_profile")
        if not isinstance(scenario_profile, str) or not scenario_profile:
            scenario_profile = None
        normalized_recent_ui_targets = list(recent_ui_targets or [])
        hint_payload = (
            dict(deterministic_hint)
            if isinstance(deterministic_hint, Mapping)
            else self._serialize_deterministic_hint(inference, normalized_recent_ui_targets)
        )
        step_ui_targets = hint_payload.get("step_ui_targets")
        missing_conditions = hint_payload.get("missing_conditions")
        gate_blockers = hint_payload.get("gate_blockers")
        has_hard_blocker = hint_has_hard_blocker(missing_conditions, gate_blockers)
        if has_hard_blocker and isinstance(step_ui_targets, list):
            step_target_allowset = {item for item in step_ui_targets if isinstance(item, str) and item}
            narrowed_allowlist = [
                item for item in allowlist if isinstance(item, str) and item in step_target_allowset
            ]
            if narrowed_allowlist:
                allowlist = narrowed_allowlist
        if scenario_profile is not None and "scenario_profile" not in hint_payload:
            hint_payload["scenario_profile"] = scenario_profile

        prompt_context = {
            "intent": request.intent if request else "help",
            "message": request.message if request else None,
            "scenario_profile": scenario_profile,
            "candidate_steps": candidate_steps,
            "overlay_target_allowlist": allowlist,
            "vars": context.get("vars"),
            "recent_deltas": context.get("recent_deltas"),
            "recent_actions": context.get("recent_actions"),
            "gates": context.get("gates"),
            "rag_topk": context.get("rag_topk"),
            "observation": {
                "procedure_hint": observation.procedure_hint,
                "source": observation.source,
            },
            "error_category_enum": schema_categories,
            "deterministic_step_hint": hint_payload,
        }
        prompt_result = build_help_prompt_result(prompt_context, self.lang)
        prompt_meta = dict(prompt_result.metadata)
        prompt_meta["deterministic_step_hint"] = hint_payload
        return (
            [
                {"role": "system", "content": "You are SimTutor. Reply with JSON only."},
                {"role": "user", "content": prompt_result.prompt},
            ],
            prompt_meta,
        )

    def _print_model_io_block(
        self,
        kind: str,
        text: str,
        *,
        request: TutorRequest | None,
        attempt: int,
    ) -> None:
        if not self.print_model_io:
            return
        request_id = request.request_id if request is not None else None
        header = f"[MODEL_IO][{kind}]"
        if isinstance(request_id, str) and request_id:
            header += f"[request_id={request_id}]"
        header += f"[attempt={attempt}]"
        print(header)
        print(text)
        print(f"{header}[END]")

    @staticmethod
    def _render_debug_messages(messages: list[dict[str, Any]]) -> str:
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
                    if item_type == "text" and isinstance(item.get("text"), str):
                        rendered.append(str(item["text"]))
                        continue
                    if item_type == "image_url":
                        image_count += 1
                        continue
                    rendered.append(str(dict(item)))
                if image_count > 0:
                    rendered.append(f"[multimodal_images={image_count}]")
                continue
            rendered.append(str(content))
        return "\n".join(rendered)

    def _compute_deterministic_inference(
        self,
        observation: Observation,
        request: TutorRequest | None,
    ) -> tuple[StepInferenceResult, list[str], Mapping[str, Any]]:
        context = request.context if request and isinstance(request.context, dict) else {}
        inference_vars = self._pick_inference_vars(observation, context)
        recent_ui_targets = extract_recent_ui_targets(context)
        if not recent_ui_targets:
            payload = observation.payload if isinstance(observation.payload, Mapping) else {}
            recent_ui_targets = normalize_recent_ui_targets(payload.get("recent_ui_targets"))

        pack_steps = load_pack_steps()
        raw_scenario_profile = context.get("scenario_profile")
        scenario_profile = raw_scenario_profile if isinstance(raw_scenario_profile, str) and raw_scenario_profile else None
        inference = infer_step_id(
            pack_steps,
            inference_vars,
            recent_ui_targets,
            scenario_profile=scenario_profile,
            vision_facts=context.get("vision_facts"),
        )
        if inference.inferred_step_id is None and isinstance(observation.procedure_hint, str) and observation.procedure_hint:
            inference = StepInferenceResult(
                inferred_step_id=observation.procedure_hint,
                missing_conditions=tuple(inference.missing_conditions),
            )
        return inference, recent_ui_targets, inference_vars

    def _pick_inference_vars(
        self,
        observation: Observation,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        vars_raw = context.get("vars")
        if isinstance(vars_raw, Mapping):
            return self._resolve_inference_vars(vars_raw, context=context)
        payload = observation.payload if isinstance(observation.payload, Mapping) else {}
        payload_vars = payload.get("vars")
        if isinstance(payload_vars, Mapping):
            return self._resolve_inference_vars(payload_vars, context=context)
        return {}

    def _resolve_inference_vars(
        self,
        vars_raw: Mapping[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        resolver = self._select_var_resolver(context)
        if resolver is None:
            return vars_raw
        context = {
            "bios": {},
            "lo": {},
            "cockpit_args": {},
            "vars": dict(vars_raw),
        }
        merged = dict(vars_raw)
        source_missing = {
            item
            for item in vars_raw.get("vars_source_missing", [])
            if isinstance(item, str) and item
        } if isinstance(vars_raw.get("vars_source_missing"), list) else set()
        try:
            for key, expr in resolver.rules.items():
                if not isinstance(key, str) or not key or key in merged or key == "vars_source_missing":
                    continue
                if expr is None:
                    merged[key] = None
                    source_missing.add(key)
                    context["vars"] = merged
                    continue
                if not isinstance(expr, str):
                    merged[key] = expr
                    context["vars"] = merged
                    continue
                expr_text = expr.strip()
                if expr_text.startswith("derived(") and expr_text.endswith(")"):
                    expr_text = expr_text[len("derived(") : -1].strip()
                missing_refs = _find_missing_refs(expr_text, context)
                value = _safe_eval(expr_text, context)
                merged[key] = value
                context["vars"] = merged
                if value is None or missing_refs:
                    source_missing.add(key)
        except VarResolverError:
            return vars_raw
        merged["vars_source_missing"] = sorted(source_missing)
        return merged

    def _select_var_resolver(self, context: Mapping[str, Any] | None) -> VarResolver | None:
        resolver_path = self._resolve_active_telemetry_map_path(context)
        if resolver_path is None:
            return None
        return _load_var_resolver_for_path(str(resolver_path))

    def _resolve_active_telemetry_map_path(self, context: Mapping[str, Any] | None) -> Path | None:
        if isinstance(context, Mapping):
            context_telemetry_map = _normalize_telemetry_map_path(context.get("telemetry_map_path"))
            if context_telemetry_map is not None:
                return context_telemetry_map
            context_pack_path = _normalize_telemetry_map_path(context.get("pack_path"))
            if context_pack_path is not None:
                candidate = context_pack_path.parent / "telemetry_map.yaml"
                if candidate.is_file():
                    return candidate.resolve()
        if self.telemetry_map_path is not None:
            return self.telemetry_map_path
        return _DEFAULT_TELEMETRY_MAP_PATH

    def _present_missing_conditions(
        self,
        missing_conditions: list[str] | tuple[str, ...],
        inference_vars: Mapping[str, Any],
    ) -> list[str]:
        presented: list[str] = []
        seen: set[str] = set()
        for condition in missing_conditions:
            if not isinstance(condition, str) or not condition:
                continue
            replacements = self._expand_presented_missing_condition(condition, inference_vars)
            for item in replacements:
                if item in seen:
                    continue
                seen.add(item)
                presented.append(item)
        return presented

    def _expand_presented_missing_condition(
        self,
        condition: str,
        inference_vars: Mapping[str, Any],
    ) -> tuple[str, ...]:
        fallback_conditions = _DERIVED_CONDITION_FALLBACKS.get(condition)
        if not fallback_conditions:
            return (condition,)
        unresolved: list[str] = []
        for fallback_condition in fallback_conditions:
            key = self._extract_condition_var_key(fallback_condition)
            if key is None:
                unresolved.append(fallback_condition)
                continue
            value = inference_vars.get(key)
            if value is True:
                continue
            unresolved.append(fallback_condition)
        if unresolved:
            return (unresolved[0],)
        return (condition,)

    @staticmethod
    def _extract_condition_var_key(condition: str) -> str | None:
        prefix = "vars."
        suffix = "==true"
        if not condition.startswith(prefix) or not condition.endswith(suffix):
            return None
        key = condition[len(prefix) : -len(suffix)]
        return key or None

    def _prioritize_inferred_step(self, candidate_steps: list[Any], inferred_step_id: str | None) -> list[str]:
        normalized: list[str] = [step for step in candidate_steps if isinstance(step, str) and step]
        if not normalized or not inferred_step_id or inferred_step_id not in normalized:
            return normalized
        return [inferred_step_id] + [step for step in normalized if step != inferred_step_id]

    def _serialize_deterministic_hint(
        self,
        inference: StepInferenceResult | None,
        recent_ui_targets: list[str],
        inference_vars: Mapping[str, Any],
    ) -> dict[str, Any]:
        if inference is None:
            inferred_step_id = None
            missing_conditions: list[str] = []
        else:
            inferred_step_id = inference.inferred_step_id
            missing_conditions = self._present_missing_conditions(inference.missing_conditions, inference_vars)
        hint = {
            "inferred_step_id": inferred_step_id,
            "missing_conditions": missing_conditions,
            "recent_ui_targets": list(recent_ui_targets),
        }
        if isinstance(inferred_step_id, str) and inferred_step_id:
            step_signal = self._lookup_step_signal(inferred_step_id)
            if step_signal:
                hint.update(step_signal)
        return hint

    def _lookup_step_signal(self, step_id: str) -> dict[str, Any]:
        for step in load_pack_steps():
            candidate_id = step.get("id") or step.get("step_id")
            if candidate_id != step_id:
                continue
            observability = normalize_observability_status(step.get("observability"))
            requirements_raw = step.get("evidence_requirements")
            requirements = (
                _dedupe_non_empty_strings(requirements_raw)
            )
            requires_visual_confirmation = compute_requires_visual_confirmation(observability, requirements)
            return {
                "observability": observability,
                "observability_status": observability,
                "step_evidence_requirements": requirements,
                "step_ui_targets": _dedupe_non_empty_strings(step.get("ui_targets")),
                "requires_visual_confirmation": requires_visual_confirmation,
            }
        return {}

    def _build_deterministic_fallback_message(
        self,
        inference: StepInferenceResult | None,
        inference_vars: Mapping[str, Any],
    ) -> str:
        inferred_step_id = inference.inferred_step_id if inference else None
        raw_missing_conditions = list(inference.missing_conditions) if inference else []
        missing_conditions = self._present_missing_conditions(raw_missing_conditions, inference_vars)
        if self.lang == "zh":
            if inferred_step_id and missing_conditions:
                return f"你大概率卡在 {inferred_step_id}，下一步请先满足：{'; '.join(missing_conditions)}。"
            if inferred_step_id:
                return f"你大概率卡在 {inferred_step_id}，下一步请按该步骤检查并执行。"
            return "无法生成模型答复，请先检查当前步骤前置条件后再触发 Help。"
        if inferred_step_id and missing_conditions:
            return (
                f"You are likely stuck at {inferred_step_id}. "
                f"Please satisfy: {'; '.join(missing_conditions)}."
            )
        if inferred_step_id:
            return f"You are likely stuck at {inferred_step_id}. Please re-check and execute that step."
        return "Unable to generate help response, please check the current system status and try again."

    def _validate_context_bounds(self, help_obj: Mapping[str, Any], request: TutorRequest | None) -> None:
        if request is None or not isinstance(request.context, dict):
            return
        context = request.context

        candidate_steps = context.get("candidate_steps")
        if isinstance(candidate_steps, list) and candidate_steps:
            allowed_steps = {sid for sid in candidate_steps if isinstance(sid, str)}
            for path in ("diagnosis.step_id", "next.step_id"):
                section, key = path.split(".")
                sid = help_obj[section][key]
                if sid not in allowed_steps:
                    raise annotate_exception(
                        ValueError(f"{path}={sid!r} not in candidate_steps"),
                        code=SCHEMA_FAIL,
                        stage="context_bounds",
                    )

        allowlist = context.get("overlay_target_allowlist")
        if isinstance(allowlist, list) and allowlist:
            allowed_targets = {target for target in allowlist if isinstance(target, str)}
            for idx, target in enumerate(help_obj["overlay"]["targets"]):
                if target not in allowed_targets:
                    raise annotate_exception(
                        ValueError(f"overlay.targets[{idx}]={target!r} not in overlay_target_allowlist"),
                        code=ALLOWLIST_FAIL,
                        stage="context_bounds",
                    )

    def _enforce_evidence_guardrail(
        self,
        help_obj: dict[str, Any],
        prompt_meta: Mapping[str, Any],
    ) -> list[str]:
        overlay = help_obj.get("overlay")
        if not isinstance(overlay, dict):
            return ["missing_overlay"]

        targets_raw = overlay.get("targets")
        if not isinstance(targets_raw, list):
            return ["invalid_overlay_targets_type"]
        targets = [target for target in targets_raw if isinstance(target, str)]
        if not targets:
            return []

        evidence_raw = overlay.get("evidence")
        if not isinstance(evidence_raw, list):
            evidence_raw = []

        targets_set = set(targets)
        refs_by_target: dict[str, set[str]] = {}
        unexpected_evidence_targets: set[str] = set()

        for item in evidence_raw:
            if not isinstance(item, Mapping):
                continue
            target = item.get("target")
            ref = item.get("ref")
            if not isinstance(target, str):
                continue
            if target not in targets_set:
                unexpected_evidence_targets.add(target)
                continue
            if isinstance(ref, str) and ref:
                refs_by_target.setdefault(target, set()).add(ref)

        allowed_refs = prompt_meta.get("allowed_evidence_refs")
        allowed_ref_set = (
            {ref for ref in allowed_refs if isinstance(ref, str) and ref}
            if isinstance(allowed_refs, list)
            else set()
        )

        missing_targets: list[str] = []
        invalid_ref_targets: list[str] = []
        for target in targets:
            refs = refs_by_target.get(target, set())
            if not refs:
                missing_targets.append(target)
                continue
            if not allowed_ref_set:
                invalid_ref_targets.append(target)
                continue
            if not refs.issubset(allowed_ref_set):
                invalid_ref_targets.append(target)

        reasons: list[str] = []
        if missing_targets:
            reasons.append(f"missing_target_evidence:{','.join(sorted(set(missing_targets)))}")
        if invalid_ref_targets:
            reasons.append(f"invalid_target_evidence_refs:{','.join(sorted(set(invalid_ref_targets)))}")
        if unexpected_evidence_targets:
            reasons.append(
                "evidence_target_not_in_overlay_targets:" + ",".join(sorted(unexpected_evidence_targets))
            )
        if not reasons:
            return []

        help_obj["overlay"]["targets"] = []
        help_obj["overlay"]["evidence"] = []
        details = sorted(set(missing_targets + invalid_ref_targets))
        clarification = self._build_clarification_message(details)
        explanations = help_obj.get("explanations")
        if isinstance(explanations, list):
            explanations.insert(0, clarification)
        else:
            help_obj["explanations"] = [clarification]
        return reasons

    def _build_clarification_message(self, details: list[str]) -> str:
        if self.lang == "zh":
            if details:
                return "\u9700\u8981\u66f4\u591a\u4fe1\u606f/\u8bf7\u786e\u8ba4: " + ", ".join(details)
            return "\u9700\u8981\u66f4\u591a\u4fe1\u606f/\u8bf7\u786e\u8ba4\u5f53\u524d\u5173\u952e\u6b65\u9aa4\u72b6\u6001\u3002"
        if details:
            return "Need more information/please confirm: " + ", ".join(details)
        return "Need more information/please confirm current critical step states."

    def _extract_delta_dropped_count(self, request: TutorRequest | None) -> int:
        if request is None or not isinstance(request.context, dict):
            return 0
        context = request.context
        direct = context.get("delta_dropped_count")
        if isinstance(direct, int) and not isinstance(direct, bool):
            return max(0, direct)
        delta_summary = context.get("delta_summary")
        if isinstance(delta_summary, Mapping):
            dropped_stats = delta_summary.get("dropped_stats")
            if isinstance(dropped_stats, Mapping):
                dropped = dropped_stats.get("dropped_total")
                if isinstance(dropped, int) and not isinstance(dropped, bool):
                    return max(0, dropped)
        return 0

    def _build_retry_messages(
        self,
        messages: list[dict[str, Any]],
        parse_error: Exception,
    ) -> list[dict[str, Any]]:
        error_text = f"{type(parse_error).__name__}: {parse_error}"
        if len(error_text) > 240:
            error_text = error_text[:240] + "..."
        if self.lang == "zh":
            retry_hint = (
                "上一次输出未通过结构化校验。"
                "请仅输出一个合法 JSON 对象，严格遵循 schema 与枚举，不要输出任何解释文本。"
                "顶层必须包含 diagnosis、next、overlay、explanations、confidence 五个字段。"
                f"上一轮错误：{error_text}"
            )
        else:
            retry_hint = (
                "Previous output failed structured validation. "
                "Return exactly one valid JSON object that strictly follows schema/enums, with no prose. "
                "The top-level object must include diagnosis, next, overlay, explanations, and confidence. "
                f"Previous error: {error_text}"
            )
        retry_messages = [dict(msg) for msg in messages]
        retry_messages.append({"role": "user", "content": retry_hint})
        return retry_messages

    def _chat_with_failure_classification(self, messages: list[dict[str, Any]]) -> str:
        try:
            return self._chat(messages)
        except Exception as exc:
            failure_code, failure_stage = self._classify_chat_exception(exc)
            if failure_code is not None and failure_stage is not None:
                annotate_exception(exc, code=failure_code, stage=failure_stage)
            raise

    def _classify_chat_exception(self, exc: Exception) -> tuple[str | None, str | None]:
        if self._is_transport_or_http_exception(exc):
            return MODEL_HTTP_FAIL, "model_http"
        if isinstance(exc, (ValueError, TypeError, KeyError)):
            return SCHEMA_FAIL, "model_response_envelope"
        return None, None

    def _is_transport_or_http_exception(self, exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        if isinstance(exc, RuntimeError):
            message = str(exc).strip().lower()
            if message.startswith("http "):
                return True
        try:
            import httpx  # type: ignore
        except ModuleNotFoundError:
            httpx = None  # type: ignore[assignment]
        if httpx is not None and isinstance(exc, (httpx.RequestError, httpx.HTTPStatusError)):
            return True
        return False

    def _empty_repair_details(self) -> dict[str, Any]:
        return {
            "repair_applied": False,
            "repaired_evidence_types": 0,
            "dropped_unrepairable_evidence": 0,
            "details": [],
        }

    def _attempt_contextual_schema_repair(
        self,
        raw_text: str,
        *,
        request: TutorRequest | None,
        deterministic_hint: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Any, dict[str, Any]] | None:
        try:
            help_obj, extraction = parse_first_json(raw_text)
        except Exception:
            return None
        if not isinstance(help_obj, dict):
            return None
        repaired_obj, details = self._repair_missing_help_fields_from_context(
            help_obj,
            request=request,
            deterministic_hint=deterministic_hint,
        )
        if not details:
            return None
        try:
            validate_help_response(repaired_obj)
        except ValidationError:
            return None
        return repaired_obj, extraction, {
            "repair_applied": True,
            "repaired_evidence_types": 0,
            "dropped_unrepairable_evidence": 0,
            "details": details,
        }

    def _repair_missing_help_fields_from_context(
        self,
        help_obj: dict[str, Any],
        *,
        request: TutorRequest | None,
        deterministic_hint: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        repaired = dict(help_obj)
        details: list[dict[str, Any]] = []
        inferred_step_id = self._pick_repair_step_id(repaired, deterministic_hint)
        missing_conditions = _dedupe_non_empty_strings(deterministic_hint.get("missing_conditions"))

        diagnosis = repaired.get("diagnosis")
        if not isinstance(diagnosis, Mapping):
            if inferred_step_id:
                repaired["diagnosis"] = {"step_id": inferred_step_id, "error_category": "OM"}
                details.append({"field": "diagnosis", "action": "filled_from_deterministic_hint"})
        else:
            diagnosis_mut = dict(diagnosis)
            changed = False
            if not isinstance(diagnosis_mut.get("step_id"), str) or not diagnosis_mut.get("step_id"):
                if inferred_step_id:
                    diagnosis_mut["step_id"] = inferred_step_id
                    changed = True
                    details.append({"field": "diagnosis.step_id", "action": "filled_from_deterministic_hint"})
            if not isinstance(diagnosis_mut.get("error_category"), str) or not diagnosis_mut.get("error_category"):
                diagnosis_mut["error_category"] = "OM"
                changed = True
                details.append({"field": "diagnosis.error_category", "action": "filled_default"})
            if changed:
                repaired["diagnosis"] = diagnosis_mut

        next_block = repaired.get("next")
        next_step_id = inferred_step_id
        if isinstance(repaired.get("diagnosis"), Mapping):
            diag_step_id = repaired["diagnosis"].get("step_id")
            if isinstance(diag_step_id, str) and diag_step_id:
                next_step_id = diag_step_id
        if not isinstance(next_block, Mapping):
            if next_step_id:
                repaired["next"] = {"step_id": next_step_id}
                details.append({"field": "next", "action": "filled_from_diagnosis"})
        else:
            next_mut = dict(next_block)
            if not isinstance(next_mut.get("step_id"), str) or not next_mut.get("step_id"):
                if next_step_id:
                    next_mut["step_id"] = next_step_id
                    repaired["next"] = next_mut
                    details.append({"field": "next.step_id", "action": "filled_from_diagnosis"})

        overlay = repaired.get("overlay")
        if not isinstance(overlay, Mapping):
            repaired["overlay"] = {"targets": [], "evidence": []}
            details.append({"field": "overlay", "action": "filled_empty"})
        else:
            overlay_mut = dict(overlay)
            changed = False
            if not isinstance(overlay_mut.get("targets"), list):
                overlay_mut["targets"] = []
                changed = True
                details.append({"field": "overlay.targets", "action": "filled_empty"})
            if not isinstance(overlay_mut.get("evidence"), list):
                overlay_mut["evidence"] = []
                changed = True
                details.append({"field": "overlay.evidence", "action": "filled_empty"})
            if changed:
                repaired["overlay"] = overlay_mut

        explanations = repaired.get("explanations")
        if not isinstance(explanations, list) or not any(isinstance(item, str) and item.strip() for item in explanations):
            repaired["explanations"] = [self._default_repair_explanation(inferred_step_id, missing_conditions)]
            details.append({"field": "explanations", "action": "filled_default"})

        confidence = repaired.get("confidence")
        if not isinstance(confidence, (int, float)):
            repaired["confidence"] = 0.51
            details.append({"field": "confidence", "action": "filled_default"})

        return repaired, details

    def _pick_repair_step_id(self, help_obj: Mapping[str, Any], deterministic_hint: Mapping[str, Any]) -> str | None:
        diagnosis = help_obj.get("diagnosis")
        if isinstance(diagnosis, Mapping):
            step_id = diagnosis.get("step_id")
            if isinstance(step_id, str) and step_id:
                return step_id
        next_block = help_obj.get("next")
        if isinstance(next_block, Mapping):
            step_id = next_block.get("step_id")
            if isinstance(step_id, str) and step_id:
                return step_id
        hint_step_id = deterministic_hint.get("inferred_step_id")
        if isinstance(hint_step_id, str) and hint_step_id:
            return hint_step_id
        return None

    def _default_repair_explanation(self, step_id: str | None, missing_conditions: list[str]) -> str:
        if self.lang == "zh":
            if step_id and missing_conditions:
                return f"当前大概率卡在 {step_id}，请先满足：{'; '.join(missing_conditions)}。"
            if step_id:
                return f"当前大概率卡在 {step_id}，请确认该步骤状态。"
            return "请确认当前步骤状态。"
        if step_id and missing_conditions:
            return f"You are likely blocked at {step_id}. Satisfy: {'; '.join(missing_conditions)}."
        if step_id:
            return f"You are likely blocked at {step_id}. Confirm that step state."
        return "Confirm the current step state."

    def _reset_runtime_metadata(self) -> None:
        return None

    def _collect_runtime_metadata(self) -> dict[str, Any]:
        return {}

    def _chat(self, messages: list[dict[str, Any]]) -> str:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError
