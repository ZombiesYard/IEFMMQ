"""
Shared base implementation for HelpResponse-capable model adapters.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

from adapters.help_response_parser import parse_help_response_with_diagnostics
from adapters.prompting import build_help_prompt_result
from adapters.response_mapping import map_help_response_to_tutor_response
from adapters.step_inference import (
    StepInferenceResult,
    extract_recent_ui_targets,
    infer_step_id,
    load_pack_steps,
    normalize_recent_ui_targets,
)
from core.llm_schema import get_help_response_schema
from core.types import Observation, TutorRequest, TutorResponse
from ports.model_port import ModelPort


class BaseHelpModel(ModelPort):
    provider: str = "unknown"

    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout_s: float,
        lang: str,
        log_raw_llm_text: bool = False,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.lang = lang
        self.log_raw_llm_text = bool(log_raw_llm_text)

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
        deterministic_inference = StepInferenceResult(
            inferred_step_id=observation.procedure_hint if isinstance(observation.procedure_hint, str) else None,
            missing_conditions=(),
        )
        recent_ui_targets: list[str] = []
        deterministic_inference_error: str | None = None
        try:
            deterministic_inference, recent_ui_targets = self._compute_deterministic_inference(observation, request)
        except Exception as exc:
            deterministic_inference_error = f"{type(exc).__name__}: {exc}"
        deterministic_hint = self._serialize_deterministic_hint(deterministic_inference, recent_ui_targets)
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
            raw_text = self._chat(messages)
            if self.log_raw_llm_text:
                raw_text_attempts.append(raw_text)
            try:
                help_obj, extraction, repair_details = parse_help_response_with_diagnostics(raw_text)
            except Exception as first_parse_exc:
                retry_count = 1
                retry_reason = f"{type(first_parse_exc).__name__}: {first_parse_exc}"
                retry_messages = self._build_retry_messages(messages, first_parse_exc)
                raw_text = self._chat(retry_messages)
                if self.log_raw_llm_text:
                    raw_text_attempts.append(raw_text)
                help_obj, extraction, repair_details = parse_help_response_with_diagnostics(raw_text)
            self._validate_context_bounds(help_obj, request)
            evidence_guardrail_reasons = self._enforce_evidence_guardrail(help_obj, prompt_meta)
            mapped = map_help_response_to_tutor_response(help_obj, request=request, status="ok")
            metadata = dict(mapped.metadata)
            metadata.update(
                {
                    "provider": self.provider,
                    "model": self.model_name,
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "help_response": help_obj,
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
            if self.log_raw_llm_text:
                metadata["raw_llm_text"] = raw_text_attempts[-1] if raw_text_attempts else ""
                metadata["raw_llm_text_attempts"] = list(raw_text_attempts)
            return TutorResponse(
                status=mapped.status,
                in_reply_to=mapped.in_reply_to,
                message=mapped.message,
                actions=list(mapped.actions),
                explanations=list(mapped.explanations),
                metadata=metadata,
            )
        except Exception as exc:
            fallback_message = self._build_deterministic_fallback_message(deterministic_inference)
            error_metadata = {
                "provider": self.provider,
                "model": self.model_name,
                "latency_ms": int((perf_counter() - start) * 1000),
                "error_type": type(exc).__name__,
                "error": str(exc),
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
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
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

    def _compute_deterministic_inference(
        self,
        observation: Observation,
        request: TutorRequest | None,
    ) -> tuple[StepInferenceResult, list[str]]:
        context = request.context if request and isinstance(request.context, dict) else {}
        inference_vars = self._pick_inference_vars(observation, context)
        recent_ui_targets = extract_recent_ui_targets(context)
        if not recent_ui_targets:
            payload = observation.payload if isinstance(observation.payload, Mapping) else {}
            recent_ui_targets = normalize_recent_ui_targets(payload.get("recent_ui_targets"))

        pack_steps = load_pack_steps()
        inference = infer_step_id(pack_steps, inference_vars, recent_ui_targets)
        if inference.inferred_step_id is None and isinstance(observation.procedure_hint, str) and observation.procedure_hint:
            inference = StepInferenceResult(
                inferred_step_id=observation.procedure_hint,
                missing_conditions=tuple(inference.missing_conditions),
            )
        return inference, recent_ui_targets

    def _pick_inference_vars(
        self,
        observation: Observation,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        vars_raw = context.get("vars")
        if isinstance(vars_raw, Mapping):
            return vars_raw
        payload = observation.payload if isinstance(observation.payload, Mapping) else {}
        payload_vars = payload.get("vars")
        if isinstance(payload_vars, Mapping):
            return payload_vars
        return {}

    def _prioritize_inferred_step(self, candidate_steps: list[Any], inferred_step_id: str | None) -> list[str]:
        normalized: list[str] = [step for step in candidate_steps if isinstance(step, str) and step]
        if not normalized or not inferred_step_id or inferred_step_id not in normalized:
            return normalized
        return [inferred_step_id] + [step for step in normalized if step != inferred_step_id]

    def _serialize_deterministic_hint(
        self,
        inference: StepInferenceResult | None,
        recent_ui_targets: list[str],
    ) -> dict[str, Any]:
        if inference is None:
            inferred_step_id = None
            missing_conditions: list[str] = []
        else:
            inferred_step_id = inference.inferred_step_id
            missing_conditions = list(inference.missing_conditions)
        return {
            "inferred_step_id": inferred_step_id,
            "missing_conditions": missing_conditions,
            "recent_ui_targets": list(recent_ui_targets),
        }

    def _build_deterministic_fallback_message(self, inference: StepInferenceResult | None) -> str:
        inferred_step_id = inference.inferred_step_id if inference else None
        missing_conditions = list(inference.missing_conditions) if inference else []
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
                    raise ValueError(f"{path}={sid!r} not in candidate_steps")

        allowlist = context.get("overlay_target_allowlist")
        if isinstance(allowlist, list) and allowlist:
            allowed_targets = {target for target in allowlist if isinstance(target, str)}
            for idx, target in enumerate(help_obj["overlay"]["targets"]):
                if target not in allowed_targets:
                    raise ValueError(f"overlay.targets[{idx}]={target!r} not in overlay_target_allowlist")

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
        messages: list[dict[str, str]],
        parse_error: Exception,
    ) -> list[dict[str, str]]:
        error_text = f"{type(parse_error).__name__}: {parse_error}"
        if len(error_text) > 240:
            error_text = error_text[:240] + "..."
        if self.lang == "zh":
            retry_hint = (
                "上一次输出未通过结构化校验。"
                "请仅输出一个合法 JSON 对象，严格遵循 schema 与枚举，不要输出任何解释文本。"
                f"上一轮错误：{error_text}"
            )
        else:
            retry_hint = (
                "Previous output failed structured validation. "
                "Return exactly one valid JSON object that strictly follows schema/enums, with no prose. "
                f"Previous error: {error_text}"
            )
        retry_messages = [dict(msg) for msg in messages]
        retry_messages.append({"role": "user", "content": retry_hint})
        return retry_messages

    def _empty_repair_details(self) -> dict[str, Any]:
        return {
            "repair_applied": False,
            "repaired_evidence_types": 0,
            "dropped_unrepairable_evidence": 0,
            "details": [],
        }

    def _chat(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError
