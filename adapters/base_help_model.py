"""
Shared base implementation for HelpResponse-capable model adapters.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

from adapters.help_response_parser import parse_help_response_with_meta
from adapters.prompting import build_help_prompt_result
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
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.lang = lang

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
        prompt_meta: dict[str, Any] = {}
        delta_dropped_count = self._extract_delta_dropped_count(request)
        prompt_budget_used = 0
        try:
            messages, prompt_meta = self._build_messages(observation, request)
            prompt_budget_used = int(prompt_meta.get("prompt_tokens_est") or 0)
            raw_text = self._chat(messages)
            help_obj, extraction = parse_help_response_with_meta(raw_text)
            self._validate_context_bounds(help_obj, request)
            evidence_guardrail_reasons = self._enforce_evidence_guardrail(help_obj, prompt_meta)
            actions = [
                {"type": "overlay", "intent": "highlight", "target": target}
                for target in help_obj["overlay"]["targets"]
            ]
            return TutorResponse(
                status="ok",
                in_reply_to=request.request_id if request else None,
                message=help_obj["explanations"][0] if help_obj["explanations"] else None,
                actions=actions,
                explanations=list(help_obj["explanations"]),
                metadata={
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
                },
            )
        except Exception as exc:
            return TutorResponse(
                status="error",
                in_reply_to=request.request_id if request else None,
                message="Unable to generate help response, please check the current system status and try again.",
                actions=[],
                metadata={
                    "provider": self.provider,
                    "model": self.model_name,
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "prompt_budget_used": prompt_budget_used,
                    "delta_dropped_count": delta_dropped_count,
                    "prompt_build": prompt_meta,
                },
            )

    def _build_messages(
        self,
        observation: Observation,
        request: TutorRequest | None,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        schema = get_help_response_schema()
        schema_step_ids = schema["properties"]["next"]["properties"]["step_id"]["enum"]
        schema_targets = schema["properties"]["overlay"]["properties"]["targets"]["items"]["enum"]
        schema_categories = schema["properties"]["diagnosis"]["properties"]["error_category"]["enum"]

        context = request.context if request and isinstance(request.context, dict) else {}
        candidate_steps = context.get("candidate_steps")
        if not isinstance(candidate_steps, list) or not candidate_steps:
            candidate_steps = list(schema_step_ids)
        allowlist = context.get("overlay_target_allowlist")
        if not isinstance(allowlist, list) or not allowlist:
            allowlist = list(schema_targets)

        prompt_context = {
            "intent": request.intent if request else "help",
            "message": request.message if request else None,
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
        }
        prompt_result = build_help_prompt_result(prompt_context, self.lang)
        return (
            [
                {"role": "system", "content": "You are SimTutor. Reply with JSON only."},
                {"role": "user", "content": prompt_result.prompt},
            ],
            prompt_result.metadata,
        )

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
        if isinstance(direct, int):
            return max(0, direct)
        delta_summary = context.get("delta_summary")
        if isinstance(delta_summary, Mapping):
            dropped_stats = delta_summary.get("dropped_stats")
            if isinstance(dropped_stats, Mapping):
                dropped = dropped_stats.get("dropped_total")
                if isinstance(dropped, int):
                    return max(0, dropped)
        return 0

    def _chat(self, messages: list[dict[str, str]]) -> str:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError
