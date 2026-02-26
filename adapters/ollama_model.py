"""
Ollama-backed ModelPort adapter for HelpResponse generation.

This adapter enforces a strict pipeline:
raw model text -> json_extract -> json.loads -> validate_help_response.
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Mapping

from core.llm_schema import get_help_response_schema, validate_help_response
from core.types import Observation, TutorRequest, TutorResponse
from ports.model_port import ModelPort


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _json_extract(raw_text: str) -> str:
    text = _strip_code_fence(raw_text)
    if text.startswith("{") and text.endswith("}"):
        return text

    in_string = False
    escaped = False
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    raise ValueError("Model output does not contain JSON object")


def _parse_help_response(raw_text: str) -> dict[str, Any]:
    extracted = _json_extract(raw_text)
    obj = json.loads(extracted)
    if not isinstance(obj, dict):
        raise ValueError("HelpResponse must be a JSON object")
    validate_help_response(obj)
    return obj


class OllamaModel(ModelPort):
    def __init__(
        self,
        model_name: str = "qwen3.5:35b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 20.0,
        lang: str = "zh",
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

    def plan_next_step(self, observation: Observation, request: TutorRequest | None = None) -> TutorResponse:
        return self.explain_error(observation, request)

    def explain_error(self, observation: Observation, request: TutorRequest | None = None) -> TutorResponse:
        start = perf_counter()
        try:
            messages = self._build_messages(observation, request)
            raw_text = self._chat(messages)
            help_obj = _parse_help_response(raw_text)
            self._validate_context_bounds(help_obj, request)
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
                    "provider": "ollama",
                    "model": self.model_name,
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "help_response": help_obj,
                },
            )
        except Exception as exc:
            return TutorResponse(
                status="error",
                in_reply_to=request.request_id if request else None,
                message="Unable to generate help response, please check the current system status and try again.",
                actions=[],
                metadata={
                    "provider": "ollama",
                    "model": self.model_name,
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    def _build_messages(
        self,
        observation: Observation,
        request: TutorRequest | None,
    ) -> list[dict[str, str]]:
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

        prompt_data = {
            "intent": "help.explain_error",
            "lang": self.lang,
            "contract": {
                "json_only": True,
                "required_fields": ["diagnosis", "next", "overlay", "explanations", "confidence"],
                "error_category_enum": schema_categories,
                "candidate_steps": candidate_steps,
                "overlay_target_allowlist": allowlist,
            },
            "observation": {
                "procedure_hint": observation.procedure_hint,
                "source": observation.source,
                "vars": context.get("vars"),
                "recent_deltas": context.get("recent_deltas"),
            },
            "request": {
                "intent": request.intent if request else "help",
                "message": request.message if request else None,
                "rag_topk": context.get("rag_topk"),
            },
        }
        user_prompt = (
            "Return exactly one JSON object and nothing else. "
            "No markdown, no code fence, no extra text.\n"
            f"{json.dumps(prompt_data, ensure_ascii=False, sort_keys=True)}"
        )
        return [
            {"role": "system", "content": "You are SimTutor. Reply with JSON only."},
            {"role": "user", "content": user_prompt},
        ]

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, Mapping):
            raise ValueError("Ollama response must be a JSON object")

        message = body.get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(body.get("response"), str):
            return body["response"]
        raise ValueError("Ollama response missing assistant content")

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


__all__ = ["OllamaModel"]

