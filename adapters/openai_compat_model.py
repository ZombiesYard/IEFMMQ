"""
OpenAI-compatible ModelPort adapter (vLLM/llama.cpp/TGI compatible).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from adapters.base_help_model import BaseHelpModel
from core.llm_schema import get_help_response_schema


class OpenAICompatModel(BaseHelpModel):
    provider = "openai_compat"
    _DEFAULT_QWEN35_MAX_TOKENS = 384

    def __init__(
        self,
        model_name: str = "Qwen3-8B-Instruct",
        base_url: str = "http://127.0.0.1:8000",
        timeout_s: float = 20.0,
        max_tokens: int | None = None,
        lang: str = "zh",
        log_raw_llm_text: bool = False,
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.api_key = api_key
        self.max_tokens = int(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else None
        self._help_response_schema = get_help_response_schema()
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            lang=lang,
            log_raw_llm_text=log_raw_llm_text,
            client=client,
        )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        include_json_schema = True
        include_request_overrides = True
        while True:
            payload = self._build_chat_payload(
                messages,
                include_json_schema=include_json_schema,
                include_request_overrides=include_request_overrides,
            )
            response = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            status_code = getattr(response, "status_code", None)
            if not isinstance(status_code, int) or status_code != 400:
                break
            if include_request_overrides and self._is_request_override_unsupported_400(response):
                include_request_overrides = False
                continue
            if include_json_schema and self._is_json_schema_unsupported_400(response):
                include_json_schema = False
                continue
            break

        response.raise_for_status()
        body = response.json()
        if not isinstance(body, Mapping):
            raise ValueError("OpenAI-compatible response must be a JSON object")
        return self._extract_content_from_body(body)

    def _extract_content_from_body(self, body: Mapping[str, object]) -> str:
        if not isinstance(body, Mapping):
            raise ValueError("OpenAI-compatible response must be a JSON object")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI-compatible response missing choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError("OpenAI-compatible choice must be an object")
        message = first.get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            return message["content"]
        raise ValueError("OpenAI-compatible response missing choices[0].message.content")

    def _build_chat_payload(
        self,
        messages: list[dict[str, str]],
        *,
        include_json_schema: bool,
        include_request_overrides: bool,
    ) -> dict[str, object]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            **self._build_generation_payload(include_request_overrides=include_request_overrides),
        }
        if include_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "HelpResponse",
                    "strict": True,
                    "schema": self._help_response_schema,
                },
            }
        return payload

    def _build_generation_payload(self, *, include_request_overrides: bool) -> dict[str, object]:
        payload: dict[str, object] = {"temperature": 0}
        effective_max_tokens = self._effective_max_tokens()
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens
        if include_request_overrides:
            payload.update(self._build_request_overrides())
        return payload

    def _build_request_overrides(self) -> dict[str, object]:
        if self._should_disable_thinking():
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def _effective_max_tokens(self) -> int | None:
        if self.max_tokens is not None:
            return self.max_tokens
        if self._is_qwen35_model():
            return self._DEFAULT_QWEN35_MAX_TOKENS
        return None

    def _should_disable_thinking(self) -> bool:
        return self._is_qwen35_model()

    def _is_qwen35_model(self) -> bool:
        return "qwen3.5" in self.model_name.lower()

    def _is_json_schema_unsupported_400(self, response: Any) -> bool:
        status_code = getattr(response, "status_code", None)
        if status_code != 400:
            return False
        error_text = self._extract_response_error_text(response)
        if not error_text:
            return False
        normalized = error_text.lower()
        grammar_error_patterns = (
            r"\bgrammar error\b",
            r"\bunimplemented keys?\b",
        )
        if any(re.search(pattern, normalized) for pattern in grammar_error_patterns):
            return True
        schema_tokens = ("response_format", "json_schema")
        if not any(token in normalized for token in schema_tokens):
            return False
        unsupported_patterns = (
            r"\bunsupported\b",
            r"\bnot\s+supported\b",
            r"\binvalid\s+response_format\b",
            r"\bunknown\s+field\b",
            r"\bunrecognized\s+field\b",
            r"\bunexpected\s+field\b",
            r"\bextra\s+inputs?\s+are\s+not\s+permitted\b",
            r"\bextra\s+fields?\s+not\s+permitted\b",
            r"\bdoes\s+not\s+support\b",
        )
        return any(re.search(pattern, normalized) for pattern in unsupported_patterns)

    def _is_request_override_unsupported_400(self, response: Any) -> bool:
        status_code = getattr(response, "status_code", None)
        if status_code != 400:
            return False
        error_text = self._extract_response_error_text(response)
        if not error_text:
            return False
        normalized = error_text.lower()
        override_tokens = ("chat_template_kwargs", "enable_thinking")
        if not any(token in normalized for token in override_tokens):
            return False
        unsupported_patterns = (
            r"\bunsupported\b",
            r"\bnot\s+supported\b",
            r"\bunknown\s+field\b",
            r"\bunrecognized\s+field\b",
            r"\bunexpected\s+field\b",
            r"\bextra\s+inputs?\s+are\s+not\s+permitted\b",
            r"\bextra\s+fields?\s+not\s+permitted\b",
            r"\bdoes\s+not\s+support\b",
        )
        return any(re.search(pattern, normalized) for pattern in unsupported_patterns)

    def _extract_response_error_text(self, response: Any) -> str:
        chunks: list[str] = []

        try:
            body = response.json()
        except Exception:
            body = None

        if isinstance(body, Mapping):
            chunks.extend(self._collect_message_fields(body))
            error_obj = body.get("error")
            if isinstance(error_obj, Mapping):
                chunks.extend(self._collect_message_fields(error_obj))
            elif isinstance(error_obj, str) and error_obj.strip():
                chunks.append(error_obj.strip())
        elif isinstance(body, str) and body.strip():
            chunks.append(body.strip())

        response_text = getattr(response, "text", None)
        if isinstance(response_text, str) and response_text.strip():
            chunks.append(response_text.strip())

        return " | ".join(chunks)

    @staticmethod
    def _collect_message_fields(obj: Mapping[str, object]) -> list[str]:
        out: list[str] = []
        for key in ("message", "detail"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
        return out


__all__ = ["OpenAICompatModel"]
