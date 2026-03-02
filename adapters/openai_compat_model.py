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

    def __init__(
        self,
        model_name: str = "Qwen3-8B-Instruct",
        base_url: str = "http://127.0.0.1:8000",
        timeout_s: float = 20.0,
        lang: str = "zh",
        log_raw_llm_text: bool = False,
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.api_key = api_key
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            lang=lang,
            log_raw_llm_text=log_raw_llm_text,
            client=client,
        )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        schema = get_help_response_schema()
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "HelpResponse",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        )
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and status_code == 400 and self._is_json_schema_unsupported_400(response):
            # Compatibility fallback for older vLLM builds that reject json_schema response_format.
            fallback_payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0,
            }
            response = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=fallback_payload,
                headers=headers,
                timeout=self.timeout_s,
            )

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

    def _is_json_schema_unsupported_400(self, response: Any) -> bool:
        status_code = getattr(response, "status_code", None)
        if status_code != 400:
            return False
        error_text = self._extract_response_error_text(response)
        if not error_text:
            return False
        normalized = error_text.lower()
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
