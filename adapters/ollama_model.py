"""
Ollama-backed ModelPort adapter for HelpResponse generation.
"""

from __future__ import annotations

from typing import Any, Mapping

from adapters.base_help_model import BaseHelpModel


class OllamaModel(BaseHelpModel):
    provider = "ollama"

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 20.0,
        lang: str = "zh",
        log_raw_llm_text: bool = False,
        print_model_io: bool = False,
        client: object | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            lang=lang,
            log_raw_llm_text=log_raw_llm_text,
            print_model_io=print_model_io,
            client=client,
        )

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        payload_messages = self._normalize_messages(messages)
        payload = {
            "model": self.model_name,
            "messages": payload_messages,
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

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role:
                raise ValueError("Ollama messages must include a string role")
            if not isinstance(content, str):
                raise ValueError("Ollama messages must have string content")
            normalized.append({"role": role, "content": content})
        return normalized


__all__ = ["OllamaModel"]
