"""
OpenAI-compatible ModelPort adapter (vLLM/llama.cpp/TGI compatible).
"""

from __future__ import annotations

from typing import Mapping

from adapters.base_help_model import BaseHelpModel


class OpenAICompatModel(BaseHelpModel):
    provider = "openai_compat"

    def __init__(
        self,
        model_name: str = "Qwen3.5-32B-Instruct",
        base_url: str = "http://127.0.0.1:8000",
        timeout_s: float = 20.0,
        lang: str = "zh",
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.api_key = api_key
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            lang=lang,
            client=client,
        )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
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
        response.raise_for_status()
        body = response.json()
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


__all__ = ["OpenAICompatModel"]
