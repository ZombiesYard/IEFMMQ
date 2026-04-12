from __future__ import annotations

import json
from typing import Any

from core.types import TutorRequest


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        status_code: int = 200,
        *,
        text: str | None = None,
        json_error: Exception | None = None,
    ) -> None:
        self._payload = payload or {}
        self._status_code = status_code
        self.text = text if isinstance(text, str) else json.dumps(self._payload, ensure_ascii=False)
        self._json_error = json_error

    @property
    def status_code(self) -> int:
        return self._status_code

    def raise_for_status(self) -> None:
        if self._status_code >= 400:
            raise RuntimeError(f"http {self._status_code}")

    def json(self) -> dict[str, Any]:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class FakeClient:
    def __init__(self, responses: list[FakeResponse] | None = None, to_raise: Exception | None = None) -> None:
        self._responses = list(responses or [])
        self._to_raise = to_raise
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        json: dict[str, Any],  # noqa: A002
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
                "headers": dict(headers or {}),
            }
        )
        if self._to_raise is not None:
            raise self._to_raise
        if not self._responses:
            raise RuntimeError("no fake response left")
        return self._responses.pop(0)


def _extract_prompt_constraints_json(prompt: str) -> dict[str, Any]:
    marker = "Context and constraints JSON:\n"
    start = prompt.index(marker) + len(marker)
    decoder = json.JSONDecoder()
    payload, _end = decoder.raw_decode(prompt[start:])
    return payload


def _help_obj_ok() -> dict[str, Any]:
    return {
        "diagnosis": {"step_id": "S02", "error_category": "OM"},
        "next": {"step_id": "S03"},
        "overlay": {
            "targets": ["apu_switch"],
            "evidence": [
                {
                    "target": "apu_switch",
                    "type": "delta",
                    "ref": "RECENT_UI_TARGETS.apu_switch",
                    "quote": "Recent delta indicates APU switch activity.",
                    "grounding_confidence": 0.9,
                }
            ],
        },
        "explanations": ["APU is off, switch APU to ON before engine crank."],
    }


def _request_help() -> TutorRequest:
    return TutorRequest(
        intent="help",
        message="I am stuck at startup.",
        context={
            "vars": {"battery_on": True, "apu_on": False},
            "recent_deltas": [{"k": "apu_on", "mapped_ui_target": "apu_switch", "from": 1, "to": 0}],
            "gates": {"S03": {"status": "allowed", "reason": "prerequisites_met"}},
            "rag_topk": [{"id": "doc_001", "snippet": "APU switch to ON and wait for APU READY."}],
            "candidate_steps": ["S02", "S03"],
            "overlay_target_allowlist": ["apu_switch", "battery_switch"],
        },
    )


def _openai_chat_payload_from_help_obj(help_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(help_obj, ensure_ascii=False),
                },
            }
        ]
    }


def _ollama_message_payload_from_help_obj(help_obj: dict[str, Any], fenced: bool = False) -> dict[str, Any]:
    content = json.dumps(help_obj, ensure_ascii=False)
    if fenced:
        content = f"```json\n{content}\n```"
    return {"message": {"content": content}}


def _ollama_response_payload_from_help_obj(help_obj: dict[str, Any]) -> dict[str, Any]:
    return {"response": json.dumps(help_obj, ensure_ascii=False)}
