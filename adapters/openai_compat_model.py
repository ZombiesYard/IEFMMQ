"""
OpenAI-compatible ModelPort adapter (vLLM/llama.cpp/TGI compatible).
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
import re
from typing import Any, Mapping

from adapters.base_help_model import BaseHelpModel
from core.llm_schema import get_help_response_schema


class OpenAICompatModel(BaseHelpModel):
    provider = "openai_compat"
    _DEFAULT_QWEN35_MAX_TOKENS = 384
    _DEFAULT_QWEN35_VLM_MAX_TOKENS = 640

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
        self._runtime_metadata = self._empty_multimodal_metadata()
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            lang=lang,
            log_raw_llm_text=log_raw_llm_text,
            client=client,
        )

    def _reset_runtime_metadata(self) -> None:
        self._runtime_metadata = self._empty_multimodal_metadata()

    def _collect_runtime_metadata(self) -> dict[str, Any]:
        return dict(self._runtime_metadata)

    def _build_messages(
        self,
        observation: Any,
        request: Any,
        *,
        deterministic_inference: Any = None,
        recent_ui_targets: list[str] | None = None,
        deterministic_hint: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages, prompt_meta = super()._build_messages(
            observation,
            request,
            deterministic_inference=deterministic_inference,
            recent_ui_targets=recent_ui_targets,
            deterministic_hint=deterministic_hint,
        )
        multimodal_spec = self._build_multimodal_spec(request)
        self._runtime_metadata.update(
            {
                "multimodal_input_present": bool(multimodal_spec["image_contents"]),
                "multimodal_primary_frame_id": multimodal_spec["primary_frame_id"],
                "multimodal_frame_ids": list(multimodal_spec["frame_ids"]),
                "multimodal_image_count": len(multimodal_spec["image_contents"]),
                "multimodal_path_attempted": False,
                "multimodal_path_success": False,
                "multimodal_fallback_to_text": False,
                "multimodal_failure_reason": multimodal_spec["failure_reason"],
            }
        )
        if not multimodal_spec["image_contents"]:
            return messages, prompt_meta

        rewritten: list[dict[str, Any]] = []
        injected = False
        for message in messages:
            if not injected and message.get("role") == "user" and isinstance(message.get("content"), str):
                user_text = str(message["content"])
                multimodal_content = [
                    *multimodal_spec["image_contents"],
                    {
                        "type": "text",
                        "text": self._build_multimodal_prompt_text(
                            user_text,
                            primary_frame_id=multimodal_spec["primary_frame_id"],
                            secondary_frame_id=multimodal_spec["secondary_frame_id"],
                        ),
                    },
                ]
                rewritten.append(
                    {
                        "role": message["role"],
                        "content": multimodal_content,
                    }
                )
                injected = True
                continue
            rewritten.append(dict(message))
        return rewritten, prompt_meta

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self._messages_contain_images(messages):
            self._runtime_metadata["multimodal_path_attempted"] = True
            try:
                content = self._chat_once(messages, headers=headers, has_vision=True)
            except Exception as exc:
                self._runtime_metadata["multimodal_path_success"] = False
                self._runtime_metadata["multimodal_fallback_to_text"] = True
                self._runtime_metadata["multimodal_failure_reason"] = f"{type(exc).__name__}: {exc}"
                return self._chat_once(self._strip_images_from_messages(messages), headers=headers, has_vision=False)
            self._runtime_metadata["multimodal_path_success"] = True
            self._runtime_metadata["multimodal_fallback_to_text"] = False
            self._runtime_metadata["multimodal_failure_reason"] = None
            return content

        self._runtime_metadata["multimodal_path_attempted"] = False
        self._runtime_metadata["multimodal_path_success"] = False
        self._runtime_metadata["multimodal_fallback_to_text"] = False
        return self._chat_once(messages, headers=headers, has_vision=False)

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        *,
        headers: Mapping[str, str],
        has_vision: bool,
    ) -> str:
        include_json_schema = True
        include_request_overrides = True
        while True:
            payload = self._build_chat_payload(
                messages,
                include_json_schema=include_json_schema,
                include_request_overrides=include_request_overrides,
                has_vision=has_vision,
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
        messages: list[dict[str, Any]],
        *,
        include_json_schema: bool,
        include_request_overrides: bool,
        has_vision: bool,
    ) -> dict[str, object]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            **self._build_generation_payload(
                include_request_overrides=include_request_overrides,
                has_vision=has_vision,
            ),
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

    def _build_generation_payload(
        self,
        *,
        include_request_overrides: bool,
        has_vision: bool,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"temperature": 0}
        effective_max_tokens = self._effective_max_tokens(has_vision=has_vision)
        if effective_max_tokens is not None:
            payload["max_tokens"] = effective_max_tokens
        if include_request_overrides:
            payload.update(self._build_request_overrides())
        return payload

    def _build_request_overrides(self) -> dict[str, object]:
        if self._should_disable_thinking():
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def _effective_max_tokens(self, *, has_vision: bool) -> int | None:
        if self.max_tokens is not None:
            return self.max_tokens
        if self._is_qwen35_model():
            if has_vision:
                return self._DEFAULT_QWEN35_VLM_MAX_TOKENS
            return self._DEFAULT_QWEN35_MAX_TOKENS
        return None

    def _build_multimodal_spec(self, request: Any) -> dict[str, Any]:
        context = request.context if request is not None and isinstance(getattr(request, "context", None), Mapping) else {}
        vision = context.get("vision")
        if not isinstance(vision, Mapping):
            return {
                "image_contents": [],
                "frame_ids": [],
                "primary_frame_id": None,
                "secondary_frame_id": None,
                "failure_reason": None,
            }

        trigger_frame = vision.get("trigger_frame")
        pre_trigger_frame = vision.get("pre_trigger_frame")
        selected_frames = vision.get("selected_frames")
        primary_frame = self._normalize_frame_payload(trigger_frame)
        if primary_frame is None:
            primary_frame = self._frame_payload_by_id(selected_frames, vision.get("frame_id"))
        if primary_frame is None:
            primary_frame = self._normalize_frame_payload(pre_trigger_frame)
        secondary_frame = self._normalize_frame_payload(pre_trigger_frame)
        if (
            secondary_frame is not None
            and primary_frame is not None
            and secondary_frame.get("frame_id") == primary_frame.get("frame_id")
        ):
            secondary_frame = None

        frames: list[dict[str, Any]] = []
        for frame in (primary_frame, secondary_frame):
            if frame is None:
                continue
            frames.append(frame)

        image_contents: list[dict[str, Any]] = []
        frame_ids: list[str] = []
        failure_reason: str | None = None
        for frame in frames:
            frame_id = frame.get("frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                continue
            try:
                data_url = self._frame_to_data_url(frame)
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
                image_contents = []
                frame_ids = []
                break
            image_contents.append({"type": "image_url", "image_url": {"url": data_url}})
            frame_ids.append(frame_id)

        return {
            "image_contents": image_contents,
            "frame_ids": frame_ids,
            "primary_frame_id": frame_ids[0] if frame_ids else None,
            "secondary_frame_id": frame_ids[1] if len(frame_ids) > 1 else None,
            "failure_reason": failure_reason,
        }

    def _build_multimodal_prompt_text(
        self,
        prompt_text: str,
        *,
        primary_frame_id: str | None,
        secondary_frame_id: str | None,
    ) -> str:
        frame_notes: list[str] = []
        if isinstance(primary_frame_id, str) and primary_frame_id:
            frame_notes.append(f"Primary visual frame: {primary_frame_id}")
        if isinstance(secondary_frame_id, str) and secondary_frame_id:
            frame_notes.append(f"Reference pre-trigger frame: {secondary_frame_id}")
        if not frame_notes:
            return prompt_text
        return "\n".join([*frame_notes, "", prompt_text])

    def _messages_contain_images(self, messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "image_url":
                    return True
        return False

    def _strip_images_from_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stripped: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                stripped.append(dict(message))
                continue
            text_parts = [
                str(item.get("text"))
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text" and isinstance(item.get("text"), str)
            ]
            stripped.append(
                {
                    "role": message.get("role"),
                    "content": "\n\n".join(part for part in text_parts if part),
                }
            )
        return stripped

    def _frame_to_data_url(self, frame: Mapping[str, Any]) -> str:
        raw_url = frame.get("image_uri") or frame.get("source_image_path")
        if not isinstance(raw_url, str) or not raw_url.strip():
            raise ValueError("vision frame is missing image_uri")
        image_url = raw_url.strip()
        if image_url.startswith("data:") or image_url.startswith("http://") or image_url.startswith("https://"):
            return image_url
        path = Path(image_url).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"vision frame image not found: {path}")
        mime_type = frame.get("mime_type")
        if not isinstance(mime_type, str) or not mime_type:
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _normalize_frame_payload(frame: Any) -> dict[str, Any] | None:
        if not isinstance(frame, Mapping):
            return None
        normalized = dict(frame)
        frame_id = normalized.get("frame_id")
        image_uri = normalized.get("image_uri") or normalized.get("source_image_path")
        if not isinstance(frame_id, str) or not frame_id:
            return None
        if not isinstance(image_uri, str) or not image_uri:
            return None
        return normalized

    def _frame_payload_by_id(self, selected_frames: Any, frame_id: Any) -> dict[str, Any] | None:
        if not isinstance(frame_id, str) or not frame_id or not isinstance(selected_frames, list):
            return None
        for item in selected_frames:
            normalized = self._normalize_frame_payload(item)
            if normalized is not None and normalized.get("frame_id") == frame_id:
                return normalized
        return None

    @staticmethod
    def _empty_multimodal_metadata() -> dict[str, Any]:
        return {
            "multimodal_input_present": False,
            "multimodal_primary_frame_id": None,
            "multimodal_frame_ids": [],
            "multimodal_image_count": 0,
            "multimodal_path_attempted": False,
            "multimodal_path_success": False,
            "multimodal_fallback_to_text": False,
            "multimodal_failure_reason": None,
        }

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
