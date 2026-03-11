"""
OpenAI-compatible ModelPort adapter (vLLM/llama.cpp/TGI compatible).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from adapters.base_help_model import BaseHelpModel
from adapters.openai_compat_multimodal import (
    MultimodalRequestRejected,
    build_multimodal_image_contents,
    coerce_frame_mapping,
    copy_messages_for_payload,
    extract_response_error_text,
    is_json_schema_unsupported_400,
    is_multimodal_unsupported_400,
    is_request_override_unsupported_400,
    messages_contain_images,
    normalize_allowed_local_image_roots,
    normalize_frame_payload,
    strip_images_from_messages,
    summarize_frame_failures,
)
from core.llm_schema import get_help_response_schema


class OpenAICompatModel(BaseHelpModel):
    provider = "openai_compat"
    _DEFAULT_QWEN35_MAX_TOKENS = 384
    _DEFAULT_QWEN35_VLM_MAX_TOKENS = 640
    _DEFAULT_MAX_LOCAL_IMAGE_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        model_name: str = "Qwen3-8B-Instruct",
        base_url: str = "http://127.0.0.1:8000",
        timeout_s: float = 20.0,
        max_tokens: int | None = None,
        lang: str = "zh",
        log_raw_llm_text: bool = False,
        print_model_io: bool = False,
        api_key: str | None = None,
        enable_multimodal: bool = False,
        allowed_local_image_roots: Sequence[str | Path] | None = None,
        max_local_image_bytes: int | None = None,
        client: object | None = None,
    ) -> None:
        self.api_key = api_key
        self.max_tokens = int(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else None
        self.enable_multimodal = bool(enable_multimodal)
        self.allowed_local_image_roots = normalize_allowed_local_image_roots(allowed_local_image_roots)
        self.max_local_image_bytes = (
            int(max_local_image_bytes)
            if isinstance(max_local_image_bytes, int) and max_local_image_bytes > 0
            else self._DEFAULT_MAX_LOCAL_IMAGE_BYTES
        )
        self._help_response_schema = get_help_response_schema()
        self._runtime_metadata = self._empty_multimodal_metadata(
            multimodal_capability_enabled=self.enable_multimodal
        )
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            lang=lang,
            log_raw_llm_text=log_raw_llm_text,
            print_model_io=print_model_io,
            client=client,
        )

    def _reset_runtime_metadata(self) -> None:
        self._runtime_metadata = self._empty_multimodal_metadata(
            multimodal_capability_enabled=self.enable_multimodal
        )

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
                "multimodal_capability_enabled": self.enable_multimodal,
                "multimodal_input_present": bool(multimodal_spec["candidate_frames"]),
                "multimodal_candidate_frame_ids": list(multimodal_spec["candidate_frame_ids"]),
                "multimodal_primary_frame_id": multimodal_spec["primary_frame_id"],
                "multimodal_frame_ids": list(multimodal_spec["frame_ids"]),
                "multimodal_failed_frame_ids": list(multimodal_spec["failed_frame_ids"]),
                "multimodal_frame_failures": dict(multimodal_spec["frame_failures"]),
                "multimodal_images_built": bool(multimodal_spec["image_contents"]),
                "multimodal_image_count": len(multimodal_spec["image_contents"]),
                "multimodal_path_attempted": False,
                "multimodal_path_success": False,
                "multimodal_fallback_to_text": False,
                "multimodal_failure_reason": multimodal_spec["failure_reason"],
            }
        )
        if not self.enable_multimodal or not multimodal_spec["image_contents"]:
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
                            secondary_frame_role=multimodal_spec["secondary_frame_role"],
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
            except MultimodalRequestRejected as exc:
                self._runtime_metadata["multimodal_path_success"] = False
                self._runtime_metadata["multimodal_fallback_to_text"] = True
                self._runtime_metadata["multimodal_failure_reason"] = str(exc)
                stripped_messages = self._strip_images_from_messages(messages)
                messages[:] = [dict(item) for item in stripped_messages]
                return self._chat_once(stripped_messages, headers=headers, has_vision=False)
            except Exception as exc:
                self._runtime_metadata["multimodal_path_success"] = False
                self._runtime_metadata["multimodal_fallback_to_text"] = False
                self._runtime_metadata["multimodal_failure_reason"] = f"{type(exc).__name__}: {exc}"
                raise
            self._runtime_metadata["multimodal_path_success"] = True
            self._runtime_metadata["multimodal_fallback_to_text"] = False
            if not self._runtime_metadata.get("multimodal_frame_failures"):
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

        if has_vision and self._is_multimodal_unsupported_400(response):
            error_text = self._extract_response_error_text(response) or "server rejected multimodal request"
            raise MultimodalRequestRejected(error_text)
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
            "messages": self._copy_messages_for_payload(messages),
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
                "candidate_frames": [],
                "candidate_frame_ids": [],
                "image_contents": [],
                "frame_ids": [],
                "failed_frame_ids": [],
                "frame_failures": {},
                "primary_frame_id": None,
                "secondary_frame_id": None,
                "secondary_frame_role": None,
                "failure_reason": None,
            }

        candidate_frames = self._candidate_multimodal_frames(vision)
        candidate_frame_ids = [
            str(frame["frame_id"])
            for frame in candidate_frames
            if isinstance(frame.get("frame_id"), str) and frame.get("frame_id")
        ]

        image_contents: list[dict[str, Any]] = []
        if not self.enable_multimodal:
            return {
                "candidate_frames": candidate_frames,
                "candidate_frame_ids": candidate_frame_ids,
                "image_contents": [],
                "frame_ids": [],
                "failed_frame_ids": [],
                "frame_failures": {},
                "primary_frame_id": candidate_frame_ids[0] if candidate_frame_ids else None,
                "secondary_frame_id": candidate_frame_ids[1] if len(candidate_frame_ids) > 1 else None,
                "secondary_frame_role": self._frame_role(candidate_frames[1]) if len(candidate_frames) > 1 else None,
                "failure_reason": None,
            }
        multimodal_built = build_multimodal_image_contents(
            candidate_frames,
            allowed_local_image_roots=self.allowed_local_image_roots,
            max_local_image_bytes=self.max_local_image_bytes,
        )
        image_contents = multimodal_built["image_contents"]
        frame_ids = multimodal_built["frame_ids"]
        successful_frames = multimodal_built["successful_frames"]
        failed_frame_ids = multimodal_built["failed_frame_ids"]
        frame_failures = multimodal_built["frame_failures"]
        failure_reason = multimodal_built["failure_reason"]

        return {
            "candidate_frames": candidate_frames,
            "candidate_frame_ids": candidate_frame_ids,
            "image_contents": image_contents,
            "frame_ids": frame_ids,
            "failed_frame_ids": failed_frame_ids,
            "frame_failures": frame_failures,
            "primary_frame_id": frame_ids[0] if frame_ids else (candidate_frame_ids[0] if candidate_frame_ids else None),
            "secondary_frame_id": frame_ids[1] if len(frame_ids) > 1 else None,
            "secondary_frame_role": self._frame_role(successful_frames[1]) if len(successful_frames) > 1 else None,
            "failure_reason": failure_reason,
        }

    def _candidate_multimodal_frames(self, vision: Mapping[str, Any]) -> list[dict[str, Any]]:
        trigger_frame = self._coerce_frame_mapping(vision.get("trigger_frame"))
        pre_trigger_frame = self._coerce_frame_mapping(vision.get("pre_trigger_frame"))
        selected_frames = vision.get("selected_frames")
        usable_trigger_frame = self._normalize_frame_payload(trigger_frame)
        usable_pre_trigger_frame = self._normalize_frame_payload(pre_trigger_frame)
        primary_frame = self._frame_payload_by_id(selected_frames, vision.get("frame_id"))
        if primary_frame is None:
            primary_frame = usable_trigger_frame
        if primary_frame is None:
            primary_frame = usable_pre_trigger_frame
        if primary_frame is None:
            primary_frame = self._raw_frame_payload_by_id(selected_frames, vision.get("frame_id"))
        if primary_frame is None:
            primary_frame = trigger_frame
        if primary_frame is None:
            primary_frame = pre_trigger_frame
        primary_frame_usable = self._normalize_frame_payload(primary_frame) is not None
        secondary_frame = None
        secondary_candidates: list[dict[str, Any] | None] = [usable_trigger_frame, usable_pre_trigger_frame]
        if not primary_frame_usable:
            secondary_candidates.extend([trigger_frame, pre_trigger_frame])
        for frame in secondary_candidates:
            if frame is None:
                continue
            if primary_frame is not None and frame.get("frame_id") == primary_frame.get("frame_id"):
                continue
            secondary_frame = frame
            break
        if (
            secondary_frame is not None
            and primary_frame is not None
            and secondary_frame.get("frame_id") == primary_frame.get("frame_id")
        ):
            secondary_frame = None
        return [frame for frame in (primary_frame, secondary_frame) if frame is not None]

    def _build_multimodal_prompt_text(
        self,
        prompt_text: str,
        *,
        primary_frame_id: str | None,
        secondary_frame_id: str | None,
        secondary_frame_role: str | None,
    ) -> str:
        frame_notes: list[str] = []
        if self.lang == "zh":
            primary_label = "主视觉帧"
            role_labels = {
                "trigger_frame": "触发帧",
                "pre_trigger_frame": "触发前帧",
            }
            default_secondary_label = "参考帧"
        else:
            primary_label = "Primary visual frame"
            role_labels = {
                "trigger_frame": "Trigger frame",
                "pre_trigger_frame": "Pre-trigger frame",
            }
            default_secondary_label = "Reference frame"
        if isinstance(primary_frame_id, str) and primary_frame_id:
            frame_notes.append(f"{primary_label}: {primary_frame_id}")
        if isinstance(secondary_frame_id, str) and secondary_frame_id:
            secondary_label = role_labels.get(secondary_frame_role, default_secondary_label)
            frame_notes.append(f"{secondary_label}: {secondary_frame_id}")
        if not frame_notes:
            return prompt_text
        return "\n".join([*frame_notes, "", prompt_text])

    @staticmethod
    def _frame_role(frame: Mapping[str, Any] | None) -> str | None:
        if not isinstance(frame, Mapping):
            return None
        role = frame.get("role")
        if isinstance(role, str) and role:
            return role
        return None

    def _messages_contain_images(self, messages: list[dict[str, Any]]) -> bool:
        return messages_contain_images(messages)

    def _strip_images_from_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return strip_images_from_messages(messages)

    def _copy_messages_for_payload(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return copy_messages_for_payload(messages)

    @staticmethod
    def _coerce_frame_mapping(frame: Any) -> dict[str, Any] | None:
        return coerce_frame_mapping(frame)

    @staticmethod
    def _normalize_frame_payload(frame: Any) -> dict[str, Any] | None:
        return normalize_frame_payload(frame)

    def _frame_payload_by_id(self, selected_frames: Any, frame_id: Any) -> dict[str, Any] | None:
        if not isinstance(frame_id, str) or not frame_id or not isinstance(selected_frames, list):
            return None
        for item in selected_frames:
            normalized = self._normalize_frame_payload(item)
            if normalized is not None and normalized.get("frame_id") == frame_id:
                return normalized
        return None

    def _raw_frame_payload_by_id(self, selected_frames: Any, frame_id: Any) -> dict[str, Any] | None:
        if not isinstance(frame_id, str) or not frame_id or not isinstance(selected_frames, list):
            return None
        for item in selected_frames:
            normalized = self._coerce_frame_mapping(item)
            if normalized is not None and normalized.get("frame_id") == frame_id:
                return normalized
        return None

    @staticmethod
    def _empty_multimodal_metadata(
        *,
        multimodal_capability_enabled: bool = False,
    ) -> dict[str, Any]:
        return {
            "multimodal_capability_enabled": bool(multimodal_capability_enabled),
            "multimodal_input_present": False,
            "multimodal_candidate_frame_ids": [],
            "multimodal_primary_frame_id": None,
            "multimodal_frame_ids": [],
            "multimodal_failed_frame_ids": [],
            "multimodal_frame_failures": {},
            "multimodal_images_built": False,
            "multimodal_image_count": 0,
            "multimodal_path_attempted": False,
            "multimodal_path_success": False,
            "multimodal_fallback_to_text": False,
            "multimodal_failure_reason": None,
        }

    @staticmethod
    def _summarize_frame_failures(frame_failures: Mapping[str, str]) -> str:
        return summarize_frame_failures(frame_failures)

    def _should_disable_thinking(self) -> bool:
        return self._is_qwen35_model()

    def _is_qwen35_model(self) -> bool:
        return "qwen3.5" in self.model_name.lower()

    def _is_json_schema_unsupported_400(self, response: Any) -> bool:
        return is_json_schema_unsupported_400(response)

    def _is_request_override_unsupported_400(self, response: Any) -> bool:
        return is_request_override_unsupported_400(response)

    def _is_multimodal_unsupported_400(self, response: Any) -> bool:
        return is_multimodal_unsupported_400(response)

    def _extract_response_error_text(self, response: Any) -> str:
        return extract_response_error_text(response)


__all__ = ["OpenAICompatModel"]
