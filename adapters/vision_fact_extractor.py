"""
VLM-backed structured vision-fact extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from adapters.json_extract import parse_first_json
from adapters.openai_compat_multimodal import (
    MultimodalRequestRejected,
    build_multimodal_image_contents,
    copy_messages_for_payload,
    extract_response_error_text,
    is_json_schema_unsupported_400,
    is_multimodal_unsupported_400,
    is_request_override_unsupported_400,
    normalize_allowed_local_image_roots,
)
from adapters.vision_fact_prompting import build_vision_fact_prompt
from core.types_v2 import VisionFact, VisionFactObservation
from core.vision_facts import VISION_FACT_IDS, build_vision_fact_summary, load_vision_facts_config


def _vision_fact_response_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["facts"],
        "properties": {
            "facts": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "fact_id",
                        "state",
                        "source_frame_id",
                        "confidence",
                        "evidence_note",
                    ],
                    "properties": {
                        "fact_id": {"type": "string", "enum": list(VISION_FACT_IDS)},
                        "state": {"type": "string", "enum": ["seen", "not_seen", "uncertain"]},
                        "source_frame_id": {"type": "string", "minLength": 1},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "evidence_note": {"type": "string", "minLength": 1, "maxLength": 240},
                    },
                },
            }
        },
    }


_RESPONSE_VALIDATOR = Draft202012Validator(_vision_fact_response_schema())


@dataclass
class VisionFactExtractionResult:
    status: str
    observation: VisionFactObservation | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionFactExtractor:
    _DEFAULT_QWEN35_VLM_MAX_TOKENS = 480
    _DEFAULT_MAX_LOCAL_IMAGE_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        *,
        model_name: str = "Qwen3.5-27B-Instruct",
        base_url: str = "http://127.0.0.1:8000",
        timeout_s: float = 20.0,
        api_key: str | None = None,
        allowed_local_image_roots: Sequence[str] | None = None,
        max_local_image_bytes: int | None = None,
        lang: str = "zh",
        client: object | None = None,
        enable_multimodal: bool = True,
        config_path: str | None = None,
        pack_path: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.api_key = api_key
        self.lang = lang if lang in {"zh", "en"} else "zh"
        self.enable_multimodal = bool(enable_multimodal)
        self.allowed_local_image_roots = normalize_allowed_local_image_roots(allowed_local_image_roots)
        self.max_local_image_bytes = (
            int(max_local_image_bytes)
            if isinstance(max_local_image_bytes, int) and max_local_image_bytes > 0
            else self._DEFAULT_MAX_LOCAL_IMAGE_BYTES
        )
        self._config = load_vision_facts_config(config_path, pack_path=pack_path)
        if client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout_s)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def extract(
        self,
        vision: Mapping[str, Any],
        *,
        session_id: str | None,
        trigger_wall_ms: int,
    ) -> VisionFactExtractionResult:
        candidate_frames = self._candidate_frames(vision)
        candidate_frame_ids = [
            str(frame["frame_id"])
            for frame in candidate_frames
            if isinstance(frame.get("frame_id"), str) and frame.get("frame_id")
        ]
        if not candidate_frames:
            return VisionFactExtractionResult(
                status="vision_unavailable",
                metadata={"frame_ids": [], "multimodal_failure_reason": "no_candidate_frames"},
            )
        if not self.enable_multimodal:
            return VisionFactExtractionResult(
                status="extractor_failed",
                error="multimodal_disabled",
                metadata={"frame_ids": candidate_frame_ids, "multimodal_failure_reason": "multimodal_disabled"},
            )

        built = build_multimodal_image_contents(
            candidate_frames,
            allowed_local_image_roots=self.allowed_local_image_roots,
            max_local_image_bytes=self.max_local_image_bytes,
        )
        frame_ids = [item for item in built["frame_ids"] if isinstance(item, str) and item]
        if not built["image_contents"]:
            return VisionFactExtractionResult(
                status="extractor_failed",
                error=built["failure_reason"] or "no_usable_frames",
                metadata={
                    "frame_ids": frame_ids,
                    "multimodal_failure_reason": built["failure_reason"] or "no_usable_frames",
                    "multimodal_failed_frame_ids": list(built["failed_frame_ids"]),
                    "multimodal_frame_failures": dict(built["frame_failures"]),
                },
            )

        prompt = build_vision_fact_prompt(
            vision=self._effective_vision_context(vision, successful_frames=built["successful_frames"], frame_ids=frame_ids),
            lang=self.lang,
            config=self._config,
        )
        messages = [
            {"role": "system", "content": "You are SimTutor visual fact extractor. Reply with JSON only."},
            {
                "role": "user",
                "content": [
                    *built["image_contents"],
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        try:
            raw_text = self._chat(messages)
            observation = self._parse_response(
                raw_text,
                frame_ids=frame_ids,
                session_id=session_id,
                trigger_wall_ms=trigger_wall_ms,
            )
        except Exception as exc:
            return VisionFactExtractionResult(
                status="extractor_failed",
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "frame_ids": frame_ids,
                    "multimodal_failure_reason": f"{type(exc).__name__}: {exc}",
                    "multimodal_failed_frame_ids": list(built["failed_frame_ids"]),
                    "multimodal_frame_failures": dict(built["frame_failures"]),
                },
            )

        result_status = "available"
        summary = build_vision_fact_summary(
            {fact.fact_id: fact.to_dict() for fact in observation.facts},
            status=result_status,
            frame_ids=frame_ids,
            fresh_fact_ids=[fact.fact_id for fact in observation.facts if fact.state == "seen"],
        )
        if summary["uncertain_fact_ids"]:
            result_status = "uncertain"
            summary["status"] = result_status
        observation.summary = summary["summary_text"]
        observation.metadata = {
            **dict(observation.metadata),
            "vision_fact_summary": summary,
        }
        return VisionFactExtractionResult(
            status=result_status,
            observation=observation,
            metadata={
                "frame_ids": frame_ids,
                "multimodal_failed_frame_ids": list(built["failed_frame_ids"]),
                "multimodal_frame_failures": dict(built["frame_failures"]),
                "vision_fact_summary": summary,
            },
        )

    def _candidate_frames(self, vision: Mapping[str, Any]) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for key in ("pre_trigger_frame", "trigger_frame"):
            raw = vision.get(key)
            if not isinstance(raw, Mapping):
                continue
            frame_id = raw.get("frame_id")
            image_uri = raw.get("image_uri") or raw.get("source_image_path")
            if isinstance(frame_id, str) and frame_id and isinstance(image_uri, str) and image_uri:
                frames.append(dict(raw))
        if frames:
            return frames
        selected_frames = vision.get("selected_frames")
        if not isinstance(selected_frames, list):
            return []
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in selected_frames:
            if not isinstance(item, Mapping):
                continue
            frame_id = item.get("frame_id")
            image_uri = item.get("image_uri") or item.get("source_image_path")
            if not isinstance(frame_id, str) or not frame_id or frame_id in seen:
                continue
            if not isinstance(image_uri, str) or not image_uri:
                continue
            seen.add(frame_id)
            out.append(dict(item))
        return out[:2]

    def _effective_vision_context(
        self,
        vision: Mapping[str, Any],
        *,
        successful_frames: Sequence[Mapping[str, Any]],
        frame_ids: Sequence[str],
    ) -> dict[str, Any]:
        effective = dict(vision)
        successful = [dict(frame) for frame in successful_frames]
        frame_id_set = {item for item in frame_ids if isinstance(item, str) and item}
        effective["frame_ids"] = [item for item in frame_ids if isinstance(item, str) and item]
        effective["frame_id"] = effective["frame_ids"][0] if effective["frame_ids"] else None
        effective["selected_frames"] = successful
        for key in ("pre_trigger_frame", "trigger_frame"):
            raw = effective.get(key)
            if not isinstance(raw, Mapping):
                continue
            raw_frame_id = raw.get("frame_id")
            if isinstance(raw_frame_id, str) and raw_frame_id in frame_id_set:
                continue
            effective.pop(key, None)
        return effective

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        include_json_schema = True
        include_request_overrides = True
        while True:
            payload = {
                "model": self.model_name,
                "messages": copy_messages_for_payload(messages),
                "temperature": 0,
                "max_tokens": self._DEFAULT_QWEN35_VLM_MAX_TOKENS,
            }
            if include_request_overrides:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            if include_json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "VisionFactResponse",
                        "strict": True,
                        "schema": _vision_fact_response_schema(),
                    },
                }
            response = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            if getattr(response, "status_code", None) != 400:
                break
            if include_request_overrides and is_request_override_unsupported_400(response):
                include_request_overrides = False
                continue
            if include_json_schema and is_json_schema_unsupported_400(response):
                include_json_schema = False
                continue
            break
        if is_multimodal_unsupported_400(response):
            raise MultimodalRequestRejected(
                extract_response_error_text(response) or "server rejected multimodal request"
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
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise ValueError("OpenAI-compatible response missing choices[0].message.content")
        return str(message["content"])

    def _parse_response(
        self,
        raw_text: str,
        *,
        frame_ids: Sequence[str],
        session_id: str | None,
        trigger_wall_ms: int,
    ) -> VisionFactObservation:
        obj, _ = parse_first_json(raw_text)
        if not isinstance(obj, Mapping):
            raise ValueError("vision fact response must be a JSON object")
        _RESPONSE_VALIDATOR.validate(obj)
        facts_raw = obj.get("facts")
        if not isinstance(facts_raw, list):
            raise ValueError("vision fact response facts must be a list")

        default_frame_id = frame_ids[0] if frame_ids else "unknown_frame"
        facts_by_id: dict[str, VisionFact] = {}
        for item in facts_raw:
            if not isinstance(item, Mapping):
                continue
            fact_id = item.get("fact_id")
            if not isinstance(fact_id, str) or fact_id not in VISION_FACT_IDS or fact_id in facts_by_id:
                continue
            spec = self._config["facts_by_id"][fact_id]
            facts_by_id[fact_id] = VisionFact(
                fact_id=fact_id,
                state=str(item.get("state")),
                source_frame_id=(
                    str(item.get("source_frame_id"))
                    if isinstance(item.get("source_frame_id"), str) and item.get("source_frame_id")
                    else default_frame_id
                ),
                confidence=float(item.get("confidence")),
                expires_after_ms=int(spec["expires_after_ms"]),
                evidence_note=str(item.get("evidence_note")).strip(),
                observed_at_wall_ms=trigger_wall_ms,
                sticky=bool(spec["sticky"]),
            )

        facts: list[VisionFact] = []
        for fact_id in VISION_FACT_IDS:
            fact = facts_by_id.get(fact_id)
            if fact is None:
                spec = self._config["facts_by_id"][fact_id]
                fact = VisionFact(
                    fact_id=fact_id,
                    state="uncertain",
                    source_frame_id=default_frame_id,
                    confidence=0.0,
                    expires_after_ms=int(spec["expires_after_ms"]),
                    evidence_note="Model omitted this fact; defaulted to uncertain.",
                    observed_at_wall_ms=trigger_wall_ms,
                    sticky=bool(spec["sticky"]),
                )
            facts.append(fact)

        return VisionFactObservation(
            session_id=session_id,
            trigger_wall_ms=trigger_wall_ms,
            frame_ids=list(frame_ids),
            facts=facts,
            metadata={
                "raw_fact_count": len(facts_raw),
                "configured_fact_count": len(VISION_FACT_IDS),
            },
        )


__all__ = ["VisionFactExtractionResult", "VisionFactExtractor"]
