"""
Shared multimodal helpers for OpenAI-compatible adapters.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


class MultimodalRequestRejected(RuntimeError):
    """Raised when the upstream server rejects multimodal content."""


def normalize_allowed_local_image_roots(
    allowed_local_image_roots: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    return tuple(
        Path(item).expanduser().resolve()
        for item in (allowed_local_image_roots or ())
    )


def frame_to_data_url(
    frame: Mapping[str, Any],
    *,
    allowed_local_image_roots: Sequence[Path],
    max_local_image_bytes: int,
) -> str:
    raw_url = frame.get("image_uri") or frame.get("source_image_path")
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("vision frame is missing image_uri/source_image_path")
    image_url = raw_url.strip()
    if image_url.startswith("data:"):
        raise ValueError("inline vision frame data URLs are not allowed")
    if image_url.startswith("http://") or image_url.startswith("https://"):
        raise ValueError("remote vision frame URLs are not allowed")
    parsed_scheme = image_url.split(":", 1)[0].lower() if ":" in image_url else ""
    if parsed_scheme and not looks_like_windows_path(image_url):
        raise ValueError(f"unsupported vision frame URI scheme: {parsed_scheme}")
    path = Path(image_url).expanduser().resolve()
    label = frame_label(frame, path=path)
    if not is_allowed_local_image_path(path, allowed_local_image_roots=allowed_local_image_roots):
        raise ValueError(f"vision frame path is outside allowed roots: {label}")
    if not path.is_file():
        raise FileNotFoundError(f"vision frame image not found: {label}")
    file_size = path.stat().st_size
    if file_size > max_local_image_bytes:
        raise ValueError(f"vision frame image exceeds max size {max_local_image_bytes} bytes: {label}")
    mime_type = frame.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_image_contents(
    candidate_frames: Sequence[Mapping[str, Any]],
    *,
    allowed_local_image_roots: Sequence[Path],
    max_local_image_bytes: int,
) -> dict[str, Any]:
    image_contents: list[dict[str, Any]] = []
    frame_ids: list[str] = []
    successful_frames: list[dict[str, Any]] = []
    failed_frame_ids: list[str] = []
    frame_failures: dict[str, str] = {}
    failure_reason: str | None = None
    for frame in candidate_frames:
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        try:
            data_url = frame_to_data_url(
                frame,
                allowed_local_image_roots=allowed_local_image_roots,
                max_local_image_bytes=max_local_image_bytes,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failed_frame_ids.append(frame_id)
            frame_failures[frame_id] = failure
            continue
        image_contents.append({"type": "image_url", "image_url": {"url": data_url}})
        frame_ids.append(frame_id)
        successful_frames.append(dict(frame))

    if frame_failures:
        failure_reason = summarize_frame_failures(frame_failures)
    return {
        "image_contents": image_contents,
        "frame_ids": frame_ids,
        "successful_frames": successful_frames,
        "failed_frame_ids": failed_frame_ids,
        "frame_failures": frame_failures,
        "failure_reason": failure_reason,
    }


def frame_label(frame: Mapping[str, Any], *, path: Path | None = None) -> str:
    frame_id = frame.get("frame_id")
    if isinstance(frame_id, str) and frame_id:
        return frame_id
    if path is not None:
        return path.name or "<frame>"
    return "<frame>"


def is_allowed_local_image_path(path: Path, *, allowed_local_image_roots: Sequence[Path]) -> bool:
    if not allowed_local_image_roots:
        return False
    for root in allowed_local_image_roots:
        if path == root or root in path.parents:
            return True
    return False


def looks_like_windows_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[2] in ("\\", "/")


def coerce_frame_mapping(frame: Any) -> dict[str, Any] | None:
    if not isinstance(frame, Mapping):
        return None
    normalized = dict(frame)
    frame_id = normalized.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        return None
    return normalized


def normalize_frame_payload(frame: Any) -> dict[str, Any] | None:
    normalized = coerce_frame_mapping(frame)
    if normalized is None:
        return None
    image_uri = normalized.get("image_uri") or normalized.get("source_image_path")
    if not isinstance(image_uri, str) or not image_uri:
        return None
    return normalized


def messages_contain_images(messages: Sequence[Mapping[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "image_url":
                return True
    return False


def strip_images_from_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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


def copy_messages_for_payload(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for message in messages:
        cloned = dict(message)
        content = cloned.get("content")
        if isinstance(content, list):
            cloned["content"] = [
                dict(item) if isinstance(item, Mapping) else item
                for item in content
            ]
        copied.append(cloned)
    return copied


def summarize_frame_failures(frame_failures: Mapping[str, str]) -> str:
    parts = [f"{frame_id}: {reason}" for frame_id, reason in frame_failures.items()]
    return "; ".join(parts)


def extract_response_error_text(response: Any) -> str:
    chunks: list[str] = []
    try:
        body = response.json()
    except Exception:
        body = None

    if isinstance(body, Mapping):
        chunks.extend(collect_message_fields(body))
        error_obj = body.get("error")
        if isinstance(error_obj, Mapping):
            chunks.extend(collect_message_fields(error_obj))
        elif isinstance(error_obj, str) and error_obj.strip():
            chunks.append(error_obj.strip())
    elif isinstance(body, str) and body.strip():
        chunks.append(body.strip())

    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text.strip():
        chunks.append(response_text.strip())
    return " | ".join(chunks)


def collect_message_fields(obj: Mapping[str, object]) -> list[str]:
    out: list[str] = []
    for key in ("message", "detail"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


def is_json_schema_unsupported_400(response: Any) -> bool:
    status_code = getattr(response, "status_code", None)
    if status_code != 400:
        return False
    error_text = extract_response_error_text(response)
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


def is_request_override_unsupported_400(response: Any) -> bool:
    status_code = getattr(response, "status_code", None)
    if status_code != 400:
        return False
    error_text = extract_response_error_text(response)
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


def is_multimodal_unsupported_400(response: Any) -> bool:
    status_code = getattr(response, "status_code", None)
    if status_code != 400:
        return False
    error_text = extract_response_error_text(response)
    if not error_text:
        return False
    normalized = error_text.lower()
    multimodal_tokens = (
        "image_url",
        "multimodal",
        "multi-modal",
        "vision",
        "image input",
        "input_image",
    )
    if not any(token in normalized for token in multimodal_tokens):
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
        r"\binvalid\b",
    )
    return any(re.search(pattern, normalized) for pattern in unsupported_patterns)


__all__ = [
    "MultimodalRequestRejected",
    "build_multimodal_image_contents",
    "coerce_frame_mapping",
    "copy_messages_for_payload",
    "extract_response_error_text",
    "frame_to_data_url",
    "is_json_schema_unsupported_400",
    "is_multimodal_unsupported_400",
    "is_request_override_unsupported_400",
    "messages_contain_images",
    "normalize_allowed_local_image_roots",
    "normalize_frame_payload",
    "strip_images_from_messages",
    "summarize_frame_failures",
]
