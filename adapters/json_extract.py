"""
Minimal and safe JSON extraction/repair helpers for model outputs.

Allowed repair set is intentionally small:
1) Remove balanced markdown code fence wrappers (```json ... ```).
2) Drop non-JSON prefix text before the first JSON object/array.
3) Drop non-JSON suffix text after the first JSON object/array.

Any transformation outside this set is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


REPAIR_REMOVE_CODE_FENCE = "removed_code_fence"
REPAIR_DROP_PREFIX = "dropped_prefix_text"
REPAIR_DROP_SUFFIX = "dropped_suffix_text"

ALLOWED_REPAIRS = (
    REPAIR_REMOVE_CODE_FENCE,
    REPAIR_DROP_PREFIX,
    REPAIR_DROP_SUFFIX,
)


@dataclass(frozen=True)
class JsonExtractionResult:
    json_text: str
    json_repaired: bool
    repair_reasons: tuple[str, ...]


def _strip_balanced_code_fence(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    if not (text.startswith("```") and text.endswith("```")):
        return text, False

    lines = text.splitlines()
    if len(lines) >= 2:
        if not lines[0].startswith("```"):
            return text, False
        inner = "\n".join(lines[1:-1]).strip()
        return inner, True

    # Inline balanced fence, e.g. ```json {"k":1} ```
    inner = text[3:-3].strip()
    if inner.lower().startswith("json"):
        rest = inner[4:]
        if not rest or rest[0].isspace() or rest[0] in "{[":
            inner = rest.strip()
    return inner, True


def _find_first_json_segment(text: str) -> tuple[int, int]:
    in_string = False
    escaped = False
    stack: list[str] = []
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
            continue

        if start < 0:
            if ch == "{":
                start = i
                stack.append("}")
            elif ch == "[":
                start = i
                stack.append("]")
            continue

        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if not stack or ch != stack[-1]:
                continue
            stack.pop()
            if not stack:
                return start, i + 1

    raise ValueError("Model output does not contain JSON object/array")


def extract_first_json(raw_text: str) -> JsonExtractionResult:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    text, fence_removed = _strip_balanced_code_fence(raw_text)
    start, end = _find_first_json_segment(text)
    extracted = text[start:end]
    reasons: list[str] = []
    if fence_removed:
        reasons.append(REPAIR_REMOVE_CODE_FENCE)
    if text[:start].strip():
        reasons.append(REPAIR_DROP_PREFIX)
    if text[end:].strip():
        reasons.append(REPAIR_DROP_SUFFIX)
    return JsonExtractionResult(
        json_text=extracted,
        json_repaired=bool(reasons),
        repair_reasons=tuple(reasons),
    )


def parse_first_json(raw_text: str) -> tuple[Any, JsonExtractionResult]:
    extraction = extract_first_json(raw_text)
    try:
        obj = json.loads(extraction.json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Extracted JSON segment is invalid: {exc.msg}") from exc
    return obj, extraction


__all__ = [
    "ALLOWED_REPAIRS",
    "JsonExtractionResult",
    "REPAIR_DROP_PREFIX",
    "REPAIR_DROP_SUFFIX",
    "REPAIR_REMOVE_CODE_FENCE",
    "extract_first_json",
    "parse_first_json",
]
