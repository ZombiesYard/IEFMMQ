"""
Shared parsing helpers for model-generated HelpResponse payloads.

Pipeline:
raw model text -> code fence strip -> JSON object extract -> json.loads -> schema validate.
"""

from __future__ import annotations

import json
from typing import Any

from core.llm_schema import validate_help_response


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return stripped


def json_extract(raw_text: str) -> str:
    text = strip_code_fence(raw_text)
    if text.startswith("{") and text.endswith("}"):
        return text

    in_string = False
    escaped = False
    depth = 0
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
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    raise ValueError("Model output does not contain JSON object")


def parse_help_response(raw_text: str) -> dict[str, Any]:
    extracted = json_extract(raw_text)
    obj = json.loads(extracted)
    if not isinstance(obj, dict):
        raise ValueError("HelpResponse must be a JSON object")
    validate_help_response(obj)
    return obj


__all__ = ["json_extract", "parse_help_response", "strip_code_fence"]
