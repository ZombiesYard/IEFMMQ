"""
Shared parsing helpers for model-generated HelpResponse payloads.

Pipeline:
raw model text -> code fence strip -> JSON object extract -> json.loads -> schema validate.
"""

from __future__ import annotations

from typing import Any

from adapters.json_extract import JsonExtractionResult, extract_first_json, parse_first_json
from core.llm_schema import validate_help_response


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return stripped


def json_extract(raw_text: str) -> str:
    return extract_first_json(raw_text).json_text


def parse_help_response(raw_text: str) -> dict[str, Any]:
    obj, _ = parse_help_response_with_meta(raw_text)
    return obj


def parse_help_response_with_meta(raw_text: str) -> tuple[dict[str, Any], JsonExtractionResult]:
    obj, extraction = parse_first_json(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("HelpResponse must be a JSON object")
    validate_help_response(obj)
    return obj, extraction


__all__ = ["json_extract", "parse_help_response", "parse_help_response_with_meta", "strip_code_fence"]
