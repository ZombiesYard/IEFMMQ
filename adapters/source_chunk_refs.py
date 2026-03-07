"""
Shared helpers for canonical source-chunk reference formatting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _extract_non_empty_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _extract_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 1:
        return value
    return None


def build_source_chunk_ref(snippet: Mapping[str, Any]) -> str | None:
    doc_id = _extract_non_empty_str(snippet.get("doc_id"))
    chunk_id = _extract_non_empty_str(snippet.get("chunk_id")) or _extract_non_empty_str(snippet.get("snippet_id"))
    if doc_id is None or chunk_id is None:
        return None

    line_start = _extract_positive_int(snippet.get("line_start"))
    line_end = _extract_positive_int(snippet.get("line_end"))
    if line_start is None or line_end is None:
        return f"{doc_id}/{chunk_id}"

    start = min(line_start, line_end)
    end = max(line_start, line_end)
    return f"{doc_id}/{chunk_id}:{start}-{end}"


def collect_source_chunk_refs(snippets: Sequence[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in snippets:
        ref = build_source_chunk_ref(item)
        if ref is None or ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
    return out


__all__ = [
    "build_source_chunk_ref",
    "collect_source_chunk_refs",
]
