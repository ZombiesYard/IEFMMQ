"""
Knowledge port defines retrieval interface for documents.
"""

from __future__ import annotations

from typing import Protocol, Any


class KnowledgePort(Protocol):
    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Returns list of retrieval dicts with JSON-serializable scalar values.

        Expected keys:
        - `doc_id` (str)
        - `section` (str | None)
        - `page_or_heading` (str | int | float | bool | None)
        - `snippet` (str)
        - `snippet_id` (str)
        - `score` (float | int, optional)

        Notes:
        - `section`/`page_or_heading` may be None when unavailable (e.g. some chunks).
        - Values should remain JSON serializable because callers may emit them into JSONL events.
        """
        ...

