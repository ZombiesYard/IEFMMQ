"""
Knowledge port defines retrieval interface for documents.
"""

from __future__ import annotations

from typing import Protocol, Any, runtime_checkable


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


@runtime_checkable
class KnowledgeRetrieveWithMetaPort(KnowledgePort, Protocol):
    def retrieve_with_meta(
        self,
        query: str,
        top_k: int = 5,
        *,
        step_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Optional richer retrieval API.

        Returns:
        - snippets: same constraints as `KnowledgePort.query()`
        - metadata: JSON-serializable mapping, recommended keys include
          `cache_hit`, `grounding_missing`, `grounding_reason`, `snippet_ids`, `index_path`.
        """
        ...

