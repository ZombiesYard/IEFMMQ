"""
Knowledge port defines retrieval interface for documents.
"""

from __future__ import annotations

from typing import Protocol, List, Dict


class KnowledgePort(Protocol):
    def query(self, text: str, k: int = 5) -> List[Dict]:
        """
        Returns list of dicts containing:
        - doc_id
        - section
        - page_or_heading
        - snippet
        - score
        """
        ...

