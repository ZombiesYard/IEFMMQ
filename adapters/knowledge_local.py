"""
Local knowledge adapter that serves BM25 retrieval from index.json.
"""

from __future__ import annotations

from pathlib import Path

from core.knowledge import BM25Retriever
from ports.knowledge_port import KnowledgePort

DEFAULT_INDEX_PATH = Path("Doc") / "Evaluation" / "index.json"


class LocalKnowledgeAdapter(KnowledgePort):
    def __init__(self, index_path: str | Path | None = None):
        self.retriever = BM25Retriever(str(index_path or DEFAULT_INDEX_PATH))

    def query(self, text: str, k: int = 5):
        return self.retriever.query(text, k)
