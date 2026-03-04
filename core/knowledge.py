"""
Lightweight deterministic BM25 retriever over pre-built index.json.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ports.knowledge_port import KnowledgePort

_SNIPPET_PREVIEW_CHARS = 320


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


class BM25Retriever(KnowledgePort):
    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.data = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.chunks: list[dict[str, Any]] = []
        for doc_idx, doc in enumerate(self.data.get("documents", [])):
            if not isinstance(doc, Mapping):
                continue
            doc_id = _normalize_optional_string(doc.get("doc_id")) or f"doc_{doc_idx}"
            chunks = doc.get("chunks")
            if not isinstance(chunks, list):
                continue
            for chunk_idx, chunk in enumerate(chunks):
                if not isinstance(chunk, Mapping):
                    continue
                content = _normalize_text(chunk.get("text")).strip()
                if not content:
                    continue
                section = _normalize_optional_string(chunk.get("section"))
                if section is None:
                    section = _normalize_optional_string(chunk.get("heading"))
                page_or_heading: Any = chunk.get("page_or_heading")
                if page_or_heading is None:
                    page_or_heading = chunk.get("page")
                if page_or_heading is None:
                    page_or_heading = section
                snippet_id = (
                    _normalize_optional_string(chunk.get("snippet_id"))
                    or _normalize_optional_string(chunk.get("chunk_id"))
                    or f"{doc_id}_{chunk_idx}"
                )
                self.chunks.append(
                    {
                        "doc_id": doc_id,
                        "section": section,
                        "page_or_heading": page_or_heading,
                        "snippet": content[:_SNIPPET_PREVIEW_CHARS],
                        "snippet_id": snippet_id,
                        "content": content,
                    }
                )
        self._build_stats()

    def _build_stats(self) -> None:
        self.docs_terms = [Counter(_tokenize(c["content"])) for c in self.chunks]
        self.doc_freq: dict[str, int] = defaultdict(int)
        for terms in self.docs_terms:
            for token in terms:
                self.doc_freq[token] += 1
        self.N = len(self.docs_terms)
        self.avgdl = sum(sum(terms.values()) for terms in self.docs_terms) / self.N if self.N else 0.0

    def _score(self, query_terms: list[str], idx: int) -> float:
        k1 = 1.5
        b = 0.75
        if self.avgdl == 0.0:
            return 0.0
        terms = self.docs_terms[idx]
        dl = sum(terms.values()) or 1
        score = 0.0
        for q in query_terms:
            if q not in terms:
                continue
            df = self.doc_freq.get(q, 0)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            tf = terms[q]
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl)))
        return score

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0 or not self.chunks:
            return []
        q_terms = _tokenize(text)
        if not q_terms:
            return []
        scored: list[dict[str, Any]] = []
        for idx, chunk in enumerate(self.chunks):
            score = self._score(q_terms, idx)
            if score <= 0:
                continue
            scored.append(
                {
                    "doc_id": chunk["doc_id"],
                    "section": chunk["section"],
                    "page_or_heading": chunk["page_or_heading"],
                    "snippet": chunk["snippet"],
                    "snippet_id": chunk["snippet_id"],
                    "score": score,
                }
            )
        scored.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item.get("doc_id") or ""),
                str(item.get("section") or ""),
                str(item.get("page_or_heading") or ""),
                str(item.get("snippet_id") or ""),
            )
        )
        return scored[:k]

