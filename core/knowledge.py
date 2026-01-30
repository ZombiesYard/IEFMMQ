"""
Lightweight BM25-based retriever over pre-built index.json.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from ports.knowledge_port import KnowledgePort


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


class BM25Retriever(KnowledgePort):
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.data = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.chunks = []
        for doc in self.data.get("documents", []):
            for chunk in doc.get("chunks", []):
                self.chunks.append(
                    {
                        "doc_id": doc["doc_id"],
                        "section": chunk.get("heading"),
                        "page_or_heading": chunk.get("page") if chunk.get("page") is not None else chunk.get("heading"),
                        "snippet": chunk["text"][:300],
                    }
                )
        self._build_stats()

    def _build_stats(self):
        self.docs_terms = [Counter(_tokenize(c["snippet"])) for c in self.chunks]
        self.doc_freq = defaultdict(int)
        for terms in self.docs_terms:
            for t in terms:
                self.doc_freq[t] += 1
        self.N = len(self.docs_terms)
        self.avgdl = sum(sum(terms.values()) for terms in self.docs_terms) / self.N if self.N else 0.0

    def _score(self, query_terms: List[str], idx: int) -> float:
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

    def query(self, text: str, k: int = 5) -> List[Dict]:
        if not self.chunks:
            return []
        q_terms = _tokenize(text)
        scored = []
        for i, chunk in enumerate(self.chunks):
            s = self._score(q_terms, i)
            if s > 0:
                item = dict(chunk)
                item["score"] = s
                scored.append(item)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

