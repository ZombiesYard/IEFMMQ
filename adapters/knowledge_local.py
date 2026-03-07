"""
Local knowledge adapter that serves BM25 retrieval from index.json.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable
from collections.abc import Mapping

from adapters.knowledge_source_policy import KnowledgeSourcePolicy
from core.knowledge import BM25Retriever
from ports.knowledge_port import KnowledgePort

DEFAULT_INDEX_PATH = Path("Doc") / "Evaluation" / "index.json"
DEFAULT_STEP_CACHE_TTL_S = 60.0
DEFAULT_TOP_K = 5
RETRIEVER_POOL_MAX_SIZE = 8
STEP_CACHE_MAX_SIZE = 256


def _dedupe_strings_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_grounding_query(
    *,
    pack_title: str,
    inferred_step: str | None,
    missing_conditions: list[str],
    recent_ui_targets: list[str],
) -> str:
    parts: list[str] = []
    title = pack_title.strip()
    if title:
        parts.append(title)
    if inferred_step:
        step = inferred_step.strip()
        if step:
            parts.append(step)
    if missing_conditions:
        parts.extend(_dedupe_strings_keep_order([item.strip() for item in missing_conditions if item.strip()]))
    if recent_ui_targets:
        parts.extend(_dedupe_strings_keep_order([item.strip() for item in recent_ui_targets if item.strip()]))
    return " | ".join(parts)


@dataclass(frozen=True)
class _StepCacheEntry:
    step_id: str
    query: str
    t_mono: float
    snippets: list[dict[str, Any]]
    requested_k: int
    is_exhaustive: bool
    metadata: dict[str, Any]


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


def _snippet_ids(snippets: list[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    for item in snippets:
        snippet_id = _extract_non_empty_str(item.get("snippet_id"))
        if snippet_id is not None:
            out.append(snippet_id)
    return out


def _source_chunk_ref(snippet: Mapping[str, Any]) -> str | None:
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


def _source_chunk_refs(snippets: list[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in snippets:
        ref = _source_chunk_ref(item)
        if ref is None or ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
    return out


class LocalKnowledgeAdapter(KnowledgePort):
    _retriever_pool: "OrderedDict[str, BM25Retriever]" = OrderedDict()
    _retriever_pool_lock = threading.RLock()

    def __init__(
        self,
        index_path: str | Path | None = None,
        *,
        source_policy: KnowledgeSourcePolicy | None = None,
        step_cache_ttl_s: float = DEFAULT_STEP_CACHE_TTL_S,
        time_fn: Callable[[], float] | None = None,
    ):
        path = DEFAULT_INDEX_PATH if index_path is None else index_path
        self.index_path = Path(path)
        self.source_policy = source_policy
        self.step_cache_ttl_s = max(0.0, float(step_cache_ttl_s))
        self._time_fn = time_fn or time.monotonic
        self._step_cache: "OrderedDict[str, _StepCacheEntry]" = OrderedDict()
        self._step_cache_lock = threading.RLock()
        self._last_retrieve_metadata: dict[str, Any] = {}
        self._index_state: str = "unknown"
        self._index_error_type: str | None = None
        self.retriever = self._load_retriever(self.index_path)

    def _load_retriever(self, path: Path) -> BM25Retriever | None:
        key = str(path.resolve())
        if not path.is_file():
            self._index_state = "missing"
            self._index_error_type = None
            return None
        with self._retriever_pool_lock:
            cached = self._retriever_pool.get(key)
            if cached is not None:
                self._retriever_pool.move_to_end(key)
                self._index_state = "ready"
                self._index_error_type = None
                return cached
            try:
                retriever = BM25Retriever(path)
            except Exception as exc:
                self._index_state = "load_error"
                self._index_error_type = type(exc).__name__
                return None
            self._retriever_pool[key] = retriever
            self._retriever_pool.move_to_end(key)
            while len(self._retriever_pool) > RETRIEVER_POOL_MAX_SIZE:
                self._retriever_pool.popitem(last=False)
            self._index_state = "ready"
            self._index_error_type = None
            return retriever

    @property
    def has_index(self) -> bool:
        return self.retriever is not None

    @property
    def last_retrieve_metadata(self) -> dict[str, Any]:
        return dict(self._last_retrieve_metadata)

    def _normalize_result(self, raw: Mapping[str, Any], fallback_idx: int) -> dict[str, Any]:
        snippet_id = (
            raw.get("snippet_id")
            if isinstance(raw.get("snippet_id"), str) and raw.get("snippet_id")
            else raw.get("id")
        )
        if not isinstance(snippet_id, str) or not snippet_id:
            snippet_id = f"snippet_{fallback_idx}"
        section = raw.get("section")
        if not isinstance(section, str) or not section:
            section = None
        page_or_heading = raw.get("page_or_heading")
        if page_or_heading is None:
            page_or_heading = raw.get("page")
        if page_or_heading is None:
            page_or_heading = section
        snippet = raw.get("snippet")
        if not isinstance(snippet, str):
            snippet = str(snippet or "")
        doc_id = raw.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            doc_id = "unknown_doc"
        normalized = {
            "doc_id": doc_id,
            "section": section,
            "page_or_heading": page_or_heading,
            "snippet": snippet,
            "snippet_id": snippet_id,
            "chunk_id": (
                raw.get("chunk_id")
                if isinstance(raw.get("chunk_id"), str) and raw.get("chunk_id")
                else snippet_id
            ),
        }
        score = raw.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            normalized["score"] = float(score)
        line_start = raw.get("line_start")
        if isinstance(line_start, int) and not isinstance(line_start, bool) and line_start >= 1:
            normalized["line_start"] = line_start
        line_end = raw.get("line_end")
        if isinstance(line_end, int) and not isinstance(line_end, bool) and line_end >= 1:
            normalized["line_end"] = line_end
        return normalized

    def _prune_step_cache(self, now: float) -> None:
        expired_keys = [
            key
            for key, entry in self._step_cache.items()
            if (now - entry.t_mono) > self.step_cache_ttl_s
        ]
        for key in expired_keys:
            self._step_cache.pop(key, None)
        while len(self._step_cache) > STEP_CACHE_MAX_SIZE:
            self._step_cache.popitem(last=False)

    def retrieve_with_meta(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        step_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        k = max(0, int(top_k))
        policy = self.source_policy
        policy_id = policy.policy_id if policy is not None else None
        policy_version = policy.policy_version if policy is not None else None
        if self.retriever is None:
            if self._index_state == "load_error":
                grounding_reason = "index_load_error"
            else:
                grounding_reason = "index_missing"
            meta = {
                "query": query,
                "top_k": k,
                "cache_hit": False,
                "grounding_missing": True,
                "grounding_reason": grounding_reason,
                "index_error_type": self._index_error_type,
                "snippet_ids": [],
                "source_chunk_refs": [],
                "index_path": str(self.index_path),
                "source_policy_applied": False,
                "source_policy_id": policy_id,
                "source_policy_version": policy_version,
                "source_policy_filtered_out_count": 0,
            }
            self._last_retrieve_metadata = dict(meta)
            return [], meta
        if k == 0:
            meta = {
                "query": query,
                "top_k": k,
                "cache_hit": False,
                "grounding_missing": False,
                "grounding_reason": None,
                "snippet_ids": [],
                "source_chunk_refs": [],
                "index_path": str(self.index_path),
                "source_policy_applied": False,
                "source_policy_id": policy_id,
                "source_policy_version": policy_version,
                "source_policy_filtered_out_count": 0,
            }
            self._last_retrieve_metadata = dict(meta)
            return [], meta

        now = self._time_fn()
        cache_key = step_id if isinstance(step_id, str) and step_id else None
        if cache_key is not None:
            with self._step_cache_lock:
                self._prune_step_cache(now)
                cached = self._step_cache.get(cache_key)
                if (
                    cached is not None
                    and (now - cached.t_mono) <= self.step_cache_ttl_s
                    and cached.query == query
                    and (len(cached.snippets) >= k or cached.is_exhaustive)
                ):
                    self._step_cache.move_to_end(cache_key)
                    snippets = [dict(item) for item in cached.snippets[:k]]
                    meta = dict(cached.metadata)
                    meta.update(
                        {
                            "query": query,
                            "top_k": k,
                            "cache_hit": True,
                            "cache_step_id": cache_key,
                            "snippet_ids": _snippet_ids(snippets),
                            "source_chunk_refs": _source_chunk_refs(snippets),
                            "index_path": str(self.index_path),
                        }
                    )
                    self._last_retrieve_metadata = dict(meta)
                    return snippets, meta

        raw_results = self.retriever.query(query, k=k)
        snippets = [self._normalize_result(item, idx) for idx, item in enumerate(raw_results)]
        policy_filtered_out_count = 0
        source_policy_applied = False
        grounding_missing = False
        grounding_reason = None
        if policy is not None:
            source_policy_applied = True
            before_filter_count = len(snippets)
            snippets = policy.filter_snippets(snippets)
            policy_filtered_out_count = max(0, before_filter_count - len(snippets))
            if before_filter_count > 0 and not snippets:
                grounding_missing = True
                grounding_reason = "policy_filtered_all"
        if cache_key is not None:
            cache_meta = {
                "query": query,
                "top_k": k,
                "cache_hit": False,
                "cache_step_id": cache_key,
                "grounding_missing": grounding_missing,
                "grounding_reason": grounding_reason,
                "snippet_ids": _snippet_ids(snippets),
                "source_chunk_refs": _source_chunk_refs(snippets),
                "index_path": str(self.index_path),
                "source_policy_applied": source_policy_applied,
                "source_policy_id": policy_id,
                "source_policy_version": policy_version,
                "source_policy_filtered_out_count": policy_filtered_out_count,
            }
            with self._step_cache_lock:
                self._step_cache[cache_key] = _StepCacheEntry(
                    step_id=cache_key,
                    query=query,
                    t_mono=now,
                    snippets=[dict(item) for item in snippets],
                    requested_k=k,
                    is_exhaustive=len(snippets) < k,
                    metadata=cache_meta,
                )
                self._step_cache.move_to_end(cache_key)
                self._prune_step_cache(now)
        meta = {
            "query": query,
            "top_k": k,
            "cache_hit": False,
            "cache_step_id": cache_key,
            "grounding_missing": grounding_missing,
            "grounding_reason": grounding_reason,
            "snippet_ids": _snippet_ids(snippets),
            "source_chunk_refs": _source_chunk_refs(snippets),
            "index_path": str(self.index_path),
            "source_policy_applied": source_policy_applied,
            "source_policy_id": policy_id,
            "source_policy_version": policy_version,
            "source_policy_filtered_out_count": policy_filtered_out_count,
        }
        self._last_retrieve_metadata = dict(meta)
        return snippets, meta

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, *, step_id: str | None = None) -> list[dict[str, Any]]:
        snippets, _ = self.retrieve_with_meta(query, top_k=top_k, step_id=step_id)
        return snippets

    def query(self, text: str, k: int = 5) -> list[dict[str, Any]]:
        return self.retrieve(text, top_k=k)


__all__ = [
    "DEFAULT_INDEX_PATH",
    "DEFAULT_STEP_CACHE_TTL_S",
    "RETRIEVER_POOL_MAX_SIZE",
    "STEP_CACHE_MAX_SIZE",
    "LocalKnowledgeAdapter",
    "build_grounding_query",
]
