"""
Knowledge-source whitelist policy for cold-start production mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path, PureWindowsPath
from typing import Any
from collections.abc import Mapping, Sequence

import yaml
from core.knowledge import load_index_data


class KnowledgeSourcePolicyError(ValueError):
    pass


def _path_filename(path_like: str | Path) -> str:
    text = str(path_like)
    if len(text) >= 3 and text[1] == ":" and text[2] in ("\\", "/"):
        name = PureWindowsPath(text).name
    elif text.startswith("\\\\"):
        name = PureWindowsPath(text).name
    else:
        name = Path(text).name
    return name or "<path>"


def _resolve_path(path_like: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir.expanduser() / path).resolve()


def _coerce_non_empty_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeSourcePolicyError(f"{field_name} must be a non-empty string")
    return value.strip()


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KnowledgeSourcePolicyError(f"{field_name} must be a positive integer")
    if value < 1:
        raise KnowledgeSourcePolicyError(f"{field_name} must be >= 1")
    return value


def _coerce_line_range(value: Any, *, field_name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 2:
        raise KnowledgeSourcePolicyError(f"{field_name} must be [start_line, end_line]")
    start = _coerce_positive_int(value[0], field_name=f"{field_name}[0]")
    end = _coerce_positive_int(value[1], field_name=f"{field_name}[1]")
    if start > end:
        raise KnowledgeSourcePolicyError(f"{field_name} start line must be <= end line")
    return start, end


@dataclass(frozen=True)
class _IndexChunkInfo:
    line_count: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _ParsedAllowEntry:
    idx: int
    doc_id: str
    chunk_id: str
    line_start: int
    line_end: int


def _parse_allow_entries(raw_allow: Any) -> list[_ParsedAllowEntry]:
    if not isinstance(raw_allow, list) or not raw_allow:
        raise KnowledgeSourcePolicyError("allow must be a non-empty list")

    parsed: list[_ParsedAllowEntry] = []
    for idx, item in enumerate(raw_allow):
        if not isinstance(item, Mapping):
            raise KnowledgeSourcePolicyError(f"allow[{idx}] must be a mapping")
        doc_id = _coerce_non_empty_str(item.get("doc_id"), field_name=f"allow[{idx}].doc_id")
        chunk_id = _coerce_non_empty_str(item.get("chunk_id"), field_name=f"allow[{idx}].chunk_id")
        line_start, line_end = _coerce_line_range(item.get("line_range"), field_name=f"allow[{idx}].line_range")
        parsed.append(
            _ParsedAllowEntry(
                idx=idx,
                doc_id=doc_id,
                chunk_id=chunk_id,
                line_start=line_start,
                line_end=line_end,
            )
        )
    return parsed


def _load_index_chunk_catalog(
    index_path: Path,
    *,
    allowed_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], _IndexChunkInfo]:
    try:
        raw = load_index_data(index_path)
    except OSError as exc:
        raise KnowledgeSourcePolicyError(f"knowledge index read failed: {_path_filename(index_path)}") from exc
    except json.JSONDecodeError as exc:
        raise KnowledgeSourcePolicyError(f"knowledge index contains invalid JSON: {_path_filename(index_path)}") from exc
    if not isinstance(raw, Mapping):
        raise KnowledgeSourcePolicyError(f"knowledge index must be a mapping: {_path_filename(index_path)}")
    documents = raw.get("documents")
    if not isinstance(documents, list):
        raise KnowledgeSourcePolicyError(f"knowledge index missing documents list: {_path_filename(index_path)}")

    if not allowed_keys:
        return {}

    out: dict[tuple[str, str], _IndexChunkInfo] = {}
    for doc_idx, doc in enumerate(documents):
        if not isinstance(doc, Mapping):
            continue
        doc_id = doc.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            doc_id = f"doc_{doc_idx}"
        chunks = doc.get("chunks")
        if not isinstance(chunks, list):
            continue
        for chunk_idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                continue
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                chunk_id = f"{doc_id}_{chunk_idx}"
            key = (doc_id, chunk_id)
            if key not in allowed_keys:
                continue
            text = chunk.get("text")
            if not isinstance(text, str):
                text = str(text or "")
            lines = tuple(text.splitlines())
            if not lines:
                lines = ("",)
            out[key] = _IndexChunkInfo(
                line_count=max(1, len(lines)),
                lines=lines,
            )
            if len(out) >= len(allowed_keys):
                return out
    return out


def _extract_chunk_id(snippet: Mapping[str, Any]) -> str | None:
    direct = snippet.get("chunk_id")
    if isinstance(direct, str) and direct:
        return direct
    fallback = snippet.get("snippet_id")
    if isinstance(fallback, str) and fallback:
        return fallback
    return None


def _extract_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 1:
        return value
    return None


@dataclass(frozen=True)
class SourceChunkRule:
    doc_id: str
    chunk_id: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class KnowledgeSourcePolicy:
    policy_id: str
    policy_path: Path
    index_path: Path
    rules: tuple[SourceChunkRule, ...]
    _rules_by_key: dict[tuple[str, str], SourceChunkRule] = field(init=False, repr=False)
    _allowed_excerpt_by_key: dict[tuple[str, str], str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        by_key: dict[tuple[str, str], SourceChunkRule] = {}
        for rule in self.rules:
            key = (rule.doc_id, rule.chunk_id)
            by_key[key] = rule
        object.__setattr__(self, "_rules_by_key", by_key)
        object.__setattr__(self, "_allowed_excerpt_by_key", {})

    @property
    def doc_count(self) -> int:
        return len({rule.doc_id for rule in self.rules})

    @property
    def chunk_count(self) -> int:
        return len(self.rules)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        index_path: str | Path | None = None,
    ) -> "KnowledgeSourcePolicy":
        policy_path = Path(path).expanduser().resolve()
        try:
            raw_text = policy_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise KnowledgeSourcePolicyError(
                f"knowledge source policy read failed: {_path_filename(policy_path)}"
            ) from exc
        try:
            raw = yaml.safe_load(raw_text) or {}
        except yaml.YAMLError as exc:
            raise KnowledgeSourcePolicyError(
                f"knowledge source policy contains invalid YAML: {_path_filename(policy_path)}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise KnowledgeSourcePolicyError(
                f"knowledge source policy must be a mapping: {_path_filename(policy_path)}"
            )

        policy_id = raw.get("policy_id", policy_path.stem)
        policy_id = _coerce_non_empty_str(policy_id, field_name="policy_id")

        yaml_index_path_raw = raw.get("index_path")
        if index_path is not None and yaml_index_path_raw is not None:
            yaml_index_path = _resolve_path(yaml_index_path_raw, base_dir=policy_path.parent).resolve()
            caller_index_path = _resolve_path(index_path, base_dir=policy_path.parent).resolve()
            if yaml_index_path != caller_index_path:
                raise KnowledgeSourcePolicyError(
                    "knowledge source policy index_path mismatch: "
                    f"policy declares {_path_filename(yaml_index_path)}, "
                    f"caller supplied {_path_filename(caller_index_path)}"
                )

        index_path_raw = index_path if index_path is not None else yaml_index_path_raw
        if index_path_raw is None:
            raise KnowledgeSourcePolicyError(
                "knowledge source policy requires index_path: provide from_yaml(..., index_path=...) "
                "or set policy.index_path in YAML"
            )
        effective_index_path = _resolve_path(index_path_raw, base_dir=policy_path.parent)
        parsed_allow_entries = _parse_allow_entries(raw.get("allow"))
        allowed_keys = {(entry.doc_id, entry.chunk_id) for entry in parsed_allow_entries}
        chunk_catalog = _load_index_chunk_catalog(
            effective_index_path,
            allowed_keys=allowed_keys,
        )

        rules: list[SourceChunkRule] = []
        allowed_excerpt_by_key: dict[tuple[str, str], str] = {}
        seen: set[tuple[str, str]] = set()
        for entry in parsed_allow_entries:
            idx = entry.idx
            key = (entry.doc_id, entry.chunk_id)
            chunk_info = chunk_catalog.get(key)
            if chunk_info is None:
                raise KnowledgeSourcePolicyError(
                    f"allow[{idx}] references unknown chunk: doc_id={entry.doc_id!r} chunk_id={entry.chunk_id!r}"
                )
            max_lines = chunk_info.line_count
            if entry.line_end > max_lines:
                raise KnowledgeSourcePolicyError(
                    f"allow[{idx}].line_range exceeds chunk lines: {entry.line_end} > {max_lines} "
                    f"for doc_id={entry.doc_id!r} chunk_id={entry.chunk_id!r}"
                )
            if key in seen:
                raise KnowledgeSourcePolicyError(
                    f"duplicate whitelist chunk entry: doc_id={entry.doc_id!r} chunk_id={entry.chunk_id!r}"
                )
            seen.add(key)
            rules.append(
                SourceChunkRule(
                    doc_id=entry.doc_id,
                    chunk_id=entry.chunk_id,
                    line_start=entry.line_start,
                    line_end=entry.line_end,
                )
            )
            allowed_excerpt_by_key[key] = "\n".join(chunk_info.lines[entry.line_start - 1 : entry.line_end])

        policy = cls(
            policy_id=policy_id,
            policy_path=policy_path,
            index_path=effective_index_path,
            rules=tuple(rules),
        )
        object.__setattr__(policy, "_allowed_excerpt_by_key", allowed_excerpt_by_key)
        return policy

    def filter_snippets(self, snippets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for snippet in snippets:
            if not isinstance(snippet, Mapping):
                continue
            doc_id = snippet.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                continue
            chunk_id = _extract_chunk_id(snippet)
            if chunk_id is None:
                continue
            rule = self._rules_by_key.get((doc_id, chunk_id))
            if rule is None:
                continue

            line_start = _extract_positive_int(snippet.get("line_start"))
            line_end = _extract_positive_int(snippet.get("line_end"))
            if line_start is not None or line_end is not None:
                span_start = line_start if line_start is not None else line_end
                span_end = line_end if line_end is not None else line_start
                if span_start is not None and span_end is not None:
                    start = min(span_start, span_end)
                    end = max(span_start, span_end)
                    if end < rule.line_start or start > rule.line_end:
                        continue
            normalized = dict(snippet)
            allowed_excerpt = self._allowed_excerpt_by_key.get((doc_id, chunk_id))
            if allowed_excerpt is not None:
                normalized["snippet"] = allowed_excerpt
                normalized["line_start"] = rule.line_start
                normalized["line_end"] = rule.line_end
            filtered.append(normalized)
        return filtered

    def public_startup_info(self) -> str:
        return (
            f"policy_id={self.policy_id} docs={self.doc_count} chunks={self.chunk_count} "
            f"policy_file={self.policy_path.name} index_file={self.index_path.name}"
        )


__all__ = [
    "KnowledgeSourcePolicy",
    "KnowledgeSourcePolicyError",
    "SourceChunkRule",
]
