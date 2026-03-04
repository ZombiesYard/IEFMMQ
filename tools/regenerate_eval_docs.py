"""
Regenerate Doc/Evaluation markdown files from index.json + source policy.

The generated markdown files are treated as derived artifacts.
Use `--check` in CI to detect manual drift.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml


class EvalDocRegenerationError(ValueError):
    """Raised when regeneration inputs are invalid."""


class EvalDocDriftError(EvalDocRegenerationError):
    """Raised when generated output differs from files on disk in check mode."""

    def __init__(self, drift_paths: list[Path]) -> None:
        super().__init__("evaluation markdown drift detected")
        self.drift_paths = drift_paths


@dataclass(frozen=True)
class _PolicyRule:
    order: int
    doc_id: str
    chunk_id: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    heading: str | None
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _MarkdownDoc:
    doc_id: str
    output_path: Path
    chunks_by_id: dict[str, _Chunk]


@dataclass(frozen=True)
class RegeneratedDoc:
    doc_id: str
    output_path: Path
    content: str


def _to_repo_relative_path(raw_path: str, *, repo_root: Path) -> Path:
    text = str(raw_path).strip()
    if not text:
        raise EvalDocRegenerationError("index document source_path must be non-empty")
    win_path = PureWindowsPath(text)
    normalized = str(win_path).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _as_non_empty_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalDocRegenerationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _as_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvalDocRegenerationError(f"{field_name} must be a positive integer")
    return value


def _load_policy(policy_path: Path) -> tuple[str, str, Path | None, list[_PolicyRule]]:
    try:
        raw_text = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalDocRegenerationError(f"failed to read policy file: {policy_path}") from exc
    try:
        raw = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise EvalDocRegenerationError(f"policy contains invalid YAML: {policy_path}") from exc
    if not isinstance(raw, dict):
        raise EvalDocRegenerationError("policy root must be a mapping")

    policy_id = _as_non_empty_str(raw.get("policy_id"), field_name="policy_id")
    policy_version = _as_non_empty_str(raw.get("version", "unknown"), field_name="version")
    index_path_raw = raw.get("index_path")
    index_path_from_policy: Path | None = None
    if index_path_raw is not None:
        index_path_from_policy = Path(str(index_path_raw)).expanduser()
        if not index_path_from_policy.is_absolute():
            index_path_from_policy = (policy_path.parent / index_path_from_policy).resolve()
        else:
            index_path_from_policy = index_path_from_policy.resolve()

    allow = raw.get("allow")
    if not isinstance(allow, list) or not allow:
        raise EvalDocRegenerationError("policy.allow must be a non-empty list")

    out_rules: list[_PolicyRule] = []
    seen_doc_chunk: set[tuple[str, str]] = set()
    for idx, item in enumerate(allow):
        if not isinstance(item, dict):
            raise EvalDocRegenerationError(f"policy.allow[{idx}] must be a mapping")
        doc_id = _as_non_empty_str(item.get("doc_id"), field_name=f"allow[{idx}].doc_id")
        chunk_id = _as_non_empty_str(item.get("chunk_id"), field_name=f"allow[{idx}].chunk_id")
        raw_line_range = item.get("line_range")
        if (
            not isinstance(raw_line_range, list)
            or len(raw_line_range) != 2
            or isinstance(raw_line_range[0], bool)
            or isinstance(raw_line_range[1], bool)
        ):
            raise EvalDocRegenerationError(f"allow[{idx}].line_range must be [start_line, end_line]")
        line_start = _as_positive_int(raw_line_range[0], field_name=f"allow[{idx}].line_range[0]")
        line_end = _as_positive_int(raw_line_range[1], field_name=f"allow[{idx}].line_range[1]")
        if line_start > line_end:
            raise EvalDocRegenerationError(f"allow[{idx}].line_range start must be <= end")
        key = (doc_id, chunk_id)
        if key in seen_doc_chunk:
            raise EvalDocRegenerationError(f"duplicate allow entry for doc/chunk: {doc_id}/{chunk_id}")
        seen_doc_chunk.add(key)
        out_rules.append(
            _PolicyRule(
                order=idx,
                doc_id=doc_id,
                chunk_id=chunk_id,
                line_start=line_start,
                line_end=line_end,
            )
        )
    return policy_id, policy_version, index_path_from_policy, out_rules


def _load_markdown_docs(index_path: Path, *, repo_root: Path) -> tuple[list[_MarkdownDoc], dict[str, _MarkdownDoc]]:
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalDocRegenerationError(f"failed to read index file: {index_path}") from exc
    except json.JSONDecodeError as exc:
        raise EvalDocRegenerationError(f"index contains invalid JSON: {index_path}") from exc
    if not isinstance(raw, dict):
        raise EvalDocRegenerationError("index root must be a mapping")
    docs = raw.get("documents")
    if not isinstance(docs, list):
        raise EvalDocRegenerationError("index.documents must be a list")

    ordered_docs: list[_MarkdownDoc] = []
    docs_by_id: dict[str, _MarkdownDoc] = {}
    for doc_idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        source_path = doc.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            continue
        if not source_path.replace("\\", "/").lower().endswith(".md"):
            continue
        doc_id = _as_non_empty_str(doc.get("doc_id"), field_name=f"documents[{doc_idx}].doc_id")
        if doc_id in docs_by_id:
            raise EvalDocRegenerationError(f"duplicate markdown doc_id in index: {doc_id}")
        chunks = doc.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise EvalDocRegenerationError(f"markdown document has no chunks: {doc_id}")

        chunks_by_id: dict[str, _Chunk] = {}
        for chunk_idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            chunk_id = _as_non_empty_str(
                chunk.get("chunk_id"),
                field_name=f"documents[{doc_idx}].chunks[{chunk_idx}].chunk_id",
            )
            if chunk_id in chunks_by_id:
                raise EvalDocRegenerationError(f"duplicate chunk_id in index: {doc_id}/{chunk_id}")
            heading_raw = chunk.get("heading")
            heading = str(heading_raw).strip() if isinstance(heading_raw, str) and heading_raw.strip() else None
            text = chunk.get("text")
            text_str = str(text or "")
            lines = tuple(text_str.splitlines()) or ("",)
            chunks_by_id[chunk_id] = _Chunk(
                chunk_id=chunk_id,
                heading=heading,
                lines=lines,
            )
        if not chunks_by_id:
            raise EvalDocRegenerationError(f"markdown document has no valid chunks: {doc_id}")
        output_path = _to_repo_relative_path(source_path, repo_root=repo_root)
        doc_meta = _MarkdownDoc(doc_id=doc_id, output_path=output_path, chunks_by_id=chunks_by_id)
        ordered_docs.append(doc_meta)
        docs_by_id[doc_id] = doc_meta
    if not ordered_docs:
        raise EvalDocRegenerationError("no markdown documents found in index")
    return ordered_docs, docs_by_id


def _compute_version_stamp(index_path: Path, policy_path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(index_path.read_bytes())
    hasher.update(b"\n--\n")
    hasher.update(policy_path.read_bytes())
    return hasher.hexdigest()[:16]


def _render_doc_content(
    doc: _MarkdownDoc,
    rules: list[_PolicyRule],
    *,
    index_path: Path,
    policy_id: str,
    policy_version: str,
    version_stamp: str,
    index_display_path: str,
) -> str:
    body_sections: list[str] = []
    chunk_refs: list[str] = []

    for pos, rule in enumerate(rules):
        chunk = doc.chunks_by_id.get(rule.chunk_id)
        if chunk is None:
            raise EvalDocRegenerationError(f"policy references unknown chunk: {doc.doc_id}/{rule.chunk_id}")
        max_line = len(chunk.lines)
        if rule.line_end > max_line:
            raise EvalDocRegenerationError(
                f"policy line range exceeds chunk lines: {doc.doc_id}/{rule.chunk_id} "
                f"{rule.line_end} > {max_line}"
            )
        selected = "\n".join(chunk.lines[rule.line_start - 1 : rule.line_end]).strip()
        heading_text = chunk.heading
        heading_prefix: str | None = None
        if heading_text:
            heading_prefix = "# " if pos == 0 else "## "
        section_lines: list[str] = []
        if heading_prefix is not None:
            section_lines.append(f"{heading_prefix}{heading_text}")
        if selected:
            if section_lines:
                section_lines.append("")
            section_lines.append(selected)
        section_text = "\n".join(section_lines).strip()
        if section_text:
            body_sections.append(section_text)
        chunk_refs.append(f"- {rule.doc_id}/{rule.chunk_id}:{rule.line_start}-{rule.line_end}")

    if not body_sections:
        raise EvalDocRegenerationError(f"generated empty body for markdown doc: {doc.doc_id}")

    header = [
        "<!--",
        "AUTO-GENERATED FILE. DO NOT EDIT.",
        "generator: tools/regenerate_eval_docs.py",
        f"source_index: {index_display_path}",
        f"policy_id: {policy_id}",
        f"policy_version: {policy_version}",
        f"version_stamp: {version_stamp}",
        "source_chunks:",
        *chunk_refs,
        "-->",
    ]
    body = "\n\n".join(body_sections).strip()
    return "\n".join(header) + "\n\n" + body + "\n"


def build_regenerated_docs(
    *,
    index_path: str | Path,
    policy_path: str | Path,
    repo_root: str | Path | None = None,
    strict_policy: bool = True,
    version_stamp_override: str | None = None,
) -> list[RegeneratedDoc]:
    index_path_resolved = Path(index_path).expanduser().resolve()
    policy_path_resolved = Path(policy_path).expanduser().resolve()
    repo_root_resolved = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()

    policy_id, policy_version, policy_index_path, rules = _load_policy(policy_path_resolved)
    if policy_index_path is not None and policy_index_path != index_path_resolved:
        raise EvalDocRegenerationError(
            "policy index_path mismatch: "
            f"policy={policy_index_path} caller={index_path_resolved}"
        )
    ordered_docs, docs_by_id = _load_markdown_docs(index_path_resolved, repo_root=repo_root_resolved)
    rules_by_doc: dict[str, list[_PolicyRule]] = {}
    for rule in rules:
        rules_by_doc.setdefault(rule.doc_id, []).append(rule)

    index_doc_ids = set(docs_by_id)
    policy_doc_ids = set(rules_by_doc)
    missing_in_policy = sorted(index_doc_ids - policy_doc_ids)
    unknown_in_policy = sorted(policy_doc_ids - index_doc_ids)
    if unknown_in_policy:
        joined = ", ".join(unknown_in_policy)
        raise EvalDocRegenerationError(f"policy references markdown doc_ids not in index: {joined}")
    if strict_policy and missing_in_policy:
        joined = ", ".join(missing_in_policy)
        raise EvalDocRegenerationError(f"markdown docs missing in policy allow list: {joined}")

    version_stamp = version_stamp_override or _compute_version_stamp(index_path_resolved, policy_path_resolved)
    try:
        index_display_path = str(index_path_resolved.relative_to(repo_root_resolved)).replace("\\", "/")
    except ValueError:
        index_display_path = str(index_path_resolved.name)
    out: list[RegeneratedDoc] = []
    for doc in ordered_docs:
        doc_rules = rules_by_doc.get(doc.doc_id)
        if not doc_rules:
            if strict_policy:
                raise EvalDocRegenerationError(f"missing policy entries for markdown doc: {doc.doc_id}")
            continue
        content = _render_doc_content(
            doc,
            doc_rules,
            index_path=index_path_resolved,
            policy_id=policy_id,
            policy_version=policy_version,
            version_stamp=version_stamp,
            index_display_path=index_display_path,
        )
        out.append(
            RegeneratedDoc(
                doc_id=doc.doc_id,
                output_path=doc.output_path,
                content=content,
            )
        )
    return out


def regenerate_eval_docs(
    *,
    index_path: str | Path,
    policy_path: str | Path,
    repo_root: str | Path | None = None,
    check: bool = False,
    strict_policy: bool = True,
) -> list[Path]:
    renders = build_regenerated_docs(
        index_path=index_path,
        policy_path=policy_path,
        repo_root=repo_root,
        strict_policy=strict_policy,
    )
    changed: list[Path] = []
    drift: list[Path] = []
    for render in renders:
        current = render.output_path.read_text(encoding="utf-8") if render.output_path.exists() else ""
        if current == render.content:
            continue
        if check:
            drift.append(render.output_path)
            continue
        render.output_path.parent.mkdir(parents=True, exist_ok=True)
        render.output_path.write_text(render.content, encoding="utf-8")
        changed.append(render.output_path)
    if drift:
        raise EvalDocDriftError(sorted(drift))
    return changed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Doc/Evaluation markdown docs from index + policy",
    )
    parser.add_argument(
        "--index",
        default="Doc/Evaluation/index.json",
        help="Path to index JSON (default: Doc/Evaluation/index.json)",
    )
    parser.add_argument(
        "--policy",
        default="knowledge_source_policy.yaml",
        help="Path to source policy YAML (default: knowledge_source_policy.yaml)",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve index source_path entries",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check drift only; do not write files",
    )
    parser.add_argument(
        "--allow-partial-policy",
        action="store_true",
        help="Allow markdown docs not listed in policy.allow",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        changed = regenerate_eval_docs(
            index_path=args.index,
            policy_path=args.policy,
            repo_root=args.repo_root,
            check=args.check,
            strict_policy=not args.allow_partial_policy,
        )
    except EvalDocDriftError as exc:
        print("eval-doc-regeneration: DRIFT")
        for path in exc.drift_paths:
            print(f"- {path}")
        return 1
    except EvalDocRegenerationError as exc:
        print(f"eval-doc-regeneration: ERROR: {exc}")
        return 2

    if args.check:
        print("eval-doc-regeneration: OK")
    else:
        print(f"eval-doc-regeneration: regenerated {len(changed)} file(s)")
        for path in changed:
            print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
