from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.knowledge_source_policy import KnowledgeSourcePolicy, KnowledgeSourcePolicyError


def _write_index(path: Path) -> None:
    index = {
        "documents": [
            {
                "doc_id": "doc_a",
                "chunks": [
                    {"chunk_id": "doc_a_0", "text": "line1\nline2\nline3"},
                    {"chunk_id": "doc_a_1", "text": "only one line"},
                ],
            },
            {
                "doc_id": "doc_b",
                "chunks": [
                    {"chunk_id": "doc_b_0", "text": "x\ny"},
                ],
            },
        ]
    }
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def test_policy_loads_and_filters_by_doc_chunk(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "policy_id: test_policy\n"
        "allow:\n"
        "  - doc_id: doc_a\n"
        "    chunk_id: doc_a_0\n"
        "    line_range: [1, 3]\n",
        encoding="utf-8",
    )

    policy = KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)
    assert policy.policy_id == "test_policy"
    assert policy.doc_count == 1
    assert policy.chunk_count == 1
    assert "docs=1" in policy.public_startup_info()
    assert "chunks=1" in policy.public_startup_info()

    snippets = [
        {"doc_id": "doc_a", "snippet_id": "doc_a_0", "snippet": "ok"},
        {"doc_id": "doc_b", "snippet_id": "doc_b_0", "snippet": "drop"},
    ]
    filtered = policy.filter_snippets(snippets)
    assert len(filtered) == 1
    assert filtered[0]["doc_id"] == "doc_a"
    assert filtered[0]["snippet_id"] == "doc_a_0"
    assert filtered[0]["line_start"] == 1
    assert filtered[0]["line_end"] == 3
    assert filtered[0]["snippet"] == "line1\nline2\nline3"


def test_policy_clips_snippet_to_allowed_line_range_when_span_metadata_missing(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    policy_path = tmp_path / "policy_clip.yaml"
    policy_path.write_text(
        "policy_id: test_clip\n"
        "allow:\n"
        "  - doc_id: doc_a\n"
        "    chunk_id: doc_a_0\n"
        "    line_range: [2, 2]\n",
        encoding="utf-8",
    )
    policy = KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)

    snippets = [
        {"doc_id": "doc_a", "snippet_id": "doc_a_0", "snippet": "line1"},
    ]
    filtered = policy.filter_snippets(snippets)
    assert len(filtered) == 1
    assert filtered[0]["snippet"] == "line2"
    assert filtered[0]["line_start"] == 2
    assert filtered[0]["line_end"] == 2


def test_policy_rejects_snippet_when_reported_span_has_no_overlap(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    policy_path = tmp_path / "policy_no_overlap.yaml"
    policy_path.write_text(
        "policy_id: test_overlap\n"
        "allow:\n"
        "  - doc_id: doc_a\n"
        "    chunk_id: doc_a_0\n"
        "    line_range: [2, 3]\n",
        encoding="utf-8",
    )
    policy = KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)

    snippets = [
        {"doc_id": "doc_a", "snippet_id": "doc_a_0", "snippet": "line1", "line_start": 1, "line_end": 1},
    ]
    filtered = policy.filter_snippets(snippets)
    assert filtered == []


def test_policy_rejects_invalid_line_range_shape(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    policy_path = tmp_path / "policy_bad_line_range.yaml"
    policy_path.write_text(
        "policy_id: test\n"
        "allow:\n"
        "  - doc_id: doc_a\n"
        "    chunk_id: doc_a_0\n"
        "    line_range: [1]\n",
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeSourcePolicyError, match="line_range"):
        KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)


def test_policy_rejects_line_range_exceeding_chunk_lines(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    policy_path = tmp_path / "policy_bad_span.yaml"
    policy_path.write_text(
        "policy_id: test\n"
        "allow:\n"
        "  - doc_id: doc_a\n"
        "    chunk_id: doc_a_1\n"
        "    line_range: [1, 2]\n",
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeSourcePolicyError, match="exceeds chunk lines"):
        KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)


def test_policy_rejects_unknown_chunk_reference(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    _write_index(index_path)
    policy_path = tmp_path / "policy_bad_chunk.yaml"
    policy_path.write_text(
        "policy_id: test\n"
        "allow:\n"
        "  - doc_id: doc_a\n"
        "    chunk_id: unknown_chunk\n"
        "    line_range: [1, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeSourcePolicyError, match="unknown chunk"):
        KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)


def test_policy_rejects_mismatched_yaml_index_and_caller_index(tmp_path: Path) -> None:
    index_a = tmp_path / "a" / "index.json"
    index_b = tmp_path / "b" / "index.json"
    index_a.parent.mkdir(parents=True, exist_ok=True)
    index_b.parent.mkdir(parents=True, exist_ok=True)
    _write_index(index_a)
    _write_index(index_b)

    policy_path = tmp_path / "policy_mismatch.yaml"
    policy_path.write_text(
        "policy_id: test\n"
        "index_path: a/index.json\n"
        "allow:\n"
        "  - doc_id: doc_a\n"
        "    chunk_id: doc_a_0\n"
        "    line_range: [1, 1]\n",
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeSourcePolicyError, match="index_path mismatch"):
        KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_b)


def test_policy_public_startup_info_does_not_expose_absolute_paths(tmp_path: Path) -> None:
    index_path = tmp_path / "nested" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    _write_index(index_path)
    policy_path = tmp_path / "nested" / "policy.yaml"
    policy_path.write_text(
        "policy_id: test\n"
        "allow:\n"
        "  - doc_id: doc_a\n"
        "    chunk_id: doc_a_0\n"
        "    line_range: [1, 1]\n",
        encoding="utf-8",
    )

    policy = KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)
    info = policy.public_startup_info()
    assert "policy_file=policy.yaml" in info
    assert "index_file=index.json" in info
    assert str(policy_path.parent) not in info
    assert str(index_path.parent) not in info


def test_repository_policy_is_valid_against_repository_index() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    policy_path = repo_root / "knowledge_source_policy.yaml"
    index_path = repo_root / "Doc" / "Evaluation" / "index.json"

    policy = KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)
    assert policy.policy_id == "fa18c_cold_start_whitelist_v1"
    assert policy.doc_count == 2
    assert policy.chunk_count > 0
