from pathlib import Path
import json

from adapters.knowledge_local import (
    RETRIEVER_POOL_MAX_SIZE,
    STEP_CACHE_MAX_SIZE,
    LocalKnowledgeAdapter,
    build_grounding_query,
)
from adapters.knowledge_source_policy import KnowledgeSourcePolicy
from tools.index_docs import build_index


def test_index_and_retrieve_markdown_returns_required_fields(tmp_path: Path):
    doc = tmp_path / "doc.md"
    doc.write_text("# Title\nEngine start checklist\nTurn battery on.\n", encoding="utf-8")
    index_path = tmp_path / "index.json"
    build_index([str(doc)], str(index_path))

    adapter = LocalKnowledgeAdapter(str(index_path))
    results = adapter.retrieve("battery on", top_k=3)
    assert results
    top = results[0]
    assert top["doc_id"] == "doc"
    assert top["section"] == "Title"
    assert top["page_or_heading"] == "Title"
    assert top["snippet_id"] == "doc_0"
    assert "battery" in top["snippet"].lower()
    assert isinstance(top["score"], float)


def test_retrieve_is_deterministic_for_same_query(tmp_path: Path) -> None:
    doc = tmp_path / "doc_deterministic.md"
    doc.write_text(
        "# Startup\nAPU switch to ON.\n# Power\nBattery ON before fire test.\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index_deterministic.json"
    build_index([str(doc)], str(index_path))

    adapter = LocalKnowledgeAdapter(str(index_path))
    query = "battery apu startup"
    first = adapter.retrieve(query, top_k=5)
    second = adapter.retrieve(query, top_k=5)

    assert [item["snippet_id"] for item in first] == [item["snippet_id"] for item in second]
    assert [item["doc_id"] for item in first] == [item["doc_id"] for item in second]


def test_retrieve_caches_same_step_within_60_seconds(tmp_path: Path) -> None:
    doc = tmp_path / "doc_cache.md"
    doc.write_text("# Step\nAPU READY required before crank.\n", encoding="utf-8")
    index_path = tmp_path / "index_cache.json"
    build_index([str(doc)], str(index_path))

    now = [100.0]
    adapter = LocalKnowledgeAdapter(str(index_path), time_fn=lambda: now[0], step_cache_ttl_s=60.0)
    assert adapter.retriever is not None

    calls = {"count": 0}
    raw_query = adapter.retriever.query

    def _counted_query(text: str, k: int = 5):
        calls["count"] += 1
        return raw_query(text, k)

    adapter.retriever.query = _counted_query  # type: ignore[assignment]

    query = "S03 apu_ready apu_switch"
    adapter.retrieve(query, top_k=1, step_id="S03")
    assert calls["count"] == 1

    now[0] += 30.0
    adapter.retrieve(query, top_k=1, step_id="S03")
    assert calls["count"] == 1

    adapter.retrieve(query, top_k=3, step_id="S03")
    assert calls["count"] == 2

    adapter.retrieve(query, top_k=2, step_id="S03")
    assert calls["count"] == 2

    alt_query = "S03 engine_crank_switch"
    adapter.retrieve(alt_query, top_k=3, step_id="S03")
    assert calls["count"] == 3

    adapter.retrieve(alt_query, top_k=3, step_id="S03")
    assert calls["count"] == 3

    adapter.retrieve(query, top_k=3, step_id="S04")
    assert calls["count"] == 4

    now[0] += 31.0
    adapter.retrieve(query, top_k=3, step_id="S03")
    assert calls["count"] == 5


def test_retrieve_without_index_marks_grounding_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing_index.json"
    adapter = LocalKnowledgeAdapter(missing)

    snippets, meta = adapter.retrieve_with_meta("battery apu", top_k=3, step_id="S03")
    assert snippets == []
    assert meta["grounding_missing"] is True
    assert meta["grounding_reason"] == "index_missing"
    assert meta["snippet_ids"] == []


def test_retrieve_with_invalid_index_marks_load_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad_index.json"
    bad.write_text("{invalid json", encoding="utf-8")
    adapter = LocalKnowledgeAdapter(bad)

    snippets, meta = adapter.retrieve_with_meta("battery apu", top_k=3, step_id="S03")
    assert snippets == []
    assert meta["grounding_missing"] is True
    assert meta["grounding_reason"] == "index_load_error"
    assert meta["index_error_type"] is not None


def test_retrieve_with_source_policy_filters_non_whitelist_chunks_and_returns_chunk_refs(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "trusted.md"
    doc.write_text(
        "# S03\nAPU switch ON and wait for READY.\n# S04\nEngine crank to RIGHT.\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index_policy.json"
    build_index([str(doc)], str(index_path))

    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "policy_id: retrieval_policy\n"
        "version: v-test\n"
        "allow:\n"
        "  - doc_id: trusted\n"
        "    chunk_id: trusted_0\n"
        "    line_range: [1, 1]\n",
        encoding="utf-8",
    )
    policy = KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)

    adapter = LocalKnowledgeAdapter(index_path, source_policy=policy)
    snippets, meta = adapter.retrieve_with_meta("apu engine crank", top_k=5, step_id="S03")

    assert [item["snippet_id"] for item in snippets] == ["trusted_0"]
    assert snippets[0]["snippet"] == "APU switch ON and wait for READY."
    assert meta["source_policy_applied"] is True
    assert meta["source_policy_id"] == "retrieval_policy"
    assert meta["source_policy_version"] == "v-test"
    assert meta["source_policy_filtered_out_count"] == 1
    assert meta["grounding_missing"] is False
    assert meta["source_chunk_refs"] == ["trusted/trusted_0:1-1"]


def test_retrieve_with_source_policy_marks_grounding_missing_when_all_results_filtered(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_filtered_all.json"
    index_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "allowed_doc",
                        "chunks": [
                            {
                                "chunk_id": "allowed_doc_0",
                                "snippet_id": "allowed_doc_0",
                                "section": "Safe",
                                "page_or_heading": "Safe",
                                "text": "hydraulic accumulator servicing guidance",
                            }
                        ],
                    },
                    {
                        "doc_id": "forbidden",
                        "chunks": [
                            {
                                "chunk_id": "forbidden_0",
                                "snippet_id": "forbidden_0",
                                "section": "Intro",
                                "page_or_heading": "Intro",
                                "text": "battery switch and apu switch are mentioned here",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy_filtered_all.yaml"
    policy_path.write_text(
        "policy_id: retrieval_policy\n"
        "allow:\n"
        "  - doc_id: allowed_doc\n"
        "    chunk_id: allowed_doc_0\n"
        "    line_range: [1, 1]\n",
        encoding="utf-8",
    )

    policy = KnowledgeSourcePolicy.from_yaml(policy_path, index_path=index_path)
    adapter = LocalKnowledgeAdapter(index_path, source_policy=policy)
    snippets, meta = adapter.retrieve_with_meta("battery apu switch", top_k=3, step_id="S03")

    assert snippets == []
    assert meta["grounding_missing"] is True
    assert meta["grounding_reason"] == "policy_filtered_all"
    assert meta["snippet_ids"] == []
    assert meta["source_chunk_refs"] == []
    assert meta["source_policy_filtered_out_count"] == 1


def test_retriever_pool_is_bounded(tmp_path: Path) -> None:
    for idx in range(RETRIEVER_POOL_MAX_SIZE + 4):
        doc = tmp_path / f"doc_{idx}.md"
        index_path = tmp_path / f"index_{idx}.json"
        doc.write_text(f"# H{idx}\nBattery on {idx}\n", encoding="utf-8")
        build_index([str(doc)], str(index_path))
        LocalKnowledgeAdapter(index_path)

    assert len(LocalKnowledgeAdapter._retriever_pool) <= RETRIEVER_POOL_MAX_SIZE


def test_step_cache_is_bounded(tmp_path: Path) -> None:
    doc = tmp_path / "doc_step_cache.md"
    index_path = tmp_path / "index_step_cache.json"
    doc.write_text("# H\nBattery on\n", encoding="utf-8")
    build_index([str(doc)], str(index_path))
    adapter = LocalKnowledgeAdapter(index_path, step_cache_ttl_s=9999.0)
    for idx in range(STEP_CACHE_MAX_SIZE + 16):
        adapter.retrieve(f"query-{idx}", top_k=1, step_id=f"S{idx:04d}")
    assert len(adapter._step_cache) <= STEP_CACHE_MAX_SIZE


def test_build_grounding_query_uses_required_components() -> None:
    query = build_grounding_query(
        pack_title="F/A-18C Cold Start",
        inferred_step="S03",
        missing_conditions=["vars.apu_ready==true", "vars.apu_ready==true"],
        recent_ui_targets=["apu_switch", "apu_switch", "engine_crank_switch"],
    )
    assert query == (
        "F/A-18C Cold Start | S03 | vars.apu_ready==true | apu_switch | engine_crank_switch"
    )


def test_build_grounding_query_ignores_blank_inferred_step() -> None:
    query = build_grounding_query(
        pack_title="F/A-18C Cold Start",
        inferred_step="   ",
        missing_conditions=["vars.apu_ready==true"],
        recent_ui_targets=["apu_switch"],
    )
    assert query == "F/A-18C Cold Start | vars.apu_ready==true | apu_switch"
