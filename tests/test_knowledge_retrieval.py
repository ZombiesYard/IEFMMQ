from pathlib import Path

from adapters.knowledge_local import LocalKnowledgeAdapter, build_grounding_query
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
    adapter.retrieve(query, top_k=3, step_id="S03")
    assert calls["count"] == 1

    now[0] += 30.0
    adapter.retrieve(query, top_k=3, step_id="S03")
    assert calls["count"] == 1

    alt_query = "S03 engine_crank_switch"
    adapter.retrieve(alt_query, top_k=3, step_id="S03")
    assert calls["count"] == 2

    adapter.retrieve(alt_query, top_k=3, step_id="S03")
    assert calls["count"] == 2

    adapter.retrieve(query, top_k=3, step_id="S04")
    assert calls["count"] == 3

    now[0] += 31.0
    adapter.retrieve(query, top_k=3, step_id="S03")
    assert calls["count"] == 4


def test_retrieve_without_index_marks_grounding_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing_index.json"
    adapter = LocalKnowledgeAdapter(missing)

    snippets, meta = adapter.retrieve_with_meta("battery apu", top_k=3, step_id="S03")
    assert snippets == []
    assert meta["grounding_missing"] is True
    assert meta["grounding_reason"] == "index_missing"
    assert meta["snippet_ids"] == []


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
