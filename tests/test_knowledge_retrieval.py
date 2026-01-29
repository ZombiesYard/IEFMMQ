from pathlib import Path

from adapters.knowledge_local import LocalKnowledgeAdapter
from tools.index_docs import build_index


def test_index_and_retrieve_markdown():
    base = Path("logs/test_index")
    base.mkdir(parents=True, exist_ok=True)
    doc = base / "doc.md"
    doc.write_text("# Title\nEngine start checklist\nTurn battery on.\n", encoding="utf-8")
    index_path = base / "index.json"
    build_index([str(doc)], str(index_path))

    adapter = LocalKnowledgeAdapter(str(index_path))
    results = adapter.query("battery on", k=3)
    assert results
    top = results[0]
    assert top["doc_id"] == "doc"
    assert "battery" in top["snippet"].lower()
