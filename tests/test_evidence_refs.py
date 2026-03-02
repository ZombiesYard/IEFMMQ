from adapters.evidence_refs import collect_evidence_refs_from_context


def test_collect_evidence_refs_from_context_collects_all_supported_families() -> None:
    context = {
        "vars": {"battery_on": True},
        "gates": [
            {"gate_id": "S01.completion", "status": "blocked"},
            {"gate_id": "S03.precondition", "status": "allowed"},
        ],
        "recent_deltas": [
            {"k": "BATTERY_SW", "mapped_ui_target": "battery_switch"},
        ],
        "delta_summary": {
            "changed_keys_sample": ["BATTERY_SW"],
            "recent_key_changes_topk": [{"key": "APU_CONTROL_SW"}],
        },
        "rag_topk": [
            {"snippet_id": "doc_1"},
            {"id": "doc_2"},
        ],
    }

    refs = collect_evidence_refs_from_context(context)

    assert "VARS.battery_on" in refs
    assert "GATES.S01.completion" in refs
    assert "GATES.S03.precondition" in refs
    assert "RECENT_UI_TARGETS.battery_switch" in refs
    assert "DELTA_KEYS.BATTERY_SW" in refs
    assert "DELTA_KEYS.APU_CONTROL_SW" in refs
    assert "RAG_SNIPPETS.doc_1" in refs
    assert "RAG_SNIPPETS.doc_2" in refs
