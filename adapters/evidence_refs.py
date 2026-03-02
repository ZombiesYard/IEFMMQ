"""
Shared evidence reference typing helpers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

# Prefix order is intentional: more-specific families should appear before less-specific ones.
EVIDENCE_PREFIX_TO_TYPE: tuple[tuple[str, str], ...] = (
    ("VARS.", "var"),
    ("GATES.", "gate"),
    ("RAG_SNIPPETS.", "rag"),
    ("RECENT_UI_TARGETS.", "delta"),
    ("DELTA_KEYS.", "delta"),
)

_prefixes_by_type: dict[str, list[str]] = defaultdict(list)
for _prefix, _evidence_type in EVIDENCE_PREFIX_TO_TYPE:
    _prefixes_by_type[_evidence_type].append(_prefix)

EVIDENCE_TYPE_PREFIXES: dict[str, tuple[str, ...]] = {
    evidence_type: tuple(prefixes)
    for evidence_type, prefixes in _prefixes_by_type.items()
}

EVIDENCE_TYPES: frozenset[str] = frozenset(EVIDENCE_TYPE_PREFIXES.keys())


def infer_evidence_type_from_ref(ref: str) -> str | None:
    for prefix, evidence_type in EVIDENCE_PREFIX_TO_TYPE:
        if ref.startswith(prefix):
            return evidence_type
    return None


def collect_evidence_refs_from_context(context: Mapping[str, Any] | None) -> set[str]:
    refs: set[str] = set()
    if not isinstance(context, Mapping):
        return refs

    vars_map = context.get("vars")
    if isinstance(vars_map, Mapping):
        for key in vars_map.keys():
            if isinstance(key, str) and key:
                refs.add(f"VARS.{key}")

    gates = context.get("gates")
    if isinstance(gates, Mapping):
        for gate_id in gates.keys():
            if isinstance(gate_id, str) and gate_id:
                refs.add(f"GATES.{gate_id}")
    elif isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, Mapping):
                continue
            gate_id = gate.get("gate_id")
            if isinstance(gate_id, str) and gate_id:
                refs.add(f"GATES.{gate_id}")

    recent_deltas = context.get("recent_deltas")
    if isinstance(recent_deltas, list):
        for item in recent_deltas:
            if not isinstance(item, Mapping):
                continue
            ui_target = item.get("ui_target")
            if not isinstance(ui_target, str) or not ui_target:
                ui_target = item.get("mapped_ui_target")
            if not isinstance(ui_target, str) or not ui_target:
                ui_target = item.get("target")
            if isinstance(ui_target, str) and ui_target:
                refs.add(f"RECENT_UI_TARGETS.{ui_target}")
            bios_key = item.get("k")
            if not isinstance(bios_key, str) or not bios_key:
                bios_key = item.get("bios_key")
            if isinstance(bios_key, str) and bios_key:
                refs.add(f"DELTA_KEYS.{bios_key}")

    delta_summary = context.get("delta_summary")
    if isinstance(delta_summary, Mapping):
        changed_keys = delta_summary.get("changed_keys_sample")
        if isinstance(changed_keys, list):
            for key in changed_keys:
                if isinstance(key, str) and key:
                    refs.add(f"DELTA_KEYS.{key}")
        topk = delta_summary.get("recent_key_changes_topk")
        if isinstance(topk, list):
            for row in topk:
                if not isinstance(row, Mapping):
                    continue
                key = row.get("key")
                if isinstance(key, str) and key:
                    refs.add(f"DELTA_KEYS.{key}")

    rag_topk = context.get("rag_topk")
    if isinstance(rag_topk, list):
        for item in rag_topk:
            if not isinstance(item, Mapping):
                continue
            snippet_id = item.get("snippet_id") or item.get("id")
            if isinstance(snippet_id, str) and snippet_id:
                refs.add(f"RAG_SNIPPETS.{snippet_id}")

    return refs


__all__ = [
    "collect_evidence_refs_from_context",
    "EVIDENCE_PREFIX_TO_TYPE",
    "EVIDENCE_TYPE_PREFIXES",
    "EVIDENCE_TYPES",
    "infer_evidence_type_from_ref",
]
