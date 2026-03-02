"""
Shared evidence reference typing helpers.
"""

from __future__ import annotations

from collections import defaultdict

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


__all__ = [
    "EVIDENCE_PREFIX_TO_TYPE",
    "EVIDENCE_TYPE_PREFIXES",
    "EVIDENCE_TYPES",
    "infer_evidence_type_from_ref",
]
