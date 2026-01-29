"""
Simple scoring engine v1.

Rules:
- Omission (OM): any canonical step up to the max observed step not completed.
- State Violation (SV): observation tags containing 'state_violation'.
- Other categories default to 0 in this version.
- Critical multiplier applied to omissions when step critical=True.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


def _load_taxonomy(taxonomy_path: str) -> dict:
    return yaml.safe_load(Path(taxonomy_path).read_text(encoding="utf-8"))


def _load_steps(pack_path: str) -> List[dict]:
    return yaml.safe_load(Path(pack_path).read_text(encoding="utf-8"))["steps"]


def _step_order(steps: List[dict]) -> Dict[str, int]:
    return {s["id"]: idx for idx, s in enumerate(steps)}


def score_log(log_events: List[dict], pack_path: str, taxonomy_path: str) -> dict:
    steps = _load_steps(pack_path)
    order = _step_order(steps)
    taxonomy = _load_taxonomy(taxonomy_path)
    weights = taxonomy["scoring"]["base_weights"]
    critical_multiplier = taxonomy["scoring"]["critical_step_multiplier"]
    rounding = taxonomy["scoring"].get("rounding", "ceil")

    completed: Set[str] = set()
    max_seen_idx = -1
    sv_count = 0

    for ev in log_events:
        kind = ev.get("kind") or ev.get("type")
        payload = ev.get("payload", {})
        if kind == "step_completed":
            sid = payload["step_id"]
            completed.add(sid)
            max_seen_idx = max(max_seen_idx, order.get(sid, -1))
        if kind == "step_activated":
            sid = payload["step_id"]
            max_seen_idx = max(max_seen_idx, order.get(sid, -1))
        if kind == "observation":
            tags = payload.get("tags") or payload.get("payload", {}).get("tags") or []
            if "state_violation" in tags:
                sv_count += 1

    target_steps = [s for s in steps if order[s["id"]] <= max_seen_idx] if max_seen_idx >= 0 else []
    om_count = sum(1 for s in target_steps if s["id"] not in completed)

    def apply_weight(count: int, weight: int, critical: bool) -> int:
        score = count * weight * (critical_multiplier if critical else 1)
        return math.ceil(score) if rounding == "ceil" else score

    counts = {"OM": om_count, "CO": 0, "OR": 0, "PA": 0, "SV": sv_count}
    errors = {
        "OM": sum(
            apply_weight(1, weights["OM"], s.get("critical", False))
            for s in target_steps
            if s["id"] not in completed
        ),
        "CO": 0,
        "OR": 0,
        "PA": 0,
        "SV": apply_weight(sv_count, weights["SV"], False),
    }

    total = sum(errors.values())

    return {
        "Count_OM": counts["OM"],
        "Count_CO": counts["CO"],
        "Count_OR": counts["OR"],
        "Count_PA": counts["PA"],
        "Count_SV": counts["SV"],
        "Error_OM": errors["OM"],
        "Error_CO": errors["CO"],
        "Error_OR": errors["OR"],
        "Error_PA": errors["PA"],
        "Error_SV": errors["SV"],
        "TotalErrorScore": total,
        "completed_steps": sorted(list(completed)),
        "max_observed_step_index": max_seen_idx,
    }
