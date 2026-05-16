"""
Normalized evidence packet for help-cycle adjudication.

This module intentionally depends only on core Python data structures. Runtime
adapters may feed it telemetry, VLM facts, gate results, and recent actions, but
the packet itself remains independent from DCS, OpenAI, and overlay transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


EARLY_STEP_IDS = frozenset({"S01", "S02", "S03"})
DEFAULT_LATE_DISPLAY_ANCHOR_FACTS = frozenset(
    {
        "tac_page_visible",
        "supt_page_visible",
        "fcs_page_visible",
        "bit_root_page_visible",
        "fcsmc_page_visible",
        "fcsmc_in_test_visible",
        "fcsmc_intermediate_result_visible",
        "fcsmc_final_go_result_visible",
        "hsi_page_visible",
        "hsi_map_layer_visible",
        "ins_grnd_alignment_text_visible",
        "ins_ok_text_visible",
    }
)
DEFAULT_VISUAL_CANDIDATE_STEP_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("tac_page_visible", "bit_root_page_visible"), ("S08", "S09")),
    (
        (
            "ins_grnd_alignment_text_visible",
            "ins_ok_text_visible",
            "hsi_page_visible",
            "hsi_map_layer_visible",
        ),
        ("S12", "S13"),
    ),
    (
        (
            "fcsmc_page_visible",
            "fcsmc_in_test_visible",
            "fcsmc_intermediate_result_visible",
            "fcsmc_final_go_result_visible",
        ),
        ("S18", "S19", "S20"),
    ),
)
CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS = "early_step_from_telemetry_vs_late_display_from_vlm"
CONFLICT_LATCH_MISSING_VS_LATER_STAGE_EVIDENCE = "latch_missing_vs_later_stage_evidence"
CONFLICT_RECENT_ACTION_VS_GATE_CONTRADICTION = "recent_action_vs_gate_contradiction"


def _string_items(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _coerce_int(raw: Any) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw


def _coerce_number(raw: Any) -> int | float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return raw


def _late_display_anchor_fact_ids(context: Mapping[str, Any]) -> set[str]:
    configured = set(_string_items(context.get("late_display_anchor_fact_ids")))
    return configured or set(DEFAULT_LATE_DISPLAY_ANCHOR_FACTS)


def _visual_candidate_step_groups(context: Mapping[str, Any]) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    raw = context.get("visual_candidate_step_groups")
    if not isinstance(raw, (list, tuple)):
        return DEFAULT_VISUAL_CANDIDATE_STEP_GROUPS
    out: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        fact_ids = tuple(_string_items(item.get("fact_ids")))
        step_ids = tuple(_string_items(item.get("step_ids")))
        if fact_ids and step_ids:
            out.append((fact_ids, step_ids))
    return tuple(out) or DEFAULT_VISUAL_CANDIDATE_STEP_GROUPS


def _build_visual_candidate_steps(anchor_ids: set[str], context: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for fact_ids, step_ids in _visual_candidate_step_groups(context):
        if set(fact_ids).intersection(anchor_ids):
            candidates.extend(step_ids)
    return _string_items(candidates)


def _status_from_bool(raw_status: Any) -> str:
    return raw_status if isinstance(raw_status, str) and raw_status else "unknown"


@dataclass(frozen=True)
class TelemetryEvidence:
    source_status: str
    confidence: str
    freshness: dict[str, Any]
    missing_source_count: int
    missing_source_ids: tuple[str, ...]
    early_vars: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_status": self.source_status,
            "confidence": self.confidence,
            "freshness": dict(self.freshness),
            "missing_source_count": self.missing_source_count,
            "missing_source_ids": list(self.missing_source_ids),
            "early_vars": dict(self.early_vars),
        }

    def to_state_harness_dict(self) -> dict[str, Any]:
        return {
            "source_status": self.source_status,
            "confidence": self.confidence,
            "observation_seq": self.freshness.get("observation_seq"),
            "vars_source_missing_count": self.missing_source_count,
            "early_vars": dict(self.early_vars),
        }


@dataclass(frozen=True)
class VisionEvidence:
    source_status: str
    confidence: str
    freshness: dict[str, Any]
    seen_fact_ids: tuple[str, ...]
    not_seen_fact_ids: tuple[str, ...]
    uncertain_fact_ids: tuple[str, ...]
    fresh_fact_ids: tuple[str, ...]
    late_display_anchors: tuple[str, ...]
    visual_candidate_steps: tuple[str, ...]
    source_frame_ids: tuple[str, ...]
    facts: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_status": self.source_status,
            "confidence": self.confidence,
            "freshness": dict(self.freshness),
            "seen_fact_ids": list(self.seen_fact_ids),
            "not_seen_fact_ids": list(self.not_seen_fact_ids),
            "uncertain_fact_ids": list(self.uncertain_fact_ids),
            "fresh_fact_ids": list(self.fresh_fact_ids),
            "late_display_anchors": list(self.late_display_anchors),
            "visual_candidate_steps": list(self.visual_candidate_steps),
            "source_frame_ids": list(self.source_frame_ids),
            "facts": [dict(item) for item in self.facts],
        }

    def to_state_harness_dict(self) -> dict[str, Any]:
        return {
            "source_status": self.source_status,
            "confidence": self.confidence,
            "late_display_anchors": list(self.late_display_anchors),
            "visual_candidate_steps": list(self.visual_candidate_steps),
            "seen_fact_ids": list(self.seen_fact_ids)[:12],
            "fresh_fact_ids": list(self.fresh_fact_ids)[:12],
            "not_seen_fact_ids": list(self.not_seen_fact_ids)[:12],
        }


@dataclass(frozen=True)
class GateEvidence:
    source_status: str
    confidence: str
    freshness: dict[str, Any]
    blocked_gate_ids: tuple[str, ...]
    satisfied_gate_ids: tuple[str, ...]
    blocked_gates: tuple[dict[str, Any], ...]
    satisfied_gates: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_status": self.source_status,
            "confidence": self.confidence,
            "freshness": dict(self.freshness),
            "blocked_gate_ids": list(self.blocked_gate_ids),
            "blocked_gate_count": len(self.blocked_gate_ids),
            "satisfied_gate_ids": list(self.satisfied_gate_ids),
            "satisfied_gate_count": len(self.satisfied_gate_ids),
            "blocked_gates": [dict(item) for item in self.blocked_gates],
            "satisfied_gates": [dict(item) for item in self.satisfied_gates],
        }

    def to_state_harness_dict(self) -> dict[str, Any]:
        legacy_blocked_gate_ids = list(self.blocked_gate_ids)[:8]
        return {
            "blocked_gate_ids": legacy_blocked_gate_ids,
            "blocked_gate_count": len(legacy_blocked_gate_ids),
        }


@dataclass(frozen=True)
class RecentActionEvidence:
    source_status: str
    confidence: str
    freshness: dict[str, Any]
    target_ids: tuple[str, ...]
    current_target_id: str | None
    actions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_status": self.source_status,
            "confidence": self.confidence,
            "freshness": dict(self.freshness),
            "target_ids": list(self.target_ids),
            "current_target_id": self.current_target_id,
            "actions": [dict(item) for item in self.actions],
        }

    def to_state_harness_dict(self) -> dict[str, Any]:
        return {
            "source_status": self.source_status,
            "recent_buttons": list(self.target_ids)[:8],
        }


@dataclass(frozen=True)
class DeterministicCandidateEvidence:
    step_id: str | None
    overlay_step_id: str | None
    missing_conditions: tuple[str, ...]
    role: str = "candidate_not_authoritative"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "overlay_step_id": self.overlay_step_id,
            "missing_conditions": list(self.missing_conditions),
            "role": self.role,
        }

    def to_state_harness_dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class EvidencePacket:
    telemetry_evidence: TelemetryEvidence
    vision_evidence: VisionEvidence
    gate_evidence: GateEvidence
    recent_action_evidence: RecentActionEvidence
    deterministic_candidate: DeterministicCandidateEvidence
    conflicts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "telemetry_evidence": self.telemetry_evidence.to_dict(),
            "vision_evidence": self.vision_evidence.to_dict(),
            "gate_evidence": self.gate_evidence.to_dict(),
            "recent_action_evidence": self.recent_action_evidence.to_dict(),
            "deterministic_candidate": self.deterministic_candidate.to_dict(),
            "conflicts": list(self.conflicts),
        }

    def to_state_harness_dict(self) -> dict[str, Any]:
        legacy_conflicts = [
            conflict
            for conflict in self.conflicts
            if conflict == CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS
        ]
        return {
            "telemetry_evidence": self.telemetry_evidence.to_state_harness_dict(),
            "vision_evidence": self.vision_evidence.to_state_harness_dict(),
            "gate_evidence": self.gate_evidence.to_state_harness_dict(),
            "recent_action_evidence": self.recent_action_evidence.to_state_harness_dict(),
            "deterministic_candidate": self.deterministic_candidate.to_state_harness_dict(),
            "conflicts": legacy_conflicts,
        }

    def compact_summary(self) -> dict[str, Any]:
        return {
            "telemetry_status": self.telemetry_evidence.source_status,
            "telemetry_missing_source_count": self.telemetry_evidence.missing_source_count,
            "vision_status": self.vision_evidence.source_status,
            "vision_seen_count": len(self.vision_evidence.seen_fact_ids),
            "vision_fresh_count": len(self.vision_evidence.fresh_fact_ids),
            "blocked_gate_count": len(self.gate_evidence.blocked_gate_ids),
            "recent_action_count": len(self.recent_action_evidence.target_ids),
            "conflicts": list(self.conflicts),
        }


def _build_telemetry_evidence(context: Mapping[str, Any]) -> TelemetryEvidence:
    vars_raw = context.get("vars")
    vars_map = vars_raw if isinstance(vars_raw, Mapping) else {}
    missing_sources = _string_items(vars_map.get("vars_source_missing"))
    missing_count = len(missing_sources)
    vision_raw = context.get("vision")
    vision = vision_raw if isinstance(vision_raw, Mapping) else {}
    telemetry_raw = context.get("telemetry")
    telemetry = telemetry_raw if isinstance(telemetry_raw, Mapping) else {}
    seq = _coerce_int(vision.get("observation_seq"))
    t_wall = _coerce_number(telemetry.get("t_wall"))

    bootstrap_like = (
        missing_count >= 20
        or (isinstance(seq, int) and seq <= 3)
        or (
            vars_map.get("battery_on") is False
            and vars_map.get("power_available") is False
            and missing_count >= 8
        )
    )
    source_status = "low_confidence_bootstrap" if bootstrap_like else "nominal"
    freshness: dict[str, Any] = {}
    if seq is not None:
        freshness["observation_seq"] = seq
    if t_wall is not None:
        freshness["t_wall"] = t_wall
    return TelemetryEvidence(
        source_status=source_status,
        confidence="low" if source_status != "nominal" else "medium",
        freshness=freshness,
        missing_source_count=missing_count,
        missing_source_ids=tuple(missing_sources[:32]),
        early_vars={
            key: vars_map.get(key)
            for key in ("battery_on", "power_available", "fire_test_a_complete", "fire_test_b_complete")
            if key in vars_map
        },
    )


def _build_vision_evidence(context: Mapping[str, Any]) -> VisionEvidence:
    summary_raw = context.get("vision_fact_summary")
    summary = summary_raw if isinstance(summary_raw, Mapping) else {}
    seen_ids = set(_string_items(summary.get("seen_fact_ids")))
    fresh_ids = set(_string_items(summary.get("fresh_fact_ids")))
    not_seen_ids = set(_string_items(summary.get("not_seen_fact_ids")))
    uncertain_ids = set(_string_items(summary.get("uncertain_fact_ids")))
    frame_ids = _string_items(summary.get("frame_ids"))

    facts: list[dict[str, Any]] = []
    facts_raw = context.get("vision_facts")
    if isinstance(facts_raw, list):
        for item in facts_raw:
            if not isinstance(item, Mapping):
                continue
            fact_id = item.get("fact_id")
            if not isinstance(fact_id, str) or not fact_id:
                continue
            fact: dict[str, Any] = {
                "fact_id": fact_id,
                "state": item.get("state"),
            }
            state = item.get("state")
            if state == "seen":
                seen_ids.add(fact_id)
            elif state == "not_seen":
                not_seen_ids.add(fact_id)
            elif state == "uncertain":
                uncertain_ids.add(fact_id)
            source_frame_id = item.get("source_frame_id")
            if isinstance(source_frame_id, str) and source_frame_id:
                fact["source_frame_id"] = source_frame_id
                if source_frame_id not in frame_ids:
                    frame_ids.append(source_frame_id)
            sticky = item.get("sticky")
            if isinstance(sticky, bool):
                fact["sticky"] = sticky
            expires_after_ms = item.get("expires_after_ms")
            if isinstance(expires_after_ms, (int, float)) and not isinstance(expires_after_ms, bool):
                fact["expires_after_ms"] = expires_after_ms
            facts.append(fact)

    late_anchors = sorted((seen_ids | fresh_ids).intersection(_late_display_anchor_fact_ids(context)))
    source_status = _status_from_bool(summary.get("status")) if summary else "vision_unavailable"
    if source_status == "vision_unavailable" and facts:
        source_status = "available"
    return VisionEvidence(
        source_status=source_status,
        confidence="high" if late_anchors else "medium",
        freshness={"fresh_fact_ids": sorted(fresh_ids), "frame_ids": list(frame_ids)},
        seen_fact_ids=tuple(sorted(seen_ids)),
        not_seen_fact_ids=tuple(sorted(not_seen_ids)),
        uncertain_fact_ids=tuple(sorted(uncertain_ids)),
        fresh_fact_ids=tuple(sorted(fresh_ids)),
        late_display_anchors=tuple(late_anchors),
        visual_candidate_steps=tuple(_build_visual_candidate_steps(set(late_anchors), context)),
        source_frame_ids=tuple(frame_ids),
        facts=tuple(facts[:16]),
    )


def _gate_entry(gate_id: str, gate: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "gate_id": gate_id,
        "status": gate.get("status"),
    }
    for key in ("step_id", "gate_type", "reason_code", "reason"):
        value = gate.get(key)
        if value is not None:
            out[key] = value
    return out


def _build_gate_evidence(context: Mapping[str, Any]) -> GateEvidence:
    gates_raw = context.get("gates")
    gates = gates_raw if isinstance(gates_raw, Mapping) else {}
    blocked: list[dict[str, Any]] = []
    satisfied: list[dict[str, Any]] = []
    for gate_id, gate in gates.items():
        if not isinstance(gate_id, str) or not isinstance(gate, Mapping):
            continue
        status = gate.get("status")
        if status == "blocked":
            blocked.append(_gate_entry(gate_id, gate))
        elif status in {"allowed", "satisfied"} or gate.get("allowed") is True:
            satisfied.append(_gate_entry(gate_id, gate))
    return GateEvidence(
        source_status="available" if gates else "empty",
        confidence="high" if gates else "low",
        freshness={"gate_count": len(gates)},
        blocked_gate_ids=tuple(item["gate_id"] for item in blocked),
        satisfied_gate_ids=tuple(item["gate_id"] for item in satisfied),
        blocked_gates=tuple(blocked[:16]),
        satisfied_gates=tuple(satisfied[:16]),
    )


def _build_recent_action_evidence(context: Mapping[str, Any]) -> RecentActionEvidence:
    recent_raw = context.get("recent_actions")
    recent = recent_raw if isinstance(recent_raw, Mapping) else {}
    target_ids = _string_items(recent.get("recent_buttons"))
    current = recent.get("current_button")
    current_target_id = current if isinstance(current, str) and current else None
    deltas_raw = context.get("recent_deltas")
    actions_by_target: dict[str, dict[str, Any]] = {}
    if isinstance(deltas_raw, list):
        for item in deltas_raw:
            if not isinstance(item, Mapping):
                continue
            target = item.get("mapped_ui_target")
            if not isinstance(target, str) or not target or target in actions_by_target:
                continue
            action: dict[str, Any] = {"target_id": target}
            seq = _coerce_int(item.get("seq"))
            if seq is not None:
                action["seq"] = seq
            t_wall = _coerce_number(item.get("t_wall"))
            if t_wall is not None:
                action["t_wall"] = t_wall
            actions_by_target[target] = action
    actions: list[dict[str, Any]] = []
    for target_id in target_ids:
        actions.append(actions_by_target.get(target_id, {"target_id": target_id}))
    seq_values = [item.get("seq") for item in actions if isinstance(item.get("seq"), int)]
    t_wall_values = [
        item.get("t_wall")
        for item in actions
        if isinstance(item.get("t_wall"), (int, float)) and not isinstance(item.get("t_wall"), bool)
    ]
    freshness: dict[str, Any] = {}
    if seq_values:
        freshness["latest_seq"] = max(seq_values)
    if t_wall_values:
        freshness["latest_t_wall"] = max(t_wall_values)
    return RecentActionEvidence(
        source_status="available" if target_ids else "empty",
        confidence="medium" if target_ids else "low",
        freshness=freshness,
        target_ids=tuple(target_ids[:8]),
        current_target_id=current_target_id,
        actions=tuple(actions[:8]),
    )


def _build_deterministic_candidate(context: Mapping[str, Any]) -> DeterministicCandidateEvidence:
    raw = context.get("deterministic_step_hint")
    hint = raw if isinstance(raw, Mapping) else {}
    step_id = hint.get("inferred_step_id")
    overlay_step_id = hint.get("overlay_step_id")
    return DeterministicCandidateEvidence(
        step_id=step_id if isinstance(step_id, str) else None,
        overlay_step_id=overlay_step_id if isinstance(overlay_step_id, str) else None,
        missing_conditions=tuple(_string_items(hint.get("missing_conditions"))),
    )


def _has_recent_action_gate_contradiction(
    gate_evidence: GateEvidence,
    recent_action_evidence: RecentActionEvidence,
) -> bool:
    recent_targets = set(recent_action_evidence.target_ids)
    if not recent_targets:
        return False
    for gate in gate_evidence.blocked_gates:
        searchable = " ".join(
            str(gate.get(key, ""))
            for key in ("gate_id", "reason_code", "reason")
        )
        if any(target in searchable for target in recent_targets):
            return True
    return False


def _build_conflicts(
    *,
    telemetry_evidence: TelemetryEvidence,
    vision_evidence: VisionEvidence,
    gate_evidence: GateEvidence,
    recent_action_evidence: RecentActionEvidence,
    deterministic_candidate: DeterministicCandidateEvidence,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    if (
        deterministic_candidate.step_id in EARLY_STEP_IDS
        and len(vision_evidence.late_display_anchors) >= 2
    ):
        conflicts.append(CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS)
    if (
        deterministic_candidate.missing_conditions
        and vision_evidence.late_display_anchors
        and any("_complete" in item or "latch" in item for item in deterministic_candidate.missing_conditions)
    ):
        conflicts.append(CONFLICT_LATCH_MISSING_VS_LATER_STAGE_EVIDENCE)
    if _has_recent_action_gate_contradiction(gate_evidence, recent_action_evidence):
        conflicts.append(CONFLICT_RECENT_ACTION_VS_GATE_CONTRADICTION)
    return tuple(_string_items(conflicts))


def build_evidence_packet(context: Mapping[str, Any]) -> EvidencePacket:
    telemetry_evidence = _build_telemetry_evidence(context)
    vision_evidence = _build_vision_evidence(context)
    gate_evidence = _build_gate_evidence(context)
    recent_action_evidence = _build_recent_action_evidence(context)
    deterministic_candidate = _build_deterministic_candidate(context)
    conflicts = _build_conflicts(
        telemetry_evidence=telemetry_evidence,
        vision_evidence=vision_evidence,
        gate_evidence=gate_evidence,
        recent_action_evidence=recent_action_evidence,
        deterministic_candidate=deterministic_candidate,
    )
    return EvidencePacket(
        telemetry_evidence=telemetry_evidence,
        vision_evidence=vision_evidence,
        gate_evidence=gate_evidence,
        recent_action_evidence=recent_action_evidence,
        deterministic_candidate=deterministic_candidate,
        conflicts=conflicts,
    )


__all__ = [
    "CONFLICT_LATCH_MISSING_VS_LATER_STAGE_EVIDENCE",
    "CONFLICT_RECENT_ACTION_VS_GATE_CONTRADICTION",
    "CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS",
    "EARLY_STEP_IDS",
    "DEFAULT_LATE_DISPLAY_ANCHOR_FACTS",
    "DEFAULT_VISUAL_CANDIDATE_STEP_GROUPS",
    "EvidencePacket",
    "build_evidence_packet",
]
