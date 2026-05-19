"""
Normalized evidence packet for help-cycle adjudication.

This module intentionally depends only on core Python data structures. Runtime
adapters may feed it telemetry, VLM facts, gate results, and recent actions,
but the packet itself remains independent from concrete transport providers.
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
CONFLICT_TELEMETRY_WINDOW_VS_SINGLE_FRAME = "telemetry_window_conflicts_with_single_frame_hint"
CONTRADICTION_BATTERY_FIRST_FRAME_REFUTED = (
    "battery_on=false only in first frame but later downstream avionics evidence is present"
)
CONTRADICTION_FIRE_TEST_LATCH_MISSING_LATER_STAGE = (
    "fire_test latch missing while later-stage telemetry evidence is present"
)
CONTRADICTION_RECENT_TRANSITION_BLOCKED_GATE = (
    "recent telemetry transition conflicts with current blocked gate"
)
MAX_DIGEST_STRING_VALUE_CHARS = 80

IMPORTANT_BOOLEAN_VARS = frozenset(
    {
        "battery_on",
        "power_available",
        "left_ddi_on",
        "right_ddi_on",
        "mpcd_on",
        "hud_on",
        "fire_test_a_complete",
        "fire_test_b_complete",
        "fire_test_complete",
        "fcs_bit_switch_up",
        "pitot_heat_on",
        "flap_auto",
    }
)
DOWNSTREAM_AVIONICS_VARS = frozenset(
    {
        "power_available",
        "left_ddi_on",
        "right_ddi_on",
        "mpcd_on",
        "hud_on",
    }
)
EARLY_LATCH_VARS = frozenset(
    {
        "fire_test_a_complete",
        "fire_test_b_complete",
        "fire_test_complete",
    }
)
TELEMETRY_WINDOW_CANDIDATE_VARS = frozenset(
    {
        "battery_on",
        "power_available",
        "left_ddi_on",
        "right_ddi_on",
        "mpcd_on",
        "hud_on",
        "fire_test_a_complete",
        "fire_test_b_complete",
        "fire_test_complete",
        "fcs_bit_switch_up",
        "ext_refuel_probe_value",
        "launch_bar_switch_value",
        "hook_handle_value",
        "pitot_heat_on",
        "flap_auto",
    }
)


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


def _safe_digest_value(raw: Any) -> Any:
    if raw is None or isinstance(raw, (bool, int, float)):
        return raw
    if isinstance(raw, str):
        return raw if len(raw) <= MAX_DIGEST_STRING_VALUE_CHARS else raw[:MAX_DIGEST_STRING_VALUE_CHARS] + "..."
    text = str(raw)
    return text if len(text) <= MAX_DIGEST_STRING_VALUE_CHARS else text[:MAX_DIGEST_STRING_VALUE_CHARS] + "..."


def _frame_vars(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    vars_raw = raw.get("vars")
    if isinstance(vars_raw, Mapping):
        return {
            key: _safe_digest_value(value)
            for key, value in vars_raw.items()
            if isinstance(key, str) and key
        }
    delta_raw = raw.get("delta")
    if isinstance(delta_raw, Mapping):
        return {
            key: _safe_digest_value(value)
            for key, value in delta_raw.items()
            if isinstance(key, str) and key
        }
    return {}


def _frame_vars_are_full_snapshot(raw: Mapping[str, Any]) -> bool:
    if raw.get("vars_is_full_snapshot") is True or raw.get("frame_kind") == "vars_snapshot":
        return True
    return isinstance(raw.get("vars"), Mapping) and not isinstance(raw.get("delta"), Mapping)


def _frame_seq_range(frames: tuple[dict[str, Any], ...]) -> tuple[int | None, int | None]:
    first_seq: int | None = None
    latest_seq: int | None = None
    for frame in frames:
        seq = _coerce_int(frame.get("seq"))
        if seq is None:
            continue
        if first_seq is None:
            first_seq = seq
        latest_seq = seq
    return first_seq, latest_seq


def _latest_frame_meta(frames: tuple[dict[str, Any], ...]) -> tuple[int | None, int | float | None]:
    latest_seq: int | None = None
    latest_t_wall: int | float | None = None
    for frame in frames:
        seq = _coerce_int(frame.get("seq"))
        if seq is not None:
            latest_seq = seq
        t_wall = _coerce_number(frame.get("t_wall"))
        if t_wall is not None:
            latest_t_wall = t_wall
    return latest_seq, latest_t_wall


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
class TelemetryWindowDigest:
    window_duration_s: float | None
    frame_count: int
    first_seq: int | None
    latest_seq: int | None
    latest_t_wall: int | float | None
    changed_vars: tuple[dict[str, Any], ...]
    stable_true_vars: tuple[str, ...]
    stable_false_vars: tuple[str, ...]
    unknown_or_missing_vars: tuple[str, ...]
    first_frame_only_values: tuple[dict[str, Any], ...]
    last_seen_true: tuple[dict[str, Any], ...]
    contradictions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_duration_s": self.window_duration_s,
            "frame_count": self.frame_count,
            "first_seq": self.first_seq,
            "latest_seq": self.latest_seq,
            "latest_t_wall": self.latest_t_wall,
            "changed_vars": [dict(item) for item in self.changed_vars],
            "stable_true_vars": list(self.stable_true_vars),
            "stable_false_vars": list(self.stable_false_vars),
            "unknown_or_missing_vars": list(self.unknown_or_missing_vars),
            "first_frame_only_values": [dict(item) for item in self.first_frame_only_values],
            "last_seen_true": [dict(item) for item in self.last_seen_true],
            "contradictions": list(self.contradictions),
        }

    def to_state_harness_dict(self) -> dict[str, Any]:
        return {
            "window_duration_s": self.window_duration_s,
            "frame_count": self.frame_count,
            "first_seq": self.first_seq,
            "latest_seq": self.latest_seq,
            "latest_t_wall": self.latest_t_wall,
            "changed_vars": [dict(item) for item in self.changed_vars[:12]],
            "stable_true_vars": list(self.stable_true_vars[:16]),
            "stable_false_vars": list(self.stable_false_vars[:16]),
            "unknown_or_missing_vars": list(self.unknown_or_missing_vars[:16]),
            "first_frame_only_values": [dict(item) for item in self.first_frame_only_values[:8]],
            "last_seen_true": [dict(item) for item in self.last_seen_true[:12]],
            "contradictions": list(self.contradictions[:8]),
        }

    def compact_summary(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "first_seq": self.first_seq,
            "latest_seq": self.latest_seq,
            "changed_var_count": len(self.changed_vars),
            "contradiction_count": len(self.contradictions),
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
class StepCandidate:
    step_id: str
    source: str
    role: str
    supporting_evidence_refs: tuple[str, ...]
    refuting_evidence_refs: tuple[str, ...]
    confidence: float
    missing_conditions: tuple[str, ...]
    proposed_next_action_target_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "source": self.source,
            "role": self.role,
            "supporting_evidence_refs": list(self.supporting_evidence_refs),
            "refuting_evidence_refs": list(self.refuting_evidence_refs),
            "confidence": self.confidence,
            "missing_conditions": list(self.missing_conditions),
            "proposed_next_action_target_ids": list(self.proposed_next_action_target_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EvidencePacket:
    telemetry_evidence: TelemetryEvidence
    telemetry_window_digest: TelemetryWindowDigest
    vision_evidence: VisionEvidence
    gate_evidence: GateEvidence
    recent_action_evidence: RecentActionEvidence
    deterministic_candidate: DeterministicCandidateEvidence
    conflicts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "telemetry_evidence": self.telemetry_evidence.to_dict(),
            "telemetry_window_digest": self.telemetry_window_digest.to_dict(),
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
            if conflict
            in {
                CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS,
                CONFLICT_TELEMETRY_WINDOW_VS_SINGLE_FRAME,
            }
        ]
        return {
            "telemetry_evidence": self.telemetry_evidence.to_state_harness_dict(),
            "telemetry_window_digest": self.telemetry_window_digest.to_state_harness_dict(),
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
            "telemetry_window_digest": self.telemetry_window_digest.compact_summary(),
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
    telemetry_seq = _coerce_int(telemetry.get("observation_seq"))
    vision_seq = _coerce_int(vision.get("observation_seq"))
    seq = vision_seq
    if seq is None:
        seq = telemetry_seq
    t_wall = _coerce_number(telemetry.get("t_wall"))

    bootstrap_like = (
        missing_count >= 20
        or (isinstance(vision_seq, int) and vision_seq <= 3)
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


def _build_telemetry_window_digest(context: Mapping[str, Any]) -> TelemetryWindowDigest:
    frames_raw = context.get("telemetry_window_frames")
    raw_frames = frames_raw if isinstance(frames_raw, list) else []
    frames: list[dict[str, Any]] = []
    for raw in raw_frames:
        if not isinstance(raw, Mapping):
            continue
        frame_vars = _frame_vars(raw)
        frames.append(
            {
                "seq": _coerce_int(raw.get("seq")),
                "t_wall": _coerce_number(raw.get("t_wall")),
                "vars": frame_vars,
                "vars_is_full_snapshot": _frame_vars_are_full_snapshot(raw),
            }
        )

    frame_tuple = tuple(frames)
    first_seq, latest_seq = _frame_seq_range(frame_tuple)
    _, latest_t_wall = _latest_frame_meta(frame_tuple)
    t_values = [frame["t_wall"] for frame in frame_tuple if _coerce_number(frame.get("t_wall")) is not None]
    window_duration_s = None
    if len(t_values) >= 2:
        window_duration_s = round(float(t_values[-1]) - float(t_values[0]), 3)

    vars_raw = context.get("vars")
    latest_vars = vars_raw if isinstance(vars_raw, Mapping) else {}
    missing_vars = set(_string_items(latest_vars.get("vars_source_missing")))
    for key, value in latest_vars.items():
        if isinstance(key, str) and key and value is None:
            missing_vars.add(key)

    observations: dict[str, list[tuple[Any, int | None, int | float | None]]] = {}
    for frame in frame_tuple:
        seq = _coerce_int(frame.get("seq"))
        t_wall = _coerce_number(frame.get("t_wall"))
        vars_map = frame.get("vars")
        if not isinstance(vars_map, Mapping):
            continue
        for key, value in vars_map.items():
            if not isinstance(key, str) or not key:
                continue
            observations.setdefault(key, []).append((_safe_digest_value(value), seq, t_wall))

    for key, value in latest_vars.items():
        if not isinstance(key, str) or not key or key == "vars_source_missing" or value is None:
            continue
        if key in observations:
            continue
        observed = observations.setdefault(key, [])
        safe_value = _safe_digest_value(value)
        if not observed or observed[-1][0] != safe_value:
            observed.append((safe_value, latest_seq, latest_t_wall))

    changed_vars: list[dict[str, Any]] = []
    stable_true_vars: list[str] = []
    stable_false_vars: list[str] = []
    last_seen_true: list[dict[str, Any]] = []
    for key in sorted(observations.keys()):
        values = observations[key]
        if not values:
            continue
        compact_values = [value for value, _, _ in values]
        transition_count = 0
        latest_transition_t = values[-1][2]
        for idx in range(1, len(compact_values)):
            if compact_values[idx] != compact_values[idx - 1]:
                transition_count += 1
                latest_transition_t = values[idx][2]
        if transition_count:
            age = None
            if latest_t_wall is not None and latest_transition_t is not None:
                age = round(float(latest_t_wall) - float(latest_transition_t), 3)
            changed_vars.append(
                {
                    "var": key,
                    "first_value": compact_values[0],
                    "last_value": compact_values[-1],
                    "transition_count": transition_count,
                    "latest_transition_age_s": age,
                }
            )

        if key in IMPORTANT_BOOLEAN_VARS and compact_values and all(
            value is True or value is False or value is None for value in compact_values
        ):
            if all(value is True for value in compact_values):
                stable_true_vars.append(key)
            elif all(value is False for value in compact_values):
                stable_false_vars.append(key)
            true_observations = [
                (seq, t_wall)
                for value, seq, t_wall in values
                if value is True
            ]
            if true_observations:
                seq, t_wall = true_observations[-1]
                entry: dict[str, Any] = {"var": key}
                if seq is not None:
                    entry["seq"] = seq
                if latest_t_wall is not None and t_wall is not None:
                    entry["age_s"] = round(float(latest_t_wall) - float(t_wall), 3)
                last_seen_true.append(entry)

    first_frame_only_values: list[dict[str, Any]] = []
    if (
        len(frame_tuple) >= 2
        and bool(frame_tuple[0].get("vars_is_full_snapshot"))
        and any(bool(frame.get("vars_is_full_snapshot")) for frame in frame_tuple[1:])
    ):
        first_vars = frame_tuple[0].get("vars")
        later_keys = {
            key
            for frame in frame_tuple[1:]
            if bool(frame.get("vars_is_full_snapshot"))
            if isinstance(frame.get("vars"), Mapping)
            for key in frame["vars"].keys()
            if isinstance(key, str)
        }
        if isinstance(first_vars, Mapping):
            later_vars = [
                frame["vars"]
                for frame in frame_tuple[1:]
                if bool(frame.get("vars_is_full_snapshot")) and isinstance(frame.get("vars"), Mapping)
            ]
            for key, value in first_vars.items():
                later_present = key in later_keys
                later_has_known_value = any(vars_map.get(key) is not None for vars_map in later_vars)
                if isinstance(key, str) and key and (not later_present or not later_has_known_value):
                    first_frame_only_values.append({"var": key, "value": _safe_digest_value(value)})

    downstream_true = any(
        latest_vars.get(key) is True
        or key in stable_true_vars
        or any(item.get("var") == key and item.get("last_value") is True for item in changed_vars)
        for key in DOWNSTREAM_AVIONICS_VARS
    )
    contradictions: list[str] = []
    if downstream_true and any(
        item.get("var") == "battery_on" and item.get("value") is False
        for item in first_frame_only_values
    ):
        contradictions.append(CONTRADICTION_BATTERY_FIRST_FRAME_REFUTED)
    if downstream_true and (
        missing_vars.intersection(EARLY_LATCH_VARS)
        or any("fire_test" in item for item in _string_items(context.get("deterministic_step_hint", {}).get("missing_conditions") if isinstance(context.get("deterministic_step_hint"), Mapping) else []))
    ):
        contradictions.append(CONTRADICTION_FIRE_TEST_LATCH_MISSING_LATER_STAGE)

    gate_raw = context.get("gates")
    gates = gate_raw if isinstance(gate_raw, Mapping) else {}
    changed_names = {
        item.get("var")
        for item in changed_vars
        if isinstance(item.get("var"), str) and item.get("var") in TELEMETRY_WINDOW_CANDIDATE_VARS
    }
    for gate in gates.values():
        if not isinstance(gate, Mapping) or gate.get("status") != "blocked":
            continue
        searchable = " ".join(str(gate.get(key, "")) for key in ("gate_id", "reason_code", "reason"))
        if any(name in searchable for name in changed_names):
            contradictions.append(CONTRADICTION_RECENT_TRANSITION_BLOCKED_GATE)
            break

    unknown_or_missing = sorted(missing_vars)
    return TelemetryWindowDigest(
        window_duration_s=window_duration_s,
        frame_count=len(frame_tuple),
        first_seq=first_seq,
        latest_seq=latest_seq,
        latest_t_wall=latest_t_wall,
        changed_vars=tuple(changed_vars[:24]),
        stable_true_vars=tuple(_string_items(stable_true_vars)[:24]),
        stable_false_vars=tuple(_string_items(stable_false_vars)[:24]),
        unknown_or_missing_vars=tuple(unknown_or_missing[:32]),
        first_frame_only_values=tuple(first_frame_only_values[:16]),
        last_seen_true=tuple(last_seen_true[:24]),
        contradictions=tuple(_string_items(contradictions)),
    )


def _build_vision_evidence(context: Mapping[str, Any]) -> VisionEvidence:
    summary_raw = context.get("vision_fact_summary")
    summary = summary_raw if isinstance(summary_raw, Mapping) else {}
    source_status = _status_from_bool(summary.get("status")) if summary else "vision_unavailable"
    frame_ids = _string_items(summary.get("frame_ids"))
    if source_status == "vision_not_required":
        return VisionEvidence(
            source_status=source_status,
            confidence="low",
            freshness={"fresh_fact_ids": [], "frame_ids": list(frame_ids)},
            seen_fact_ids=(),
            not_seen_fact_ids=(),
            uncertain_fact_ids=(),
            fresh_fact_ids=(),
            late_display_anchors=(),
            visual_candidate_steps=(),
            source_frame_ids=tuple(frame_ids),
            facts=(),
        )

    seen_ids = set(_string_items(summary.get("seen_fact_ids")))
    fresh_ids = set(_string_items(summary.get("fresh_fact_ids")))
    not_seen_ids = set(_string_items(summary.get("not_seen_fact_ids")))
    uncertain_ids = set(_string_items(summary.get("uncertain_fact_ids")))

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
        satisfied_gates=tuple(satisfied),
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
    telemetry_window_digest: TelemetryWindowDigest,
    vision_evidence: VisionEvidence,
    gate_evidence: GateEvidence,
    recent_action_evidence: RecentActionEvidence,
    deterministic_candidate: DeterministicCandidateEvidence,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    if any(
        item in telemetry_window_digest.contradictions
        for item in {
            CONTRADICTION_BATTERY_FIRST_FRAME_REFUTED,
            CONTRADICTION_FIRE_TEST_LATCH_MISSING_LATER_STAGE,
        }
    ):
        conflicts.append(CONFLICT_TELEMETRY_WINDOW_VS_SINGLE_FRAME)
    if (
        deterministic_candidate.step_id in EARLY_STEP_IDS
        and len(vision_evidence.late_display_anchors) >= 2
    ):
        conflicts.append(CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS)
    if (
        deterministic_candidate.missing_conditions
        and (
            vision_evidence.late_display_anchors
            or CONTRADICTION_FIRE_TEST_LATCH_MISSING_LATER_STAGE
            in telemetry_window_digest.contradictions
        )
        and any("_complete" in item or "latch" in item for item in deterministic_candidate.missing_conditions)
    ):
        conflicts.append(CONFLICT_LATCH_MISSING_VS_LATER_STAGE_EVIDENCE)
    if _has_recent_action_gate_contradiction(gate_evidence, recent_action_evidence):
        conflicts.append(CONFLICT_RECENT_ACTION_VS_GATE_CONTRADICTION)
    return tuple(_string_items(conflicts))


def build_evidence_packet(context: Mapping[str, Any]) -> EvidencePacket:
    telemetry_evidence = _build_telemetry_evidence(context)
    telemetry_window_digest = _build_telemetry_window_digest(context)
    vision_evidence = _build_vision_evidence(context)
    gate_evidence = _build_gate_evidence(context)
    recent_action_evidence = _build_recent_action_evidence(context)
    deterministic_candidate = _build_deterministic_candidate(context)
    conflicts = _build_conflicts(
        telemetry_evidence=telemetry_evidence,
        telemetry_window_digest=telemetry_window_digest,
        vision_evidence=vision_evidence,
        gate_evidence=gate_evidence,
        recent_action_evidence=recent_action_evidence,
        deterministic_candidate=deterministic_candidate,
    )
    return EvidencePacket(
        telemetry_evidence=telemetry_evidence,
        telemetry_window_digest=telemetry_window_digest,
        vision_evidence=vision_evidence,
        gate_evidence=gate_evidence,
        recent_action_evidence=recent_action_evidence,
        deterministic_candidate=deterministic_candidate,
        conflicts=conflicts,
    )


def _visual_refs(vision_evidence: VisionEvidence) -> tuple[str, ...]:
    return tuple(f"VISION_FACTS.{fact_id}" for fact_id in vision_evidence.late_display_anchors)


def _gate_refs_for_step(gate_evidence: GateEvidence, step_id: str) -> tuple[str, ...]:
    refs: list[str] = []
    prefix = f"{step_id}."
    for gate_id in gate_evidence.blocked_gate_ids:
        if gate_id == step_id or gate_id.startswith(prefix):
            refs.append(f"GATES.{gate_id}")
    return tuple(refs)


def _targets_for_step(step_id: str, step_harness_specs: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(step_harness_specs, Mapping):
        return ()
    spec = step_harness_specs.get(step_id)
    if spec is None:
        return ()
    allowed = getattr(spec, "allowed_overlay_targets", None)
    if allowed:
        return tuple(_string_items(allowed))
    declared = getattr(spec, "declared_ui_targets", None)
    return tuple(_string_items(declared))


def _visual_candidate_step_ids(packet: EvidencePacket) -> tuple[str, ...]:
    anchors = set(packet.vision_evidence.late_display_anchors)
    if "fcsmc_final_go_result_visible" in anchors:
        return ("S20",)
    if anchors.intersection(
        {
            "fcsmc_page_visible",
            "fcsmc_in_test_visible",
            "fcsmc_intermediate_result_visible",
        }
    ):
        return ("S19", "S18", "S20")
    return tuple(packet.vision_evidence.visual_candidate_steps)


def _candidate_reason(source: str, step_id: str) -> str:
    if source == "deterministic":
        return "forward deterministic inference fallback"
    if source == "sticky_state":
        return "sticky VLM completion state supports advancing to the next procedure step"
    if source == "visual_anchor":
        return "fresh VLM visual anchors support this procedure step"
    if source == "gate_blocker":
        return "blocked gate evidence points at this procedure step"
    if source == "recent_action":
        return "recent cockpit action matches this step target set"
    if source == "telemetry_window":
        return "recent telemetry window sequence supports this procedure region"
    return f"{source} supports {step_id}"


def _telemetry_window_refs_for_var(kind: str, var_name: str) -> tuple[str, ...]:
    return (f"TELEMETRY_WINDOW.{kind}.{var_name}",)


def _changed_var_map(digest: TelemetryWindowDigest) -> dict[str, dict[str, Any]]:
    return {
        item["var"]: item
        for item in digest.changed_vars
        if isinstance(item.get("var"), str)
    }


def _has_first_frame_false(digest: TelemetryWindowDigest, var_name: str) -> bool:
    return any(
        item.get("var") == var_name and item.get("value") is False
        for item in digest.first_frame_only_values
    )


def _supports_later_avionics(digest: TelemetryWindowDigest) -> bool:
    changed = _changed_var_map(digest)
    for var_name in DOWNSTREAM_AVIONICS_VARS:
        if var_name in digest.stable_true_vars:
            return True
        item = changed.get(var_name)
        if item is not None and item.get("last_value") is True:
            return True
    return False


def _telemetry_progression_candidates(
    digest: TelemetryWindowDigest,
    *,
    deterministic_step_id: str | None = None,
) -> tuple[tuple[str, str, str, str, bool], ...]:
    changed = _changed_var_map(digest)
    out: list[tuple[str, str, str, str, bool]] = []

    probe = changed.get("ext_refuel_probe_value")
    if probe is not None:
        first = _coerce_number(probe.get("first_value"))
        last = _coerce_number(probe.get("last_value"))
        transition_count = _coerce_number(probe.get("transition_count")) or 0
        if last is not None and last >= 60000:
            out.append(("S21", "changed_vars", "ext_refuel_probe_value", "", False))
        elif last is not None and last <= 5000:
            out.append(("S22", "changed_vars", "ext_refuel_probe_value", "", False))
        elif (
            deterministic_step_id == "S20"
            and transition_count > 0
            and first is not None
            and last is not None
            and last > first
        ):
            out.append((
                "S20",
                "changed_vars",
                "ext_refuel_probe_value",
                "refueling probe extending in progress",
                True,
            ))
        elif (
            deterministic_step_id == "S21"
            and transition_count > 0
            and first is not None
            and last is not None
            and last < first
        ):
            out.append((
                "S21",
                "changed_vars",
                "ext_refuel_probe_value",
                "refueling probe retracting in progress",
                True,
            ))

    launch_bar = changed.get("launch_bar_switch_value")
    if launch_bar is not None:
        last = _coerce_number(launch_bar.get("last_value"))
        if last == 1:
            out.append(("S23", "changed_vars", "launch_bar_switch_value", "", False))
        elif last == 0:
            out.append(("S24", "changed_vars", "launch_bar_switch_value", "", False))

    hook = changed.get("hook_handle_value")
    if hook is not None:
        last = _coerce_number(hook.get("last_value"))
        if last == 0:
            out.append(("S25", "changed_vars", "hook_handle_value", "", False))
        elif last == 1:
            out.append(("S26", "changed_vars", "hook_handle_value", "", False))

    pitot = changed.get("pitot_heat_on")
    if pitot is not None and pitot.get("last_value") is True:
        out.append(("S27", "changed_vars", "pitot_heat_on", "", False))
    changed = _changed_var_map(digest)

    def _age(item: tuple[str, str, str, str, bool]) -> float:
        changed_item = changed.get(item[2])
        if changed_item is None:
            return 999999.0
        raw_age = changed_item.get("latest_transition_age_s")
        return float(raw_age) if isinstance(raw_age, (int, float)) and not isinstance(raw_age, bool) else 999999.0

    return tuple(sorted(out, key=_age))


def build_step_candidates(
    packet: EvidencePacket,
    *,
    step_harness_specs: Mapping[str, Any] | None = None,
    ordered_step_ids: list[str] | tuple[str, ...] | None = None,
    max_candidates: int = 8,
) -> tuple[StepCandidate, ...]:
    """
    Generate ordered adjudication candidates for a help cycle.

    Deterministic inference remains present, but it is explicitly marked as
    non-authoritative so downstream code can compare it with visual, gate, and
    recent-action evidence instead of inheriting a single final truth.
    """
    candidates: list[StepCandidate] = []
    seen_candidates: set[tuple[str, str]] = set()
    visual_refs = _visual_refs(packet.vision_evidence)
    has_visual_conflict = CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS in packet.conflicts

    def _append(candidate: StepCandidate) -> None:
        key = (candidate.step_id, candidate.source)
        if not candidate.step_id or key in seen_candidates:
            return
        seen_candidates.add(key)
        candidates.append(candidate)

    visual_source = (
        "sticky_state"
        if "fcsmc_final_go_result_visible" in packet.vision_evidence.late_display_anchors
        else "visual_anchor"
    )
    for step_id in _visual_candidate_step_ids(packet):
        _append(
            StepCandidate(
                step_id=step_id,
                source=visual_source,
                role="candidate",
                supporting_evidence_refs=visual_refs,
                refuting_evidence_refs=(),
                confidence=0.92 if visual_source == "sticky_state" else 0.86,
                missing_conditions=(),
                proposed_next_action_target_ids=_targets_for_step(step_id, step_harness_specs),
                reason=_candidate_reason(visual_source, step_id),
            )
        )

    deterministic_step_id = packet.deterministic_candidate.step_id
    telemetry_digest = packet.telemetry_window_digest
    if telemetry_digest.frame_count:
        if _has_first_frame_false(telemetry_digest, "battery_on") and _supports_later_avionics(telemetry_digest):
            refs = []
            if "left_ddi_on" in telemetry_digest.stable_true_vars:
                refs.extend(_telemetry_window_refs_for_var("stable_true_vars", "left_ddi_on"))
            elif "power_available" in telemetry_digest.stable_true_vars:
                refs.extend(_telemetry_window_refs_for_var("stable_true_vars", "power_available"))
            else:
                refs.extend(_telemetry_window_refs_for_var("last_seen_true", "power_available"))
            _append(
                StepCandidate(
                    step_id="S08",
                    source="telemetry_window",
                    role="candidate",
                    supporting_evidence_refs=tuple(refs),
                    refuting_evidence_refs=_telemetry_window_refs_for_var(
                        "first_frame_only_values",
                        "battery_on",
                    ),
                    confidence=0.82,
                    missing_conditions=(),
                    proposed_next_action_target_ids=_targets_for_step("S08", step_harness_specs),
                    reason=_candidate_reason("telemetry_window", "S08"),
                )
            )
        elif (
            "fire_test latch missing while later-stage telemetry evidence is present"
            in telemetry_digest.contradictions
            and _supports_later_avionics(telemetry_digest)
        ):
            _append(
                StepCandidate(
                    step_id="S08",
                    source="telemetry_window",
                    role="candidate",
                    supporting_evidence_refs=_telemetry_window_refs_for_var(
                        "stable_true_vars",
                        "left_ddi_on",
                    ),
                    refuting_evidence_refs=tuple(
                        f"TELEMETRY_WINDOW.unknown_or_missing_vars.{var_name}"
                        for var_name in telemetry_digest.unknown_or_missing_vars
                        if var_name in EARLY_LATCH_VARS
                    ),
                    confidence=0.78,
                    missing_conditions=(),
                    proposed_next_action_target_ids=_targets_for_step("S08", step_harness_specs),
                    reason=_candidate_reason("telemetry_window", "S08"),
                )
            )

        changed = _changed_var_map(telemetry_digest)
        fcs_bit_switch = changed.get("fcs_bit_switch_up")
        if fcs_bit_switch is not None and fcs_bit_switch.get("last_value") is True:
            _append(
                StepCandidate(
                    step_id="S19",
                    source="telemetry_window",
                    role="candidate",
                    supporting_evidence_refs=_telemetry_window_refs_for_var(
                        "changed_vars",
                        "fcs_bit_switch_up",
                    ),
                    refuting_evidence_refs=_gate_refs_for_step(packet.gate_evidence, "S19"),
                    confidence=0.74,
                    missing_conditions=(),
                    proposed_next_action_target_ids=_targets_for_step("S19", step_harness_specs),
                    reason=_candidate_reason("telemetry_window", "S19"),
                )
            )

        for step_id, ref_kind, var_name, reason, suppress_targets in _telemetry_progression_candidates(
            telemetry_digest,
            deterministic_step_id=deterministic_step_id,
        ):
            _append(
                StepCandidate(
                    step_id=step_id,
                    source="telemetry_window",
                    role="candidate",
                    supporting_evidence_refs=_telemetry_window_refs_for_var(ref_kind, var_name),
                    refuting_evidence_refs=(),
                    confidence=0.7,
                    missing_conditions=(),
                    proposed_next_action_target_ids=(
                        () if suppress_targets else _targets_for_step(step_id, step_harness_specs)
                    ),
                    reason=reason or _candidate_reason("telemetry_window", step_id),
                )
            )

    if isinstance(deterministic_step_id, str) and deterministic_step_id:
        _append(
            StepCandidate(
                step_id=deterministic_step_id,
                source="deterministic",
                role=packet.deterministic_candidate.role,
                supporting_evidence_refs=_gate_refs_for_step(packet.gate_evidence, deterministic_step_id),
                refuting_evidence_refs=visual_refs if has_visual_conflict else (),
                confidence=0.35 if has_visual_conflict else 0.55,
                missing_conditions=packet.deterministic_candidate.missing_conditions,
                proposed_next_action_target_ids=_targets_for_step(deterministic_step_id, step_harness_specs),
                reason=_candidate_reason("deterministic", deterministic_step_id),
            )
        )

    for gate in packet.gate_evidence.blocked_gates:
        step_id = gate.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            gate_id = gate.get("gate_id")
            if isinstance(gate_id, str) and "." in gate_id:
                step_id = gate_id.split(".", 1)[0]
        if not isinstance(step_id, str) or not step_id:
            continue
        gate_id = gate.get("gate_id")
        refs = (f"GATES.{gate_id}",) if isinstance(gate_id, str) and gate_id else ()
        _append(
            StepCandidate(
                step_id=step_id,
                source="gate_blocker",
                role="candidate",
                supporting_evidence_refs=refs,
                refuting_evidence_refs=(),
                confidence=0.66,
                missing_conditions=(),
                proposed_next_action_target_ids=_targets_for_step(step_id, step_harness_specs),
                reason=_candidate_reason("gate_blocker", step_id),
            )
        )

    recent_targets = set(packet.recent_action_evidence.target_ids)
    if recent_targets and isinstance(step_harness_specs, Mapping):
        for step_id in ordered_step_ids or tuple(step_harness_specs.keys()):
            if not isinstance(step_id, str) or (step_id, "recent_action") in seen_candidates:
                continue
            matched_targets = sorted(recent_targets.intersection(_targets_for_step(step_id, step_harness_specs)))
            if not matched_targets:
                continue
            _append(
                StepCandidate(
                    step_id=step_id,
                    source="recent_action",
                    role="candidate",
                    supporting_evidence_refs=tuple(f"RECENT_ACTIONS.{target}" for target in matched_targets),
                    refuting_evidence_refs=(),
                    confidence=0.5,
                    missing_conditions=(),
                    proposed_next_action_target_ids=tuple(matched_targets),
                    reason=_candidate_reason("recent_action", step_id),
                )
            )

    if ordered_step_ids and not candidates:
        for step_id in ordered_step_ids:
            if not isinstance(step_id, str) or not step_id:
                continue
            _append(
                StepCandidate(
                    step_id=step_id,
                    source="procedure_order",
                    role="fallback",
                    supporting_evidence_refs=(),
                    refuting_evidence_refs=(),
                    confidence=0.1,
                    missing_conditions=(),
                    proposed_next_action_target_ids=_targets_for_step(step_id, step_harness_specs),
                    reason="procedure order fallback candidate",
                )
            )
            break

    return tuple(candidates[: max(1, int(max_candidates))])


__all__ = [
    "CONFLICT_LATCH_MISSING_VS_LATER_STAGE_EVIDENCE",
    "CONFLICT_RECENT_ACTION_VS_GATE_CONTRADICTION",
    "CONFLICT_STALE_TELEMETRY_VS_FRESH_VISUAL_FACTS",
    "CONFLICT_TELEMETRY_WINDOW_VS_SINGLE_FRAME",
    "EARLY_STEP_IDS",
    "DEFAULT_LATE_DISPLAY_ANCHOR_FACTS",
    "DEFAULT_VISUAL_CANDIDATE_STEP_GROUPS",
    "EvidencePacket",
    "StepCandidate",
    "TelemetryWindowDigest",
    "build_evidence_packet",
    "build_step_candidates",
]
