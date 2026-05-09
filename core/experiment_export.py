"""
Experiment-ready data export: contracts and post-processor.

Reads runtime event logs and produces study-ready artifacts with participant
behavior traces, system response metadata, and experiment-layer annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from core.help_cycle_audit import normalize_help_cycle_audit_fields
from core.interaction_metrics import InteractionMetrics, compute_interaction_metrics


def _str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item]


def _opt_str(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _opt_float(raw: Any) -> float | None:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return None


def _opt_int(raw: Any) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


def _opt_bool(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    return None


# ── experiment-layer metadata ──────────────────────────────────────────


@dataclass
class SessionMeta:
    participant_id: str = ""
    session_id: str = ""
    condition: str = ""
    group: str = ""
    questionnaire_ref: str | None = None
    experimenter_notes: str | None = None
    started_at: str | None = None
    ended_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SessionMeta":
        return cls(
            participant_id=_opt_str(raw.get("participant_id")) or "",
            session_id=_opt_str(raw.get("session_id")) or "",
            condition=_opt_str(raw.get("condition")) or "",
            group=_opt_str(raw.get("group")) or "",
            questionnaire_ref=_opt_str(raw.get("questionnaire_ref")),
            experimenter_notes=_opt_str(raw.get("experimenter_notes")),
            started_at=_opt_str(raw.get("started_at")),
            ended_at=_opt_str(raw.get("ended_at")),
        )


# ── per-cycle behavioural record ───────────────────────────────────────


@dataclass
class HelpCycleRecord:
    cycle_index: int = 0
    help_cycle_id: str = ""
    trigger_wall_s: float | None = None
    generation_mode: str | None = None
    vision_used: bool | None = None
    vision_fallback_reason: str | None = None
    vision_status: str | None = None
    sync_delta_ms: int | None = None
    frame_ids: list[str] = field(default_factory=list)
    layout_id: str | None = None
    fused_step_id: str | None = None
    fused_missing_conditions: list[str] = field(default_factory=list)
    model_next_step_id: str | None = None
    overlay_targets: list[str] = field(default_factory=list)
    overlay_executed: int = 0
    overlay_rejected: int = 0
    overlay_dropped: int = 0
    overlay_dry_run_count: int = 0
    response_mapping_failure_codes: list[str] = field(default_factory=list)
    response_status: str | None = None
    fallback_overlay_used: bool | None = None
    fallback_overlay_reason: str | None = None
    observability_status: str | None = None
    requires_visual_confirmation: bool | None = None
    scenario_profile: str | None = None
    vision_fact_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── session timeline snapshot ──────────────────────────────────────────


@dataclass
class TimelineSnapshot:
    wall_s: float = 0.0
    active_step_id: str | None = None
    completed_step_ids: list[str] = field(default_factory=list)
    blocked_step_ids: list[str] = field(default_factory=list)
    help_request_count: int = 0
    vision_frames_seen: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── top-level export ───────────────────────────────────────────────────


@dataclass
class ExperimentExport:
    meta: SessionMeta = field(default_factory=SessionMeta)
    summary: InteractionMetrics = field(default_factory=InteractionMetrics)
    help_cycles: list[HelpCycleRecord] = field(default_factory=list)
    scoring: dict[str, Any] | None = None
    timeline: list[TimelineSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "summary": self.summary.to_dict(),
            "help_cycles": [c.to_dict() for c in self.help_cycles],
            "scoring": self.scoring,
            "timeline": [t.to_dict() for t in self.timeline],
        }


# ── extraction helpers ─────────────────────────────────────────────────


def _extract_help_cycles(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group events by help_cycle_id and build per-cycle records."""

    cycles: dict[str, dict[str, Any]] = {}
    cycle_order: list[str] = []

    for ev in events:
        kind = ev.get("kind") or ev.get("type") or ""
        ev_meta = ev.get("metadata") if isinstance(ev.get("metadata"), Mapping) else {}
        payload = ev.get("payload") if isinstance(ev.get("payload"), Mapping) else {}

        cid = ev_meta.get("help_cycle_id") or payload.get("help_cycle_id")
        if not isinstance(cid, str) or not cid:
            continue

        if cid not in cycles:
            cycles[cid] = {}
            cycle_order.append(cid)

        bucket = cycles[cid]

        if kind == "tutor_request":
            bucket["request"] = ev
        elif kind == "tutor_response":
            bucket["response"] = ev
        elif kind == "overlay_dry_run":
            bucket.setdefault("overlay_dry_runs", []).append(ev)
        elif kind == "overlay_rejected":
            bucket.setdefault("overlay_rejected_list", []).append(ev)

    records: list[dict[str, Any]] = []
    for idx, cid in enumerate(cycle_order):
        bucket = cycles[cid]

        request_ev = bucket.get("request", {})
        response_ev = bucket.get("response", {})

        # Merge metadata from both event-level and payload-level.
        # In real logs, rich fields (help_response, response_mapping,
        # fallback_overlay_used, observability_status, etc.) reside in
        # payload["metadata"] — see live_dcs._sanitize_*_payload_for_event.
        def _merged_meta(ev: Mapping[str, Any]) -> dict[str, Any]:
            ev_meta = ev.get("metadata") if isinstance(ev.get("metadata"), Mapping) else {}
            pl_meta = ev.get("payload", {}).get("metadata") if isinstance(ev.get("payload", {}).get("metadata"), Mapping) else {}
            return {**ev_meta, **pl_meta}

        request_meta = _merged_meta(request_ev)
        response_meta = _merged_meta(response_ev)
        response_payload = response_ev.get("payload") if isinstance(response_ev.get("payload"), Mapping) else {}

        # overlay targets: prefer dry_run events; fall back to response
        # payload actions and help_response overlay targets.
        dry_runs = bucket.get("overlay_dry_runs", [])
        overlay_targets: list[str] = []
        for dr in dry_runs:
            dr_payload = dr.get("payload") if isinstance(dr.get("payload"), Mapping) else {}
            target = dr_payload.get("target") or dr_payload.get("element_id")
            if isinstance(target, str) and target and target not in overlay_targets:
                overlay_targets.append(target)
        if not overlay_targets:
            # Fallback 1: response.actions
            actions = response_payload.get("actions")
            if isinstance(actions, list):
                for act in actions:
                    if isinstance(act, Mapping):
                        target = act.get("target") or act.get("element_id")
                        if isinstance(target, str) and target and target not in overlay_targets:
                            overlay_targets.append(target)
            # Fallback 2: help_response.overlay.targets
            if not overlay_targets:
                help_resp = response_meta.get("help_response")
                if isinstance(help_resp, Mapping):
                    overlay = help_resp.get("overlay")
                    if isinstance(overlay, Mapping):
                        targets = overlay.get("targets")
                        if isinstance(targets, list):
                            for t in targets:
                                if isinstance(t, str) and t and t not in overlay_targets:
                                    overlay_targets.append(t)

        # overlay report from response_mapping (payload metadata)
        response_mapping = response_meta.get("response_mapping")
        if isinstance(response_mapping, Mapping):
            report = response_mapping
        else:
            report = {}
        overlay_exec = len(report.get("executed", [])) if isinstance(report.get("executed"), list) else 0
        overlay_rej = len(report.get("rejected", [])) if isinstance(report.get("rejected"), list) else 0
        overlay_drop = len(report.get("dropped", [])) if isinstance(report.get("dropped"), list) else 0

        # Fallback: if response_mapping report is empty, use overlay_rejected_list count
        if overlay_rej == 0 and overlay_exec == 0 and overlay_drop == 0:
            rejected_list = bucket.get("overlay_rejected_list", [])
            if rejected_list:
                overlay_rej = len(rejected_list)
                for rj_ev in rejected_list:
                    rj_payload = rj_ev.get("payload") if isinstance(rj_ev.get("payload"), Mapping) else {}
                    rj_target = rj_payload.get("target")
                    if isinstance(rj_target, str) and rj_target and rj_target not in overlay_targets:
                        overlay_targets.append(rj_target)

        # audit fields
        audit = normalize_help_cycle_audit_fields({**response_meta, **request_meta})

        # model next step from help_response
        model_next = None
        help_resp = response_meta.get("help_response")
        if isinstance(help_resp, Mapping):
            next_payload = help_resp.get("next")
            if isinstance(next_payload, Mapping):
                model_next = _opt_str(next_payload.get("step_id"))

        record = {
            "cycle_index": idx,
            "help_cycle_id": cid,
            "trigger_wall_s": _opt_float(request_ev.get("t_wall")),
            "generation_mode": audit.get("generation_mode") or response_meta.get("generation_mode"),
            "vision_used": _opt_bool(audit.get("vision_used")),
            "vision_fallback_reason": audit.get("vision_fallback_reason"),
            "vision_status": request_meta.get("vision_status"),
            "sync_delta_ms": _opt_int(audit.get("sync_delta_ms")),
            "frame_ids": _str_list(request_ev.get("vision_refs")),
            "layout_id": audit.get("layout_id"),
            "fused_step_id": audit.get("fused_step_id"),
            "fused_missing_conditions": _str_list(audit.get("fused_missing_conditions")),
            "model_next_step_id": model_next,
            "overlay_targets": overlay_targets,
            "overlay_executed": overlay_exec,
            "overlay_rejected": overlay_rej,
            "overlay_dropped": overlay_drop,
            "overlay_dry_run_count": len(dry_runs),
            "response_mapping_failure_codes": _str_list(response_meta.get("response_mapping_failure_codes")),
            "response_status": response_payload.get("status"),
            "fallback_overlay_used": _opt_bool(response_meta.get("fallback_overlay_used")),
            "fallback_overlay_reason": response_meta.get("fallback_overlay_reason"),
            "observability_status": response_meta.get("observability_status"),
            "requires_visual_confirmation": _opt_bool(response_meta.get("requires_visual_confirmation")),
            "scenario_profile": response_meta.get("scenario_profile"),
            "vision_fact_status": request_meta.get("vision_fact_status"),
        }
        records.append(record)

    return records


def _extract_session_meta(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first_ts: str | None = None
    last_ts: str | None = None
    session_id: str | None = None

    for ev in events:
        ts = ev.get("timestamp")
        if isinstance(ts, str) and ts:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        sid = ev.get("session_id")
        if isinstance(sid, str) and sid:
            session_id = sid

    return {
        "session_id": session_id or "",
        "started_at": first_ts,
        "ended_at": last_ts,
    }


def _build_timeline(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    active_step: str | None = None
    completed: set[str] = set()
    blocked: set[str] = set()
    help_count = 0
    vision_frames_seen = 0

    for ev in events:
        kind = ev.get("kind") or ev.get("type") or ""
        payload = ev.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}

        t_wall = _opt_float(ev.get("t_wall"))
        if t_wall is None:
            continue

        state_changed = False

        if kind == "step_activated":
            sid = payload.get("step_id")
            if isinstance(sid, str) and sid:
                active_step = sid
                state_changed = True
        elif kind == "step_completed":
            sid = payload.get("step_id")
            if isinstance(sid, str) and sid:
                completed.add(sid)
                if active_step == sid:
                    active_step = None
                state_changed = True
        elif kind == "step_blocked":
            sid = payload.get("step_id")
            if isinstance(sid, str) and sid:
                blocked.add(sid)
                if active_step == sid:
                    active_step = None
                state_changed = True
        elif kind == "tutor_request":
            intent = (payload.get("intent") or "").lower()
            if "help" in intent or "hint" in intent:
                help_count += 1
                state_changed = True
        elif kind == "observation":
            metadata = ev.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("observation_kind") in ("vision", "vision_fact"):
                vision_frames_seen += 1
                state_changed = True

        if state_changed:
            snapshots.append({
                "wall_s": t_wall,
                "active_step_id": active_step,
                "completed_step_ids": sorted(completed),
                "blocked_step_ids": sorted(blocked),
                "help_request_count": help_count,
                "vision_frames_seen": vision_frames_seen,
            })

    return snapshots


# ── main entry point ───────────────────────────────────────────────────


def build_experiment_export(
    events: Sequence[Mapping[str, Any]],
    *,
    meta_overrides: Mapping[str, Any] | None = None,
    scoring: Mapping[str, Any] | None = None,
) -> ExperimentExport:
    """Build a complete experiment export from a list of event dicts."""

    extracted_meta = _extract_session_meta(events)
    if isinstance(meta_overrides, Mapping):
        extracted_meta.update(
            {k: v for k, v in meta_overrides.items() if v is not None}
        )

    meta = SessionMeta(
        participant_id=_opt_str(extracted_meta.get("participant_id")) or "",
        session_id=_opt_str(extracted_meta.get("session_id")) or "",
        condition=_opt_str(extracted_meta.get("condition")) or "",
        group=_opt_str(extracted_meta.get("group")) or "",
        questionnaire_ref=_opt_str(extracted_meta.get("questionnaire_ref")),
        experimenter_notes=_opt_str(extracted_meta.get("experimenter_notes")),
        started_at=_opt_str(extracted_meta.get("started_at")),
        ended_at=_opt_str(extracted_meta.get("ended_at")),
    )

    raw_cycles = _extract_help_cycles(events)
    help_cycles = [
        HelpCycleRecord(
            cycle_index=rc["cycle_index"],
            help_cycle_id=rc["help_cycle_id"],
            trigger_wall_s=rc["trigger_wall_s"],
            generation_mode=rc["generation_mode"],
            vision_used=rc["vision_used"],
            vision_fallback_reason=rc["vision_fallback_reason"],
            vision_status=rc["vision_status"],
            sync_delta_ms=rc["sync_delta_ms"],
            frame_ids=rc["frame_ids"],
            layout_id=rc["layout_id"],
            fused_step_id=rc["fused_step_id"],
            fused_missing_conditions=rc["fused_missing_conditions"],
            model_next_step_id=rc["model_next_step_id"],
            overlay_targets=rc["overlay_targets"],
            overlay_executed=rc["overlay_executed"],
            overlay_rejected=rc["overlay_rejected"],
            overlay_dropped=rc["overlay_dropped"],
            overlay_dry_run_count=rc["overlay_dry_run_count"],
            response_mapping_failure_codes=rc["response_mapping_failure_codes"],
            response_status=rc["response_status"],
            fallback_overlay_used=rc["fallback_overlay_used"],
            fallback_overlay_reason=rc["fallback_overlay_reason"],
            observability_status=rc["observability_status"],
            requires_visual_confirmation=rc["requires_visual_confirmation"],
            scenario_profile=rc["scenario_profile"],
            vision_fact_status=rc["vision_fact_status"],
        )
        for rc in raw_cycles
    ]

    summary = compute_interaction_metrics(events)

    raw_timeline = _build_timeline(events)
    timeline = [
        TimelineSnapshot(
            wall_s=ts["wall_s"],
            active_step_id=ts["active_step_id"],
            completed_step_ids=ts["completed_step_ids"],
            blocked_step_ids=ts["blocked_step_ids"],
            help_request_count=ts["help_request_count"],
            vision_frames_seen=ts["vision_frames_seen"],
        )
        for ts in raw_timeline
    ]

    return ExperimentExport(
        meta=meta,
        summary=summary,
        help_cycles=help_cycles,
        scoring=dict(scoring) if isinstance(scoring, Mapping) else None,
        timeline=timeline,
    )


__all__ = [
    "SessionMeta",
    "HelpCycleRecord",
    "TimelineSnapshot",
    "ExperimentExport",
    "build_experiment_export",
]
