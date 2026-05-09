"""
Experiment-ready data export: contracts and post-processor.

Reads runtime event logs and produces study-ready artifacts with participant
behavior traces, system response metadata, and experiment-layer annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from core.help_cycle_audit import HELP_CYCLE_AUDIT_FIELDS, normalize_help_cycle_audit_fields
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
    sync_status: str | None = None
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

    # index: help_cycle_id → events
    cycles: dict[str, dict[str, Any]] = {}
    cycle_meta_index: dict[str, dict[str, Any]] = {}
    cycle_order: list[str] = []

    for ev in events:
        kind = ev.get("kind") or ev.get("type") or ""
        metadata = ev.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        payload = ev.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}

        cid = metadata.get("help_cycle_id")
        if not isinstance(cid, str) or not cid:
            # try payload-level
            cid = payload.get("help_cycle_id")
        if not isinstance(cid, str) or not cid:
            continue

        if cid not in cycles:
            cycles[cid] = {}
            cycle_order.append(cid)
            cycle_meta_index[cid] = metadata

        bucket = cycles[cid]

        if kind == "tutor_request":
            bucket["request"] = ev
            # merge request audit metadata
            request_meta = ev.get("metadata") if isinstance(ev.get("metadata"), Mapping) else {}
            cycle_meta_index[cid] = {**cycle_meta_index.get(cid, {}), **dict(request_meta)}
        elif kind == "tutor_response":
            bucket["response"] = ev
            response_meta = ev.get("metadata") if isinstance(ev.get("metadata"), Mapping) else {}
            cycle_meta_index[cid] = {**cycle_meta_index.get(cid, {}), **dict(response_meta)}
            # per-action overlay events are keyed by target
        elif kind == "overlay_dry_run":
            bucket.setdefault("overlay_dry_runs", []).append(ev)
        elif kind == "overlay_rejected":
            bucket.setdefault("overlay_rejected_list", []).append(ev)

    records: list[dict[str, Any]] = []
    for idx, cid in enumerate(cycle_order):
        bucket = cycles[cid]
        meta = cycle_meta_index.get(cid, {})

        request_ev = bucket.get("request", {})
        response_ev = bucket.get("response", {})
        request_payload = request_ev.get("payload") if isinstance(request_ev.get("payload"), Mapping) else {}
        response_payload = response_ev.get("payload") if isinstance(response_ev.get("payload"), Mapping) else {}
        response_meta = response_ev.get("metadata") if isinstance(response_ev.get("metadata"), Mapping) else {}
        request_meta = request_ev.get("metadata") if isinstance(request_ev.get("metadata"), Mapping) else {}

        # overlay targets from dry_run events
        dry_runs = bucket.get("overlay_dry_runs", [])
        overlay_targets: list[str] = []
        for dr in dry_runs:
            dr_payload = dr.get("payload") if isinstance(dr.get("payload"), Mapping) else {}
            target = dr_payload.get("target") or dr_payload.get("element_id")
            if isinstance(target, str) and target:
                overlay_targets.append(target)

        # overlay report from response metadata
        response_mapping = response_meta.get("response_mapping")
        if isinstance(response_mapping, Mapping):
            report = response_mapping
        else:
            report = response_meta.get("overlay_report")
            if not isinstance(report, Mapping):
                report = {}
        overlay_exec = len(report.get("executed", [])) if isinstance(report.get("executed"), list) else 0
        overlay_rej = len(report.get("rejected", [])) if isinstance(report.get("rejected"), list) else 0
        overlay_drop = len(report.get("dropped", [])) if isinstance(report.get("dropped"), list) else 0

        # audit fields from request metadata (primary) or response metadata (fallback)
        audit = normalize_help_cycle_audit_fields({**response_meta, **request_meta})

        # response action payload for executed/dropped from response
        actions = response_payload.get("actions")
        if isinstance(actions, list):
            # actions → response_mapping report
            pass

        # model next step from response help_response
        model_next = None
        help_resp = response_meta.get("help_response")
        if isinstance(help_resp, Mapping):
            next_payload = help_resp.get("next")
            if isinstance(next_payload, Mapping):
                model_next = _opt_str(next_payload.get("step_id"))

        # mapping failure codes
        mapping_failure_codes = _str_list(response_meta.get("response_mapping_failure_codes"))

        record = {
            "cycle_index": idx,
            "help_cycle_id": cid,
            "trigger_wall_s": _opt_float(request_ev.get("t_wall")),
            "generation_mode": audit.get("generation_mode") or response_meta.get("generation_mode"),
            "vision_used": _opt_bool(audit.get("vision_used")),
            "vision_fallback_reason": audit.get("vision_fallback_reason"),
            "sync_status": audit.get("sync_status"),
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
            "response_mapping_failure_codes": mapping_failure_codes,
            "response_status": response_payload.get("status"),
            "fallback_overlay_used": _opt_bool(response_meta.get("fallback_overlay_used")),
            "fallback_overlay_reason": response_meta.get("fallback_overlay_reason"),
            "observability_status": response_meta.get("observability_status"),
            "requires_visual_confirmation": _opt_bool(response_meta.get("requires_visual_confirmation")),
            "scenario_profile": response_meta.get("scenario_profile"),
            "vision_fact_status": audit.get("vision_fact_status"),
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


def _extract_overlay_execution_results(
    events: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    """Return (executed, rejected, dropped) totals from overlay events."""
    executed = 0
    rejected = 0
    dropped = 0
    for ev in events:
        kind = ev.get("kind") or ev.get("type") or ""
        payload = ev.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if kind == "overlay_dry_run":
            pass  # these are previews, not execution results
        elif kind == "overlay_rejected":
            rejected += 1
    return executed, rejected, dropped


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
            sync_status=rc["sync_status"],
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
