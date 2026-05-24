"""
Experiment-ready data export: contracts and post-processor.

Reads runtime event logs and produces study-ready artifacts with participant
behavior traces, system response metadata, and experiment-layer annotations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from core.gating import GatingEngine
from core.help_cycle_audit import normalize_help_cycle_audit_fields
from core.interaction_metrics import InteractionMetrics, compute_interaction_metrics
from core.vars import VarResolver, VarResolverError


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
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in ("1", "true", "yes", "y"):
            return True
        if normalized in ("0", "false", "no", "n"):
            return False
    return None


# ── experiment-layer metadata ──────────────────────────────────────────

EXPECTED_PACK_STEP_IDS = [f"S{i:02d}" for i in range(1, 34)]

HELP_CYCLES_CSV_FIELDS = [
    "cycle_index", "help_cycle_id", "trigger_wall_s", "generation_mode",
    "vision_used", "vision_status", "vision_fact_status", "vlm_call_status",
    "vision_fallback_reason", "sync_delta_ms",
    "frame_ids", "layout_id", "fused_step_id", "fused_missing_conditions", "model_next_step_id",
    "overlay_targets", "overlay_executed", "overlay_rejected",
    "overlay_dropped", "overlay_dry_run_count", "response_status", "fallback_overlay_used",
    "fallback_overlay_reason", "response_mapping_failure_codes",
    "observability_status", "requires_visual_confirmation",
    "scenario_profile",
]

ACTION_TIMELINE_CSV_FIELDS = [
    "ParticipantID",
    "Condition",
    "TrialID",
    "EventIndex",
    "Timestamp",
    "TWall",
    "Source",
    "RawKey",
    "RawValueBefore",
    "RawValueAfter",
    "Delta",
    "MappedTarget",
    "CandidateStepID",
    "ActiveStepID",
    "FusedStepID",
    "ExpectedForStep",
    "BeforeHelpCycleID",
    "AfterHelpCycleID",
    "NearestHelpCycleID",
    "SecondsSinceLastHelp",
    "StepCompletedByThisEvent",
    "GateViolationCandidate",
    "AutoCodingHint",
]

STEP_CODING_CSV_FIELDS = [
    "ParticipantID",
    "Condition",
    "TrialID",
    "StepID",
    "StepTitle",
    "Phase",
    "Critical",
    "Performed",
    "Completed",
    "FirstHelpTime_sec",
    "HelpCount",
    "FirstOverlayTargets",
    "LastOverlayTargets",
    "FirstFusedStepID",
    "LastFusedStepID",
    "EvidenceRefs",
    "Auto_Error_OM",
    "Auto_Error_CO",
    "Auto_Error_OR",
    "Auto_Error_PA",
    "Auto_Error_SV",
    "AutoConfidence",
    "AutoEvidenceRefs",
    "NeedsHumanReview",
    "Error_OM",
    "Error_CO",
    "Error_OR",
    "Error_PA",
    "Error_SV",
    "CoderID",
    "CoderNotes",
    "AutoCodingNotes",
]

TRIAL_SUMMARY_CSV_FIELDS = [
    "ParticipantID",
    "Condition",
    "TrialID",
    "Completed",
    "TaskTime_sec",
    "HelpRequests",
    "LLMTriggers",
    "VLMCalls",
    "OverlayExecuted",
    "OverlayRejected",
    "FallbackCount",
    "CriticalStepsCompleted",
    "TotalStepsCompleted",
    "StepCompletionAccuracy",
]

CRITICAL_EXPORT_META_FIELDS = [
    "trial_id",
    "study_id",
    "participant_id",
    "condition",
    "group",
    "experimenter_id",
    "questionnaire_ref",
    "recording_ref",
    "git_commit",
    "git_dirty",
    "pack_path",
    "pack_hash",
    "taxonomy_path",
    "taxonomy_hash",
    "ui_map_hash",
    "bios_to_ui_hash",
    "model_provider",
    "model_name",
    "vision_model_name",
    "scenario_profile",
    "dcs_mission",
    "dcs_aircraft",
    "raw_log_ref",
    "raw_log_sha256",
]

RECOMMENDED_EXPORT_META_FIELDS: list[str] = []

CRITICAL_ONE_OF_META_FIELDS = [
    ("prompt_version or prompt_hash", ("prompt_version", "prompt_hash")),
    ("vr_setup or monitor_setup", ("vr_setup", "monitor_setup")),
]


@dataclass
class SessionMeta:
    trial_id: str = ""
    study_id: str = ""
    participant_id: str = ""
    session_id: str = ""
    condition: str = ""
    group: str = ""
    experimenter_id: str | None = None
    questionnaire_ref: str | None = None
    recording_ref: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    pack_path: str | None = None
    pack_hash: str | None = None
    taxonomy_path: str | None = None
    taxonomy_hash: str | None = None
    ui_map_hash: str | None = None
    bios_to_ui_hash: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    vision_model_name: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    scenario_profile: str | None = None
    dcs_mission: str | None = None
    dcs_aircraft: str | None = None
    vr_setup: str | None = None
    monitor_setup: str | None = None
    raw_log_ref: str | None = None
    raw_log_sha256: str | None = None
    experimenter_notes: str | None = None
    started_at: str | None = None
    ended_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SessionMeta":
        return cls(
            trial_id=_opt_str(raw.get("trial_id")) or "",
            study_id=_opt_str(raw.get("study_id")) or "",
            participant_id=_opt_str(raw.get("participant_id")) or "",
            session_id=_opt_str(raw.get("session_id")) or "",
            condition=_opt_str(raw.get("condition")) or "",
            group=_opt_str(raw.get("group")) or "",
            experimenter_id=_opt_str(raw.get("experimenter_id")),
            questionnaire_ref=_opt_str(raw.get("questionnaire_ref")),
            recording_ref=_opt_str(raw.get("recording_ref")),
            git_commit=_opt_str(raw.get("git_commit")),
            git_dirty=_opt_bool(raw.get("git_dirty")),
            pack_path=_opt_str(raw.get("pack_path")),
            pack_hash=_opt_str(raw.get("pack_hash")),
            taxonomy_path=_opt_str(raw.get("taxonomy_path")),
            taxonomy_hash=_opt_str(raw.get("taxonomy_hash")),
            ui_map_hash=_opt_str(raw.get("ui_map_hash")),
            bios_to_ui_hash=_opt_str(raw.get("bios_to_ui_hash")),
            model_provider=_opt_str(raw.get("model_provider")),
            model_name=_opt_str(raw.get("model_name")),
            vision_model_name=_opt_str(raw.get("vision_model_name")),
            prompt_version=_opt_str(raw.get("prompt_version")),
            prompt_hash=_opt_str(raw.get("prompt_hash")),
            scenario_profile=_opt_str(raw.get("scenario_profile")),
            dcs_mission=_opt_str(raw.get("dcs_mission")),
            dcs_aircraft=_opt_str(raw.get("dcs_aircraft")),
            vr_setup=_opt_str(raw.get("vr_setup")),
            monitor_setup=_opt_str(raw.get("monitor_setup")),
            raw_log_ref=_opt_str(raw.get("raw_log_ref")),
            raw_log_sha256=_opt_str(raw.get("raw_log_sha256")),
            experimenter_notes=_opt_str(raw.get("experimenter_notes")),
            started_at=_opt_str(raw.get("started_at")),
            ended_at=_opt_str(raw.get("ended_at")),
        )


@dataclass
class ExportQualityReport:
    strict: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "strict": self.strict,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
        }


def build_file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _meta_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _load_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("YAML root must be a mapping")
    return payload


def _append_gate_issue(report: ExportQualityReport, message: str, *, strict: bool) -> None:
    if strict:
        report.errors.append(message)
    else:
        report.warnings.append(message)


def build_export_quality_report(
    meta: SessionMeta,
    *,
    events: Sequence[Mapping[str, Any]],
    strict: bool = False,
    pack_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
    ui_map_path: str | Path | None = None,
    bios_to_ui_path: str | Path | None = None,
    raw_log_copied: bool = False,
    expected_help_cycle_rows: int | None = None,
    actual_help_cycle_rows: int | None = None,
    help_cycle_csv_headers: Sequence[str] | None = None,
) -> ExportQualityReport:
    report = ExportQualityReport(strict=bool(strict))
    report.checks["event_count"] = len(events)

    for field_name in CRITICAL_EXPORT_META_FIELDS:
        if not _meta_value_present(getattr(meta, field_name)):
            _append_gate_issue(
                report,
                f"missing critical metadata field: {field_name}",
                strict=strict,
            )

    for group_name, field_names in CRITICAL_ONE_OF_META_FIELDS:
        if not any(_meta_value_present(getattr(meta, field_name)) for field_name in field_names):
            _append_gate_issue(
                report,
                f"missing critical metadata field: {group_name}",
                strict=strict,
            )

    missing_recommended = [
        field_name
        for field_name in RECOMMENDED_EXPORT_META_FIELDS
        if not _meta_value_present(getattr(meta, field_name))
    ]
    report.checks["missing_recommended_metadata_fields"] = missing_recommended
    for field_name in missing_recommended:
        report.warnings.append(f"missing recommended metadata field: {field_name}")

    if not raw_log_copied:
        _append_gate_issue(
            report,
            "raw_log was not copied into the export directory",
            strict=strict,
        )
    report.checks["raw_log_copied"] = bool(raw_log_copied)

    def _validate_hash(path: str | Path | None, meta_hash_field: str) -> None:
        if path is None:
            return
        try:
            actual_hash = build_file_sha256(path)
        except Exception as exc:
            _append_gate_issue(
                report,
                f"failed to hash {Path(path).name}: {exc}",
                strict=strict,
            )
            return
        recorded_hash = getattr(meta, meta_hash_field)
        report.checks[f"actual_{meta_hash_field}"] = actual_hash
        if _meta_value_present(recorded_hash) and recorded_hash != actual_hash:
            _append_gate_issue(
                report,
                f"{meta_hash_field} does not match file content",
                strict=strict,
            )

    if pack_path is None:
        _append_gate_issue(report, "pack_path is required for S01-S33 quality gate", strict=strict)
    else:
        _validate_hash(pack_path, "pack_hash")
        try:
            pack = _load_yaml_mapping(pack_path)
            steps = pack.get("steps")
            if not isinstance(steps, list):
                raise ValueError("pack.yaml missing top-level steps list")
            step_ids = [
                item.get("id")
                for item in steps
                if isinstance(item, Mapping) and isinstance(item.get("id"), str)
            ]
            report.checks["pack_step_ids"] = step_ids
            if step_ids != EXPECTED_PACK_STEP_IDS:
                _append_gate_issue(
                    report,
                    "pack steps must be exactly S01-S33 in order",
                    strict=strict,
                )
        except Exception as exc:
            _append_gate_issue(report, f"failed to validate pack steps: {exc}", strict=strict)

    if taxonomy_path is None:
        _append_gate_issue(report, "taxonomy_path is required for scoring quality gate", strict=strict)
    else:
        _validate_hash(taxonomy_path, "taxonomy_hash")
        try:
            taxonomy = _load_yaml_mapping(taxonomy_path)
            scoring = taxonomy.get("scoring")
            taxonomy_root = taxonomy.get("taxonomy")
            categories = taxonomy_root.get("categories") if isinstance(taxonomy_root, Mapping) else None
            if not isinstance(scoring, Mapping):
                raise ValueError("taxonomy.yaml missing scoring mapping")
            if not isinstance(categories, list) or not categories:
                raise ValueError("taxonomy.yaml missing taxonomy.categories")
            weights = scoring.get("base_weights")
            if not isinstance(weights, Mapping):
                raise ValueError("taxonomy.yaml missing scoring.base_weights")
            category_codes = [
                item.get("code")
                for item in categories
                if isinstance(item, Mapping) and isinstance(item.get("code"), str)
            ]
            report.checks["taxonomy_category_codes"] = category_codes
            missing_weights = [code for code in category_codes if code not in weights]
            if missing_weights:
                _append_gate_issue(
                    report,
                    "taxonomy categories missing scoring weights: " + ", ".join(missing_weights),
                    strict=strict,
                )
        except Exception as exc:
            _append_gate_issue(report, f"failed to validate taxonomy/scoring docs: {exc}", strict=strict)

    _validate_hash(ui_map_path, "ui_map_hash")
    _validate_hash(bios_to_ui_path, "bios_to_ui_hash")

    if help_cycle_csv_headers is not None:
        header_list = list(help_cycle_csv_headers)
        report.checks["help_cycles_csv_headers"] = header_list
        if header_list != HELP_CYCLES_CSV_FIELDS:
            _append_gate_issue(
                report,
                "help_cycles.csv headers do not match expected export contract",
                strict=strict,
            )
    elif expected_help_cycle_rows is not None:
        _append_gate_issue(
            report,
            "help_cycles.csv was not generated for row/header validation",
            strict=strict,
        )

    if actual_help_cycle_rows is not None and expected_help_cycle_rows is not None:
        report.checks["help_cycle_row_count"] = actual_help_cycle_rows
        report.checks["expected_help_cycle_row_count"] = expected_help_cycle_rows
        if actual_help_cycle_rows != expected_help_cycle_rows:
            _append_gate_issue(
                report,
                (
                    "help_cycles.csv row count mismatch: "
                    f"expected {expected_help_cycle_rows}, got {actual_help_cycle_rows}"
                ),
                strict=strict,
            )
    elif expected_help_cycle_rows is not None:
        _append_gate_issue(
            report,
            "help_cycles.csv row count was not available for validation",
            strict=strict,
        )

    return report


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
    vlm_call_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepCodingRecord:
    ParticipantID: str = ""
    Condition: str = ""
    TrialID: str = ""
    StepID: str = ""
    StepTitle: str = ""
    Phase: str = ""
    Critical: str = ""
    Performed: str = ""
    Completed: str = ""
    FirstHelpTime_sec: float | None = None
    HelpCount: int = 0
    FirstOverlayTargets: str = ""
    LastOverlayTargets: str = ""
    FirstFusedStepID: str = ""
    LastFusedStepID: str = ""
    EvidenceRefs: str = ""
    Auto_Error_OM: str = "0"
    Auto_Error_CO: str = "0"
    Auto_Error_OR: str = "0"
    Auto_Error_PA: str = "0"
    Auto_Error_SV: str = "0"
    AutoConfidence: str = ""
    AutoEvidenceRefs: str = ""
    NeedsHumanReview: str = "no"
    Error_OM: str = ""
    Error_CO: str = ""
    Error_OR: str = ""
    Error_PA: str = ""
    Error_SV: str = ""
    CoderID: str = ""
    CoderNotes: str = ""
    AutoCodingNotes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionTimelineRecord:
    ParticipantID: str = ""
    Condition: str = ""
    TrialID: str = ""
    EventIndex: int = 0
    Timestamp: str = ""
    TWall: float | None = None
    Source: str = ""
    RawKey: str = ""
    RawValueBefore: str = ""
    RawValueAfter: str = ""
    Delta: str = ""
    MappedTarget: str = ""
    CandidateStepID: str = ""
    ActiveStepID: str = ""
    FusedStepID: str = ""
    ExpectedForStep: str = ""
    BeforeHelpCycleID: str = ""
    AfterHelpCycleID: str = ""
    NearestHelpCycleID: str = ""
    SecondsSinceLastHelp: float | None = None
    StepCompletedByThisEvent: str = ""
    GateViolationCandidate: str = ""
    AutoCodingHint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrialSummaryRecord:
    ParticipantID: str = ""
    Condition: str = ""
    TrialID: str = ""
    Completed: str = ""
    TaskTime_sec: float | None = None
    HelpRequests: int = 0
    LLMTriggers: int = 0
    VLMCalls: int = 0
    OverlayExecuted: int = 0
    OverlayRejected: int = 0
    FallbackCount: int = 0
    CriticalStepsCompleted: int = 0
    TotalStepsCompleted: int = 0
    StepCompletionAccuracy: float | None = None

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
    action_timeline: list[ActionTimelineRecord] = field(default_factory=list)
    step_coding: list[StepCodingRecord] = field(default_factory=list)
    trial_summary: list[TrialSummaryRecord] = field(default_factory=list)
    scoring: dict[str, Any] | None = None
    timeline: list[TimelineSnapshot] = field(default_factory=list)
    quality_gate: ExportQualityReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "summary": self.summary.to_dict(),
            "help_cycles": [c.to_dict() for c in self.help_cycles],
            "action_timeline": [a.to_dict() for a in self.action_timeline],
            "step_coding": [c.to_dict() for c in self.step_coding],
            "trial_summary": [s.to_dict() for s in self.trial_summary],
            "scoring": self.scoring,
            "timeline": [t.to_dict() for t in self.timeline],
            "quality_gate": self.quality_gate.to_dict() if self.quality_gate is not None else None,
        }


# ── extraction helpers ─────────────────────────────────────────────────


def _extract_help_cycles(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group events by help_cycle_id and build per-cycle records."""

    cycles: dict[str, dict[str, Any]] = {}
    cycle_order: list[str] = []

    for ev in events:
        kind = ev.get("kind") or ""
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
            pl = ev.get("payload")
            pl_meta = pl.get("metadata") if isinstance(pl, Mapping) and isinstance(pl.get("metadata"), Mapping) else {}
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

        # Overlay execution counts.
        # Production event logs carry execution results in two places:
        # 1. payload.actions — all actions the tutor response attempted
        # 2. payload.metadata.response_mapping — rejected/dropped targets
        #    (keys: rejected_targets, dropped_targets — both list[str])
        # There is no "executed" key in response_mapping; actions that
        # passed are counted as (len(actions) - rejected - dropped).
        response_mapping = response_meta.get("response_mapping")
        if not isinstance(response_mapping, Mapping):
            response_mapping = {}
        actions = response_payload.get("actions")
        total_actions = len(actions) if isinstance(actions, list) else 0
        overlay_rej = len(response_mapping.get("rejected_targets", [])) if isinstance(response_mapping.get("rejected_targets"), list) else 0
        overlay_drop = len(response_mapping.get("dropped_targets", [])) if isinstance(response_mapping.get("dropped_targets"), list) else 0
        overlay_exec = max(0, total_actions - overlay_rej - overlay_drop)

        # Fallback: if response_mapping is empty, use overlay_rejected_list
        if overlay_rej == 0 and overlay_exec == 0 and overlay_drop == 0:
            rejected_list = bucket.get("overlay_rejected_list", [])
            if rejected_list:
                fallback_rej = 0
                for rj_ev in rejected_list:
                    rj_payload = rj_ev.get("payload") if isinstance(rj_ev.get("payload"), Mapping) else {}
                    rj_targets = rj_payload.get("rejected_targets")
                    if isinstance(rj_targets, list):
                        for t in rj_targets:
                            if isinstance(t, str) and t:
                                fallback_rej += 1
                                if t not in overlay_targets:
                                    overlay_targets.append(t)
                    else:
                        rj_target = rj_payload.get("target")
                        if isinstance(rj_target, str) and rj_target:
                            fallback_rej += 1
                            if rj_target not in overlay_targets:
                                overlay_targets.append(rj_target)
                overlay_rej = fallback_rej

        # audit fields
        audit = normalize_help_cycle_audit_fields({**request_meta, **response_meta})

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
            "vlm_call_status": audit.get("vlm_call_status"),
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
        kind = ev.get("kind") or ""
        payload = ev.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}

        t_wall = _opt_float(ev.get("t_wall"))
        if t_wall is None:
            # Fall back to sequential ordering for events without wall-clock
            # time (e.g. procedure-engine step events produced by mock run).
            t_wall = float(len(snapshots))

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


def _load_pack_steps_for_export(pack_path: str | Path | None) -> list[Mapping[str, Any]]:
    if pack_path is None:
        return []
    try:
        pack = _load_yaml_mapping(pack_path)
    except Exception:
        return []
    steps = pack.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def _load_bios_to_ui_rules_for_export(
    bios_to_ui_path: str | Path | None,
    ui_map_path: str | Path | None,
) -> dict[str, tuple[str, ...]]:
    if bios_to_ui_path is None:
        return {}
    allowed_targets: set[str] | None = None
    if ui_map_path is not None:
        ui_map = _load_yaml_mapping(ui_map_path)
        cockpit_elements = ui_map.get("cockpit_elements")
        if not isinstance(cockpit_elements, Mapping):
            raise ValueError("ui_map.yaml missing cockpit_elements mapping")
        allowed_targets = {
            key for key in cockpit_elements.keys() if isinstance(key, str) and key
        }

    bios_to_ui = _load_yaml_mapping(bios_to_ui_path)
    mappings = bios_to_ui.get("mappings")
    if not isinstance(mappings, Mapping):
        raise ValueError("bios_to_ui.yaml missing mappings")

    rules: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_value in mappings.items():
        if not isinstance(raw_key, str) or not raw_key:
            continue
        targets: list[str] = []
        if isinstance(raw_value, str):
            targets = [raw_value]
        elif isinstance(raw_value, list):
            targets = [item for item in raw_value if isinstance(item, str) and item]
        elif isinstance(raw_value, Mapping):
            raw_targets = raw_value.get("targets")
            if isinstance(raw_targets, list):
                targets = [item for item in raw_targets if isinstance(item, str) and item]
        if not targets:
            continue
        ordered: list[str] = []
        seen: set[str] = set()
        for target in targets:
            if target in seen:
                continue
            if allowed_targets is not None and target not in allowed_targets:
                raise ValueError(
                    f"bios_to_ui key {raw_key!r} references unknown ui target {target!r}"
                )
            seen.add(target)
            ordered.append(target)
        rules[raw_key] = tuple(ordered)
    return rules


def _build_step_targets_index(pack_steps: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    by_target: dict[str, list[str]] = {}
    for step in pack_steps:
        step_id = _opt_str(step.get("id"))
        if not step_id:
            continue
        targets = step.get("ui_targets")
        if not isinstance(targets, list):
            continue
        for target in targets:
            if not isinstance(target, str) or not target:
                continue
            by_target.setdefault(target, [])
            if step_id not in by_target[target]:
                by_target[target].append(step_id)
    return by_target


def _step_title(step: Mapping[str, Any]) -> str:
    for key in ("title", "name", "short_title", "label"):
        value = _opt_str(step.get(key))
        if value:
            return value
    conditions = step.get("completion_conditions")
    if isinstance(conditions, list):
        titles = [item.strip() for item in conditions if isinstance(item, str) and item.strip()]
        if titles:
            return " ".join(titles)
    prompts = step.get("tutor_prompts")
    if isinstance(prompts, list):
        for prompt in prompts:
            value = _opt_str(prompt)
            if value:
                return value
    return _opt_str(step.get("id")) or ""


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _event_step_id(ev: Mapping[str, Any]) -> str | None:
    payload = ev.get("payload")
    if isinstance(payload, Mapping):
        step_id = _opt_str(payload.get("step_id"))
        if step_id:
            return step_id
    return _opt_str(ev.get("step_id"))


def _event_wall_time(ev: Mapping[str, Any]) -> float | None:
    payload = ev.get("payload")
    if isinstance(payload, Mapping):
        t_wall = _opt_float(payload.get("t_wall"))
        if t_wall is not None:
            return t_wall
    return _opt_float(ev.get("t_wall"))


def _join_csv_values(values: Sequence[str]) -> str:
    return ";".join(value for value in values if value)


def _stringify_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    try:
        return yaml.safe_dump(value, default_flow_style=True, sort_keys=True).strip()
    except Exception:
        return str(value)


def _record_text(value: str | None) -> str:
    return value or ""


def _help_cycle_matches_step(cycle: HelpCycleRecord, step_id: str) -> bool:
    cycle_step_id = cycle.fused_step_id or cycle.model_next_step_id
    return cycle_step_id == step_id


def _step_order_index(pack_steps: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    order: dict[str, int] = {}
    for idx, step in enumerate(pack_steps):
        step_id = _opt_str(step.get("id"))
        if step_id:
            order[step_id] = idx
    return order


def _help_cycle_step_id(cycle: HelpCycleRecord) -> str | None:
    return cycle.fused_step_id or cycle.model_next_step_id


def _metadata_step_id(metadata: Mapping[str, Any]) -> str | None:
    step_id = _opt_str(metadata.get("fused_step_id")) or _opt_str(metadata.get("model_next_step_id"))
    if step_id:
        return step_id
    help_response = metadata.get("help_response")
    if isinstance(help_response, Mapping):
        next_payload = help_response.get("next")
        if isinstance(next_payload, Mapping):
            step_id = _opt_str(next_payload.get("step_id"))
            if step_id:
                return step_id
        diagnosis = help_response.get("diagnosis")
        if isinstance(diagnosis, Mapping):
            step_id = _opt_str(diagnosis.get("step_id"))
            if step_id:
                return step_id
    final_public = metadata.get("final_public_response")
    if isinstance(final_public, Mapping):
        next_payload = final_public.get("next")
        if isinstance(next_payload, Mapping):
            step_id = _opt_str(next_payload.get("step_id"))
            if step_id:
                return step_id
    return None


def _terminal_completion_event(
    ev: Mapping[str, Any],
    *,
    terminal_step_id: str | None = None,
) -> tuple[str, str] | None:
    kind = ev.get("kind") or ev.get("type") or ""
    if kind != "tutor_response":
        return None
    metadata = _merged_event_metadata(ev)
    step_id = _metadata_step_id(metadata)
    if step_id is None or (terminal_step_id is not None and step_id != terminal_step_id):
        return None

    payload = ev.get("payload") if isinstance(ev.get("payload"), Mapping) else {}
    category = _opt_str(metadata.get("final_public_instruction_category"))
    final_public = metadata.get("final_public_response")
    if isinstance(final_public, Mapping):
        category = category or _opt_str(final_public.get("instruction_category"))
    reasons = _str_list(metadata.get("harness_validation_reasons"))

    completed = (
        category == "completed"
        or f"completion_gate_already_satisfied:{step_id}" in reasons
    )
    if not completed:
        return None

    help_cycle_id = _event_help_cycle_id(ev, metadata) or ""
    return step_id, help_cycle_id


def _inferred_step_completion_notes(
    events: Sequence[Mapping[str, Any]],
    *,
    help_cycles: Sequence[HelpCycleRecord],
    pack_steps: Sequence[Mapping[str, Any]],
    explicit_completed: set[str],
) -> dict[str, tuple[str, str]]:
    order = _step_order_index(pack_steps)
    ordered_steps = [
        _opt_str(step.get("id"))
        for step in pack_steps
        if _opt_str(step.get("id")) in order
    ]
    inferred: dict[str, tuple[str, str]] = {}

    previous_idx: int | None = None
    for cycle in help_cycles:
        step_id = _help_cycle_step_id(cycle)
        current_idx = order.get(step_id or "")
        if current_idx is None:
            continue
        if previous_idx is None:
            previous_idx = current_idx
            continue
        progression_gap = current_idx - previous_idx
        if not 0 < progression_gap <= 3:
            previous_idx = current_idx
            continue
        for prior_step in ordered_steps[previous_idx:current_idx]:
            if prior_step is None or prior_step in explicit_completed or prior_step in inferred:
                continue
            inferred[prior_step] = (
                f"progression_to:{step_id}",
                "completed_inferred_from_progression",
            )
        previous_idx = current_idx

    terminal_step_id = next((step_id for step_id in reversed(ordered_steps) if step_id), None)
    terminal = None
    for ev in events:
        terminal = _terminal_completion_event(ev, terminal_step_id=terminal_step_id) or terminal
    if terminal is None:
        return inferred

    terminal_step_id, terminal_help_cycle_id = terminal
    terminal_idx = order.get(terminal_step_id)
    if terminal_idx is None:
        return inferred
    terminal_ref = (
        f"terminal_s33:{terminal_help_cycle_id}"
        if terminal_help_cycle_id
        else "terminal_s33"
    )
    for step_id in ordered_steps[: terminal_idx + 1]:
        if step_id is None or step_id in explicit_completed or step_id in inferred:
            continue
        inferred[step_id] = (terminal_ref, "completed_inferred_from_terminal_s33")

    return inferred


PassiveCompletionNotes = Mapping[str, tuple[Sequence[str], str]]


def _build_step_coding(
    events: Sequence[Mapping[str, Any]],
    *,
    meta: SessionMeta,
    help_cycles: Sequence[HelpCycleRecord],
    pack_steps: Sequence[Mapping[str, Any]],
    passive_completion_notes: PassiveCompletionNotes | None = None,
) -> list[StepCodingRecord]:
    completed_steps: set[str] = set()
    activated_steps: set[str] = set()
    completed_event_seen: set[str] = set()
    activated_event_seen: set[str] = set()

    for ev in events:
        kind = ev.get("kind") or ev.get("type") or ""
        sid = _event_step_id(ev)
        if not sid:
            continue
        if kind == "step_activated":
            activated_steps.add(sid)
            activated_event_seen.add(sid)
        elif kind == "step_completed":
            completed_steps.add(sid)
            completed_event_seen.add(sid)

    inferred_completion_notes = _inferred_step_completion_notes(
        events,
        help_cycles=help_cycles,
        pack_steps=pack_steps,
        explicit_completed=completed_event_seen,
    )

    rows: list[StepCodingRecord] = []
    for step in pack_steps:
        step_id = _opt_str(step.get("id"))
        if not step_id:
            continue
        cycles = [cycle for cycle in help_cycles if _help_cycle_matches_step(cycle, step_id)]
        completed = step_id in completed_steps
        inferred_completion = inferred_completion_notes.get(step_id)
        passive_completion = (
            passive_completion_notes.get(step_id)
            if passive_completion_notes is not None
            else None
        )
        if not completed and inferred_completion is not None:
            completed = True
        if not completed and passive_completion is not None:
            completed = True
        performed = completed or step_id in activated_steps or bool(cycles)

        evidence_refs: list[str] = []
        if step_id in activated_event_seen:
            evidence_refs.append("step_activated")
        if step_id in completed_event_seen:
            evidence_refs.append("step_completed")
        for cycle in cycles:
            evidence_refs.append(f"help_cycle:{cycle.help_cycle_id}")
            if cycle.frame_ids:
                evidence_refs.append("frames:" + _join_csv_values(cycle.frame_ids))
        if inferred_completion is not None:
            evidence_refs.append(inferred_completion[0])
        if passive_completion is not None:
            evidence_refs.extend(passive_completion[0])

        auto_notes: list[str] = ["human_error_columns_blank"]
        if step_id in completed_event_seen:
            auto_notes.append("completed_from_step_completed")
        elif passive_completion is not None:
            auto_notes.append(passive_completion[1])
        elif inferred_completion is not None:
            auto_notes.append(inferred_completion[1])
        elif performed:
            auto_notes.append("performed_inferred_from_step_or_help_evidence")
        else:
            auto_notes.append("no_step_evidence_found")

        first_cycle = cycles[0] if cycles else None
        last_cycle = cycles[-1] if cycles else None

        rows.append(
            StepCodingRecord(
                ParticipantID=meta.participant_id,
                Condition=meta.condition,
                TrialID=meta.trial_id,
                StepID=step_id,
                StepTitle=_step_title(step),
                Phase=_opt_str(step.get("phase")) or "",
                Critical=_yes_no(_opt_bool(step.get("critical")) is True),
                Performed=_yes_no(performed),
                Completed=_yes_no(completed),
                FirstHelpTime_sec=first_cycle.trigger_wall_s if first_cycle is not None else None,
                HelpCount=len(cycles),
                FirstOverlayTargets=_join_csv_values(first_cycle.overlay_targets) if first_cycle is not None else "",
                LastOverlayTargets=_join_csv_values(last_cycle.overlay_targets) if last_cycle is not None else "",
                FirstFusedStepID=_record_text(first_cycle.fused_step_id) if first_cycle is not None else "",
                LastFusedStepID=_record_text(last_cycle.fused_step_id) if last_cycle is not None else "",
                EvidenceRefs=_join_csv_values(evidence_refs),
                AutoCodingNotes=_join_csv_values(auto_notes),
            )
        )
    return rows


def _event_source(ev: Mapping[str, Any]) -> str:
    source = _opt_str(ev.get("source"))
    if source:
        return source
    for payload in _event_payload_layers(ev):
        source = _opt_str(payload.get("source"))
        if source:
            return source
    metadata = ev.get("metadata")
    if isinstance(metadata, Mapping):
        source = _opt_str(metadata.get("source")) or _opt_str(metadata.get("observation_kind"))
        if source:
            return source
    return ""


def _event_payload_layers(ev: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = ev.get("payload")
    if not isinstance(payload, Mapping):
        return []
    layers: list[Mapping[str, Any]] = [payload]
    nested = payload.get("payload")
    if isinstance(nested, Mapping):
        layers.append(nested)
    return layers


def _extract_event_delta_items(ev: Mapping[str, Any]) -> list[tuple[str, Any]]:
    for payload in reversed(_event_payload_layers(ev)):
        delta = payload.get("delta")
        if isinstance(delta, Mapping):
            return [(key, value) for key, value in delta.items() if isinstance(key, str) and key]

        delta_summary = payload.get("delta_summary")
        if not isinstance(delta_summary, Mapping):
            continue
        recent = delta_summary.get("recent_key_changes_topk")
        if not isinstance(recent, list):
            continue

        items: list[tuple[str, Any]] = []
        for row in recent:
            if not isinstance(row, Mapping):
                continue
            key = row.get("key")
            if isinstance(key, str) and key:
                items.append((key, row.get("value")))
        if items:
            return items
    return []


def _event_bios_map(ev: Mapping[str, Any]) -> Mapping[str, Any]:
    for payload in reversed(_event_payload_layers(ev)):
        bios = payload.get("bios")
        if isinstance(bios, Mapping):
            return bios
    return {}


_PASSIVE_TIMELINE_KEYS = {
    "COMM1",
    "COMM2",
    "EXT_HOOK",
    "EXT_REFUEL_PROBE",
    "EXT_NOZZLE_POS_L",
    "EXT_NOZZLE_POS_R",
    "HMD_OFF_BRT",
    "HUD_BLACK_LVL",
    "HUD_LTDR",
    "HYD_IND_BRAKE",
    "HYD_IND_LEFT",
    "HYD_IND_RIGHT",
    "IFEI_BINGO",
    "IFEI_TIME_SET_MODE",
    "LANDING_GEAR_HANDLE_LT",
    "LOW_ALT_WARN_LT",
    "MASTER_CAUTION_LT",
    "RADALT_ALT_PTR",
    "RADALT_HEIGHT",
    "RADALT_OFF_FLAG",
    "SAI_BANK",
    "SAI_MAN_PITCH_ADJ",
    "SAI_POINTER_HOR",
    "SAI_POINTER_VER",
    "SAI_SET",
    "SAI_SLIP_BALL",
}

_PASSIVE_TIMELINE_PREFIXES = (
    "AOA_INDEXER",
    "CLIP_",
    "EMERG_INSTR_",
    "ENG_INSTR_",
    "FIRE_",
    "FLP_LG_",
    "IFEI_DD_",
    "IFEI_DISP_",
    "IFEI_TEMP_",
    "IFEI_RPM_",
    "IFEI_FF_",
    "IFEI_FUEL_",
    "IFEI_OIL_PRESS_",
    "LH_ADV_",
    "LS_",
    "SAI_ATT_",
    "UFC_OPTION_CUEING_",
    "VOLT_",
)


def _is_passive_timeline_key(raw_key: str) -> bool:
    if raw_key in _PASSIVE_TIMELINE_KEYS:
        return True
    if raw_key.endswith("_LT"):
        return True
    if raw_key.endswith("_DISPLAY") or "_DISPLAY_" in raw_key:
        return True
    return any(raw_key.startswith(prefix) for prefix in _PASSIVE_TIMELINE_PREFIXES)


def _candidate_steps_for_targets(
    targets: Sequence[str],
    step_ids_by_target: Mapping[str, Sequence[str]],
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for target in targets:
        for step_id in step_ids_by_target.get(target, ()):
            if step_id in seen:
                continue
            seen.add(step_id)
            candidates.append(step_id)
    return candidates


def _help_cycle_links(
    t_wall: float | None,
    help_cycles: Sequence[HelpCycleRecord],
) -> tuple[str, str, str, float | None, str]:
    if t_wall is None:
        return "", "", "", None, ""

    timed_cycles = [
        cycle
        for cycle in help_cycles
        if cycle.trigger_wall_s is not None and cycle.help_cycle_id
    ]
    if not timed_cycles:
        return "", "", "", None, ""

    before_candidates = [cycle for cycle in timed_cycles if cycle.trigger_wall_s is not None and cycle.trigger_wall_s > t_wall]
    after_candidates = [cycle for cycle in timed_cycles if cycle.trigger_wall_s is not None and cycle.trigger_wall_s <= t_wall]

    before = min(before_candidates, key=lambda cycle: float(cycle.trigger_wall_s)) if before_candidates else None
    after = max(after_candidates, key=lambda cycle: float(cycle.trigger_wall_s)) if after_candidates else None
    nearest = min(timed_cycles, key=lambda cycle: abs(float(cycle.trigger_wall_s) - t_wall))
    seconds_since = None if after is None else round(t_wall - float(after.trigger_wall_s), 6)
    fused_step_id = nearest.fused_step_id or nearest.model_next_step_id or ""

    return (
        before.help_cycle_id if before is not None else "",
        after.help_cycle_id if after is not None else "",
        nearest.help_cycle_id,
        seconds_since,
        fused_step_id,
    )


def _completed_by_action_event(
    events: Sequence[Mapping[str, Any]],
    *,
    event_index: int,
    event_wall: float | None,
    candidate_step_ids: Sequence[str],
) -> bool:
    if not candidate_step_ids:
        return False
    candidate_set = set(candidate_step_ids)

    current_step = _event_step_id(events[event_index])
    current_kind = events[event_index].get("kind") or events[event_index].get("type") or ""
    if current_kind == "step_completed" and current_step in candidate_set:
        return True

    for next_ev in events[event_index + 1:event_index + 4]:
        kind = next_ev.get("kind") or next_ev.get("type") or ""
        if kind not in ("step_completed", "step_blocked", "step_activated", "observation"):
            continue
        if kind == "observation":
            break
        next_step = _event_step_id(next_ev)
        if kind == "step_completed" and next_step in candidate_set:
            next_wall = _event_wall_time(next_ev)
            if event_wall is None or next_wall is None or 0 <= next_wall - event_wall <= 3.0:
                return True
            return False
        if kind in ("step_blocked", "step_activated"):
            break
    return False


def _build_action_timeline(
    events: Sequence[Mapping[str, Any]],
    *,
    meta: SessionMeta,
    help_cycles: Sequence[HelpCycleRecord],
    pack_steps: Sequence[Mapping[str, Any]],
    bios_to_ui_path: str | Path | None,
    ui_map_path: str | Path | None,
) -> list[ActionTimelineRecord]:
    bios_to_ui = _load_bios_to_ui_rules_for_export(bios_to_ui_path, ui_map_path)
    step_ids_by_target = _build_step_targets_index(pack_steps)
    rows: list[ActionTimelineRecord] = []
    active_step_id = ""
    last_values: dict[str, Any] = {}

    for event_index, ev in enumerate(events):
        kind = ev.get("kind") or ev.get("type") or ""

        if kind == "step_activated":
            active_step_id = _event_step_id(ev) or active_step_id
        elif kind in ("step_completed", "step_blocked"):
            sid = _event_step_id(ev)
            if sid and active_step_id == sid:
                active_step_id = ""

        delta_items = _extract_event_delta_items(ev)
        if not delta_items:
            for key, value in _event_bios_map(ev).items():
                if isinstance(key, str) and key:
                    last_values[key] = value
            continue

        t_wall = _event_wall_time(ev)
        before_help, after_help, nearest_help, seconds_since_help, fused_step_id = _help_cycle_links(
            t_wall,
            help_cycles,
        )
        source = _event_source(ev)
        timestamp = _opt_str(ev.get("timestamp")) or ""

        bios_map = _event_bios_map(ev)

        for raw_key, raw_after in delta_items:
            raw_before = last_values.get(raw_key)
            if raw_after is None and raw_key in bios_map:
                raw_after = bios_map.get(raw_key)

            mapped_targets = list(bios_to_ui.get(raw_key, ()))
            candidate_step_ids = _candidate_steps_for_targets(mapped_targets, step_ids_by_target)
            if _is_passive_timeline_key(raw_key):
                continue
            expected_for_step = ""
            if active_step_id and candidate_step_ids:
                expected_for_step = _yes_no(active_step_id in candidate_step_ids)
            elif active_step_id and not candidate_step_ids:
                expected_for_step = "no"

            gate_violation = bool(
                active_step_id
                and mapped_targets
                and candidate_step_ids
                and active_step_id not in candidate_step_ids
            )
            completed_by_event = _completed_by_action_event(
                events,
                event_index=event_index,
                event_wall=t_wall,
                candidate_step_ids=candidate_step_ids,
            )

            hints: list[str] = []
            if not mapped_targets:
                hints.append("unmapped_raw_key")
            if mapped_targets and not candidate_step_ids:
                hints.append("mapped_target_without_candidate_step")
            if active_step_id and candidate_step_ids and active_step_id not in candidate_step_ids:
                hints.append("unexpected_for_active_step")
            if active_step_id and candidate_step_ids and active_step_id in candidate_step_ids:
                hints.append("expected_for_active_step")
            if seconds_since_help is not None:
                hints.append("after_help")
            if completed_by_event:
                hints.append("completed_step")

            delta_text = ""
            if raw_before is not None and isinstance(raw_before, (int, float)) and isinstance(raw_after, (int, float)):
                delta_text = _stringify_csv_value(raw_after - raw_before)
            elif raw_after is not None:
                delta_text = _stringify_csv_value(raw_after)

            rows.append(
                ActionTimelineRecord(
                    ParticipantID=meta.participant_id,
                    Condition=meta.condition,
                    TrialID=meta.trial_id,
                    EventIndex=event_index,
                    Timestamp=timestamp,
                    TWall=t_wall,
                    Source=source,
                    RawKey=raw_key,
                    RawValueBefore=_stringify_csv_value(raw_before),
                    RawValueAfter=_stringify_csv_value(raw_after),
                    Delta=delta_text,
                    MappedTarget=_join_csv_values(mapped_targets),
                    CandidateStepID=_join_csv_values(candidate_step_ids),
                    ActiveStepID=active_step_id,
                    FusedStepID=fused_step_id,
                    ExpectedForStep=expected_for_step,
                    BeforeHelpCycleID=before_help,
                    AfterHelpCycleID=after_help,
                    NearestHelpCycleID=nearest_help,
                    SecondsSinceLastHelp=seconds_since_help,
                    StepCompletedByThisEvent=_yes_no(completed_by_event),
                    GateViolationCandidate=_yes_no(gate_violation),
                    AutoCodingHint=_join_csv_values(hints),
                )
            )

        for key, value in delta_items:
            last_values[key] = value
        for key, value in bios_map.items():
            if isinstance(key, str) and key:
                last_values[key] = value

    return rows


def _passive_completion_enabled(
    *,
    meta: SessionMeta,
    scoring: Mapping[str, Any] | None,
) -> bool:
    condition = meta.condition.strip().lower()
    if condition in {"without_tutor", "baseline", "no_tutor"}:
        return True
    if isinstance(scoring, Mapping):
        return _opt_bool(scoring.get("passive_step_inference")) is True
    return False


def _pack_telemetry_map_path(pack_path: str | Path | None) -> Path | None:
    if pack_path is None:
        return None
    candidate = Path(pack_path).parent / "telemetry_map.yaml"
    return candidate if candidate.exists() else None


def _load_var_resolver_for_export(pack_path: str | Path | None) -> VarResolver | None:
    telemetry_map_path = _pack_telemetry_map_path(pack_path)
    if telemetry_map_path is None:
        return None
    try:
        return VarResolver.from_yaml(telemetry_map_path)
    except (OSError, VarResolverError, yaml.YAMLError):
        return None


def _is_missing_logged_var_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"unknown", "unk", "missing", "n/a", "na"}
    return False


def _passive_vars_history_by_event(
    events: Sequence[Mapping[str, Any]],
    *,
    pack_path: str | Path | None,
) -> list[dict[str, Any]]:
    resolver = _load_var_resolver_for_export(pack_path)
    current_bios: dict[str, Any] = {}
    logged_vars: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    for ev in events:
        for key, value in _event_bios_map(ev).items():
            if isinstance(key, str) and key:
                current_bios[key] = value
        for key, value in _extract_event_delta_items(ev):
            current_bios[key] = value
        for key, value in _event_vars(ev).items():
            if isinstance(key, str) and key:
                logged_vars[key] = value

        if resolver is not None:
            try:
                current_vars = resolver.resolve({"bios": current_bios, "vars": logged_vars})
            except VarResolverError:
                current_vars = dict(logged_vars)
            logged_overrides = {
                key: value
                for key, value in logged_vars.items()
                if key != "vars_source_missing" and not _is_missing_logged_var_value(value)
            }
            current_vars.update(logged_overrides)
            source_missing = current_vars.get("vars_source_missing")
            if isinstance(source_missing, list):
                current_vars["vars_source_missing"] = sorted(
                    key
                    for key in source_missing
                    if isinstance(key, str) and key not in logged_overrides
                )
        else:
            current_vars = dict(logged_vars)

        vars_snapshot = dict(current_vars)
        bios_snapshot = dict(current_bios)
        history.append(
            {
                "vars": vars_snapshot,
                "bios": bios_snapshot,
                "payload": {"vars": vars_snapshot, "bios": bios_snapshot},
            }
        )

    return history


def _actions_by_event(
    action_timeline: Sequence[ActionTimelineRecord],
) -> dict[int, list[ActionTimelineRecord]]:
    by_event: dict[int, list[ActionTimelineRecord]] = {}
    for action in action_timeline:
        if not action.RawKey:
            continue
        by_event.setdefault(action.EventIndex, []).append(action)
    return by_event


def _gate_rules(raw_rules: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rules, list) or not raw_rules:
        return []
    return [dict(rule) for rule in raw_rules if isinstance(rule, Mapping)]


def _passive_gate_refs(step_id: str, rules: Sequence[Mapping[str, Any]]) -> list[str]:
    refs: list[str] = []
    for rule in rules:
        reason_code = _opt_str(rule.get("reason_code"))
        var_path = _opt_str(rule.get("var"))
        if reason_code:
            refs.append(f"passive_gate:{step_id}:{reason_code}")
        elif var_path:
            refs.append(f"passive_gate:{step_id}:{var_path}")
        if var_path:
            refs.append(f"telemetry_var:{var_path}")
    return refs


def _passive_completion_notes(
    events: Sequence[Mapping[str, Any]],
    *,
    action_timeline: Sequence[ActionTimelineRecord],
    pack_steps: Sequence[Mapping[str, Any]],
    pack_path: str | Path | None,
    scenario_profile: str | None,
) -> dict[str, tuple[Sequence[str], str]]:
    gate_config = _load_gate_config_for_export(pack_path, scenario_profile=scenario_profile)
    completion_gates = gate_config["completion_gates"]
    if not completion_gates:
        return {}

    step_ids = [
        step_id
        for step in pack_steps
        if (step_id := _opt_str(step.get("id"))) is not None
    ]
    step_order = {step_id: idx for idx, step_id in enumerate(step_ids)}
    rules_by_step = {
        step_id: rules
        for step_id in step_ids
        if (rules := _gate_rules(completion_gates.get(step_id)))
    }
    vars_history = _passive_vars_history_by_event(events, pack_path=pack_path)
    actions_by_event = _actions_by_event(action_timeline)
    last_allowed: dict[str, bool] = {}
    notes: dict[str, tuple[Sequence[str], str]] = {}

    for event_index, vars_snapshot in enumerate(vars_history):
        allowed_by_step = {
            step_id: GatingEngine(rules).evaluate([vars_snapshot]).allowed
            for step_id, rules in rules_by_step.items()
        }
        action_candidates_for_event: set[str] = set()
        action_refs_for_step: dict[str, list[str]] = {}
        for action in actions_by_event.get(event_index, []):
            candidates = sorted(
                _split_csv_values(action.CandidateStepID),
                key=lambda step_id: step_order.get(step_id, len(step_order)),
            )
            action_candidates_for_event.update(candidates)
            for candidate in candidates:
                if candidate in notes:
                    continue
                if allowed_by_step.get(candidate) is not True:
                    break
                action_refs_for_step.setdefault(candidate, [])
                if action.RawKey not in action_refs_for_step[candidate]:
                    action_refs_for_step[candidate].append(action.RawKey)
                break

        for step_id in step_ids:
            if step_id in notes:
                continue
            rules = rules_by_step.get(step_id)
            if rules is None:
                continue
            allowed = allowed_by_step[step_id]
            previous_allowed = last_allowed.get(step_id)
            action_refs = action_refs_for_step.get(step_id, [])
            gate_transition = (
                previous_allowed is False
                and allowed
                and step_id not in action_candidates_for_event
            )
            action_supported = bool(action_refs)

            if allowed and (gate_transition or action_supported):
                refs = _passive_gate_refs(step_id, rules)
                refs.extend(f"action:{raw_key}" for raw_key in action_refs)
                notes[step_id] = (refs, "completed_from_passive_gate")

            last_allowed[step_id] = allowed

    return notes


_AUTO_CONFIDENCE_RANK = {"": 0, "low": 1, "medium": 2, "high": 3}


def _split_csv_values(value: str) -> list[str]:
    return [item for item in value.split(";") if item]


def _append_unique_ref(existing: str, ref: str) -> str:
    refs = _split_csv_values(existing)
    if ref not in refs:
        refs.append(ref)
    return _join_csv_values(refs)


def _set_auto_candidate(
    row: StepCodingRecord,
    category: str,
    evidence_ref: str,
    *,
    confidence: str = "medium",
) -> None:
    field_name = f"Auto_Error_{category}"
    if not hasattr(row, field_name):
        return
    setattr(row, field_name, "1")
    row.NeedsHumanReview = "yes"
    row.AutoEvidenceRefs = _append_unique_ref(row.AutoEvidenceRefs, evidence_ref)
    if _AUTO_CONFIDENCE_RANK.get(confidence, 0) > _AUTO_CONFIDENCE_RANK.get(row.AutoConfidence, 0):
        row.AutoConfidence = confidence


def _completion_event_indices(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    completed: dict[str, int] = {}
    for event_index, ev in enumerate(events):
        kind = ev.get("kind") or ev.get("type") or ""
        if kind != "step_completed":
            continue
        step_id = _event_step_id(ev)
        if step_id and step_id not in completed:
            completed[step_id] = event_index
    return completed


def _load_gate_config_for_export(
    pack_path: str | Path | None,
    *,
    scenario_profile: str | None,
) -> dict[str, Mapping[str, Any]]:
    if pack_path is None:
        return {"precondition_gates": {}, "completion_gates": {}}
    try:
        pack = _load_yaml_mapping(pack_path)
    except Exception:
        return {"precondition_gates": {}, "completion_gates": {}}

    config: dict[str, dict[str, Any]] = {"precondition_gates": {}, "completion_gates": {}}
    for field_name in config:
        raw = pack.get(field_name)
        if isinstance(raw, Mapping):
            config[field_name] = {
                step_id: list(rules) if isinstance(rules, list) else rules
                for step_id, rules in raw.items()
                if isinstance(step_id, str)
            }

    profile = scenario_profile if isinstance(scenario_profile, str) and scenario_profile else "airfield"
    overrides_root = pack.get("profile_overrides")
    if isinstance(overrides_root, Mapping):
        overrides = overrides_root.get(profile)
        if isinstance(overrides, Mapping):
            for field_name in config:
                raw_overrides = overrides.get(field_name)
                if not isinstance(raw_overrides, Mapping):
                    continue
                for step_id, rules in raw_overrides.items():
                    if isinstance(step_id, str) and isinstance(rules, list):
                        config[field_name][step_id] = list(rules)

    return config


def _mark_om_candidates(rows: Sequence[StepCodingRecord]) -> None:
    for row in rows:
        if row.Completed == "yes":
            continue
        evidence = "om:not_completed"
        _set_auto_candidate(row, "OM", evidence, confidence="high")
        if row.Performed == "yes":
            row.AutoEvidenceRefs = _append_unique_ref(row.AutoEvidenceRefs, "om:partial_or_unfinished")


def _event_vars(ev: Mapping[str, Any]) -> Mapping[str, Any]:
    for payload in reversed(_event_payload_layers(ev)):
        vars_payload = payload.get("vars")
        if isinstance(vars_payload, Mapping):
            return vars_payload
    vars_top = ev.get("vars")
    return vars_top if isinstance(vars_top, Mapping) else {}


def _vars_history_by_event(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    current_vars: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for ev in events:
        for key, value in _event_vars(ev).items():
            if isinstance(key, str) and key:
                current_vars[key] = value
        snapshot = dict(current_vars)
        history.append({"vars": snapshot, "payload": {"vars": snapshot}})
    return history


def _gate_failure_has_evidence(reason: str | None) -> bool:
    if not reason:
        return False
    lowered = reason.lower()
    return "missing" not in lowered and "unknown" not in lowered


def _precondition_failure(
    *,
    step_id: str,
    event_index: int,
    precondition_gates: Mapping[str, Any],
    vars_history: Sequence[dict[str, Any]],
) -> tuple[str, str | None, str] | None:
    raw_rules = precondition_gates.get(step_id)
    if not isinstance(raw_rules, list) or not raw_rules:
        return None
    if event_index < 0 or event_index >= len(vars_history):
        return None
    rules = [dict(rule) for rule in raw_rules if isinstance(rule, Mapping)]
    result, failure_index = GatingEngine(rules).evaluate_with_failure_index(
        [vars_history[event_index]]
    )
    if result.allowed or not _gate_failure_has_evidence(result.reason):
        return None
    failed_rule = rules[failure_index] if failure_index is not None and failure_index < len(rules) else {}
    failed_var = _opt_str(failed_rule.get("var"))
    reason_code = _opt_str(failed_rule.get("reason_code")) or failed_var or "precondition_unsatisfied"
    return result.reason or "precondition_unsatisfied", failed_var, reason_code


def _dependency_step_for_failed_var(
    *,
    step_id: str,
    event_index: int,
    failed_var: str | None,
    step_order: Mapping[str, int],
    completion_indices: Mapping[str, int],
    completion_gates: Mapping[str, Any],
) -> str | None:
    if not failed_var:
        return None
    candidate_order = step_order.get(step_id)
    if candidate_order is None:
        return None
    ordered_prior = sorted(
        (
            (prior_order, prior_step)
            for prior_step, prior_order in step_order.items()
            if prior_order < candidate_order
        )
    )
    for _, prior_step in ordered_prior:
        raw_rules = completion_gates.get(prior_step)
        if not isinstance(raw_rules, list):
            continue
        if not any(isinstance(rule, Mapping) and rule.get("var") == failed_var for rule in raw_rules):
            continue
        completed_at = completion_indices.get(prior_step)
        if completed_at is None or completed_at > event_index:
            return prior_step
    return None


def _mark_action_candidates(
    *,
    rows_by_step: Mapping[str, StepCodingRecord],
    step_order: Mapping[str, int],
    action_timeline: Sequence[ActionTimelineRecord],
    completion_indices: Mapping[str, int],
    precondition_gates: Mapping[str, Any],
    completion_gates: Mapping[str, Any],
    vars_history: Sequence[dict[str, Any]],
) -> None:
    for action in action_timeline:
        candidate_steps = _split_csv_values(action.CandidateStepID)
        active_step = action.ActiveStepID or action.FusedStepID
        blocked_candidate_seen = False

        for step_id in candidate_steps:
            row = rows_by_step.get(step_id)
            if row is None:
                continue
            failure_reason = _precondition_failure(
                step_id=step_id,
                event_index=action.EventIndex,
                precondition_gates=precondition_gates,
                vars_history=vars_history,
            )
            if failure_reason is None:
                continue
            _failure_text, failed_var, reason_code = failure_reason
            blocked_candidate_seen = True
            _set_auto_candidate(
                row,
                "SV",
                f"action:{action.EventIndex}:precondition_gate:{reason_code}",
                confidence="medium",
            )
            missing_prior = _dependency_step_for_failed_var(
                step_id=step_id,
                event_index=action.EventIndex,
                failed_var=failed_var,
                step_order=step_order,
                completion_indices=completion_indices,
                completion_gates=completion_gates,
            )
            if missing_prior is not None:
                _set_auto_candidate(
                    row,
                    "OR",
                    f"action:{action.EventIndex}:out_of_order_before:{missing_prior}",
                    confidence="medium",
                )

        hints = set(_split_csv_values(action.AutoCodingHint))
        if "unmapped_raw_key" in hints and active_step:
            row = rows_by_step.get(active_step)
            if row is not None:
                _set_auto_candidate(
                    row,
                    "CO",
                    f"action:{action.EventIndex}:unmapped_raw_key",
                    confidence="low",
                )
        if (
            "unexpected_for_active_step" in hints
            or "mapped_target_without_candidate_step" in hints
        ) and active_step and not blocked_candidate_seen:
            row = rows_by_step.get(active_step)
            if row is not None:
                _set_auto_candidate(
                    row,
                    "CO",
                    f"action:{action.EventIndex}:unexpected_for_active_step",
                    confidence="medium",
                )


def _payload_tags(ev: Mapping[str, Any]) -> list[str]:
    for payload in _event_payload_layers(ev):
        tags = payload.get("tags")
        if isinstance(tags, list):
            return [tag for tag in tags if isinstance(tag, str) and tag]
    tags = ev.get("tags")
    if isinstance(tags, list):
        return [tag for tag in tags if isinstance(tag, str) and tag]
    return []


def _payload_procedure_hint(ev: Mapping[str, Any]) -> str | None:
    for payload in _event_payload_layers(ev):
        hint = _opt_str(payload.get("procedure_hint"))
        if hint:
            return hint
    return _opt_str(ev.get("procedure_hint"))


def _mark_explicit_sv_candidates(
    *,
    events: Sequence[Mapping[str, Any]],
    rows_by_step: Mapping[str, StepCodingRecord],
) -> None:
    active_step = ""
    for event_index, ev in enumerate(events):
        kind = ev.get("kind") or ev.get("type") or ""
        step_id = _event_step_id(ev)
        if kind == "step_activated" and step_id:
            active_step = step_id
        elif kind in ("step_completed", "step_blocked") and step_id == active_step:
            active_step = ""

        if kind != "observation" or "state_violation" not in _payload_tags(ev):
            continue
        target_step = _payload_procedure_hint(ev) or active_step or step_id
        row = rows_by_step.get(target_step or "")
        if row is not None:
            _set_auto_candidate(
                row,
                "SV",
                f"observation:{event_index}:state_violation",
                confidence="medium",
            )


def _vars_by_event(events: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    by_event: dict[int, dict[str, Any]] = {}
    for event_index, ev in enumerate(events):
        event_vars: dict[str, Any] = {}
        for key, value in _event_vars(ev).items():
            if isinstance(key, str) and key:
                event_vars[key] = value
                event_vars[f"vars.{key}"] = value
        if event_vars:
            by_event[event_index] = event_vars
    return by_event


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mark_pa_candidates(
    *,
    rows_by_step: Mapping[str, StepCodingRecord],
    completion_gates: Mapping[str, Any],
    action_timeline: Sequence[ActionTimelineRecord],
    completion_indices: Mapping[str, int],
    step_order: Mapping[str, int],
    vars_by_event: Mapping[int, Mapping[str, Any]],
) -> None:
    candidate_actions_by_step: dict[str, list[int]] = {}
    for action in action_timeline:
        for step_id in _split_csv_values(action.CandidateStepID):
            candidate_actions_by_step.setdefault(step_id, []).append(action.EventIndex)

    for step_id, raw_rules in completion_gates.items():
        if not isinstance(step_id, str) or not isinstance(raw_rules, list):
            continue
        row = rows_by_step.get(step_id)
        if row is None:
            continue
        for rule in raw_rules:
            if not isinstance(rule, Mapping) or rule.get("op") != "arg_in_range":
                continue
            var_path = _opt_str(rule.get("var"))
            min_value = rule.get("min")
            max_value = rule.get("max")
            if not var_path or not _is_numeric(min_value) or not _is_numeric(max_value):
                continue
            values: list[Any] = []
            relevant_indices = _pa_relevant_event_indices(
                step_id=step_id,
                var_path=var_path,
                candidate_actions_by_step=candidate_actions_by_step,
                completion_indices=completion_indices,
                step_order=step_order,
                vars_by_event=vars_by_event,
            )
            for event_index in relevant_indices:
                event_vars = vars_by_event.get(event_index, {})
                value = event_vars.get(var_path)
                if value is None and var_path.startswith("vars."):
                    value = event_vars.get(var_path[len("vars.") :])
                if _is_numeric(value):
                    values.append(value)
            if not values:
                continue
            if any(min_value <= value <= max_value for value in values):
                continue
            final_value = values[-1]
            _set_auto_candidate(
                row,
                "PA",
                f"pa:{var_path}={final_value} not_in:[{min_value},{max_value}]",
                confidence="high",
            )


def _pa_relevant_event_indices(
    *,
    step_id: str,
    var_path: str,
    candidate_actions_by_step: Mapping[str, Sequence[int]],
    completion_indices: Mapping[str, int],
    step_order: Mapping[str, int],
    vars_by_event: Mapping[int, Mapping[str, Any]],
) -> list[int]:
    action_indices = sorted(set(candidate_actions_by_step.get(step_id, ())))
    completed_at = completion_indices.get(step_id)
    previous_completed_at: int | None = None
    current_order = step_order.get(step_id)
    if current_order is not None:
        prior_completion_indices = [
            completed
            for prior_step, prior_order in step_order.items()
            if prior_order < current_order
            if (completed := completion_indices.get(prior_step)) is not None
        ]
        if prior_completion_indices:
            previous_completed_at = max(prior_completion_indices)

    start_at = previous_completed_at if previous_completed_at is not None else -1
    end_at = completed_at
    if end_at is None and action_indices:
        end_at = max(max(vars_by_event.keys(), default=action_indices[-1]), action_indices[-1])

    relevant = []
    for event_index in action_indices:
        if event_index <= start_at:
            continue
        if completed_at is not None and event_index > completed_at:
            continue
        relevant.append(event_index)
    for event_index, event_vars in vars_by_event.items():
        if event_index <= start_at:
            continue
        if end_at is not None and event_index > end_at:
            continue
        has_var = var_path in event_vars or (
            var_path.startswith("vars.") and var_path[len("vars.") :] in event_vars
        )
        if has_var:
            relevant.append(event_index)
    if completed_at is not None:
        relevant.append(completed_at)
    return sorted(set(relevant))


def _apply_pre_scoring_candidates(
    events: Sequence[Mapping[str, Any]],
    *,
    step_coding: Sequence[StepCodingRecord],
    action_timeline: Sequence[ActionTimelineRecord],
    pack_steps: Sequence[Mapping[str, Any]],
    pack_path: str | Path | None,
    scenario_profile: str | None,
) -> None:
    rows_by_step = {row.StepID: row for row in step_coding if row.StepID}
    step_order = {
        step_id: idx
        for idx, step in enumerate(pack_steps)
        if (step_id := _opt_str(step.get("id"))) is not None
    }
    completion_indices = _completion_event_indices(events)
    gate_config = _load_gate_config_for_export(pack_path, scenario_profile=scenario_profile)
    vars_history = _vars_history_by_event(events)
    vars_by_event = _vars_by_event(events)

    _mark_om_candidates(step_coding)
    _mark_action_candidates(
        rows_by_step=rows_by_step,
        step_order=step_order,
        action_timeline=action_timeline,
        completion_indices=completion_indices,
        precondition_gates=gate_config["precondition_gates"],
        completion_gates=gate_config["completion_gates"],
        vars_history=vars_history,
    )
    _mark_explicit_sv_candidates(events=events, rows_by_step=rows_by_step)
    _mark_pa_candidates(
        rows_by_step=rows_by_step,
        completion_gates=gate_config["completion_gates"],
        action_timeline=action_timeline,
        completion_indices=completion_indices,
        step_order=step_order,
        vars_by_event=vars_by_event,
    )


def _is_task_time_event(ev: Mapping[str, Any]) -> bool:
    kind = ev.get("kind") or ev.get("type") or ""
    if kind in (
        "step_activated",
        "step_completed",
        "step_blocked",
        "tutor_request",
        "tutor_response",
        "overlay_requested",
        "overlay_dry_run",
        "overlay_rejected",
    ):
        return True
    if kind != "observation":
        return False

    source = _event_source(ev)
    if source == "vision_frame_manifest":
        return False
    return bool(_event_bios_map(ev) or _event_vars(ev) or _extract_event_delta_items(ev))


def _terminal_task_end_time(
    events: Sequence[Mapping[str, Any]],
    *,
    terminal_step_id: str | None,
) -> float | None:
    end_time = None
    for ev in events:
        if _terminal_completion_event(ev, terminal_step_id=terminal_step_id) is None:
            continue
        end_time = _event_wall_time(ev) or end_time
    return end_time


def _task_time_seconds(
    events: Sequence[Mapping[str, Any]],
    *,
    terminal_step_id: str | None = None,
) -> float | None:
    times = [
        t
        for ev in events
        if _is_task_time_event(ev)
        if (t := _event_wall_time(ev)) is not None
    ]
    if not times:
        return None
    start_time = min(times)
    end_time = _terminal_task_end_time(events, terminal_step_id=terminal_step_id)
    if end_time is None:
        end_time = max(times)
    if end_time < start_time:
        return None
    return end_time - start_time


def _merged_event_metadata(ev: Mapping[str, Any]) -> dict[str, Any]:
    metadata = ev.get("metadata") if isinstance(ev.get("metadata"), Mapping) else {}
    payload = ev.get("payload")
    payload_metadata = (
        payload.get("metadata")
        if isinstance(payload, Mapping) and isinstance(payload.get("metadata"), Mapping)
        else {}
    )
    return {**metadata, **payload_metadata}


def _event_help_cycle_id(ev: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
    help_cycle_id = _opt_str(metadata.get("help_cycle_id"))
    if help_cycle_id:
        return help_cycle_id
    payload = ev.get("payload")
    if isinstance(payload, Mapping):
        help_cycle_id = _opt_str(payload.get("help_cycle_id"))
        if help_cycle_id:
            return help_cycle_id
    return _opt_str(ev.get("related_id"))


def _count_vlm_calls(
    events: Sequence[Mapping[str, Any]],
    help_cycles: Sequence[HelpCycleRecord],
) -> int:
    saw_vlm_status = False
    called_cycle_ids: set[str] = set()
    called_without_cycle_id = 0
    for ev in events:
        metadata = _merged_event_metadata(ev)
        status = metadata.get("vlm_call_status")
        if not isinstance(status, str):
            continue
        saw_vlm_status = True
        if status != "called":
            continue
        help_cycle_id = _event_help_cycle_id(ev, metadata)
        if help_cycle_id:
            called_cycle_ids.add(help_cycle_id)
        else:
            called_without_cycle_id += 1

    if saw_vlm_status:
        return len(called_cycle_ids) + called_without_cycle_id

    vision_fact_observations = sum(
        1
        for ev in events
        if _merged_event_metadata(ev).get("observation_kind") == "vision_fact"
    )
    if vision_fact_observations:
        return vision_fact_observations

    return sum(1 for cycle in help_cycles if cycle.vision_used is True)


def _build_trial_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    meta: SessionMeta,
    summary: InteractionMetrics,
    help_cycles: Sequence[HelpCycleRecord],
    step_coding: Sequence[StepCodingRecord],
) -> list[TrialSummaryRecord]:
    if not step_coding:
        return []

    total_steps_completed = sum(1 for row in step_coding if row.Completed == "yes")
    critical_steps_completed = sum(
        1 for row in step_coding if row.Critical == "yes" and row.Completed == "yes"
    )
    all_completed = total_steps_completed == len(step_coding)
    fallback_count = sum(
        1
        for cycle in help_cycles
        if cycle.fallback_overlay_used is True or cycle.generation_mode == "fallback"
    )

    return [
        TrialSummaryRecord(
            ParticipantID=meta.participant_id,
            Condition=meta.condition,
            TrialID=meta.trial_id,
            Completed=_yes_no(all_completed),
            TaskTime_sec=_task_time_seconds(events, terminal_step_id=step_coding[-1].StepID),
            HelpRequests=summary.help_requests,
            LLMTriggers=summary.llm_triggers,
            VLMCalls=_count_vlm_calls(events, help_cycles),
            OverlayExecuted=sum(cycle.overlay_executed for cycle in help_cycles),
            OverlayRejected=sum(cycle.overlay_rejected for cycle in help_cycles),
            FallbackCount=fallback_count,
            CriticalStepsCompleted=critical_steps_completed,
            TotalStepsCompleted=total_steps_completed,
            StepCompletionAccuracy=round(total_steps_completed / len(step_coding), 6),
        )
    ]


# ── main entry point ───────────────────────────────────────────────────


def build_experiment_export(
    events: Sequence[Mapping[str, Any]],
    *,
    meta_overrides: Mapping[str, Any] | None = None,
    scoring: Mapping[str, Any] | None = None,
    pack_path: str | Path | None = None,
    bios_to_ui_path: str | Path | None = None,
    ui_map_path: str | Path | None = None,
) -> ExperimentExport:
    """Build a complete experiment export from a list of event dicts."""

    extracted_meta = _extract_session_meta(events)
    if isinstance(meta_overrides, Mapping):
        extracted_meta.update(
            {k: v for k, v in meta_overrides.items() if v is not None}
        )

    meta = SessionMeta.from_dict(extracted_meta)

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
            vlm_call_status=rc["vlm_call_status"],
        )
        for rc in raw_cycles
    ]

    summary = compute_interaction_metrics(events)
    pack_steps = _load_pack_steps_for_export(pack_path)
    action_timeline = _build_action_timeline(
        events,
        meta=meta,
        help_cycles=help_cycles,
        pack_steps=pack_steps,
        bios_to_ui_path=bios_to_ui_path,
        ui_map_path=ui_map_path,
    )
    passive_completion_notes = (
        _passive_completion_notes(
            events,
            action_timeline=action_timeline,
            pack_steps=pack_steps,
            pack_path=pack_path,
            scenario_profile=meta.scenario_profile,
        )
        if _passive_completion_enabled(meta=meta, scoring=scoring)
        else None
    )
    step_coding = _build_step_coding(
        events,
        meta=meta,
        help_cycles=help_cycles,
        pack_steps=pack_steps,
        passive_completion_notes=passive_completion_notes,
    )
    _apply_pre_scoring_candidates(
        events,
        step_coding=step_coding,
        action_timeline=action_timeline,
        pack_steps=pack_steps,
        pack_path=pack_path,
        scenario_profile=meta.scenario_profile,
    )
    trial_summary = _build_trial_summary(
        events,
        meta=meta,
        summary=summary,
        help_cycles=help_cycles,
        step_coding=step_coding,
    )

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
        action_timeline=action_timeline,
        step_coding=step_coding,
        trial_summary=trial_summary,
        scoring=dict(scoring) if isinstance(scoring, Mapping) else None,
        timeline=timeline,
    )


__all__ = [
    "SessionMeta",
    "ExportQualityReport",
    "HelpCycleRecord",
    "ActionTimelineRecord",
    "StepCodingRecord",
    "TrialSummaryRecord",
    "TimelineSnapshot",
    "ExperimentExport",
    "HELP_CYCLES_CSV_FIELDS",
    "ACTION_TIMELINE_CSV_FIELDS",
    "STEP_CODING_CSV_FIELDS",
    "TRIAL_SUMMARY_CSV_FIELDS",
    "build_export_quality_report",
    "build_experiment_export",
    "build_file_sha256",
]
