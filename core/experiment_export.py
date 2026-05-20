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
    "vision_used", "vision_status", "vision_fact_status", "vision_fallback_reason", "sync_delta_ms",
    "frame_ids", "layout_id", "fused_step_id", "fused_missing_conditions", "model_next_step_id",
    "overlay_targets", "overlay_executed", "overlay_rejected",
    "overlay_dropped", "overlay_dry_run_count", "response_status", "fallback_overlay_used",
    "fallback_overlay_reason", "response_mapping_failure_codes",
    "observability_status", "requires_visual_confirmation",
    "scenario_profile",
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
    "Error_OM",
    "Error_CO",
    "Error_OR",
    "Error_PA",
    "Error_SV",
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
    Error_OM: str = ""
    Error_CO: str = ""
    Error_OR: str = ""
    Error_PA: str = ""
    Error_SV: str = ""
    CoderNotes: str = ""
    AutoCodingNotes: str = ""

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


def _record_text(value: str | None) -> str:
    return value or ""


def _help_cycle_matches_step(cycle: HelpCycleRecord, step_id: str) -> bool:
    cycle_step_id = cycle.fused_step_id or cycle.model_next_step_id
    return cycle_step_id == step_id


def _build_step_coding(
    events: Sequence[Mapping[str, Any]],
    *,
    meta: SessionMeta,
    help_cycles: Sequence[HelpCycleRecord],
    pack_steps: Sequence[Mapping[str, Any]],
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

    rows: list[StepCodingRecord] = []
    for step in pack_steps:
        step_id = _opt_str(step.get("id"))
        if not step_id:
            continue
        cycles = [cycle for cycle in help_cycles if _help_cycle_matches_step(cycle, step_id)]
        completed = step_id in completed_steps
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

        auto_notes: list[str] = ["human_error_columns_blank"]
        if completed:
            auto_notes.append("completed_from_step_completed")
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


def _task_time_seconds(events: Sequence[Mapping[str, Any]]) -> float | None:
    times = [t for ev in events if (t := _event_wall_time(ev)) is not None]
    if not times:
        return None
    return max(times) - min(times)


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
            TaskTime_sec=_task_time_seconds(events),
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
        )
        for rc in raw_cycles
    ]

    summary = compute_interaction_metrics(events)
    pack_steps = _load_pack_steps_for_export(pack_path)
    step_coding = _build_step_coding(
        events,
        meta=meta,
        help_cycles=help_cycles,
        pack_steps=pack_steps,
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
        step_coding=step_coding,
        trial_summary=trial_summary,
        scoring=dict(scoring) if isinstance(scoring, Mapping) else None,
        timeline=timeline,
    )


__all__ = [
    "SessionMeta",
    "ExportQualityReport",
    "HelpCycleRecord",
    "StepCodingRecord",
    "TrialSummaryRecord",
    "TimelineSnapshot",
    "ExperimentExport",
    "HELP_CYCLES_CSV_FIELDS",
    "STEP_CODING_CSV_FIELDS",
    "TRIAL_SUMMARY_CSV_FIELDS",
    "build_export_quality_report",
    "build_experiment_export",
    "build_file_sha256",
]
