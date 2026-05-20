"""
Lightweight post-export analysis for thesis-ready experiment tables.

This module consumes the CSV artifacts produced by ``experiment-export`` and
keeps the aggregation deliberately boring: deterministic rows, no pandas, and
optional figures only when matplotlib is available.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


STUDY_SUMMARY_CSV_FIELDS = [
    "StudyID",
    "ParticipantCount",
    "TrialCount",
    "ConditionCount",
    "CompletionRate",
    "MeanTaskTime_sec",
    "MedianTaskTime_sec",
    "MeanStepCompletionAccuracy",
    "CriticalStepSuccessRate",
    "HelpRequestsPerTrial",
    "RecoveryAfterHelpCandidateRate",
    "OverlayRejectionRate",
    "FallbackRate",
    "VLMCalls",
    "VLMNotRequiredLeakageCount",
]

CONDITION_SUMMARY_CSV_FIELDS = [
    "StudyID",
    "Condition",
    "ParticipantCount",
    "TrialCount",
    "CompletionRate",
    "MeanTaskTime_sec",
    "MedianTaskTime_sec",
    "MeanStepCompletionAccuracy",
    "CriticalStepSuccessRate",
    "HelpRequestsPerTrial",
    "RecoveryAfterHelpCandidateRate",
    "OverlayRejectionRate",
    "FallbackRate",
    "VLMCalls",
    "VLMNotRequiredLeakageCount",
]

STEP_ACCURACY_BY_CONDITION_CSV_FIELDS = [
    "StudyID",
    "Condition",
    "StepID",
    "StepTitle",
    "Critical",
    "TrialCount",
    "PerformedCount",
    "CompletedCount",
    "CompletionRate",
]

HELP_QUALITY_SUMMARY_CSV_FIELDS = [
    "StudyID",
    "Condition",
    "TrialCount",
    "HelpCycleCount",
    "HelpRequestsPerTrial",
    "OverlayExecuted",
    "OverlayRejected",
    "OverlayRejectionRate",
    "FallbackCount",
    "FallbackRate",
    "VisionUsedCount",
    "VisionFactOkRate",
    "RecoveryAfterHelpCandidateRate",
    "VLMNotRequiredLeakageCount",
]


@dataclass
class _TrialExportTables:
    path: Path
    study_id: str
    trial_rows: list[dict[str, str]] = field(default_factory=list)
    step_rows: list[dict[str, str]] = field(default_factory=list)
    help_rows: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ExperimentAnalysis:
    study_summary: list[dict[str, Any]] = field(default_factory=list)
    condition_summary: list[dict[str, Any]] = field(default_factory=list)
    step_accuracy_by_condition: list[dict[str, Any]] = field(default_factory=list)
    help_quality_summary: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisWriteResult:
    csv_paths: list[Path] = field(default_factory=list)
    figure_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_study_id(path: Path) -> str:
    session_path = path / "session.json"
    if not session_path.exists():
        return ""
    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    meta = payload.get("meta") if isinstance(payload, Mapping) else None
    study_id = meta.get("study_id") if isinstance(meta, Mapping) else None
    return study_id if isinstance(study_id, str) else ""


def _discover_trial_exports(input_dir: str | Path) -> list[_TrialExportTables]:
    root = Path(input_dir)
    trial_summary_paths = sorted(root.rglob("trial_summary.csv"))
    exports: list[_TrialExportTables] = []
    for trial_summary_path in trial_summary_paths:
        trial_dir = trial_summary_path.parent
        exports.append(
            _TrialExportTables(
                path=trial_dir,
                study_id=_read_study_id(trial_dir),
                trial_rows=_read_csv_dicts(trial_summary_path),
                step_rows=_read_csv_dicts(trial_dir / "step_coding.csv"),
                help_rows=_read_csv_dicts(trial_dir / "help_cycles.csv"),
            )
        )
    return exports


def _nonempty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_float(value: Any) -> float | None:
    text = _nonempty(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_int(value: Any) -> int:
    number = _as_float(value)
    return int(number) if number is not None else 0


def _is_yes(value: Any) -> bool:
    return _nonempty(value).lower() in {"1", "true", "yes", "y"}


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _mean(values: Sequence[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return round(median(values), 6) if values else 0.0


def _study_id(exports: Sequence[_TrialExportTables], fallback: str) -> str:
    study_ids = sorted({export.study_id for export in exports if export.study_id})
    if not study_ids:
        return fallback
    return ";".join(study_ids)


def _condition(row: Mapping[str, str]) -> str:
    return _nonempty(row.get("Condition")) or "unknown"


def _participant(row: Mapping[str, str]) -> str:
    return _nonempty(row.get("ParticipantID"))


def _step_sort_key(step_id: str) -> tuple[str, int, str]:
    if len(step_id) >= 2 and step_id[0].upper() == "S" and step_id[1:].isdigit():
        return ("S", int(step_id[1:]), step_id)
    return ("", 0, step_id)


def _help_cycle_rejected(row: Mapping[str, str]) -> bool:
    return _as_int(row.get("overlay_rejected")) > 0


def _help_cycle_fallback(row: Mapping[str, str]) -> bool:
    return _is_yes(row.get("fallback_overlay_used")) or _nonempty(row.get("generation_mode")).lower() == "fallback"


def _help_cycle_vision_ok(row: Mapping[str, str]) -> bool:
    return _nonempty(row.get("vision_fact_status")).lower() in {"ok", "available"}


def _help_cycle_vlm_leak(row: Mapping[str, str]) -> bool:
    status = _nonempty(row.get("vlm_call_status")).lower()
    vision_status = _nonempty(row.get("vision_status")).lower()
    vision_fact_status = _nonempty(row.get("vision_fact_status")).lower()
    return status == "called" and (
        vision_status in {"not_required", "vision_not_required"}
        or vision_fact_status in {"not_required", "vision_not_required"}
    )


def _rows_for_conditions(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_condition(row), []).append(row)
    return grouped


def _summary_row(
    *,
    study_id: str,
    condition: str | None,
    trial_rows: Sequence[dict[str, str]],
    step_rows: Sequence[dict[str, str]],
    help_rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    task_times = [
        value
        for value in (_as_float(row.get("TaskTime_sec")) for row in trial_rows)
        if value is not None
    ]
    step_accuracies = [
        value
        for value in (_as_float(row.get("StepCompletionAccuracy")) for row in trial_rows)
        if value is not None
    ]
    critical_steps = [row for row in step_rows if _is_yes(row.get("Critical"))]
    helped_steps = [row for row in step_rows if _as_int(row.get("HelpCount")) > 0]
    completed_helped_steps = [row for row in helped_steps if _is_yes(row.get("Completed"))]
    rejected_cycles = [row for row in help_rows if _help_cycle_rejected(row)]
    fallback_cycles = [row for row in help_rows if _help_cycle_fallback(row)]
    participants = {_participant(row) for row in trial_rows if _participant(row)}
    conditions = {_condition(row) for row in trial_rows}

    row: dict[str, Any] = {
        "StudyID": study_id,
        "ParticipantCount": len(participants),
        "TrialCount": len(trial_rows),
        "CompletionRate": _safe_rate(sum(1 for row in trial_rows if _is_yes(row.get("Completed"))), len(trial_rows)),
        "MeanTaskTime_sec": _mean(task_times),
        "MedianTaskTime_sec": _median(task_times),
        "MeanStepCompletionAccuracy": _mean(step_accuracies),
        "CriticalStepSuccessRate": _safe_rate(
            sum(1 for row in critical_steps if _is_yes(row.get("Completed"))),
            len(critical_steps),
        ),
        "HelpRequestsPerTrial": _safe_rate(sum(_as_int(row.get("HelpRequests")) for row in trial_rows), len(trial_rows)),
        "RecoveryAfterHelpCandidateRate": _safe_rate(len(completed_helped_steps), len(helped_steps)),
        "OverlayRejectionRate": _safe_rate(len(rejected_cycles), len(help_rows)),
        "FallbackRate": _safe_rate(len(fallback_cycles), len(help_rows)),
        "VLMCalls": sum(_as_int(row.get("VLMCalls")) for row in trial_rows),
        "VLMNotRequiredLeakageCount": sum(1 for row in help_rows if _help_cycle_vlm_leak(row)),
    }
    if condition is None:
        row["ConditionCount"] = len(conditions)
        return {field: row.get(field, "") for field in STUDY_SUMMARY_CSV_FIELDS}
    row["Condition"] = condition
    return {field: row.get(field, "") for field in CONDITION_SUMMARY_CSV_FIELDS}


def _build_step_accuracy_rows(
    *,
    study_id: str,
    step_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in step_rows:
        step_id = _nonempty(row.get("StepID"))
        if not step_id:
            continue
        grouped.setdefault((_condition(row), step_id), []).append(row)

    output: list[dict[str, Any]] = []
    for (condition, step_id), rows in sorted(grouped.items(), key=lambda item: (item[0][0], _step_sort_key(item[0][1]))):
        completed_count = sum(1 for row in rows if _is_yes(row.get("Completed")))
        performed_count = sum(1 for row in rows if _is_yes(row.get("Performed")))
        row = {
            "StudyID": study_id,
            "Condition": condition,
            "StepID": step_id,
            "StepTitle": _nonempty(rows[0].get("StepTitle")),
            "Critical": _nonempty(rows[0].get("Critical")),
            "TrialCount": len(rows),
            "PerformedCount": performed_count,
            "CompletedCount": completed_count,
            "CompletionRate": _safe_rate(completed_count, len(rows)),
        }
        output.append({field: row.get(field, "") for field in STEP_ACCURACY_BY_CONDITION_CSV_FIELDS})
    return output


def _build_help_quality_rows(
    *,
    study_id: str,
    trial_rows: Sequence[dict[str, str]],
    step_rows: Sequence[dict[str, str]],
    help_rows: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    trials_by_condition = _rows_for_conditions(trial_rows)
    steps_by_condition = _rows_for_conditions(step_rows)
    help_by_condition: dict[str, list[dict[str, str]]] = {}
    for row in help_rows:
        condition = _nonempty(row.get("Condition"))
        if not condition:
            condition = _condition_for_trial(
                trial_rows,
                participant=_nonempty(row.get("ParticipantID")),
                trial=_nonempty(row.get("TrialID")),
            )
        help_by_condition.setdefault(condition or "unknown", []).append(row)

    conditions = sorted(set(trials_by_condition) | set(help_by_condition))
    output: list[dict[str, Any]] = []
    for condition in conditions:
        condition_trials = trials_by_condition.get(condition, [])
        condition_steps = steps_by_condition.get(condition, [])
        condition_help = help_by_condition.get(condition, [])
        helped_steps = [row for row in condition_steps if _as_int(row.get("HelpCount")) > 0]
        completed_helped_steps = [row for row in helped_steps if _is_yes(row.get("Completed"))]
        rejected_cycles = [row for row in condition_help if _help_cycle_rejected(row)]
        fallback_cycles = [row for row in condition_help if _help_cycle_fallback(row)]
        vision_used_cycles = [row for row in condition_help if _is_yes(row.get("vision_used"))]
        vision_ok_cycles = [row for row in condition_help if _help_cycle_vision_ok(row)]
        row = {
            "StudyID": study_id,
            "Condition": condition,
            "TrialCount": len(condition_trials),
            "HelpCycleCount": len(condition_help),
            "HelpRequestsPerTrial": _safe_rate(
                sum(_as_int(row.get("HelpRequests")) for row in condition_trials),
                len(condition_trials),
            ),
            "OverlayExecuted": sum(_as_int(row.get("overlay_executed")) for row in condition_help),
            "OverlayRejected": sum(_as_int(row.get("overlay_rejected")) for row in condition_help),
            "OverlayRejectionRate": _safe_rate(len(rejected_cycles), len(condition_help)),
            "FallbackCount": len(fallback_cycles),
            "FallbackRate": _safe_rate(len(fallback_cycles), len(condition_help)),
            "VisionUsedCount": len(vision_used_cycles),
            "VisionFactOkRate": _safe_rate(len(vision_ok_cycles), len(vision_used_cycles)),
            "RecoveryAfterHelpCandidateRate": _safe_rate(len(completed_helped_steps), len(helped_steps)),
            "VLMNotRequiredLeakageCount": sum(1 for row in condition_help if _help_cycle_vlm_leak(row)),
        }
        output.append({field: row.get(field, "") for field in HELP_QUALITY_SUMMARY_CSV_FIELDS})
    return output


def _condition_for_trial(
    trial_rows: Sequence[dict[str, str]],
    *,
    participant: str,
    trial: str,
) -> str:
    for row in trial_rows:
        if participant and _nonempty(row.get("ParticipantID")) != participant:
            continue
        if trial and _nonempty(row.get("TrialID")) != trial:
            continue
        return _condition(row)
    return ""


def _copy_condition_to_help_rows(
    help_rows: Sequence[dict[str, str]],
    trial_rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    if not help_rows:
        return []
    default_condition = _condition(trial_rows[0]) if len(trial_rows) == 1 else "unknown"
    rows: list[dict[str, str]] = []
    for row in help_rows:
        copied = dict(row)
        if not _nonempty(copied.get("Condition")):
            copied["Condition"] = default_condition
        if not _nonempty(copied.get("ParticipantID")) and len(trial_rows) == 1:
            copied["ParticipantID"] = _participant(trial_rows[0])
        if not _nonempty(copied.get("TrialID")) and len(trial_rows) == 1:
            copied["TrialID"] = _nonempty(trial_rows[0].get("TrialID"))
        rows.append(copied)
    return rows


def build_experiment_analysis(input_dir: str | Path) -> ExperimentAnalysis:
    exports = _discover_trial_exports(input_dir)
    if not exports:
        raise ValueError(f"no experiment-export trial_summary.csv files found under {input_dir}")

    study_id = _study_id(exports, fallback=Path(input_dir).name)
    trial_rows = [row for export in exports for row in export.trial_rows]
    step_rows = [row for export in exports for row in export.step_rows]
    help_rows = [
        row
        for export in exports
        for row in _copy_condition_to_help_rows(export.help_rows, export.trial_rows)
    ]

    condition_summary = [
        _summary_row(
            study_id=study_id,
            condition=condition,
            trial_rows=rows,
            step_rows=[row for row in step_rows if _condition(row) == condition],
            help_rows=[row for row in help_rows if _condition(row) == condition],
        )
        for condition, rows in sorted(_rows_for_conditions(trial_rows).items())
    ]

    return ExperimentAnalysis(
        study_summary=[
            _summary_row(
                study_id=study_id,
                condition=None,
                trial_rows=trial_rows,
                step_rows=step_rows,
                help_rows=help_rows,
            )
        ],
        condition_summary=condition_summary,
        step_accuracy_by_condition=_build_step_accuracy_rows(study_id=study_id, step_rows=step_rows),
        help_quality_summary=_build_help_quality_rows(
            study_id=study_id,
            trial_rows=trial_rows,
            step_rows=step_rows,
            help_rows=help_rows,
        ),
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bar_chart(path: Path, labels: Sequence[str], values: Sequence[float], *, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_figures(analysis: ExperimentAnalysis, output_dir: Path) -> list[Path]:
    labels = [str(row["Condition"]) for row in analysis.condition_summary]
    if not labels:
        return []
    specs = [
        (
            "fig_completion_rate.png",
            [float(row["CompletionRate"]) for row in analysis.condition_summary],
            "Completion rate by condition",
            "completion rate",
        ),
        (
            "fig_task_time.png",
            [float(row["MeanTaskTime_sec"]) for row in analysis.condition_summary],
            "Mean task time by condition",
            "seconds",
        ),
        (
            "fig_step_accuracy.png",
            [float(row["MeanStepCompletionAccuracy"]) for row in analysis.condition_summary],
            "Mean step accuracy by condition",
            "accuracy",
        ),
        (
            "fig_help_requests.png",
            [float(row["HelpRequestsPerTrial"]) for row in analysis.condition_summary],
            "Help requests per trial by condition",
            "requests/trial",
        ),
    ]
    paths: list[Path] = []
    for filename, values, title, ylabel in specs:
        path = output_dir / filename
        _bar_chart(path, labels, values, title=title, ylabel=ylabel)
        paths.append(path)
    return paths


def write_experiment_analysis(
    analysis: ExperimentAnalysis,
    output_dir: str | Path,
    *,
    make_figures: bool = True,
) -> AnalysisWriteResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        ("study_summary.csv", STUDY_SUMMARY_CSV_FIELDS, analysis.study_summary),
        ("condition_summary.csv", CONDITION_SUMMARY_CSV_FIELDS, analysis.condition_summary),
        ("step_accuracy_by_condition.csv", STEP_ACCURACY_BY_CONDITION_CSV_FIELDS, analysis.step_accuracy_by_condition),
        ("help_quality_summary.csv", HELP_QUALITY_SUMMARY_CSV_FIELDS, analysis.help_quality_summary),
    ]
    csv_paths: list[Path] = []
    for filename, fieldnames, rows in outputs:
        path = out_dir / filename
        _write_csv(path, fieldnames, rows)
        csv_paths.append(path)

    warnings = list(analysis.warnings)
    figure_paths: list[Path] = []
    if make_figures:
        try:
            figure_paths = _write_figures(analysis, out_dir)
        except Exception as exc:
            message = f"figures skipped: {type(exc).__name__}: {exc}"
            warnings.append(message)
            for name in (
                "fig_completion_rate.png",
                "fig_task_time.png",
                "fig_step_accuracy.png",
                "fig_help_requests.png",
            ):
                try:
                    (out_dir / name).unlink()
                except FileNotFoundError:
                    pass
            (out_dir / "figures_skipped.txt").write_text(message + "\n", encoding="utf-8")

    return AnalysisWriteResult(csv_paths=csv_paths, figure_paths=figure_paths, warnings=warnings)


__all__ = [
    "CONDITION_SUMMARY_CSV_FIELDS",
    "HELP_QUALITY_SUMMARY_CSV_FIELDS",
    "STEP_ACCURACY_BY_CONDITION_CSV_FIELDS",
    "STUDY_SUMMARY_CSV_FIELDS",
    "AnalysisWriteResult",
    "ExperimentAnalysis",
    "build_experiment_analysis",
    "write_experiment_analysis",
]
