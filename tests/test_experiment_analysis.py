"""
Tests for thesis-ready experiment analysis exports.
"""

import argparse
import csv
import json
from pathlib import Path

import core.experiment_analysis as experiment_analysis
from core.experiment_analysis import (
    CONDITION_SUMMARY_CSV_FIELDS,
    HELP_QUALITY_SUMMARY_CSV_FIELDS,
    STEP_ACCURACY_BY_CONDITION_CSV_FIELDS,
    STUDY_SUMMARY_CSV_FIELDS,
    build_experiment_analysis,
    write_experiment_analysis,
)
from simtutor.__main__ import _run_experiment_analyze


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_trial_export(
    root: Path,
    *,
    participant: str,
    condition: str,
    trial: str,
    completed: str,
    task_time: float,
    help_requests: int,
    vlm_calls: int,
    overlay_rejected: int,
    fallback_count: int,
    step_accuracy: float,
    s01_completed: str,
    s02_completed: str,
) -> None:
    trial_dir = root / participant / trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "session.json").write_text(
        json.dumps(
            {
                "meta": {
                    "study_id": "study-alpha",
                    "participant_id": participant,
                    "condition": condition,
                    "trial_id": trial,
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        trial_dir / "trial_summary.csv",
        [
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
        ],
        [
            {
                "ParticipantID": participant,
                "Condition": condition,
                "TrialID": trial,
                "Completed": completed,
                "TaskTime_sec": task_time,
                "HelpRequests": help_requests,
                "LLMTriggers": 1,
                "VLMCalls": vlm_calls,
                "OverlayExecuted": 2,
                "OverlayRejected": overlay_rejected,
                "FallbackCount": fallback_count,
                "CriticalStepsCompleted": 1 if s01_completed == "yes" else 0,
                "TotalStepsCompleted": (1 if s01_completed == "yes" else 0)
                + (1 if s02_completed == "yes" else 0),
                "StepCompletionAccuracy": step_accuracy,
            }
        ],
    )
    _write_csv(
        trial_dir / "step_coding.csv",
        [
            "ParticipantID",
            "Condition",
            "TrialID",
            "StepID",
            "StepTitle",
            "Critical",
            "Performed",
            "Completed",
            "HelpCount",
        ],
        [
            {
                "ParticipantID": participant,
                "Condition": condition,
                "TrialID": trial,
                "StepID": "S01",
                "StepTitle": "Battery switch",
                "Critical": "yes",
                "Performed": "yes",
                "Completed": s01_completed,
                "HelpCount": 1,
            },
            {
                "ParticipantID": participant,
                "Condition": condition,
                "TrialID": trial,
                "StepID": "S02",
                "StepTitle": "Display power",
                "Critical": "no",
                "Performed": s02_completed,
                "Completed": s02_completed,
                "HelpCount": 0,
            },
        ],
    )
    _write_csv(
        trial_dir / "help_cycles.csv",
        [
            "help_cycle_id",
            "vision_used",
            "vision_fact_status",
            "overlay_executed",
            "overlay_rejected",
            "fallback_overlay_used",
            "generation_mode",
            "vlm_call_status",
        ],
        [
            {
                "help_cycle_id": f"{participant}-{trial}-h1",
                "vision_used": "True",
                "vision_fact_status": "ok",
                "overlay_executed": "1",
                "overlay_rejected": str(overlay_rejected),
                "fallback_overlay_used": "False" if fallback_count == 0 else "True",
                "generation_mode": "model" if fallback_count == 0 else "fallback",
                "vlm_call_status": "called",
            }
        ],
    )


def test_build_experiment_analysis_combines_conditions_and_steps(tmp_path: Path):
    exports = tmp_path / "experiments"
    _write_trial_export(
        exports,
        participant="P01",
        condition="with_tutor",
        trial="T01",
        completed="yes",
        task_time=100.0,
        help_requests=2,
        vlm_calls=1,
        overlay_rejected=0,
        fallback_count=0,
        step_accuracy=1.0,
        s01_completed="yes",
        s02_completed="yes",
    )
    _write_trial_export(
        exports,
        participant="P02",
        condition="without_tutor",
        trial="T01",
        completed="no",
        task_time=200.0,
        help_requests=0,
        vlm_calls=0,
        overlay_rejected=1,
        fallback_count=1,
        step_accuracy=0.5,
        s01_completed="yes",
        s02_completed="no",
    )
    _write_csv(
        exports / "P01" / "T01" / "help_cycles.csv",
        [
            "help_cycle_id",
            "vision_used",
            "vision_status",
            "vision_fact_status",
            "overlay_executed",
            "overlay_rejected",
            "fallback_overlay_used",
            "generation_mode",
            "vlm_call_status",
        ],
        [
            {
                "help_cycle_id": "P01-T01-h1",
                "vision_used": "True",
                "vision_status": "available",
                "vision_fact_status": "ok",
                "overlay_executed": "1",
                "overlay_rejected": "0",
                "fallback_overlay_used": "False",
                "generation_mode": "model",
                "vlm_call_status": "called",
            },
            {
                "help_cycle_id": "P01-T01-h2",
                "vision_used": "False",
                "vision_status": "not_required",
                "vision_fact_status": "vision_not_required",
                "overlay_executed": "1",
                "overlay_rejected": "0",
                "fallback_overlay_used": "False",
                "generation_mode": "model",
                "vlm_call_status": "not_required",
            },
        ],
    )

    analysis = build_experiment_analysis(exports)

    assert list(analysis.study_summary[0].keys()) == STUDY_SUMMARY_CSV_FIELDS
    assert analysis.study_summary[0]["StudyID"] == "study-alpha"
    assert analysis.study_summary[0]["ParticipantCount"] == 2
    assert analysis.study_summary[0]["TrialCount"] == 2
    assert analysis.study_summary[0]["CompletionRate"] == 0.5
    assert analysis.study_summary[0]["MeanTaskTime_sec"] == 150.0
    assert analysis.study_summary[0]["MedianTaskTime_sec"] == 150.0
    assert analysis.study_summary[0]["MeanStepCompletionAccuracy"] == 0.75

    condition_rows = {row["Condition"]: row for row in analysis.condition_summary}
    assert list(analysis.condition_summary[0].keys()) == CONDITION_SUMMARY_CSV_FIELDS
    assert condition_rows["with_tutor"]["ParticipantCount"] == 1
    assert condition_rows["with_tutor"]["CompletionRate"] == 1.0
    assert condition_rows["without_tutor"]["CompletionRate"] == 0.0
    assert condition_rows["without_tutor"]["OverlayRejectionRate"] == 1.0
    assert condition_rows["without_tutor"]["FallbackRate"] == 1.0

    step_rows = {
        (row["Condition"], row["StepID"]): row
        for row in analysis.step_accuracy_by_condition
    }
    assert list(analysis.step_accuracy_by_condition[0].keys()) == STEP_ACCURACY_BY_CONDITION_CSV_FIELDS
    assert step_rows[("with_tutor", "S02")]["CompletionRate"] == 1.0
    assert step_rows[("without_tutor", "S02")]["CompletionRate"] == 0.0

    help_rows = {row["Condition"]: row for row in analysis.help_quality_summary}
    assert list(analysis.help_quality_summary[0].keys()) == HELP_QUALITY_SUMMARY_CSV_FIELDS
    assert help_rows["with_tutor"]["HelpRequestsPerTrial"] == 2.0
    assert help_rows["with_tutor"]["RecoveryAfterHelpCandidateRate"] == 1.0
    assert help_rows["with_tutor"]["VisionUsedCount"] == 1
    assert help_rows["with_tutor"]["VisionFactOkRate"] == 1.0
    assert help_rows["without_tutor"]["RecoveryAfterHelpCandidateRate"] == 1.0


def test_write_experiment_analysis_creates_csvs_even_without_figures(tmp_path: Path):
    exports = tmp_path / "experiments"
    output = tmp_path / "analysis"
    _write_trial_export(
        exports,
        participant="P01",
        condition="with_tutor",
        trial="T01",
        completed="yes",
        task_time=100.0,
        help_requests=1,
        vlm_calls=1,
        overlay_rejected=0,
        fallback_count=0,
        step_accuracy=1.0,
        s01_completed="yes",
        s02_completed="yes",
    )

    analysis = build_experiment_analysis(exports)
    result = write_experiment_analysis(analysis, output, make_figures=False)

    assert result.csv_paths == [
        output / "study_summary.csv",
        output / "condition_summary.csv",
        output / "step_accuracy_by_condition.csv",
        output / "help_quality_summary.csv",
    ]
    assert result.figure_paths == []
    assert (output / "study_summary.csv").exists()
    assert (output / "condition_summary.csv").exists()
    assert not (output / "fig_completion_rate.png").exists()


def test_write_experiment_analysis_keeps_csvs_when_figures_fail(tmp_path: Path, monkeypatch):
    exports = tmp_path / "experiments"
    output = tmp_path / "analysis"
    _write_trial_export(
        exports,
        participant="P01",
        condition="with_tutor",
        trial="T01",
        completed="yes",
        task_time=100.0,
        help_requests=1,
        vlm_calls=1,
        overlay_rejected=0,
        fallback_count=0,
        step_accuracy=1.0,
        s01_completed="yes",
        s02_completed="yes",
    )

    def _raise_figure_error(*args, **kwargs):
        raise ImportError("matplotlib missing")

    monkeypatch.setattr(experiment_analysis, "_write_figures", _raise_figure_error)
    analysis = build_experiment_analysis(exports)

    result = write_experiment_analysis(analysis, output, make_figures=True)

    assert (output / "study_summary.csv").exists()
    assert (output / "figures_skipped.txt").exists()
    assert result.figure_paths == []
    assert result.warnings == ["figures skipped: ImportError: matplotlib missing"]


def test_experiment_analyze_cli_writes_analysis_outputs(tmp_path: Path):
    exports = tmp_path / "experiments"
    output = tmp_path / "analysis"
    _write_trial_export(
        exports,
        participant="P01",
        condition="with_tutor",
        trial="T01",
        completed="yes",
        task_time=100.0,
        help_requests=1,
        vlm_calls=1,
        overlay_rejected=0,
        fallback_count=0,
        step_accuracy=1.0,
        s01_completed="yes",
        s02_completed="yes",
    )

    args = argparse.Namespace(input_dir=str(exports), output_dir=str(output), no_figures=True)

    assert _run_experiment_analyze(args) == 0
    assert (output / "study_summary.csv").exists()
    assert (output / "condition_summary.csv").exists()
    assert (output / "step_accuracy_by_condition.csv").exists()
    assert (output / "help_quality_summary.csv").exists()
