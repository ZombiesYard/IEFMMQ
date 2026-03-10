from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from simtutor.__main__ import main
from simtutor.replay_eval import (
    ReplayEvalCase,
    ReplayEvalExpectation,
    ReplayEvalOracleModel,
    _extract_case_outcome,
    load_replay_eval_suite,
    run_replay_eval_suite,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPO_ROOT / "replay_eval" / "fa18c_startup_v04" / "suite.yaml"


def test_load_replay_eval_suite_exposes_expected_cases() -> None:
    suite = load_replay_eval_suite(SUITE_PATH)

    assert suite.suite_id == "fa18c_startup_v04_replay_regression"
    assert [case.case_id for case in suite.cases] == [
        "noop_2min",
        "batteryon_2min",
        "Batteryon_enginGenoff_2min",
        "fcs_reset_fcs_bit_2min",
        "ins_2min",
    ]


def test_run_replay_eval_suite_oracle_emits_fixed_summary(tmp_path: Path) -> None:
    suite = load_replay_eval_suite(SUITE_PATH)

    report = run_replay_eval_suite(suite, output_dir=tmp_path / "oracle")

    assert report["model_provider"] == "replay_eval_oracle"
    assert report["summary"] == {
        "case_count": 5,
        "passed_case_count": 5,
        "step_accuracy": 1.0,
        "overlay_target_accuracy": 1.0,
        "requires_visual_confirmation_accuracy": 1.0,
        "fallback_rate": 0.0,
        "vision_unavailable_rate": 0.6,
        "sync_failure_rate": 0.2,
    }
    assert [case["status"] for case in report["cases"]] == ["passed"] * 5


def test_run_replay_eval_suite_is_stable_across_repeated_runs(tmp_path: Path) -> None:
    suite = load_replay_eval_suite(SUITE_PATH)

    first = run_replay_eval_suite(suite, output_dir=tmp_path / "run_a")
    second = run_replay_eval_suite(suite, output_dir=tmp_path / "run_b")

    assert first == second


def test_run_replay_eval_suite_does_not_print_dry_run_actions(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    suite = load_replay_eval_suite(SUITE_PATH)

    run_replay_eval_suite(suite, output_dir=tmp_path / "quiet")

    captured = capsys.readouterr()
    assert "dry_run_actions" not in captured.out


def test_cli_replay_eval_writes_report(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "replay_eval_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simtutor",
            "replay-eval",
            "--suite",
            str(SUITE_PATH),
            "--output-dir",
            str(tmp_path / "logs"),
            "--report",
            str(report_path),
        ],
    )

    code = main()

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model_provider"] == "replay_eval_oracle"
    assert report["summary"]["case_count"] == 5
    assert report["summary"]["passed_case_count"] == 5


def test_load_replay_eval_suite_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "schema_version: v999\n"
        "suite_id: bad_suite\n"
        "dataset_kind: synthetic\n"
        "cases: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported replay eval suite schema_version"):
        load_replay_eval_suite(suite_path)


def test_load_replay_eval_suite_treats_null_required_path_as_missing_not_literal_none(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "schema_version: v1\n"
        "suite_id: null_path_suite\n"
        "dataset_kind: synthetic\n"
        "pack_path:\n"
        "ui_map_path: packs/fa18c_startup/ui_map.yaml\n"
        "telemetry_map_path: packs/fa18c_startup/telemetry_map.yaml\n"
        "bios_to_ui_path: packs/fa18c_startup/bios_to_ui.yaml\n"
        "knowledge_index_path: Doc/Evaluation/index.json\n"
        "cases:\n"
        "  - case_id: c1\n"
        "    input: replay_eval/fa18c_startup_v04/cases/noop_2min/dcs_bios_raw.jsonl\n"
        "    expected:\n"
        "      step_id: S01\n"
        "      overlay_target: battery_switch\n"
        "      requires_visual_confirmation: false\n"
        "      vision_status: vision_unavailable\n"
        "      sync_status:\n"
        "      sync_delta_ms:\n"
        "      frame_ids: []\n",
        encoding="utf-8",
    )

    suite = load_replay_eval_suite(suite_path)

    assert suite.pack_path == REPO_ROOT / "packs" / "fa18c_startup" / "pack.yaml"


def test_load_replay_eval_suite_rejects_invalid_default_integer_budget(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "schema_version: v1\n"
        "suite_id: bad_defaults\n"
        "dataset_kind: synthetic\n"
        "ui_map_path: packs/fa18c_startup/ui_map.yaml\n"
        "telemetry_map_path: packs/fa18c_startup/telemetry_map.yaml\n"
        "bios_to_ui_path: packs/fa18c_startup/bios_to_ui.yaml\n"
        "knowledge_index_path: Doc/Evaluation/index.json\n"
        "defaults:\n"
        "  vision_sync_window_ms: true\n"
        "cases:\n"
        "  - case_id: c1\n"
        "    input: replay_eval/fa18c_startup_v04/cases/noop_2min/dcs_bios_raw.jsonl\n"
        "    expected:\n"
        "      step_id: S01\n"
        "      overlay_target: battery_switch\n"
        "      requires_visual_confirmation: false\n"
        "      vision_status: vision_unavailable\n"
        "      sync_status:\n"
        "      sync_delta_ms:\n"
        "      frame_ids: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="defaults.vision_sync_window_ms must be a non-negative integer"):
        load_replay_eval_suite(suite_path)


def test_load_replay_eval_suite_rejects_negative_case_max_frames(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "schema_version: v1\n"
        "suite_id: bad_case_budget\n"
        "dataset_kind: synthetic\n"
        "ui_map_path: packs/fa18c_startup/ui_map.yaml\n"
        "telemetry_map_path: packs/fa18c_startup/telemetry_map.yaml\n"
        "bios_to_ui_path: packs/fa18c_startup/bios_to_ui.yaml\n"
        "knowledge_index_path: Doc/Evaluation/index.json\n"
        "cases:\n"
        "  - case_id: c1\n"
        "    input: replay_eval/fa18c_startup_v04/cases/noop_2min/dcs_bios_raw.jsonl\n"
        "    max_frames: -1\n"
        "    expected:\n"
        "      step_id: S01\n"
        "      overlay_target: battery_switch\n"
        "      requires_visual_confirmation: false\n"
        "      vision_status: vision_unavailable\n"
        "      sync_status:\n"
        "      sync_delta_ms:\n"
        "      frame_ids: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="c1.max_frames must be >= 0"):
        load_replay_eval_suite(suite_path)


def test_load_replay_eval_suite_rejects_boolean_vision_sync_values(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "schema_version: v1\n"
        "suite_id: bad_vision_budget\n"
        "dataset_kind: synthetic\n"
        "ui_map_path: packs/fa18c_startup/ui_map.yaml\n"
        "telemetry_map_path: packs/fa18c_startup/telemetry_map.yaml\n"
        "bios_to_ui_path: packs/fa18c_startup/bios_to_ui.yaml\n"
        "knowledge_index_path: Doc/Evaluation/index.json\n"
        "cases:\n"
        "  - case_id: c1\n"
        "    input: replay_eval/fa18c_startup_v04/cases/noop_2min/dcs_bios_raw.jsonl\n"
        "    vision:\n"
        "      saved_games_dir: replay_eval/fa18c_startup_v04/cases/ins_2min/Saved Games/DCS\n"
        "      sync_window_ms: false\n"
        "    expected:\n"
        "      step_id: S01\n"
        "      overlay_target: battery_switch\n"
        "      requires_visual_confirmation: false\n"
        "      vision_status: vision_unavailable\n"
        "      sync_status:\n"
        "      sync_delta_ms:\n"
        "      frame_ids: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="c1.vision.sync_window_ms must be a non-negative integer"):
        load_replay_eval_suite(suite_path)


def test_extract_case_outcome_preserves_missing_boolean_metadata_as_none() -> None:
    case = ReplayEvalCase(
        case_id="c1",
        input_path=REPO_ROOT / "replay_eval" / "fa18c_startup_v04" / "cases" / "noop_2min" / "dcs_bios_raw.jsonl",
        session_id="sess-c1",
        scenario_profile="airfield",
        max_frames=2,
        expectation=ReplayEvalExpectation(
            step_id="S01",
            overlay_target="battery_switch",
            requires_visual_confirmation=False,
            vision_status="vision_unavailable",
            sync_status=None,
            sync_delta_ms=None,
            frame_ids=(),
        ),
    )
    events = [
        {
            "kind": "tutor_request",
            "payload": {
                "context": {
                    "vision": {
                        "status": "vision_unavailable",
                        "sync_status": None,
                        "sync_delta_ms": None,
                        "frame_ids": [],
                    }
                }
            },
        },
        {
            "kind": "tutor_response",
            "payload": {
                "actions": [{"target": "battery_switch"}],
                "metadata": {
                    "diagnosis": {"step_id": "S01"},
                    "generation_mode": "model",
                },
            },
        },
    ]

    outcome = _extract_case_outcome(events, case=case)

    assert outcome["actual"]["requires_visual_confirmation"] is None
    assert outcome["actual"]["multimodal_fallback_to_text"] is None
    assert outcome["checks"]["requires_visual_confirmation_match"] is False
    assert outcome["fallback_used"] is False
    assert outcome["status"] == "failed"


def test_extract_case_outcome_falls_back_to_help_response_next_step_id() -> None:
    case = ReplayEvalCase(
        case_id="c2",
        input_path=REPO_ROOT / "replay_eval" / "fa18c_startup_v04" / "cases" / "noop_2min" / "dcs_bios_raw.jsonl",
        session_id="sess-c2",
        scenario_profile="airfield",
        max_frames=2,
        expectation=ReplayEvalExpectation(
            step_id="S03",
            overlay_target="apu_switch",
            requires_visual_confirmation=False,
            vision_status="vision_unavailable",
            sync_status=None,
            sync_delta_ms=None,
            frame_ids=(),
        ),
    )
    events = [
        {
            "kind": "tutor_request",
            "payload": {
                "context": {
                    "vision": {
                        "status": "vision_unavailable",
                        "sync_status": None,
                        "sync_delta_ms": None,
                        "frame_ids": [],
                    }
                }
            },
        },
        {
            "kind": "tutor_response",
            "payload": {
                "actions": [{"target": "apu_switch"}],
                "metadata": {
                    "help_response": {
                        "next": {"step_id": "S03"},
                    },
                    "requires_visual_confirmation": False,
                    "generation_mode": "model",
                },
            },
        },
    ]

    outcome = _extract_case_outcome(events, case=case)

    assert outcome["actual"]["step_id"] == "S03"
    assert outcome["checks"]["step_match"] is True


def test_run_replay_eval_suite_continues_after_case_error(tmp_path: Path) -> None:
    suite = load_replay_eval_suite(SUITE_PATH)

    def _factory(case: ReplayEvalCase):
        if case.case_id == "batteryon_2min":
            raise RuntimeError("synthetic case failure")
        return ReplayEvalOracleModel(case, lang=suite.lang)

    report = run_replay_eval_suite(
        suite,
        output_dir=tmp_path / "error_tolerant",
        model_factory=_factory,
    )

    assert report["summary"]["case_count"] == 5
    assert report["summary"]["passed_case_count"] == 4
    failed_case = next(case for case in report["cases"] if case["case_id"] == "batteryon_2min")
    assert failed_case["status"] == "error"
    assert failed_case["error"]["stage"] == "execution"
    assert failed_case["error"]["type"] == "RuntimeError"
    assert failed_case["error"]["message"] == "synthetic case failure"
    passed_case_ids = [case["case_id"] for case in report["cases"] if case["status"] == "passed"]
    assert "noop_2min" in passed_case_ids
    assert "ins_2min" in passed_case_ids
