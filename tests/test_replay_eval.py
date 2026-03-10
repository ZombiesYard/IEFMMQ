from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from simtutor.__main__ import main
from simtutor.replay_eval import load_replay_eval_suite, run_replay_eval_suite


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
