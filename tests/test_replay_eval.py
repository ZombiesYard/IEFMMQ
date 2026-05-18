from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from core.llm_schema import validate_help_response
from core.types import Observation, TutorRequest
from simtutor.__main__ import main
from simtutor.replay_eval import (
    ReplayEvalCase,
    ReplayEvalExpectation,
    ReplayEvalOracleModel,
    ReplayEvalSuite,
    _extract_case_outcome,
    build_harness_coverage_matrix,
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
    summary = report["summary"]
    assert summary["case_count"] == 5
    assert summary["step_accuracy"] == 1.0
    assert 0.0 <= summary["overlay_target_accuracy"] <= 1.0
    assert 0.0 <= summary["requires_visual_confirmation_accuracy"] <= 1.0
    assert summary["vision_unavailable_rate"] == 0.6
    assert summary["sync_failure_rate"] == 0.2
    assert summary["message_category_evaluated_count"] == 0
    assert summary["message_category_accuracy"] is None
    assert summary["repair_path_evaluated_count"] == 0
    assert summary["repair_path_accuracy"] is None
    assert summary["harness_trace_coverage"] == 1.0
    assert len(report["cases"]) == 5
    assert {case["status"] for case in report["cases"]}.issubset({"passed", "failed"})


def test_harness_coverage_matrix_marks_all_s01_s33_categories() -> None:
    suite = load_replay_eval_suite(SUITE_PATH)

    matrix = build_harness_coverage_matrix(suite)

    assert matrix["step_count"] == 33
    assert matrix["missing_cell_count"] == 0
    assert matrix["contract_only_cell_count"] > 0
    assert matrix["state_categories"] == [
        "normal_progression",
        "omission_missing_action",
        "completion_already_true",
        "stale_telemetry",
        "moving_settling_control",
        "recent_action_gate_conflict",
        "wrong_target_prevention",
        "vlm_not_required",
        "vlm_required",
        "vlm_unavailable",
        "vlm_failed",
    ]
    assert sorted(matrix["steps"].keys()) == [f"S{i:02d}" for i in range(1, 34)]
    for step_id, row in matrix["steps"].items():
        for category in matrix["state_categories"]:
            cell = row[category]
            assert cell["status"] in {
                "covered",
                "contract_only",
                "regression_reference",
                "not_applicable",
            }, (step_id, category, cell)
            assert cell["sources"] or cell["reason"], (step_id, category, cell)
    assert matrix["steps"]["S01"]["vlm_not_required"]["status"] == "contract_only"
    assert matrix["steps"]["S01"]["vlm_unavailable"]["status"] == "not_applicable"
    assert matrix["steps"]["S21"]["moving_settling_control"]["status"] in {
        "contract_only",
        "regression_reference",
    }

    unexecuted_matrix = build_harness_coverage_matrix(suite, case_results=[])
    assert unexecuted_matrix["covered_cell_count"] == 0
    assert unexecuted_matrix["regression_reference_cell_count"] >= 0


def test_replay_eval_report_includes_issue_312_fixture_regression_sources(tmp_path: Path) -> None:
    suite = load_replay_eval_suite(SUITE_PATH)

    report = run_replay_eval_suite(suite, output_dir=tmp_path / "issue312")
    matrix = report["coverage_matrix"]

    issue_sources: dict[int, set[str]] = {}
    for row in matrix["steps"].values():
        for cell in row.values():
            for source in cell.get("sources", []):
                issue = source.get("issue")
                fixture = source.get("fixture")
                if isinstance(issue, int) and isinstance(fixture, str):
                    issue_sources.setdefault(issue, set()).add(fixture)

    assert set(issue_sources) >= {294, 298, 300, 306, 310}
    for issue in (294, 298, 300, 306, 310):
        assert all((REPO_ROOT / fixture).exists() for fixture in issue_sources[issue])
    assert matrix["steps"]["S01"]["vlm_unavailable"]["status"] == "not_applicable"
    assert matrix["steps"]["S21"]["moving_settling_control"]["status"] == "contract_only"
    assert any(
        source.get("issue") == 306
        for source in matrix["steps"]["S21"]["moving_settling_control"]["sources"]
    )


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


def test_run_replay_eval_suite_wires_noop_tutor_text_sender_into_loop(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    suite = ReplayEvalSuite(
        suite_path=tmp_path / "suite.yaml",
        suite_id="issue231",
        dataset_kind="synthetic",
        pack_path=tmp_path / "pack.yaml",
        ui_map_path=tmp_path / "ui_map.yaml",
        telemetry_map_path=tmp_path / "telemetry_map.yaml",
        bios_to_ui_path=tmp_path / "bios_to_ui.yaml",
        knowledge_index_path=tmp_path / "index.json",
        knowledge_source_policy_path=None,
        lang="zh",
        scenario_profile="cold_start",
        cases=(
            ReplayEvalCase(
                case_id="case_1",
                input_path=tmp_path / "input.jsonl",
                session_id="sess-1",
                scenario_profile="cold_start",
                max_frames=1,
                expectation=ReplayEvalExpectation(
                    step_id="S01",
                    overlay_target="battery_switch",
                    requires_visual_confirmation=False,
                    vision_status="vision_unavailable",
                    sync_status=None,
                    sync_delta_ms=None,
                    frame_ids=(),
                ),
            ),
        ),
    )

    class FakeLoop:
        def __init__(self, **kwargs) -> None:
            captured["loop_kwargs"] = dict(kwargs)

        def run(self, **_kwargs) -> None:
            return None

        def close(self) -> None:
            return

    class FakeStore:
        def __init__(self, path: Path, mode: str = "w") -> None:
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.path.write_text("[]", encoding="utf-8")
            return None

        def append(self, _event) -> None:
            return

        @staticmethod
        def load(_path: Path) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr("live_dcs.ReplayBiosReceiver", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("live_dcs.LiveDcsTutorLoop", FakeLoop)
    monkeypatch.setattr("simtutor.replay_eval.JsonlEventStore", FakeStore)
    monkeypatch.setattr(
        "simtutor.replay_eval.OverlayActionExecutor",
        lambda **_kwargs: type(
            "FakeExecutor",
            (),
            {
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: None,
            },
        )(),
    )
    monkeypatch.setattr(
        "simtutor.replay_eval._extract_case_outcome",
        lambda _events, *, case: {"case_id": case.case_id, "status": "passed"},
    )

    run_replay_eval_suite(suite, output_dir=tmp_path / "out", model_factory=lambda _case: object())

    sender = captured["loop_kwargs"]["tutor_text_sender"]
    assert sender is not None
    assert sender.send_text("Turn on APU.")["status"] == "skipped"


def test_replay_eval_oracle_help_response_matches_schema_without_top_level_confidence() -> None:
    suite = load_replay_eval_suite(SUITE_PATH)
    case = suite.cases[0]
    model = ReplayEvalOracleModel(case, lang=suite.lang)
    request = TutorRequest(
        context={
            "recent_deltas": [
                {
                    "ui_target": case.expectation.overlay_target,
                }
            ]
        }
    )

    response = model.explain_error(Observation(), request)
    help_response = response.metadata["help_response"]

    assert "confidence" not in help_response
    validate_help_response(help_response)


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
    assert 0 <= report["summary"]["passed_case_count"] <= 5


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


def test_load_replay_eval_suite_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "schema_version: v1\n"
        "suite_id: duplicate_cases\n"
        "dataset_kind: synthetic\n"
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
        "      frame_ids: []\n"
        "  - case_id: c1\n"
        "    input: replay_eval/fa18c_startup_v04/cases/batteryon_2min/dcs_bios_raw.jsonl\n"
        "    expected:\n"
        "      step_id: S02\n"
        "      overlay_target: battery_switch\n"
        "      requires_visual_confirmation: false\n"
        "      vision_status: vision_unavailable\n"
        "      sync_status:\n"
        "      sync_delta_ms:\n"
        "      frame_ids: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case_id: c1"):
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


def test_load_replay_eval_suite_rejects_unsupported_case_scenario_profile(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "schema_version: v1\n"
        "suite_id: bad_profile_suite\n"
        "dataset_kind: synthetic\n"
        "ui_map_path: packs/fa18c_startup/ui_map.yaml\n"
        "telemetry_map_path: packs/fa18c_startup/telemetry_map.yaml\n"
        "bios_to_ui_path: packs/fa18c_startup/bios_to_ui.yaml\n"
        "knowledge_index_path: Doc/Evaluation/index.json\n"
        "cases:\n"
        "  - case_id: c1\n"
        "    scenario_profile: orbit\n"
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

    with pytest.raises(ValueError, match="c1.scenario_profile: unsupported scenario_profile 'orbit'"):
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


def test_extract_case_outcome_ignores_blank_diagnosis_step_id_and_falls_back_to_next_step() -> None:
    case = ReplayEvalCase(
        case_id="c3",
        input_path=REPO_ROOT / "replay_eval" / "fa18c_startup_v04" / "cases" / "noop_2min" / "dcs_bios_raw.jsonl",
        session_id="sess-c3",
        scenario_profile="airfield",
        max_frames=2,
        expectation=ReplayEvalExpectation(
            step_id="S04",
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
                    "diagnosis": {"step_id": "   "},
                    "next": {"step_id": "S04"},
                    "requires_visual_confirmation": False,
                    "generation_mode": "model",
                },
            },
        },
    ]

    outcome = _extract_case_outcome(events, case=case)

    assert outcome["actual"]["step_id"] == "S04"
    assert outcome["checks"]["step_match"] is True


def test_extract_case_outcome_checks_harness_trace_message_and_repair_path() -> None:
    case = ReplayEvalCase(
        case_id="trace-c1",
        input_path=REPO_ROOT / "replay_eval" / "fa18c_startup_v04" / "cases" / "noop_2min" / "dcs_bios_raw.jsonl",
        session_id="sess-trace-c1",
        scenario_profile="airfield",
        max_frames=2,
        expectation=ReplayEvalExpectation(
            step_id="S08",
            overlay_target="left_mdi_pb15",
            requires_visual_confirmation=True,
            vision_status="ok",
            sync_status="matched",
            sync_delta_ms=12,
            frame_ids=("frame-1",),
            message_category="harness_validator_repair",
            repair_path="validator_repair",
        ),
    )
    events = [
        {
            "kind": "tutor_request",
            "payload": {
                "context": {
                    "vision": {
                        "status": "ok",
                        "sync_status": "matched",
                        "sync_delta_ms": 12,
                        "frame_ids": ["frame-1"],
                    }
                }
            },
        },
        {
            "kind": "tutor_response",
            "payload": {
                "actions": [{"target": "left_mdi_pb15"}],
                "metadata": {
                    "diagnosis": {"step_id": "S08"},
                    "requires_visual_confirmation": True,
                    "generation_mode": "repair",
                    "message_category": "harness_validator_repair",
                    "harness_trace": {
                        "schema_version": "v1",
                        "message_category": "harness_validator_repair",
                        "repair_result": {
                            "applied": True,
                            "path": "validator_repair",
                        },
                        "final_action_plan": {
                            "source": "validator_repair",
                            "targets": ["left_mdi_pb15"],
                        },
                        "vlm_call": {"status": "called", "reason": None},
                    },
                },
            },
        },
    ]

    outcome = _extract_case_outcome(events, case=case)

    assert outcome["actual"]["message_category"] == "harness_validator_repair"
    assert outcome["actual"]["repair_path"] == "validator_repair"
    assert outcome["actual"]["harness_trace_present"] is True
    assert outcome["actual"]["vlm_call_status"] == "called"
    assert outcome["checks"]["message_category_match"] is True
    assert outcome["checks"]["repair_path_match"] is True
    assert outcome["status"] == "passed"


def test_extract_case_outcome_keeps_legacy_event_without_trace_passed() -> None:
    case = ReplayEvalCase(
        case_id="legacy-c1",
        input_path=REPO_ROOT / "replay_eval" / "fa18c_startup_v04" / "cases" / "noop_2min" / "dcs_bios_raw.jsonl",
        session_id="sess-legacy-c1",
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
                    "diagnosis": {"step_id": "S03"},
                    "requires_visual_confirmation": False,
                    "generation_mode": "model",
                },
            },
        },
    ]

    outcome = _extract_case_outcome(events, case=case)

    assert outcome["actual"]["harness_trace_present"] is False
    assert outcome["status"] == "passed"


def test_extract_case_outcome_fails_when_expected_message_category_mismatches() -> None:
    case = ReplayEvalCase(
        case_id="trace-mismatch-c1",
        input_path=REPO_ROOT / "replay_eval" / "fa18c_startup_v04" / "cases" / "noop_2min" / "dcs_bios_raw.jsonl",
        session_id="sess-trace-mismatch-c1",
        scenario_profile="airfield",
        max_frames=2,
        expectation=ReplayEvalExpectation(
            step_id="S08",
            overlay_target="left_mdi_pb15",
            requires_visual_confirmation=True,
            vision_status="ok",
            sync_status="matched",
            sync_delta_ms=12,
            frame_ids=("frame-1",),
            message_category="harness_validator_repair",
            repair_path="validator_repair",
        ),
    )
    events = [
        {
            "kind": "tutor_request",
            "payload": {
                "context": {
                    "vision": {
                        "status": "ok",
                        "sync_status": "matched",
                        "sync_delta_ms": 12,
                        "frame_ids": ["frame-1"],
                    }
                }
            },
        },
        {
            "kind": "tutor_response",
            "payload": {
                "actions": [{"target": "left_mdi_pb15"}],
                "metadata": {
                    "diagnosis": {"step_id": "S08"},
                    "requires_visual_confirmation": True,
                    "message_category": "model",
                    "harness_trace": {
                        "message_category": "model",
                        "repair_result": {"path": "validator_repair"},
                    },
                },
            },
        },
    ]

    outcome = _extract_case_outcome(events, case=case)

    assert outcome["checks"]["message_category_match"] is False
    assert outcome["checks"]["repair_path_match"] is True
    assert outcome["status"] == "failed"


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
    assert any(case["status"] == "error" for case in report["cases"])
    assert any(case["status"] == "passed" for case in report["cases"])
    failed_case = next(case for case in report["cases"] if case["case_id"] == "batteryon_2min")
    assert failed_case["status"] == "error"
    assert failed_case["error"]["stage"] == "execution"
    assert failed_case["error"]["type"] == "RuntimeError"
    assert failed_case["error"]["message"] == "synthetic case failure"
    assert report["coverage_matrix"]["steps"]["S02"]["normal_progression"]["status"] != "covered"
    passed_case_ids = [case["case_id"] for case in report["cases"] if case["status"] == "passed"]
    assert "noop_2min" in passed_case_ids
    assert len(passed_case_ids) >= 1
