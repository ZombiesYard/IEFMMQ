from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adapters.step_inference import StepInferenceResult
from core.types import TutorRequest, TutorResponse
from live_dcs import LiveDcsTutorLoop, ReplayBiosReceiver


class _NoopModel:
    def explain_error(self, obs, request):  # pragma: no cover
        raise AssertionError("model is not used by these progress-state tests")


class _NoopExecutor:
    def execute_actions(self, actions):  # pragma: no cover
        return {"executed": [], "failed": [], "dry_run": []}


def _write_replay(path: Path, frames: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(frame) for frame in frames), encoding="utf-8")


def _bios_frame(seq: int, t_wall: float) -> dict[str, Any]:
    return {
        "schema_version": "v2",
        "seq": seq,
        "t_wall": t_wall,
        "aircraft": "FA-18C_hornet",
        "bios": {"BATTERY_SW": 2, "L_GEN_SW": 1, "R_GEN_SW": 1},
        "delta": {},
    }


def _loop(tmp_path: Path, session_id: str) -> LiveDcsTutorLoop:
    replay_path = tmp_path / f"{session_id}.jsonl"
    _write_replay(replay_path, [_bios_frame(1, 10.0)])
    return LiveDcsTutorLoop(
        source=ReplayBiosReceiver(replay_path, speed=0.0),
        model=_NoopModel(),
        action_executor=_NoopExecutor(),
        session_id=session_id,
    )


def _request(step_id: str, missing_conditions: list[str]) -> TutorRequest:
    return TutorRequest(
        request_id=f"cycle-{step_id.lower()}",
        actor="learner",
        intent="help",
        message="help",
        observation_ref=f"obs-{step_id.lower()}",
        context={
            "vars": {},
            "gates": {},
            "recent_actions": {"recent_buttons": []},
            "deterministic_step_hint": {
                "inferred_step_id": step_id,
                "missing_conditions": list(missing_conditions),
            },
            "vision_facts": [],
        },
        metadata={},
    )


def _trusted_response(step_id: str) -> TutorResponse:
    return TutorResponse(
        status="ok",
        in_reply_to=f"cycle-{step_id.lower()}",
        message=f"Proceed with {step_id}.",
        actions=[],
        metadata={
            "harness_action_plan": {
                "step_id": step_id,
                "overlay_step_id": step_id,
                "targets": [],
                "text_only": True,
                "source": "final_evidence_consistency_validator",
            }
        },
    )


def test_issue_298_keeps_s08_active_when_page_navigation_unconfirmed(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-s08-page-unconfirmed")
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)

        active = loop._active_step_ids_for_vision_facts(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
        )
    finally:
        loop.close()

    assert active == ["S08"]


def test_issue_298_remembers_final_response_progress_for_next_vision_gate(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-final-progress")
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)

        updated = loop._remember_final_response_progress(
            _trusted_response("S09"),
            _request("S08", ["vision_facts.fcs_page_visible==seen"]),
        )
        active = loop._active_step_ids_for_vision_facts(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
        )
    finally:
        loop.close()

    assert updated is True
    assert loop._last_inferred_step_id == "S09"
    assert loop._sticky_inference_step_id == "S09"
    assert active == ["S09"]


def test_issue_298_does_not_remember_untrusted_next_as_progress(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-untrusted-next")
    try:
        loop._last_inferred_step_id = "S08"
        loop._sticky_inference_step_id = "S08"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcs_page_visible==seen",)
        response = TutorResponse(
            status="ok",
            in_reply_to="cycle-untrusted-next",
            message="Model says a later step.",
            actions=[],
            metadata={
                "next": {"step_id": "S20"},
                "final_action_plan": {
                    "step_id": "S20",
                    "targets": [],
                    "text_only": True,
                    "source": "model",
                },
            },
        )

        updated = loop._remember_final_response_progress(
            response,
            _request("S08", ["vision_facts.fcs_page_visible==seen"]),
        )
        active = loop._active_step_ids_for_vision_facts(
            StepInferenceResult("S08", ("vision_facts.fcs_page_visible==seen",)),
        )
    finally:
        loop.close()

    assert updated is False
    assert loop._last_inferred_step_id == "S08"
    assert loop._sticky_inference_step_id == "S08"
    assert active == ["S08"]


def test_issue_298_remembers_visual_final_progress_as_visual_hold(tmp_path: Path) -> None:
    loop = _loop(tmp_path, "issue-298-visual-final-progress")
    try:
        loop._last_inferred_step_id = "S18"
        loop._sticky_inference_step_id = "S18"
        loop._sticky_inference_missing_conditions = ("vision_facts.fcsmc_page_visible==seen",)

        updated = loop._remember_final_response_progress(
            _trusted_response("S19"),
            _request("S19", ["vision_facts.fcsmc_final_go_result_visible==seen"]),
        )
        active = loop._active_step_ids_for_vision_facts(StepInferenceResult("S17", ()))
    finally:
        loop.close()

    assert updated is True
    assert loop._sticky_inference_step_id == "S19"
    assert loop._sticky_inference_missing_conditions == ("vision_facts.fcsmc_final_go_result_visible==seen",)
    assert active == ["S19"]
