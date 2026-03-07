from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from adapters.action_executor import OverlayActionExecutor
from core.types import Observation, TutorResponse
from live_dcs import LiveDcsTutorLoop, ReplayBiosReceiver
from tests.adapters.socket_stubs import DummySocket
from tools.build_coldstart_state_matrix import build_coldstart_state_matrix_dataset


BASE_DIR = Path(__file__).resolve().parents[2]
PACK_PATH = BASE_DIR / "packs" / "fa18c_startup" / "pack.yaml"
UI_MAP_PATH = BASE_DIR / "packs" / "fa18c_startup" / "ui_map.yaml"
TELEMETRY_MAP_PATH = BASE_DIR / "packs" / "fa18c_startup" / "telemetry_map.yaml"
BIOS_TO_UI_PATH = BASE_DIR / "packs" / "fa18c_startup" / "bios_to_ui.yaml"
_VERIFIABLE_EVIDENCE_TYPES = {"var", "gate", "delta"}


class FailingHelpModel:
    def plan_next_step(self, observation: Observation, request=None) -> TutorResponse:  # pragma: no cover
        raise RuntimeError("forced fallback for cold-start regression case")

    def explain_error(self, observation: Observation, request=None) -> TutorResponse:
        raise RuntimeError("forced fallback for cold-start regression case")


def _pick_mapping_event(events: list[Any], kind: str) -> Mapping[str, Any]:
    matches = [event.payload for event in events if getattr(event, "kind", None) == kind]
    assert len(matches) == 1, f"expected exactly one {kind} event, got {len(matches)}"
    payload = matches[0]
    assert isinstance(payload, Mapping)
    return payload


def _expected_executable_result(loop: LiveDcsTutorLoop, inferred_step_id: str | None) -> tuple[bool, list[str]]:
    if not isinstance(inferred_step_id, str) or not inferred_step_id:
        return False, []

    fallback_profile = loop.step_fallback_profiles.get(inferred_step_id, {})
    raw_targets = fallback_profile.get("ui_targets", [])
    available_targets = [
        target
        for target in raw_targets
        if isinstance(target, str) and target in loop.overlay_allowset
    ]

    signal_profile = loop.step_signal_profiles.get(inferred_step_id, {})
    requirements = signal_profile.get("evidence_requirements")
    if isinstance(requirements, list) and requirements:
        can_verify = any(
            isinstance(item, str) and item in _VERIFIABLE_EVIDENCE_TYPES
            for item in requirements
        )
    else:
        can_verify = True

    return bool(available_targets) and can_verify, available_targets


def _run_matrix_case(
    replay_path: Path,
    *,
    frame_count: int,
    scenario_profile: str,
) -> tuple[LiveDcsTutorLoop, dict[str, int], list[Any]]:
    events: list[Any] = []
    source = ReplayBiosReceiver(replay_path, speed=0.0)
    executor = OverlayActionExecutor(
        ui_map_path=UI_MAP_PATH,
        pack_path=PACK_PATH,
        dry_run=True,
        event_sink=events.append,
    )
    loop = LiveDcsTutorLoop(
        source=source,
        model=FailingHelpModel(),
        action_executor=executor,
        pack_path=PACK_PATH,
        ui_map_path=UI_MAP_PATH,
        telemetry_map_path=TELEMETRY_MAP_PATH,
        bios_to_ui_path=BIOS_TO_UI_PATH,
        scenario_profile=scenario_profile,
        event_sink=events.append,
        dry_run_overlay=False,
        rag_top_k=0,
        cooldown_s=0.0,
        lang="en",
    )
    try:
        stats = loop.run(max_frames=frame_count, auto_help_every_n_frames=frame_count)
        return loop, stats, events
    except Exception:
        loop.close()
        raise


def test_coldstart_help_loop_accepts_state_matrix_cases(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("adapters.dcs.overlay.sender.socket.socket", lambda *args, **kwargs: DummySocket())

    output_dir = tmp_path / "coldstart_matrix"
    manifest = build_coldstart_state_matrix_dataset(
        output_dir=output_dir,
        pack_path=PACK_PATH,
        telemetry_map_path=TELEMETRY_MAP_PATH,
        bios_to_ui_path=BIOS_TO_UI_PATH,
        scenario_profile="airfield",
    )

    assert manifest["scenario_profile"] == "airfield"
    assert manifest["case_count"] == 50

    expected_steps = {f"S{i:02d}" for i in range(1, 26)}
    accepted_steps: set[str] = set()

    for case in manifest["cases"]:
        replay_path = output_dir / case["replay_input"]
        loop, stats, events = _run_matrix_case(
            replay_path,
            frame_count=case["frame_count"],
            scenario_profile=case["scenario_profile"],
        )
        try:
            assert stats["frames"] == case["frame_count"], case["case_id"]
            assert stats["help_cycles"] == 1, case["case_id"]
            assert stats["model_calls"] == 1, case["case_id"]
            assert stats["cache_hits"] == 0, case["case_id"]

            request_payload = _pick_mapping_event(events, "tutor_request")
            response_payload = _pick_mapping_event(events, "tutor_response")

            request_meta = request_payload["metadata"]
            response_meta = response_payload["metadata"]
            hint = request_payload["context"]["deterministic_step_hint"]

            assert request_meta["scenario_profile"] == case["scenario_profile"], case["case_id"]
            assert response_meta["scenario_profile"] == case["scenario_profile"], case["case_id"]
            assert request_meta["help_cycle_id"] == response_meta["help_cycle_id"], case["case_id"]
            assert hint["scenario_profile"] == case["scenario_profile"], case["case_id"]
            assert isinstance(hint["missing_conditions"], list), case["case_id"]
            assert isinstance(hint["gate_blockers"], list), case["case_id"]
            assert isinstance(hint["recent_ui_targets"], list), case["case_id"]
            if case["recent_ui_targets"]:
                assert set(hint["recent_ui_targets"]).intersection(case["recent_ui_targets"]), case["case_id"]

            assert response_payload["status"] == "error", case["case_id"]
            assert response_meta["provider"] == "fallback", case["case_id"]
            assert response_meta["generation_mode"] == "fallback", case["case_id"]
            assert isinstance(response_payload["message"], str) and response_payload["message"], case["case_id"]
            assert isinstance(response_meta["fallback_overlay_reason"], str), case["case_id"]
            assert isinstance(response_meta["prompt_build"], dict), case["case_id"]

            executable, available_targets = _expected_executable_result(
                loop,
                hint["inferred_step_id"],
            )
            dry_run_payloads = [event.payload for event in events if getattr(event, "kind", None) == "overlay_dry_run"]

            if executable:
                assert response_meta["fallback_overlay_used"] is True, case["case_id"]
                assert len(response_payload["actions"]) == 1, case["case_id"]
                action = response_payload["actions"][0]
                assert action["type"] == "overlay", case["case_id"]
                assert action["target"] in available_targets, case["case_id"]
                assert len(dry_run_payloads) == 1, case["case_id"]
                assert dry_run_payloads[0]["target"] == action["target"], case["case_id"]
            else:
                assert response_meta["fallback_overlay_used"] is False, case["case_id"]
                assert response_payload["actions"] == [], case["case_id"]
                assert dry_run_payloads == [], case["case_id"]
                assert response_payload["message"].startswith("Fallback:"), case["case_id"]

            accepted_steps.add(case["step_id"])
        finally:
            loop.close()

    assert accepted_steps == expected_steps
