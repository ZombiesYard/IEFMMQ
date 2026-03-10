from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from adapters.action_executor import OverlayActionExecutor
from adapters.evidence_refs import collect_evidence_refs_from_context, infer_evidence_type_from_ref
from adapters.vision_frames import DEFAULT_FRAME_CHANNEL, FrameDirectoryVisionPort
from adapters.vision_prompting import DEFAULT_LAYOUT_ID
from core.event_store import JsonlEventStore
from core.types import TutorRequest, TutorResponse


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_repo_or_suite_path(*, suite_dir: Path, raw_path: str | None) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    suite_path = (suite_dir / candidate).resolve()
    if suite_path.exists():
        return suite_path
    repo_path = (_repo_root() / candidate).resolve()
    return repo_path


def _ensure_text(raw: Any, *, field_name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return raw.strip()


def _ensure_bool(raw: Any, *, field_name: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return raw


def _ensure_optional_int(raw: Any, *, field_name: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field_name} must be an integer or null")
    return raw


def _ensure_non_negative_int(raw: Any, *, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if raw < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return raw


def _normalize_string_list(raw: Any, *, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list of strings")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        out.append(item.strip())
    return tuple(out)


def _extract_optional_bool(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    return None


@dataclass(frozen=True)
class ReplayEvalVisionConfig:
    saved_games_dir: Path
    session_id: str
    channel: str
    layout_id: str
    sync_window_ms: int
    trigger_wait_ms: int


@dataclass(frozen=True)
class ReplayEvalExpectation:
    step_id: str
    overlay_target: str
    requires_visual_confirmation: bool
    vision_status: str
    sync_status: str | None
    sync_delta_ms: int | None
    frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReplayEvalCase:
    case_id: str
    input_path: Path
    session_id: str
    scenario_profile: str
    max_frames: int
    expectation: ReplayEvalExpectation
    vision: ReplayEvalVisionConfig | None = None


@dataclass(frozen=True)
class ReplayEvalSuite:
    suite_path: Path
    suite_id: str
    dataset_kind: str
    pack_path: Path
    ui_map_path: Path
    telemetry_map_path: Path
    bios_to_ui_path: Path
    knowledge_index_path: Path
    knowledge_source_policy_path: Path | None
    lang: str
    scenario_profile: str
    cases: tuple[ReplayEvalCase, ...]


class ReplayEvalOracleModel:
    provider = "replay_eval_oracle"

    def __init__(self, case: ReplayEvalCase, *, lang: str) -> None:
        self.case = case
        self.lang = lang if lang in {"zh", "en"} else "zh"

    def close(self) -> None:
        return

    def plan_next_step(self, observation, request: TutorRequest | None = None) -> TutorResponse:
        return self.explain_error(observation, request)

    def explain_error(self, observation, request: TutorRequest | None = None) -> TutorResponse:
        evidence_ref, evidence_type = self._pick_evidence(request, self.case.expectation.overlay_target)
        help_response = {
            "diagnosis": {
                "step_id": self.case.expectation.step_id,
                "error_category": "OM",
            },
            "next": {
                "step_id": self.case.expectation.step_id,
            },
            "overlay": {
                "targets": [self.case.expectation.overlay_target],
                "evidence": [
                    {
                        "target": self.case.expectation.overlay_target,
                        "type": evidence_type,
                        "ref": evidence_ref,
                        "quote": self._evidence_quote(),
                    }
                ],
            },
            "explanations": [self._message_text()],
            "confidence": 0.86 if not self.case.expectation.requires_visual_confirmation else 0.62,
        }
        return TutorResponse(
            status="ok",
            in_reply_to=request.request_id if request else None,
            message=self._message_text(),
            actions=[],
            explanations=[self._message_text()],
            metadata={
                "provider": self.provider,
                "generation_mode": "model",
                "help_response": help_response,
            },
        )

    def _pick_evidence(self, request: TutorRequest | None, target: str) -> tuple[str, str]:
        context = request.context if request and isinstance(request.context, Mapping) else {}
        allowed_refs = collect_evidence_refs_from_context(context)
        preferred = f"RECENT_UI_TARGETS.{target}"
        if preferred in allowed_refs:
            return preferred, "delta"

        gates = context.get("gates")
        if isinstance(gates, Mapping):
            for gate_id in gates.keys():
                if not isinstance(gate_id, str) or not gate_id:
                    continue
                candidate = f"GATES.{gate_id}"
                if candidate in allowed_refs:
                    return candidate, "gate"

        for ref in sorted(allowed_refs):
            evidence_type = infer_evidence_type_from_ref(ref)
            if evidence_type is not None:
                return ref, evidence_type

        raise ValueError(f"no verifiable evidence ref available for replay-eval case {self.case.case_id}")

    def _message_text(self) -> str:
        if self.lang == "en":
            return f"Replay eval guidance: focus on {self.case.expectation.overlay_target}."
        return f"回放评测建议：请先关注 {self.case.expectation.overlay_target}。"

    def _evidence_quote(self) -> str:
        if self.lang == "en":
            return f"Replay regression suite anchors this hint to {self.case.case_id}."
        return f"回放回归集将该提示锚定到样例 {self.case.case_id}。"


class _NoopOverlaySender:
    enabled = False
    event_sink = None

    def close(self) -> None:
        return


def _resolve_required_path(
    *,
    suite_dir: Path,
    raw_value: Any,
    default_value: Any,
    field_name: str,
) -> Path:
    effective_raw = default_value if raw_value is None else raw_value
    if not isinstance(effective_raw, str) or not effective_raw.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    resolved = _resolve_repo_or_suite_path(suite_dir=suite_dir, raw_path=effective_raw)
    if resolved is None:
        raise ValueError(f"{field_name} could not be resolved")
    if not resolved.exists():
        raise ValueError(f"{field_name} does not exist: {resolved}")
    return resolved


def load_replay_eval_suite(path: str | Path) -> ReplayEvalSuite:
    suite_path = Path(path).expanduser().resolve()
    suite_dir = suite_path.parent
    raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"replay eval suite must be a mapping: {suite_path}")
    schema_version = raw.get("schema_version")
    if schema_version != "v1":
        raise ValueError(f"unsupported replay eval suite schema_version {schema_version!r}; expected 'v1'")

    defaults = raw.get("defaults")
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, Mapping):
        raise ValueError("suite.defaults must be a mapping")

    suite_id = _ensure_text(raw.get("suite_id"), field_name="suite_id")
    dataset_kind = _ensure_text(raw.get("dataset_kind"), field_name="dataset_kind")
    lang = raw.get("lang", "zh")
    if lang not in {"zh", "en"}:
        raise ValueError("lang must be zh or en")
    scenario_profile = _ensure_text(
        raw.get("scenario_profile", defaults.get("scenario_profile", "airfield")),
        field_name="scenario_profile",
    )

    pack_path = _resolve_required_path(
        suite_dir=suite_dir,
        raw_value=raw.get("pack_path"),
        default_value=defaults.get("pack_path", "packs/fa18c_startup/pack.yaml"),
        field_name="pack_path",
    )
    ui_map_path = _resolve_required_path(
        suite_dir=suite_dir,
        raw_value=raw.get("ui_map_path"),
        default_value=defaults.get("ui_map_path", "packs/fa18c_startup/ui_map.yaml"),
        field_name="ui_map_path",
    )
    telemetry_map_path = _resolve_required_path(
        suite_dir=suite_dir,
        raw_value=raw.get("telemetry_map_path"),
        default_value=defaults.get("telemetry_map_path", "packs/fa18c_startup/telemetry_map.yaml"),
        field_name="telemetry_map_path",
    )
    bios_to_ui_path = _resolve_required_path(
        suite_dir=suite_dir,
        raw_value=raw.get("bios_to_ui_path"),
        default_value=defaults.get("bios_to_ui_path", "packs/fa18c_startup/bios_to_ui.yaml"),
        field_name="bios_to_ui_path",
    )
    knowledge_index_path = _resolve_required_path(
        suite_dir=suite_dir,
        raw_value=raw.get("knowledge_index_path"),
        default_value=defaults.get("knowledge_index_path", "Doc/Evaluation/index.json"),
        field_name="knowledge_index_path",
    )
    knowledge_source_policy_path = _resolve_repo_or_suite_path(
        suite_dir=suite_dir,
        raw_path=raw.get("knowledge_source_policy_path", defaults.get("knowledge_source_policy_path")),
    )
    if knowledge_source_policy_path is not None and not knowledge_source_policy_path.exists():
        raise ValueError(f"knowledge_source_policy_path does not exist: {knowledge_source_policy_path}")

    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("suite.cases must be a non-empty list")

    default_sync_window_ms = _ensure_non_negative_int(
        defaults.get("vision_sync_window_ms", 100),
        field_name="defaults.vision_sync_window_ms",
    )
    default_trigger_wait_ms = _ensure_non_negative_int(
        defaults.get("vision_trigger_wait_ms", 0),
        field_name="defaults.vision_trigger_wait_ms",
    )
    default_max_frames = _ensure_non_negative_int(
        defaults.get("max_frames", 2),
        field_name="defaults.max_frames",
    )

    cases: list[ReplayEvalCase] = []
    for item in cases_raw:
        if not isinstance(item, Mapping):
            raise ValueError("each suite case must be a mapping")
        case_id = _ensure_text(item.get("case_id"), field_name="case_id")
        input_path = _resolve_required_path(
            suite_dir=suite_dir,
            raw_value=item.get("input"),
            default_value="",
            field_name=f"{case_id}.input",
        )
        session_id = _ensure_text(item.get("session_id", case_id), field_name=f"{case_id}.session_id")
        case_profile = _ensure_text(item.get("scenario_profile", scenario_profile), field_name=f"{case_id}.scenario_profile")
        max_frames = _ensure_non_negative_int(
            item.get("max_frames", default_max_frames),
            field_name=f"{case_id}.max_frames",
        )

        expected = item.get("expected")
        if not isinstance(expected, Mapping):
            raise ValueError(f"{case_id}.expected must be a mapping")
        expectation = ReplayEvalExpectation(
            step_id=_ensure_text(expected.get("step_id"), field_name=f"{case_id}.expected.step_id"),
            overlay_target=_ensure_text(expected.get("overlay_target"), field_name=f"{case_id}.expected.overlay_target"),
            requires_visual_confirmation=_ensure_bool(
                expected.get("requires_visual_confirmation"),
                field_name=f"{case_id}.expected.requires_visual_confirmation",
            ),
            vision_status=_ensure_text(expected.get("vision_status"), field_name=f"{case_id}.expected.vision_status"),
            sync_status=(
                _ensure_text(expected.get("sync_status"), field_name=f"{case_id}.expected.sync_status")
                if expected.get("sync_status") is not None
                else None
            ),
            sync_delta_ms=_ensure_optional_int(expected.get("sync_delta_ms"), field_name=f"{case_id}.expected.sync_delta_ms"),
            frame_ids=_normalize_string_list(expected.get("frame_ids"), field_name=f"{case_id}.expected.frame_ids"),
        )

        vision = item.get("vision")
        vision_config: ReplayEvalVisionConfig | None = None
        if vision is not None:
            if not isinstance(vision, Mapping):
                raise ValueError(f"{case_id}.vision must be a mapping when provided")
            saved_games_dir = _resolve_required_path(
                suite_dir=suite_dir,
                raw_value=vision.get("saved_games_dir"),
                default_value="",
                field_name=f"{case_id}.vision.saved_games_dir",
            )
            vision_config = ReplayEvalVisionConfig(
                saved_games_dir=saved_games_dir,
                session_id=_ensure_text(
                    vision.get("session_id", session_id),
                    field_name=f"{case_id}.vision.session_id",
                ),
                channel=_ensure_text(
                    vision.get("channel", DEFAULT_FRAME_CHANNEL),
                    field_name=f"{case_id}.vision.channel",
                ),
                layout_id=_ensure_text(
                    vision.get("layout_id", DEFAULT_LAYOUT_ID),
                    field_name=f"{case_id}.vision.layout_id",
                ),
                sync_window_ms=_ensure_non_negative_int(
                    vision.get("sync_window_ms", default_sync_window_ms),
                    field_name=f"{case_id}.vision.sync_window_ms",
                ),
                trigger_wait_ms=_ensure_non_negative_int(
                    vision.get("trigger_wait_ms", default_trigger_wait_ms),
                    field_name=f"{case_id}.vision.trigger_wait_ms",
                ),
            )

        cases.append(
            ReplayEvalCase(
                case_id=case_id,
                input_path=input_path,
                session_id=session_id,
                scenario_profile=case_profile,
                max_frames=max_frames,
                expectation=expectation,
                vision=vision_config,
            )
        )

    return ReplayEvalSuite(
        suite_path=suite_path,
        suite_id=suite_id,
        dataset_kind=dataset_kind,
        pack_path=pack_path,
        ui_map_path=ui_map_path,
        telemetry_map_path=telemetry_map_path,
        bios_to_ui_path=bios_to_ui_path,
        knowledge_index_path=knowledge_index_path,
        knowledge_source_policy_path=knowledge_source_policy_path,
        lang=lang,
        scenario_profile=scenario_profile,
        cases=tuple(cases),
    )


def _extract_case_outcome(events: Sequence[Mapping[str, Any]], *, case: ReplayEvalCase) -> dict[str, Any]:
    request_payload = next(
        (
            event.get("payload")
            for event in events
            if event.get("kind") == "tutor_request" and isinstance(event.get("payload"), Mapping)
        ),
        None,
    )
    response_payload = next(
        (
            event.get("payload")
            for event in events
            if event.get("kind") == "tutor_response" and isinstance(event.get("payload"), Mapping)
        ),
        None,
    )
    if not isinstance(request_payload, Mapping):
        raise ValueError(f"replay-eval case {case.case_id} did not emit tutor_request")
    if not isinstance(response_payload, Mapping):
        raise ValueError(f"replay-eval case {case.case_id} did not emit tutor_response")

    request_context = request_payload.get("context")
    if not isinstance(request_context, Mapping):
        request_context = {}
    vision = request_context.get("vision")
    if not isinstance(vision, Mapping):
        vision = {}
    response_meta = response_payload.get("metadata")
    if not isinstance(response_meta, Mapping):
        response_meta = {}

    help_response = response_meta.get("help_response")
    if not isinstance(help_response, Mapping):
        help_response = {}

    diagnosis = response_meta.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        diagnosis = help_response.get("diagnosis")
        if not isinstance(diagnosis, Mapping):
            diagnosis = {}
    next_step = response_meta.get("next")
    if not isinstance(next_step, Mapping):
        next_step = help_response.get("next")
        if not isinstance(next_step, Mapping):
            next_step = {}

    actions = response_payload.get("actions")
    if not isinstance(actions, list):
        actions = []
    overlay_target = None
    if actions:
        first = actions[0]
        if isinstance(first, Mapping):
            target = first.get("target")
            if isinstance(target, str) and target:
                overlay_target = target

    actual = {
        "step_id": diagnosis.get("step_id") if isinstance(diagnosis.get("step_id"), str) else next_step.get("step_id"),
        "overlay_target": overlay_target,
        "requires_visual_confirmation": _extract_optional_bool(response_meta.get("requires_visual_confirmation")),
        "vision_status": vision.get("status"),
        "sync_status": vision.get("sync_status"),
        "sync_delta_ms": vision.get("sync_delta_ms"),
        "frame_ids": tuple(
            item for item in vision.get("frame_ids", ())
            if isinstance(item, str) and item
        )
        if isinstance(vision.get("frame_ids"), (list, tuple))
        else (),
        "generation_mode": response_meta.get("generation_mode"),
        "multimodal_fallback_to_text": _extract_optional_bool(response_meta.get("multimodal_fallback_to_text")),
    }
    checks = {
        "step_match": actual["step_id"] == case.expectation.step_id,
        "overlay_target_match": actual["overlay_target"] == case.expectation.overlay_target,
        "requires_visual_confirmation_match": (
            actual["requires_visual_confirmation"] == case.expectation.requires_visual_confirmation
        ),
        "vision_status_match": actual["vision_status"] == case.expectation.vision_status,
        "sync_status_match": actual["sync_status"] == case.expectation.sync_status,
        "sync_delta_ms_match": actual["sync_delta_ms"] == case.expectation.sync_delta_ms,
        "frame_ids_match": tuple(actual["frame_ids"]) == tuple(case.expectation.frame_ids),
    }
    fallback_used = bool(
        actual["generation_mode"] == "fallback" or actual["multimodal_fallback_to_text"] is True
    )
    return {
        "case_id": case.case_id,
        "expected": {
            "step_id": case.expectation.step_id,
            "overlay_target": case.expectation.overlay_target,
            "requires_visual_confirmation": case.expectation.requires_visual_confirmation,
            "vision_status": case.expectation.vision_status,
            "sync_status": case.expectation.sync_status,
            "sync_delta_ms": case.expectation.sync_delta_ms,
            "frame_ids": list(case.expectation.frame_ids),
        },
        "actual": {
            "step_id": actual["step_id"],
            "overlay_target": actual["overlay_target"],
            "requires_visual_confirmation": actual["requires_visual_confirmation"],
            "vision_status": actual["vision_status"],
            "sync_status": actual["sync_status"],
            "sync_delta_ms": actual["sync_delta_ms"],
            "frame_ids": list(actual["frame_ids"]),
            "generation_mode": actual["generation_mode"],
            "multimodal_fallback_to_text": actual["multimodal_fallback_to_text"],
        },
        "checks": checks,
        "vision_sidecar_configured": case.vision is not None,
        "fallback_used": fallback_used,
        "vision_unavailable": actual["vision_status"] == "vision_unavailable",
        "sync_failed": bool(case.vision is not None and actual["sync_status"] is None),
        "status": "passed" if all(checks.values()) else "failed",
    }


def _build_summary(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    if total <= 0:
        return {
            "case_count": 0,
            "passed_case_count": 0,
            "step_accuracy": 0.0,
            "overlay_target_accuracy": 0.0,
            "requires_visual_confirmation_accuracy": 0.0,
            "fallback_rate": 0.0,
            "vision_unavailable_rate": 0.0,
            "sync_failure_rate": 0.0,
        }

    def _count(check: Callable[[Mapping[str, Any]], bool]) -> int:
        return sum(1 for item in case_results if check(item))

    step_hits = _count(lambda item: bool(item.get("checks", {}).get("step_match")))
    overlay_hits = _count(lambda item: bool(item.get("checks", {}).get("overlay_target_match")))
    visual_hits = _count(lambda item: bool(item.get("checks", {}).get("requires_visual_confirmation_match")))
    passed = _count(lambda item: item.get("status") == "passed")
    fallback_count = _count(lambda item: bool(item.get("fallback_used")))
    vision_unavailable_count = _count(lambda item: bool(item.get("vision_unavailable")))
    sync_failure_count = _count(lambda item: bool(item.get("sync_failed")))
    return {
        "case_count": total,
        "passed_case_count": passed,
        "step_accuracy": round(step_hits / total, 4),
        "overlay_target_accuracy": round(overlay_hits / total, 4),
        "requires_visual_confirmation_accuracy": round(visual_hits / total, 4),
        "fallback_rate": round(fallback_count / total, 4),
        "vision_unavailable_rate": round(vision_unavailable_count / total, 4),
        "sync_failure_rate": round(sync_failure_count / total, 4),
    }


def _default_model_factory(case: ReplayEvalCase, *, lang: str) -> ReplayEvalOracleModel:
    return ReplayEvalOracleModel(case, lang=lang)


def _error_case_result(
    *,
    case: ReplayEvalCase,
    primary_stage: str,
    primary_exc: Exception,
    event_log_path: Path,
    secondary_stage: str | None = None,
    secondary_exc: Exception | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "expected": {
            "step_id": case.expectation.step_id,
            "overlay_target": case.expectation.overlay_target,
            "requires_visual_confirmation": case.expectation.requires_visual_confirmation,
            "vision_status": case.expectation.vision_status,
            "sync_status": case.expectation.sync_status,
            "sync_delta_ms": case.expectation.sync_delta_ms,
            "frame_ids": list(case.expectation.frame_ids),
        },
        "actual": {
            "step_id": None,
            "overlay_target": None,
            "requires_visual_confirmation": None,
            "vision_status": None,
            "sync_status": None,
            "sync_delta_ms": None,
            "frame_ids": [],
            "generation_mode": None,
            "multimodal_fallback_to_text": None,
        },
        "checks": {
            "step_match": False,
            "overlay_target_match": False,
            "requires_visual_confirmation_match": False,
            "vision_status_match": False,
            "sync_status_match": False,
            "sync_delta_ms_match": False,
            "frame_ids_match": False,
        },
        "vision_sidecar_configured": case.vision is not None,
        "fallback_used": False,
        "vision_unavailable": False,
        "sync_failed": False,
        "status": "error",
        "error": {
            "stage": primary_stage,
            "type": type(primary_exc).__name__,
            "message": str(primary_exc),
            "event_log_path": str(event_log_path),
        },
        "secondary_error": (
            {
                "stage": secondary_stage,
                "type": type(secondary_exc).__name__,
                "message": str(secondary_exc),
            }
            if secondary_stage is not None and secondary_exc is not None
            else None
        ),
    }


def run_replay_eval_suite(
    suite: ReplayEvalSuite,
    *,
    output_dir: str | Path,
    report_path: str | Path | None = None,
    model_factory: Callable[[ReplayEvalCase], Any] | None = None,
    provider_name: str = "replay_eval_oracle",
) -> dict[str, Any]:
    from live_dcs import LiveDcsTutorLoop, ReplayBiosReceiver

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    case_results: list[dict[str, Any]] = []
    factory = model_factory or (lambda case: _default_model_factory(case, lang=suite.lang))

    for case in suite.cases:
        case_output_dir = resolved_output_dir / case.case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)
        event_log_path = case_output_dir / "events.jsonl"

        source = ReplayBiosReceiver(case.input_path, speed=0.0)
        model = None
        loop = None
        execution_error: Exception | None = None
        try:
            model = factory(case)
            with JsonlEventStore(event_log_path, mode="w") as store:
                with OverlayActionExecutor(
                    sender=_NoopOverlaySender(),
                    ui_map_path=suite.ui_map_path,
                    pack_path=suite.pack_path,
                    dry_run=True,
                    session_id=case.session_id,
                    event_sink=store.append,
                ) as executor:
                    loop = LiveDcsTutorLoop(
                        source=source,
                        model=model,
                        action_executor=executor,
                        pack_path=suite.pack_path,
                        ui_map_path=suite.ui_map_path,
                        telemetry_map_path=suite.telemetry_map_path,
                        bios_to_ui_path=suite.bios_to_ui_path,
                        knowledge_index_path=suite.knowledge_index_path,
                        rag_top_k=5,
                        cold_start_production=False,
                        knowledge_source_policy_path=suite.knowledge_source_policy_path,
                        cooldown_s=0.0,
                        session_id=case.session_id,
                        lang=suite.lang,
                        scenario_profile=case.scenario_profile,
                        event_sink=store.append,
                        dry_run_overlay=False,
                        vision_port=(
                            None
                            if case.vision is None
                            else FrameDirectoryVisionPort(
                                saved_games_dir=case.vision.saved_games_dir,
                                channel=case.vision.channel,
                                layout_id=case.vision.layout_id,
                            )
                        ),
                        vision_session_id=None if case.vision is None else case.vision.session_id,
                        vision_mode="replay",
                        vision_sync_window_ms=None if case.vision is None else case.vision.sync_window_ms,
                        vision_trigger_wait_ms=None if case.vision is None else case.vision.trigger_wait_ms,
                    )
                    loop.run(
                        max_frames=case.max_frames,
                        duration_s=0.0,
                        auto_help_on_first_frame=True,
                        auto_help_every_n_frames=0,
                        help_trigger=None,
                    )
        except Exception as exc:
            execution_error = exc
        finally:
            if loop is not None:
                loop.close()
            else:
                source.close()
                if model is not None and hasattr(model, "close"):
                    model.close()
        load_error: Exception | None = None
        events: list[dict[str, Any]] | None = None
        try:
            events = JsonlEventStore.load(event_log_path)
        except Exception as exc:
            load_error = exc

        if execution_error is not None:
            case_results.append(
                _error_case_result(
                    case=case,
                    primary_stage="execution",
                    primary_exc=execution_error,
                    event_log_path=event_log_path,
                    secondary_stage="event_load" if load_error is not None else None,
                    secondary_exc=load_error,
                )
            )
            continue
        if load_error is not None or events is None:
            case_results.append(
                _error_case_result(
                    case=case,
                    primary_stage="event_load",
                    primary_exc=(load_error if load_error is not None else RuntimeError("event log unavailable")),
                    event_log_path=event_log_path,
                )
            )
            continue
        try:
            case_results.append(_extract_case_outcome(events, case=case))
        except Exception as exc:
            case_results.append(
                _error_case_result(
                    case=case,
                    primary_stage="outcome_extract",
                    primary_exc=exc,
                    event_log_path=event_log_path,
                )
            )

    case_results.sort(key=lambda item: str(item.get("case_id")))
    report = {
        "schema_version": "v1",
        "suite_id": suite.suite_id,
        "dataset_kind": suite.dataset_kind,
        "lang": suite.lang,
        "model_provider": provider_name,
        "summary": _build_summary(case_results),
        "cases": case_results,
    }
    if report_path is None:
        resolved_report_path = resolved_output_dir / "report.json"
    else:
        resolved_report_path = Path(report_path).expanduser().resolve()
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "ReplayEvalCase",
    "ReplayEvalExpectation",
    "ReplayEvalOracleModel",
    "ReplayEvalSuite",
    "ReplayEvalVisionConfig",
    "load_replay_eval_suite",
    "run_replay_eval_suite",
]
