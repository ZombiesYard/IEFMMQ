from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from adapters.action_executor import OverlayActionExecutor
from adapters.dcs.tutor_text import NoopTutorTextSender
from adapters.evidence_refs import collect_evidence_refs_from_context, infer_evidence_type_from_ref
from adapters.pack_gates import normalize_scenario_profile
from adapters.step_harness_specs import load_step_harness_specs
from adapters.vision_frames import DEFAULT_FRAME_CHANNEL, FrameDirectoryVisionPort
from adapters.vision_prompting import DEFAULT_LAYOUT_ID
from core.event_store import JsonlEventStore
from core.types import TutorRequest, TutorResponse


HARNESS_COVERAGE_STATE_CATEGORIES: tuple[str, ...] = (
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
)


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


def _ensure_optional_positive_issue(raw: Any, *, field_name: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"{field_name} must be a positive integer when provided")
    return raw


def _extract_optional_bool(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    return None


def _extract_optional_text(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    normalized = raw.strip()
    if not normalized:
        return None
    return normalized


def _ensure_scenario_profile(raw: Any, *, field_name: str) -> str:
    value = _ensure_text(raw, field_name=field_name)
    try:
        return normalize_scenario_profile(value)
    except ValueError as exc:
        raise ValueError(f"{field_name}: {exc}") from exc


def _normalize_coverage_tags(
    raw: Any,
    *,
    field_name: str,
    default_case_id: str | None = None,
) -> tuple[ReplayEvalCoverageTag, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list")
    tags: list[ReplayEvalCoverageTag] = []
    for idx, item in enumerate(raw):
        item_field = f"{field_name}[{idx}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{item_field} must be a mapping")
        state_category = _ensure_text(item.get("state_category"), field_name=f"{item_field}.state_category")
        if state_category not in HARNESS_COVERAGE_STATE_CATEGORIES:
            allowed = ", ".join(HARNESS_COVERAGE_STATE_CATEGORIES)
            raise ValueError(f"{item_field}.state_category must be one of {{{allowed}}}")
        case_id = _extract_optional_text(item.get("case_id")) or default_case_id
        tags.append(
            ReplayEvalCoverageTag(
                step_id=_ensure_text(item.get("step_id"), field_name=f"{item_field}.step_id"),
                state_category=state_category,
                source=_ensure_text(item.get("source"), field_name=f"{item_field}.source"),
                case_id=case_id,
                issue=_ensure_optional_positive_issue(item.get("issue"), field_name=f"{item_field}.issue"),
                fixture=_extract_optional_text(item.get("fixture")),
                test_id=_extract_optional_text(item.get("test_id")),
                reason=_extract_optional_text(item.get("reason")),
            )
        )
    return tuple(tags)


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
    message_category: str | None = None
    repair_path: str | None = None


@dataclass(frozen=True)
class ReplayEvalCoverageTag:
    step_id: str
    state_category: str
    source: str
    case_id: str | None = None
    issue: int | None = None
    fixture: str | None = None
    test_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ReplayEvalCase:
    case_id: str
    input_path: Path
    session_id: str
    scenario_profile: str
    max_frames: int
    expectation: ReplayEvalExpectation
    vision: ReplayEvalVisionConfig | None = None
    coverage_tags: tuple[ReplayEvalCoverageTag, ...] = ()


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
    coverage_tags: tuple[ReplayEvalCoverageTag, ...] = ()


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
        grounding_confidence = 0.86 if not self.case.expectation.requires_visual_confirmation else 0.62
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
                        "grounding_confidence": grounding_confidence,
                    }
                ],
            },
            "explanations": [self._message_text()],
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
    scenario_profile = _ensure_scenario_profile(
        raw.get("scenario_profile", defaults.get("scenario_profile", "airfield")),
        field_name="scenario_profile",
    )
    suite_coverage_tags = _normalize_coverage_tags(raw.get("coverage"), field_name="suite.coverage")

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
    seen_case_ids: set[str] = set()
    for item in cases_raw:
        if not isinstance(item, Mapping):
            raise ValueError("each suite case must be a mapping")
        case_id = _ensure_text(item.get("case_id"), field_name="case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        input_path = _resolve_required_path(
            suite_dir=suite_dir,
            raw_value=item.get("input"),
            default_value="",
            field_name=f"{case_id}.input",
        )
        session_id = _ensure_text(item.get("session_id", case_id), field_name=f"{case_id}.session_id")
        case_profile = _ensure_scenario_profile(
            item.get("scenario_profile", scenario_profile),
            field_name=f"{case_id}.scenario_profile",
        )
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
            message_category=(
                _ensure_text(expected.get("message_category"), field_name=f"{case_id}.expected.message_category")
                if expected.get("message_category") is not None
                else None
            ),
            repair_path=(
                _ensure_text(expected.get("repair_path"), field_name=f"{case_id}.expected.repair_path")
                if expected.get("repair_path") is not None
                else None
            ),
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
                coverage_tags=_normalize_coverage_tags(
                    item.get("coverage"),
                    field_name=f"{case_id}.coverage",
                    default_case_id=case_id,
                ),
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
        coverage_tags=suite_coverage_tags,
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
    harness_trace = response_meta.get("harness_trace")
    if not isinstance(harness_trace, Mapping):
        harness_trace = {}

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

    diagnosis_step_id = _extract_optional_text(diagnosis.get("step_id"))
    next_step_id = _extract_optional_text(next_step.get("step_id"))
    repair_result = harness_trace.get("repair_result")
    if not isinstance(repair_result, Mapping):
        repair_result = {}
    vlm_call = harness_trace.get("vlm_call")
    if not isinstance(vlm_call, Mapping):
        vlm_call = {}
    message_category = _extract_optional_text(response_meta.get("message_category"))
    if message_category is None:
        message_category = _extract_optional_text(harness_trace.get("message_category"))
    repair_path = _extract_optional_text(repair_result.get("path"))

    actual = {
        "step_id": diagnosis_step_id or next_step_id,
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
        "message_category": message_category,
        "repair_path": repair_path,
        "harness_trace_present": bool(harness_trace),
        "vlm_call_status": _extract_optional_text(vlm_call.get("status")),
        "vlm_call_reason": _extract_optional_text(vlm_call.get("reason")),
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
        "message_category_match": (
            True
            if case.expectation.message_category is None
            else actual["message_category"] == case.expectation.message_category
        ),
        "repair_path_match": (
            True
            if case.expectation.repair_path is None
            else actual["repair_path"] == case.expectation.repair_path
        ),
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
            "message_category": case.expectation.message_category,
            "repair_path": case.expectation.repair_path,
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
            "message_category": actual["message_category"],
            "repair_path": actual["repair_path"],
            "harness_trace_present": actual["harness_trace_present"],
            "vlm_call_status": actual["vlm_call_status"],
            "vlm_call_reason": actual["vlm_call_reason"],
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
            "message_category_accuracy": 0.0,
            "repair_path_accuracy": 0.0,
            "harness_trace_coverage": 0.0,
            "message_category_evaluated_count": 0,
            "repair_path_evaluated_count": 0,
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
    message_category_evaluated = _count(
        lambda item: item.get("expected", {}).get("message_category") is not None
    )
    repair_path_evaluated = _count(
        lambda item: item.get("expected", {}).get("repair_path") is not None
    )
    message_category_hits = _count(
        lambda item: item.get("expected", {}).get("message_category") is not None
        and bool(item.get("checks", {}).get("message_category_match"))
    )
    repair_path_hits = _count(
        lambda item: item.get("expected", {}).get("repair_path") is not None
        and bool(item.get("checks", {}).get("repair_path_match"))
    )
    harness_trace_count = _count(lambda item: bool(item.get("actual", {}).get("harness_trace_present")))
    return {
        "case_count": total,
        "passed_case_count": passed,
        "step_accuracy": round(step_hits / total, 4),
        "overlay_target_accuracy": round(overlay_hits / total, 4),
        "requires_visual_confirmation_accuracy": round(visual_hits / total, 4),
        "fallback_rate": round(fallback_count / total, 4),
        "vision_unavailable_rate": round(vision_unavailable_count / total, 4),
        "sync_failure_rate": round(sync_failure_count / total, 4),
        "message_category_accuracy": (
            round(message_category_hits / message_category_evaluated, 4)
            if message_category_evaluated
            else None
        ),
        "repair_path_accuracy": (
            round(repair_path_hits / repair_path_evaluated, 4)
            if repair_path_evaluated
            else None
        ),
        "harness_trace_coverage": round(harness_trace_count / total, 4),
        "message_category_evaluated_count": message_category_evaluated,
        "repair_path_evaluated_count": repair_path_evaluated,
    }


def _coverage_source_dict(tag: ReplayEvalCoverageTag | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(tag, ReplayEvalCoverageTag):
        raw: dict[str, Any] = {
            "source": tag.source,
            "case_id": tag.case_id,
            "issue": tag.issue,
            "fixture": tag.fixture,
            "test_id": tag.test_id,
            "reason": tag.reason,
        }
    else:
        raw = dict(tag)
    return {key: value for key, value in raw.items() if value is not None}


def _new_coverage_cell(*, status: str = "missing", reason: str | None = None) -> dict[str, Any]:
    return {"status": status, "sources": [], "reason": reason}


def _mark_coverage_contract(
    steps: dict[str, dict[str, dict[str, Any]]],
    *,
    step_id: str,
    state_category: str,
    source: Mapping[str, Any],
) -> None:
    row = steps.get(step_id)
    if row is None or state_category not in row:
        return
    cell = row[state_category]
    if cell["status"] in {"covered", "not_applicable"}:
        return
    cell["status"] = "contract_only"
    cell["reason"] = None
    source_dict = _coverage_source_dict(source)
    if source_dict not in cell["sources"]:
        cell["sources"].append(source_dict)


def _add_regression_reference(
    steps: dict[str, dict[str, dict[str, Any]]],
    *,
    step_id: str,
    state_category: str,
    source: Mapping[str, Any],
) -> None:
    row = steps.get(step_id)
    if row is None or state_category not in row:
        return
    cell = row[state_category]
    if cell["status"] == "not_applicable":
        return
    if cell["status"] not in {"covered", "contract_only"}:
        cell["status"] = "regression_reference"
        cell["reason"] = None
    source_dict = _coverage_source_dict(source)
    if source_dict not in cell["sources"]:
        cell["sources"].append(source_dict)


def _add_coverage_source(
    steps: dict[str, dict[str, dict[str, Any]]],
    *,
    step_id: str,
    state_category: str,
    source: Mapping[str, Any],
) -> None:
    row = steps.get(step_id)
    if row is None or state_category not in row:
        return
    cell = row[state_category]
    if cell["status"] == "not_applicable":
        return
    cell["status"] = "covered"
    cell["reason"] = None
    source_dict = _coverage_source_dict(source)
    if source_dict not in cell["sources"]:
        cell["sources"].append(source_dict)


def _mark_coverage_not_applicable(
    steps: dict[str, dict[str, dict[str, Any]]],
    *,
    step_id: str,
    state_category: str,
    reason: str,
) -> None:
    row = steps.get(step_id)
    if row is None or state_category not in row:
        return
    cell = row[state_category]
    if cell["status"] == "covered":
        return
    cell["status"] = "not_applicable"
    cell["reason"] = reason


def _case_result_statuses(case_results: Sequence[Mapping[str, Any]] | None) -> dict[str, str]:
    if case_results is None:
        return {}
    statuses: dict[str, str] = {}
    for result in case_results:
        case_id = result.get("case_id")
        status = result.get("status")
        if isinstance(case_id, str) and isinstance(status, str):
            statuses[case_id] = status
    return statuses


def _all_suite_coverage_tags(
    suite: ReplayEvalSuite,
    *,
    case_results: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[ReplayEvalCoverageTag, ...]:
    case_statuses = _case_result_statuses(case_results)
    tags: list[ReplayEvalCoverageTag] = []
    for case in suite.cases:
        if case_statuses.get(case.case_id) == "passed":
            tags.extend(case.coverage_tags)
            tags.append(
                ReplayEvalCoverageTag(
                    step_id=case.expectation.step_id,
                    state_category="normal_progression",
                    source="replay_eval_case",
                    case_id=case.case_id,
                    reason="case passed final harness outcome assertions",
                )
            )
    return tuple(tags)


def _build_case_only_coverage_matrix(
    suite: ReplayEvalSuite,
    *,
    reason: str,
    case_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    step_ids = sorted(
        {
            tag.step_id
            for tag in _all_suite_coverage_tags(suite, case_results=case_results)
            if isinstance(tag.step_id, str) and tag.step_id
        },
        key=lambda step_id: (int(step_id[1:]) if step_id.startswith("S") and step_id[1:].isdigit() else 10_000, step_id),
    )
    steps = {
        step_id: {
            category: _new_coverage_cell(status="not_applicable", reason=reason)
            for category in HARNESS_COVERAGE_STATE_CATEGORIES
        }
        for step_id in step_ids
    }
    for tag in _all_suite_coverage_tags(suite, case_results=case_results):
        _add_coverage_source(
            steps,
            step_id=tag.step_id,
            state_category=tag.state_category,
            source=_coverage_source_dict(tag),
        )
    for tag in suite.coverage_tags:
        _add_regression_reference(
            steps,
            step_id=tag.step_id,
            state_category=tag.state_category,
            source=_coverage_source_dict(tag),
        )
    covered_count = sum(
        1
        for row in steps.values()
        for cell in row.values()
        if cell["status"] == "covered"
    )
    contract_only_count = sum(
        1
        for row in steps.values()
        for cell in row.values()
        if cell["status"] == "contract_only"
    )
    regression_reference_count = sum(
        1
        for row in steps.values()
        for cell in row.values()
        if cell["status"] == "regression_reference"
    )
    return {
        "schema_version": "harness_coverage_matrix.v1",
        "step_count": len(step_ids),
        "state_categories": list(HARNESS_COVERAGE_STATE_CATEGORIES),
        "missing_cell_count": 0,
        "missing_cells": [],
        "covered_cell_count": covered_count,
        "contract_only_cell_count": contract_only_count,
        "regression_reference_cell_count": regression_reference_count,
        "scenario_profiles": [suite.scenario_profile],
        "steps": steps,
    }


def build_harness_coverage_matrix(
    suite: ReplayEvalSuite,
    *,
    case_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not suite.pack_path.exists():
        return _build_case_only_coverage_matrix(
            suite,
            reason=f"pack_path unavailable for spec-derived coverage: {suite.pack_path}",
            case_results=case_results,
        )
    scenario_profiles = tuple(
        sorted({suite.scenario_profile, *(case.scenario_profile for case in suite.cases)})
    )
    specs_by_profile = {
        profile: load_step_harness_specs(suite.pack_path, scenario_profile=profile)
        for profile in scenario_profiles
    }
    step_specs = specs_by_profile[suite.scenario_profile]
    ordered_step_ids = sorted(
        step_specs.keys(),
        key=lambda step_id: (int(step_id[1:]) if step_id.startswith("S") and step_id[1:].isdigit() else 10_000, step_id),
    )
    steps: dict[str, dict[str, dict[str, Any]]] = {
        step_id: {
            category: _new_coverage_cell()
            for category in HARNESS_COVERAGE_STATE_CATEGORIES
        }
        for step_id in ordered_step_ids
    }

    for step_id in ordered_step_ids:
        specs_for_step = tuple(
            profile_specs[step_id]
            for profile_specs in specs_by_profile.values()
            if step_id in profile_specs
        )
        has_completion_predicates = any(spec.completion_predicates for spec in specs_for_step)
        has_telemetry_facts = any(spec.telemetry_facts for spec in specs_for_step)
        has_recent_action_facts = any(spec.recent_action_facts for spec in specs_for_step)
        has_allowed_overlay_targets = any(spec.allowed_overlay_targets for spec in specs_for_step)
        visual_step = any(spec.requires_visual_confirmation or spec.vision_facts for spec in specs_for_step)
        has_moving_settling_control = step_id in {"S20", "S21"}
        _mark_coverage_contract(
            steps,
            step_id=step_id,
            state_category="normal_progression",
            source={
                "source": "coldstart_state_matrix",
                "state_kind": "just_completed",
                "reason": "synthetic replay input exists; run the coldstart matrix test for executable coverage",
            },
        )
        _mark_coverage_contract(
            steps,
            step_id=step_id,
            state_category="omission_missing_action",
            source={
                "source": "coldstart_state_matrix",
                "state_kind": "blocked",
                "reason": "synthetic replay input exists; run the coldstart matrix test for executable coverage",
            },
        )
        if has_completion_predicates:
            _mark_coverage_contract(
                steps,
                step_id=step_id,
                state_category="completion_already_true",
                source={
                    "source": "final_evidence_consistency_contract",
                    "reason": "step declares completion predicates checked by final validator",
                },
            )
        else:
            _mark_coverage_not_applicable(
                steps,
                step_id=step_id,
                state_category="completion_already_true",
                reason="step has no declared completion predicate",
            )

        if has_telemetry_facts:
            _mark_coverage_contract(
                steps,
                step_id=step_id,
                state_category="stale_telemetry",
                source={
                    "source": "state_harness_telemetry_contract",
                    "reason": "step declares telemetry facts tracked in evidence packet",
                },
            )
        else:
            _mark_coverage_not_applicable(
                steps,
                step_id=step_id,
                state_category="stale_telemetry",
                reason="step has no telemetry facts",
            )

        if has_moving_settling_control:
            _mark_coverage_contract(
                steps,
                step_id=step_id,
                state_category="moving_settling_control",
                source={
                    "source": "refuel_probe_motion_contract",
                    "reason": "step can be in motion/settling while the refuel probe moves toward its threshold",
                },
            )
        else:
            _mark_coverage_not_applicable(
                steps,
                step_id=step_id,
                state_category="moving_settling_control",
                reason="step has no moving/settling control contract",
            )

        if has_recent_action_facts:
            _mark_coverage_contract(
                steps,
                step_id=step_id,
                state_category="recent_action_gate_conflict",
                source={
                    "source": "recent_action_fact_contract",
                    "reason": "step declares recent action facts for gate conflict handling",
                },
            )
        else:
            _mark_coverage_not_applicable(
                steps,
                step_id=step_id,
                state_category="recent_action_gate_conflict",
                reason="step has no recent-action fact contract",
            )

        if has_allowed_overlay_targets:
            _mark_coverage_contract(
                steps,
                step_id=step_id,
                state_category="wrong_target_prevention",
                source={
                    "source": "harness_action_planner_allowlist",
                    "reason": "step declares allowed overlay targets enforced by action planner",
                },
            )
        else:
            _mark_coverage_not_applicable(
                steps,
                step_id=step_id,
                state_category="wrong_target_prevention",
                reason="step does not allow overlay targets",
            )

        if visual_step:
            _mark_coverage_not_applicable(
                steps,
                step_id=step_id,
                state_category="vlm_not_required",
                reason="step requires visual confirmation or declares vision facts",
            )
            for category, source_name in (
                ("vlm_required", "step_harness_visual_contract"),
                ("vlm_unavailable", "vision_unavailable_fallback_contract"),
                ("vlm_failed", "vision_failure_trace_contract"),
            ):
                _mark_coverage_contract(
                    steps,
                    step_id=step_id,
                    state_category=category,
                    source={
                        "source": source_name,
                        "reason": "visual step is covered by replay-eval VLM status contract",
                    },
                )
        else:
            _mark_coverage_contract(
                steps,
                step_id=step_id,
                state_category="vlm_not_required",
                source={
                    "source": "step_harness_non_visual_contract",
                    "reason": "step has no visual confirmation requirement",
                },
            )
            for category in ("vlm_required", "vlm_unavailable", "vlm_failed"):
                _mark_coverage_not_applicable(
                    steps,
                    step_id=step_id,
                    state_category=category,
                    reason="step has no visual confirmation requirement",
                )

    for tag in _all_suite_coverage_tags(suite, case_results=case_results):
        _add_coverage_source(
            steps,
            step_id=tag.step_id,
            state_category=tag.state_category,
            source=_coverage_source_dict(tag),
        )
    for tag in suite.coverage_tags:
        _add_regression_reference(
            steps,
            step_id=tag.step_id,
            state_category=tag.state_category,
            source=_coverage_source_dict(tag),
        )

    missing_cells = [
        {"step_id": step_id, "state_category": category}
        for step_id, row in steps.items()
        for category, cell in row.items()
        if cell["status"] == "missing"
    ]
    covered_count = sum(
        1
        for row in steps.values()
        for cell in row.values()
        if cell["status"] == "covered"
    )
    contract_only_count = sum(
        1
        for row in steps.values()
        for cell in row.values()
        if cell["status"] == "contract_only"
    )
    regression_reference_count = sum(
        1
        for row in steps.values()
        for cell in row.values()
        if cell["status"] == "regression_reference"
    )
    return {
        "schema_version": "harness_coverage_matrix.v1",
        "step_count": len(ordered_step_ids),
        "state_categories": list(HARNESS_COVERAGE_STATE_CATEGORIES),
        "missing_cell_count": len(missing_cells),
        "missing_cells": missing_cells,
        "covered_cell_count": covered_count,
        "contract_only_cell_count": contract_only_count,
        "regression_reference_cell_count": regression_reference_count,
        "scenario_profiles": list(scenario_profiles),
        "steps": steps,
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
            "message_category": case.expectation.message_category,
            "repair_path": case.expectation.repair_path,
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
            "message_category": None,
            "repair_path": None,
            "harness_trace_present": False,
            "vlm_call_status": None,
            "vlm_call_reason": None,
        },
        "checks": {
            "step_match": False,
            "overlay_target_match": False,
            "requires_visual_confirmation_match": False,
            "vision_status_match": False,
            "sync_status_match": False,
            "sync_delta_ms_match": False,
            "frame_ids_match": False,
            "message_category_match": False,
            "repair_path_match": False,
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
                        vision_model_name="simtutor-vision",
                        tutor_text_sender=NoopTutorTextSender(),
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
        "coverage_matrix": build_harness_coverage_matrix(suite, case_results=case_results),
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
    "HARNESS_COVERAGE_STATE_CATEGORIES",
    "ReplayEvalCase",
    "ReplayEvalCoverageTag",
    "ReplayEvalExpectation",
    "ReplayEvalOracleModel",
    "ReplayEvalSuite",
    "ReplayEvalVisionConfig",
    "build_harness_coverage_matrix",
    "load_replay_eval_suite",
    "run_replay_eval_suite",
]
