from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import queue
import re
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlparse
from uuid import UUID, uuid4

import yaml

from adapters.action_executor import OverlayActionExecutor
from adapters.dcs.overlay.config import (
    build_multi_target_overlay_config_warning,
    simtutor_config_path_from_saved_games_dir,
)
from adapters.dcs_bios.bios_ui_map import BiosUiMapper
from adapters.dcs_bios.receiver import DcsBiosRawReceiver, DcsBiosReceiver
from adapters.dcs.tutor_text import DcsTutorTextSender
from adapters.evidence_refs import collect_evidence_refs_from_context, infer_evidence_type_from_ref
from adapters.knowledge_source_policy import KnowledgeSourcePolicy, KnowledgeSourcePolicyError
from adapters.knowledge_local import DEFAULT_INDEX_PATH, LocalKnowledgeAdapter, build_grounding_query
from adapters.model_stub import ModelStub
from adapters.ollama_model import OllamaModel
from adapters.openai_compat_model import OpenAICompatModel
from adapters.vision_fact_extractor import VisionFactExtractor
from adapters.pack_gates import (
    DEFAULT_SCENARIO_PROFILE,
    SUPPORTED_SCENARIO_PROFILES,
    evaluate_pack_gates,
    load_pack_gate_config,
    normalize_scenario_profile,
)
from adapters.prompting import HARNESS_LATE_VLM_CONFLICT, build_help_prompt_result
from adapters.recent_actions import (
    RecentDeltaRingBuffer,
    build_prompt_recent_deltas,
    build_recent_button_signal,
    project_recent_ui_targets,
)
from adapters.response_mapping import map_help_response_to_tutor_response
from adapters.source_chunk_refs import build_source_chunk_ref
from adapters.step_harness_specs import (
    load_step_harness_specs,
    step_fallback_profiles_from_specs,
    step_signal_profiles_from_specs,
)
from adapters.step_inference import StepInferenceResult, infer_step_id, load_pack_steps
from adapters.telemetry_pipeline import enrich_bios_observation
from adapters.vision_capture_trigger import (
    DEFAULT_VISION_CAPTURE_TRIGGER_HOST,
    DEFAULT_VISION_CAPTURE_TRIGGER_PORT,
    build_capture_request_payload,
)
from adapters.vision_frames import DEFAULT_FRAME_CHANNEL, FrameDirectoryVisionPort, build_frames_root
from adapters.vision_prompting import DEFAULT_LAYOUT_ID
from adapters.vision_sync import (
    DEFAULT_LIVE_SYNC_WINDOW_MS,
    DEFAULT_LIVE_TRIGGER_WAIT_MS,
    DEFAULT_REPLAY_SYNC_WINDOW_MS,
    BufferedVisionSession,
    HelpCycleVisionSelection,
)
from adapters.windows_global_help_trigger import DEFAULT_GLOBAL_HELP_COOLDOWN_MS, WindowsGlobalHelpTrigger
from core.constants import ENV_COLD_START_PRODUCTION
from core.env_bool import parse_env_bool
from core.event_store import JsonlEventStore
from core.help_cycle_audit import normalize_help_cycle_audit_fields
from core.help_failure import (
    VISION_CONFLICT_UNRESOLVED,
    VISION_PARSE_FAIL,
    VISION_SYNC_MISS,
    VISION_TEXT_FALLBACK,
    VISION_UNAVAILABLE,
    classify_mapping_failure,
    merge_failure_metadata,
    overlay_rejection_payload,
)
from core.harness_validation import (
    HarnessActionHintFactRule,
    HarnessCompletionAdvance,
    HarnessTextGuidanceRule,
    plan_harness_action,
)
from core.help_orchestrator import (
    HelpCycleDecisionResult,
    HelpCycleOrchestrator,
    PreparedHelpCycle,
)
from core.security import (
    redact_sensitive_text,
    sanitize_help_response_for_log,
    sanitize_public_model_text,
    validate_model_base_url_security,
)
from core.step_signal_metadata import (
    STEP_EVIDENCE_REQUIREMENT_VALUES,
    STEP_OBSERVABILITY_VALUES,
    compute_requires_visual_confirmation,
    normalize_observability_status,
)
from core.step_hint import hint_has_hard_blocker

_MISSING_CONDITION_VAR_RE = re.compile(r"(?:payload\.)?vars\.([A-Za-z0-9_]+)")
_MISSING_CONDITION_TARGET_HINTS: dict[str, tuple[str, ...]] = {
    "battery_on": ("battery_switch",),
    "bleed_air_cycle_complete": ("bleed_air_knob",),
    "bleed_air_norm": ("bleed_air_knob",),
    "engine_crank_left": ("eng_crank_switch",),
    "engine_crank_right": ("eng_crank_switch",),
    "engine_crank_right_complete": ("eng_crank_switch",),
    "fire_test_a_complete": ("fire_test_switch",),
    "fire_test_b_complete": ("fire_test_switch",),
    "fire_test_complete": ("fire_test_switch",),
    "hud_on": ("hud_symbology_brightness_knob",),
    "left_ddi_on": ("left_mdi_brightness_selector",),
    "lights_test_complete": ("lights_test_button",),
    "mpcd_on": ("ampcd_off_brightness_knob",),
    "r_gen_on": ("generator_right_switch",),
    "right_ddi_on": ("right_mdi_brightness_selector",),
    "rpm_r": ("eng_crank_switch", "throttle_quadrant_reference"),
    "rpm_r_gte_25": ("eng_crank_switch", "throttle_quadrant_reference"),
    "rpm_r_gte_60": (),
    "throttle_r_idle_complete": ("throttle_quadrant_reference",),
}
_S08_POWER_SEQUENCE: tuple[tuple[str, str, str], ...] = (
    (
        "left_ddi_on",
        "left_mdi_brightness_selector",
        "Left DDI is still OFF. Set the left DDI brightness selector to NIGHT/DAY; the screen may take a moment to illuminate.",
    ),
    (
        "right_ddi_on",
        "right_mdi_brightness_selector",
        "Right DDI is still OFF. Set the right DDI brightness selector to NIGHT/DAY; the screen may take a moment to illuminate.",
    ),
    (
        "mpcd_on",
        "ampcd_off_brightness_knob",
        "Both DDIs are powered. Increase the AMPCD brightness knob next.",
    ),
    (
        "hud_on",
        "hud_symbology_brightness_knob",
        "DDIs and AMPCD are powered. Increase HUD symbology brightness next.",
    ),
)
from core.types import Event, Observation, TutorRequest, TutorResponse
from core.vision_facts import (
    VisionFactsConfigError,
    build_vision_fact_summary,
    load_vision_facts_config,
    merge_vision_fact_observation,
    prune_expired_facts,
    snapshot_to_list,
)
from core.evidence_packet import build_evidence_packet, build_step_candidates
from core.vars import VarResolver
from ports.knowledge_port import KnowledgePort, KnowledgeRetrieveWithMetaPort
from simtutor.cli_parsing import parse_env_int, parse_non_negative_int_arg

VISION_NOT_REQUIRED = "vision_not_required"
DEFAULT_VISION_PRIORITY_STEP_IDS: tuple[str, ...] = ("S08", "S15", "S18", "S19")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_pack_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "pack.yaml"


def _default_ui_map_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "ui_map.yaml"


def _default_telemetry_map_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "telemetry_map.yaml"


def _default_bios_to_ui_path() -> Path:
    return _repo_root() / "packs" / "fa18c_startup" / "bios_to_ui.yaml"


def _default_knowledge_index_path() -> Path:
    return _repo_root() / DEFAULT_INDEX_PATH


def _default_knowledge_source_policy_path() -> Path:
    return _repo_root() / "knowledge_source_policy.yaml"


def _normalize_fs_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _path_like_to_uri(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    if not text:
        return None
    windows_path = PureWindowsPath(text)
    if windows_path.drive and windows_path.is_absolute():
        return windows_path.as_uri()
    parsed = urlparse(text)
    if parsed.scheme:
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve().as_uri()
    return None


def _normalize_uuid_text(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    if not text:
        return None
    try:
        return str(UUID(text))
    except ValueError:
        return None


def _build_vision_event_attachments(observation: Any) -> list[str]:
    attachments: list[str] = []
    for candidate in (
        getattr(observation, "image_uri", None),
        getattr(observation, "source_image_path", None),
    ):
        uri = _path_like_to_uri(candidate)
        if uri is None or uri in attachments:
            continue
        attachments.append(uri)
    return attachments


def _emit_vision_observation_event(
    *,
    observation: Any,
    event_sink: Callable[[Event], None] | None,
    fallback_session_id: str | None,
) -> None:
    if event_sink is None:
        return
    payload = observation.to_dict()
    observation_id = _normalize_uuid_text(payload.get("observation_ref"))
    if observation_id is None:
        observation_id = str(uuid4())
    payload["observation_ref"] = observation_id
    attachments = _build_vision_event_attachments(observation)
    wrapped = Observation(
        observation_id=observation_id,
        timestamp=(
            payload.get("timestamp")
            if isinstance(payload.get("timestamp"), str)
            else datetime.now(tz=timezone.utc).isoformat()
        ),
        source=payload.get("source") if isinstance(payload.get("source"), str) else "vision",
        payload=payload,
        attachments=attachments,
        metadata={
            "observation_kind": "vision",
            "frame_id": payload.get("frame_id"),
            "layout_id": payload.get("layout_id"),
            "channel": payload.get("channel"),
        },
    )
    capture_wall_ms = payload.get("capture_wall_ms")
    t_wall = None
    if isinstance(capture_wall_ms, int) and capture_wall_ms >= 0:
        t_wall = capture_wall_ms / 1000.0
    event_sink(
        Event(
            kind="observation",
            payload=wrapped.to_dict(),
            related_id=wrapped.observation_id,
            t_wall=t_wall,
            session_id=fallback_session_id,
            vision_refs=[
                payload["frame_id"],
            ]
            if isinstance(payload.get("frame_id"), str) and payload.get("frame_id")
            else [],
            metadata={
                "observation_kind": "vision",
            },
        )
    )


def _emit_vision_fact_observation_event(
    *,
    observation: Any,
    event_sink: Callable[[Event], None] | None,
    fallback_session_id: str | None,
) -> None:
    if event_sink is None:
        return
    payload = observation.to_dict()
    observation_id = _normalize_uuid_text(payload.get("observation_id"))
    if observation_id is None:
        observation_id = str(uuid4())
    payload["observation_id"] = observation_id
    wrapped = Observation(
        observation_id=observation_id,
        timestamp=(
            payload.get("timestamp")
            if isinstance(payload.get("timestamp"), str)
            else datetime.now(tz=timezone.utc).isoformat()
        ),
        source=payload.get("source") if isinstance(payload.get("source"), str) else "vision_fact",
        payload=payload,
        metadata={
            "observation_kind": "vision_fact",
            "frame_ids": [
                item for item in payload.get("frame_ids", [])
                if isinstance(item, str) and item
            ],
        },
    )
    trigger_wall_ms = payload.get("trigger_wall_ms")
    t_wall = None
    if isinstance(trigger_wall_ms, int) and trigger_wall_ms >= 0:
        t_wall = trigger_wall_ms / 1000.0
    event_sink(
        Event(
            kind="observation",
            payload=wrapped.to_dict(),
            related_id=wrapped.observation_id,
            t_wall=t_wall,
            session_id=fallback_session_id,
            vision_refs=[
                item for item in payload.get("frame_ids", [])
                if isinstance(item, str) and item
            ],
            metadata={
                "observation_kind": "vision_fact",
            },
        )
    )


def _build_vision_fact_extractor_from_model(
    *,
    model: Any,
    lang: str,
    pack_path: str | Path | None = None,
    vision_model_name: str | None = None,
) -> VisionFactExtractor | None:
    if not isinstance(model, OpenAICompatModel):
        return None
    if not getattr(model, "enable_multimodal", False):
        return None
    effective_vision_name = vision_model_name or model.model_name
    shared_client = getattr(model, "http_client", None)
    try:
        return VisionFactExtractor(
            model_name=effective_vision_name,
            base_url=model.base_url,
            timeout_s=model.timeout_s,
            api_key=model.api_key,
            allowed_local_image_roots=[str(path) for path in model.allowed_local_image_roots],
            max_local_image_bytes=model.max_local_image_bytes,
            lang=lang,
            log_raw_llm_text=getattr(model, "log_raw_llm_text", False),
            print_model_io=getattr(model, "print_model_io", False),
            pack_path=pack_path,
            client=shared_client,
        )
    except (FileNotFoundError, OSError, ValueError, VisionFactsConfigError):
        return None


def _resolve_vision_fact_config(
    *,
    extractor: Any | None,
    pack_path: Path,
) -> dict[str, Any]:
    extractor_config = getattr(extractor, "config", None)
    if isinstance(extractor_config, Mapping):
        return dict(extractor_config)
    try:
        return load_vision_facts_config(pack_path=pack_path)
    except (FileNotFoundError, OSError, ValueError, VisionFactsConfigError):
        return {
            "schema_version": "v1",
            "layout_id": None,
            "facts_by_id": {},
            "step_bindings": {},
        }


class ObservationSource(Protocol):
    def get_observation(self) -> Observation | None:
        ...


class ActionExecutorLike(Protocol):
    def execute_actions(self, actions: Sequence[Mapping[str, Any] | Any]) -> Any:
        ...

    def close(self) -> None:
        ...


@dataclass
class _StaticHelpAdjudicator:
    response: TutorResponse

    def adjudicate(self, observation: Observation, request: TutorRequest) -> TutorResponse:
        return self.response


@dataclass
class _StaticDecisionValidator:
    fallback_overlay_used: bool
    fallback_overlay_reason: str

    def validate_and_repair(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> HelpCycleDecisionResult:
        return HelpCycleDecisionResult(
            response=response,
            fallback_overlay_used=self.fallback_overlay_used,
            fallback_overlay_reason=self.fallback_overlay_reason,
        )


@dataclass
class _CallableHelpAdjudicator:
    adjudicate_response: Callable[[Observation, TutorRequest], TutorResponse]

    def adjudicate(self, observation: Observation, request: TutorRequest) -> TutorResponse:
        return self.adjudicate_response(observation, request)


@dataclass
class _CallableDecisionValidator:
    validate_response: Callable[[TutorResponse, TutorRequest], HelpCycleDecisionResult]

    def validate_and_repair(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> HelpCycleDecisionResult:
        return self.validate_response(response, request)


@dataclass
class _CallableActionPlanner:
    execute_actions: Callable[[Sequence[Mapping[str, Any] | Any]], Mapping[str, Any]]

    def execute(self, actions: Sequence[Mapping[str, Any] | Any]) -> Mapping[str, Any]:
        return self.execute_actions(actions)


class TutorTextSenderLike(Protocol):
    def send_text(
        self,
        text: str,
        *,
        display_time_s: float = 12.0,
        clear_view: bool = False,
        expect_ack: bool = True,
        cmd_id: str | None = None,
    ) -> Any:
        ...

    def close(self) -> None:
        ...


class HelpTriggerLike(Protocol):
    def poll(self) -> bool:
        ...


def _basename_from_path_like(path_text: str) -> str:
    if len(path_text) >= 3 and path_text[1] == ":" and path_text[2] in ("\\", "/"):
        name = PureWindowsPath(path_text).name
    elif path_text.startswith("\\\\"):
        name = PureWindowsPath(path_text).name
    else:
        name = Path(path_text).name
    return name or "<path>"


def _path_text_variants(path_like: str | Path) -> list[str]:
    raw = str(path_like)
    variants: set[str] = set()
    if raw:
        variants.add(raw)
    try:
        resolved = str(Path(path_like).expanduser().resolve())
        if resolved:
            variants.add(resolved)
    except OSError:
        pass

    expanded: set[str] = set(variants)
    for item in variants:
        expanded.add(item.replace("\\", "/"))
        expanded.add(item.replace("/", "\\"))
    return [item for item in expanded if item]


def _sanitize_policy_error_for_user(
    message: str,
    *,
    path_hints: Sequence[str | Path] = (),
) -> str:
    if not isinstance(message, str) or not message.strip():
        return "invalid policy configuration"

    sanitized = message
    for hint in path_hints:
        for variant in _path_text_variants(hint):
            sanitized = sanitized.replace(variant, _basename_from_path_like(variant))
    return sanitized


def _summarize_rag_snippets_for_event(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        summarized: dict[str, Any] = {}
        for key in ("snippet_id", "id", "doc_id", "section", "page_or_heading", "page", "chunk_id", "score"):
            value = item.get(key)
            if value is None:
                continue
            summarized[key] = value
        if summarized:
            out.append(summarized)
    return out


def _sanitize_deterministic_hint_for_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    for key in (
        "inferred_step_id",
        "overlay_step_id",
        "observability",
        "observability_status",
        "requires_visual_confirmation",
        "scenario_profile",
    ):
        value = raw.get(key)
        if value is not None:
            sanitized[key] = value
    missing_conditions = raw.get("missing_conditions")
    if isinstance(missing_conditions, (list, tuple)):
        sanitized["missing_conditions_count"] = len([item for item in missing_conditions if isinstance(item, str) and item])
    gate_blockers = raw.get("gate_blockers")
    if isinstance(gate_blockers, (list, tuple)):
        sanitized["gate_blocker_count"] = len(gate_blockers)
    recent_ui_targets = raw.get("recent_ui_targets")
    if isinstance(recent_ui_targets, list):
        sanitized["recent_ui_targets"] = [
            item for item in recent_ui_targets if isinstance(item, str) and item
        ][:8]
    step_ui_targets = raw.get("step_ui_targets")
    if isinstance(step_ui_targets, list):
        sanitized["step_ui_targets"] = [
            item for item in step_ui_targets if isinstance(item, str) and item
        ][:8]
    visual_action_hint = raw.get("visual_action_hint")
    if isinstance(visual_action_hint, Mapping):
        target = visual_action_hint.get("target")
        if isinstance(target, str) and target:
            sanitized["visual_action_hint"] = {"target": target}
    action_hint = raw.get("action_hint")
    if isinstance(action_hint, Mapping):
        target = action_hint.get("target")
        if isinstance(target, str) and target:
            sanitized["action_hint"] = {"target": target}
    return sanitized


_SAFE_REQUEST_METADATA_FIELDS: tuple[str, ...] = (
    "prompt_hash",
    "prompt_tokens_est",
    "prompt_trimmed",
    "grounding_missing",
    "grounding_reason",
    "grounding_missing_requested",
    "grounding_reason_requested",
    "grounding_error_type",
    "grounding_snippet_ids",
    "source_chunk_refs",
    "grounding_cache_hit",
    "grounding_policy_id",
    "grounding_policy_version",
    "grounding_policy_filtered_out_count",
    "scenario_profile",
    "vision_status",
    "vision_frame_ids",
    "vision_fact_status",
    "vision_fact_seen_ids",
    "state_harness_conflicts",
    "state_harness_telemetry_status",
    "evidence_packet_summary",
    "help_cycle_id",
    "generation_mode",
    "vision_used",
    "frame_id",
    "sync_delta_ms",
    "vision_fact_summary",
    "fused_step_id",
    "fused_missing_conditions",
    "vision_fallback_reason",
    "layout_id",
)


def _sanitize_request_metadata_for_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}

    sanitized: dict[str, Any] = {}
    for key in _SAFE_REQUEST_METADATA_FIELDS:
        if key not in raw:
            continue
        value = raw.get(key)
        if key == "evidence_packet_summary":
            sanitized[key] = _sanitize_evidence_packet_summary_for_event(value)
            continue
        if isinstance(value, Mapping):
            sanitized[key] = dict(value)
        elif isinstance(value, tuple):
            sanitized[key] = list(value)
        elif isinstance(value, list):
            sanitized[key] = list(value)
        else:
            sanitized[key] = value
    return sanitized


def _copy_event_field(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return list(value)
    return value


def _sanitize_vision_context_for_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}

    sanitized: dict[str, Any] = {}
    for key in (
        "status",
        "observation_ref",
        "observation_seq",
        "observation_t_wall_s",
        "observation_t_wall_ms",
        "trigger_wall_ms",
        "sync_window_ms",
        "vision_used",
        "frame_id",
        "sync_status",
        "sync_delta_ms",
        "frame_stale",
        "frame_ids",
        "sync_miss_reason",
    ):
        if key not in raw:
            continue
        value = raw.get(key)
        if isinstance(value, list):
            sanitized[key] = list(value)
        else:
            sanitized[key] = value
    return sanitized


_SAFE_PROMPT_BUILD_FIELDS: tuple[str, ...] = (
    "max_overlay_targets",
    "max_prompt_chars",
    "max_prompt_tokens_est",
    "prompt_chars",
    "prompt_tokens_est",
    "prompt_trimmed",
    "trim_reasons",
    "delta_summary_top_k",
    "delta_summary_items",
    "evidence_refs_count",
    "allowed_evidence_refs",
    "preferred_overlay_target",
    "rag_snippet_count",
    "rag_snippet_ids",
    "grounding_applied",
    "grounding_missing_requested",
    "grounding_missing",
    "grounding_reason",
    "vision_fact_status",
    "vision_fact_seen_ids",
    "state_harness_conflicts",
    "state_harness_telemetry_status",
    "state_harness_visual_candidate_steps",
)


def _sanitize_prompt_build_for_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}

    sanitized: dict[str, Any] = {}
    for key in _SAFE_PROMPT_BUILD_FIELDS:
        if key not in raw:
            continue
        value = raw.get(key)
        if isinstance(value, Mapping):
            sanitized[key] = dict(value)
        elif isinstance(value, tuple):
            sanitized[key] = list(value)
        elif isinstance(value, list):
            sanitized[key] = list(value)
        else:
            sanitized[key] = value
    return sanitized


def _sanitize_vision_fact_summary_for_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}

    sanitized: dict[str, Any] = {}
    for key in (
        "status",
        "frame_ids",
        "fresh_fact_ids",
        "seen_fact_ids",
        "uncertain_fact_ids",
        "not_seen_fact_ids",
        "summary_text",
    ):
        if key not in raw:
            continue
        value = raw.get(key)
        if isinstance(value, list):
            sanitized[key] = list(value)
        else:
            sanitized[key] = value
    return sanitized


def _sanitize_state_harness_for_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    conflicts = raw.get("conflicts")
    if isinstance(conflicts, list):
        sanitized["conflicts"] = [item for item in conflicts if isinstance(item, str)]
    for key in ("telemetry_evidence", "vision_evidence", "deterministic_candidate"):
        value = raw.get(key)
        if isinstance(value, Mapping):
            sanitized[key] = dict(value)
    telemetry_window_digest = _sanitize_telemetry_window_digest_for_event(raw.get("telemetry_window_digest"))
    if telemetry_window_digest:
        sanitized["telemetry_window_digest"] = telemetry_window_digest
    return sanitized


def _sanitize_digest_scalar(raw: Any) -> Any:
    if raw is None or isinstance(raw, bool) or isinstance(raw, (int, float)):
        return raw
    if isinstance(raw, str):
        return raw if len(raw) <= 80 else raw[:80] + "..."
    return None


def _sanitize_telemetry_window_digest_for_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    for key in ("window_duration_s", "latest_t_wall"):
        value = raw.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = None
    for key in ("frame_count", "latest_seq"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = None
    for key in ("stable_true_vars", "stable_false_vars", "unknown_or_missing_vars", "contradictions"):
        values = raw.get(key)
        if isinstance(values, list):
            sanitized[key] = [item for item in values if isinstance(item, str) and len(item) <= 120][:16]
    for key in ("changed_vars", "first_frame_only_values", "last_seen_true"):
        values = raw.get(key)
        if not isinstance(values, list):
            continue
        items: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            entry: dict[str, Any] = {}
            for field in (
                "var",
                "first_value",
                "last_value",
                "value",
                "transition_count",
                "latest_transition_age_s",
                "seq",
                "age_s",
            ):
                if field not in item:
                    continue
                value = _sanitize_digest_scalar(item.get(field))
                if value is not None:
                    entry[field] = value
            if entry:
                items.append(entry)
            if len(items) >= 8:
                break
        sanitized[key] = items
    return sanitized


def _telemetry_window_frames_from_var_snapshots(frames: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in frames:
        if not isinstance(item, Mapping):
            continue
        delta = item.get("delta")
        if not isinstance(delta, Mapping):
            continue
        out.append(
            {
                "seq": item.get("seq"),
                "t_wall": item.get("t_wall"),
                "vars": dict(delta),
                "vars_is_full_snapshot": True,
            }
        )
    return out


_TELEMETRY_WINDOW_SIGNATURE_VARS = {
    "battery_on",
    "power_available",
    "left_ddi_on",
    "right_ddi_on",
    "mpcd_on",
    "hud_on",
    "fire_test_a_complete",
    "fire_test_b_complete",
    "fire_test_complete",
    "fcs_bit_switch_up",
    "ext_refuel_probe_value",
    "launch_bar_switch_value",
    "hook_handle_value",
    "pitot_heat_on",
    "flap_auto",
}


def _telemetry_window_signature(state_harness: Mapping[str, Any]) -> dict[str, Any]:
    raw = state_harness.get("telemetry_window_digest")
    digest = raw if isinstance(raw, Mapping) else {}
    changed: list[dict[str, Any]] = []
    for item in digest.get("changed_vars", []):
        if not isinstance(item, Mapping):
            continue
        var_name = item.get("var")
        if not isinstance(var_name, str) or not var_name:
            continue
        if var_name not in _TELEMETRY_WINDOW_SIGNATURE_VARS:
            continue
        changed.append(
            {
                "var": var_name,
                "first_value": _sanitize_digest_scalar(item.get("first_value")),
                "last_value": _sanitize_digest_scalar(item.get("last_value")),
                "transition_count": item.get("transition_count"),
            }
        )
    first_frame_only: list[dict[str, Any]] = []
    for item in digest.get("first_frame_only_values", []):
        if not isinstance(item, Mapping):
            continue
        var_name = item.get("var")
        if not isinstance(var_name, str) or not var_name:
            continue
        if var_name not in _TELEMETRY_WINDOW_SIGNATURE_VARS:
            continue
        first_frame_only.append(
            {
                "var": var_name,
                "value": _sanitize_digest_scalar(item.get("value")),
            }
        )
    contradictions = [
        item for item in digest.get("contradictions", []) if isinstance(item, str)
    ][:8]
    if not changed:
        contradictions = [
            item
            for item in contradictions
            if item != "recent telemetry transition conflicts with current blocked gate"
        ]
    return {
        "changed_vars": changed[:12],
        "first_frame_only_values": first_frame_only[:8],
        "contradictions": contradictions,
    }


def _sanitize_evidence_packet_summary_for_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    for key in ("telemetry_status", "vision_status"):
        value = raw.get(key)
        if isinstance(value, str) and len(value) <= 80:
            sanitized[key] = value
    for key in (
        "telemetry_missing_source_count",
        "vision_seen_count",
        "vision_fresh_count",
        "blocked_gate_count",
        "recent_action_count",
    ):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            sanitized[key] = value
    conflicts = raw.get("conflicts")
    if isinstance(conflicts, list):
        sanitized["conflicts"] = [
            item for item in conflicts if isinstance(item, str) and len(item) <= 120
        ][:8]
    telemetry_window_digest = raw.get("telemetry_window_digest")
    if isinstance(telemetry_window_digest, Mapping):
        sanitized["telemetry_window_digest"] = {
            key: value
            for key, value in telemetry_window_digest.items()
            if key in {"frame_count", "latest_seq", "changed_var_count", "contradiction_count"}
            and (isinstance(value, int) or value is None)
            and not isinstance(value, bool)
        }
    return sanitized


def _sanitize_request_context_for_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}

    sanitized: dict[str, Any] = {}
    for key in ("scenario_profile", "grounding_missing", "grounding_reason", "delta_dropped_count"):
        if key in raw:
            sanitized[key] = _copy_event_field(raw.get(key))
    if "grounding_query" in raw:
        sanitized["grounding_query"] = "[REDACTED_GROUNDING_QUERY]"
    if "rag_topk" in raw:
        sanitized["rag_topk"] = _summarize_rag_snippets_for_event(raw.get("rag_topk"))
    if "deterministic_step_hint" in raw:
        sanitized["deterministic_step_hint"] = _sanitize_deterministic_hint_for_event(
            raw.get("deterministic_step_hint")
        )
    if "state_harness" in raw:
        sanitized["state_harness"] = _sanitize_state_harness_for_event(raw.get("state_harness"))
    if "evidence_packet_summary" in raw:
        sanitized["evidence_packet_summary"] = _sanitize_evidence_packet_summary_for_event(
            raw.get("evidence_packet_summary")
        )
    if "vision" in raw:
        sanitized["vision"] = _sanitize_vision_context_for_event(raw.get("vision"))
    if "vision_facts" in raw:
        vision_facts = raw.get("vision_facts")
        if isinstance(vision_facts, list):
            sanitized["vision_facts"] = [
                {
                    "fact_id": item.get("fact_id"),
                    "status": item.get("status"),
                    "frame_ids": item.get("frame_ids"),
                }
                for item in vision_facts
                if isinstance(item, Mapping)
            ]
        else:
            sanitized["vision_facts"] = []
    if "vision_fact_summary" in raw:
        sanitized["vision_fact_summary"] = _sanitize_vision_fact_summary_for_event(raw.get("vision_fact_summary"))
    return sanitized


def _sanitize_request_payload_for_event(request: TutorRequest) -> dict[str, Any]:
    raw_message = request.message
    if raw_message is None:
        sanitized_message = None
    elif raw_message == "":
        sanitized_message = ""
    else:
        sanitized_message = "[REDACTED_USER_MESSAGE]"

    sanitized_context = _sanitize_request_context_for_event(request.context)
    return {
        "request_id": request.request_id,
        "timestamp": request.timestamp,
        "actor": request.actor,
        "intent": request.intent,
        "version": request.version,
        "message": sanitized_message,
        "observation_ref": request.observation_ref,
        "context": sanitized_context,
        "metadata": _sanitize_request_metadata_for_event(request.metadata),
    }


def _sanitize_response_payload_for_event(response: TutorResponse, *, lang: str) -> dict[str, Any]:
    sanitized_message = sanitize_public_model_text(response.message, lang=lang)
    sanitized_explanations = [
        sanitize_public_model_text(item, lang=lang) for item in response.explanations if isinstance(item, str)
    ]
    sanitized_metadata = dict(response.metadata) if isinstance(response.metadata, Mapping) else {}
    for sensitive_key in ("raw_llm_text", "raw_llm_text_attempts"):
        sanitized_metadata.pop(sensitive_key, None)
    error_value = sanitized_metadata.get("error")
    if isinstance(error_value, str):
        sanitized_metadata["error"] = redact_sensitive_text(error_value)
    help_response = sanitized_metadata.get("help_response")
    if isinstance(help_response, Mapping):
        sanitized_metadata["help_response"] = sanitize_help_response_for_log(help_response, lang=lang)
    model_raw_help_response = sanitized_metadata.get("model_raw_help_response")
    if isinstance(model_raw_help_response, Mapping):
        sanitized_metadata["model_raw_help_response"] = sanitize_help_response_for_log(
            model_raw_help_response,
            lang=lang,
        )
    final_public_response = sanitized_metadata.get("final_public_response")
    if isinstance(final_public_response, Mapping):
        sanitized_final_public_response = dict(final_public_response)
        message = sanitized_final_public_response.get("message")
        if isinstance(message, str):
            sanitized_final_public_response["message"] = sanitize_public_model_text(message, lang=lang)
        explanations = sanitized_final_public_response.get("explanations")
        if isinstance(explanations, list):
            sanitized_final_public_response["explanations"] = [
                sanitize_public_model_text(item, lang=lang)
                for item in explanations
                if isinstance(item, str)
            ]
        sanitized_metadata["final_public_response"] = sanitized_final_public_response
    prompt_build = sanitized_metadata.get("prompt_build")
    if isinstance(prompt_build, Mapping):
        sanitized_metadata["prompt_build"] = _sanitize_prompt_build_for_event(prompt_build)
    sanitized_actions = [
        dict(action) for action in response.actions if isinstance(action, Mapping)
    ]
    return {
        "response_id": response.response_id,
        "timestamp": response.timestamp,
        "actor": response.actor,
        "status": response.status,
        "version": response.version,
        "in_reply_to": response.in_reply_to,
        "message": sanitized_message,
        "actions": sanitized_actions,
        "explanations": sanitized_explanations,
        "metadata": {key: _copy_event_field(value) for key, value in sanitized_metadata.items()},
    }


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a YAML mapping: {path}")
    return data


def _load_step_ids(pack_path: Path) -> list[str]:
    steps = load_pack_steps(pack_path)
    out: list[str] = []
    seen: set[str] = set()
    for step in steps:
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id:
            continue
        if step_id in seen:
            continue
        seen.add(step_id)
        out.append(step_id)
    return out


def _load_step_signal_profiles(pack_path: Path) -> dict[str, dict[str, Any]]:
    pack = _load_yaml_mapping(pack_path, "pack.yaml")
    steps = pack.get("steps")
    if not isinstance(steps, list):
        return {}

    profiles: dict[str, dict[str, Any]] = {}
    for step_idx, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"pack.steps[{step_idx}] must be a mapping: {pack_path}")
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"pack.steps[{step_idx}].id must be non-empty string: {pack_path}")

        profile: dict[str, Any] = {}

        observability_raw = step.get("observability")
        if observability_raw is not None:
            observability = normalize_observability_status(observability_raw)
            if observability is None:
                allowed = ", ".join(sorted(STEP_OBSERVABILITY_VALUES))
                raise ValueError(
                    f"pack.steps[{step_idx}].observability must be one of {{{allowed}}}: {pack_path}"
                )
            profile["observability"] = observability
            profile["observability_status"] = observability

        evidence_requirements_raw = step.get("evidence_requirements")
        if evidence_requirements_raw is not None:
            if not isinstance(evidence_requirements_raw, list):
                raise ValueError(f"pack.steps[{step_idx}].evidence_requirements must be a list: {pack_path}")
            evidence_requirements: list[str] = []
            seen: set[str] = set()
            for req_idx, req in enumerate(evidence_requirements_raw):
                if not isinstance(req, str) or not req:
                    raise ValueError(
                        f"pack.steps[{step_idx}].evidence_requirements[{req_idx}] must be non-empty string: "
                        f"{pack_path}"
                    )
                if req not in STEP_EVIDENCE_REQUIREMENT_VALUES:
                    allowed = ", ".join(sorted(STEP_EVIDENCE_REQUIREMENT_VALUES))
                    raise ValueError(
                        f"pack.steps[{step_idx}].evidence_requirements[{req_idx}] must be one of "
                        f"{{{allowed}}}: {pack_path}"
                    )
                if req in seen:
                    continue
                seen.add(req)
                evidence_requirements.append(req)
            profile["evidence_requirements"] = evidence_requirements

        ui_targets_raw = step.get("ui_targets")
        if ui_targets_raw is not None:
            if not isinstance(ui_targets_raw, list):
                raise ValueError(f"pack.steps[{step_idx}].ui_targets must be a list: {pack_path}")
            ui_targets: list[str] = []
            seen_targets: set[str] = set()
            for target_idx, target in enumerate(ui_targets_raw):
                if not isinstance(target, str) or not target:
                    raise ValueError(
                        f"pack.steps[{step_idx}].ui_targets[{target_idx}] must be non-empty string: {pack_path}"
                    )
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                ui_targets.append(target)
            profile["ui_targets"] = ui_targets

        overlay_enabled_raw = step.get("overlay_enabled")
        if overlay_enabled_raw is not None:
            if not isinstance(overlay_enabled_raw, bool):
                raise ValueError(f"pack.steps[{step_idx}].overlay_enabled must be a bool: {pack_path}")
            profile["overlay_enabled"] = overlay_enabled_raw

        if profile:
            observability_value = profile.get("observability")
            evidence_requirements_value = profile.get("evidence_requirements", [])
            requires_visual_confirmation = compute_requires_visual_confirmation(
                observability_value if isinstance(observability_value, str) else None,
                evidence_requirements_value if isinstance(evidence_requirements_value, list) else [],
            )
            profile["requires_visual_confirmation"] = bool(requires_visual_confirmation)
            profiles[step_id] = profile

    return profiles


def _load_overlay_allowlist(pack_path: Path, ui_map_path: Path) -> list[str]:
    ui_map = _load_yaml_mapping(ui_map_path, "ui_map.yaml")
    cockpit_elements = ui_map.get("cockpit_elements")
    if not isinstance(cockpit_elements, Mapping):
        raise ValueError(f"ui_map.yaml missing cockpit_elements mapping: {ui_map_path}")
    base = {key for key in cockpit_elements.keys() if isinstance(key, str) and key}

    pack = _load_yaml_mapping(pack_path, "pack.yaml")
    pack_targets = pack.get("ui_targets")
    if pack_targets is None:
        step_targets: set[str] = set()
        steps = pack.get("steps")
        if isinstance(steps, list):
            for step_idx, step in enumerate(steps):
                if not isinstance(step, Mapping):
                    raise ValueError(f"pack.steps[{step_idx}] must be a mapping: {pack_path}")
                ui_targets = step.get("ui_targets")
                if ui_targets is None:
                    continue
                if not isinstance(ui_targets, list):
                    raise ValueError(f"pack.steps[{step_idx}].ui_targets must be a list: {pack_path}")
                for target_idx, target in enumerate(ui_targets):
                    if not isinstance(target, str) or not target:
                        raise ValueError(
                            f"pack.steps[{step_idx}].ui_targets[{target_idx}] must be non-empty string: {pack_path}"
                        )
                    if target not in base:
                        raise ValueError(
                            f"pack.steps[{step_idx}].ui_targets[{target_idx}]={target!r} not found in ui_map: "
                            f"{pack_path}"
                        )
                    step_targets.add(target)
        if step_targets:
            return sorted(step_targets)
        return sorted(base)
    if not isinstance(pack_targets, list):
        raise ValueError(f"pack.ui_targets must be a list: {pack_path}")

    narrowed: set[str] = set()
    invalid_targets: list[str] = []
    for idx, target in enumerate(pack_targets):
        if not isinstance(target, str) or not target:
            raise ValueError(f"pack.ui_targets[{idx}] must be non-empty string: {pack_path}")
        if target not in base:
            invalid_targets.append(f"pack.ui_targets[{idx}]={target!r}")
            continue
        narrowed.add(target)
    if invalid_targets:
        joined = ", ".join(invalid_targets)
        raise ValueError(f"{joined} not found in ui_map: {pack_path}")
    if not narrowed:
        raise ValueError(
            "pack.ui_targets narrows overlay allowlist to zero valid targets; "
            f"check ui_map consistency: {pack_path}"
        )
    return sorted(narrowed)


def _load_pack_title(pack_path: Path) -> str:
    pack = _load_yaml_mapping(pack_path, "pack.yaml")
    title = pack.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    pack_id = pack.get("pack_id")
    if isinstance(pack_id, str) and pack_id.strip():
        return pack_id.strip()
    return pack_path.stem


def _load_vision_priority_steps(pack_path: Path) -> tuple[str, ...]:
    try:
        pack = _load_yaml_mapping(pack_path, "pack.yaml")
    except (FileNotFoundError, OSError, ValueError, yaml.YAMLError):
        return DEFAULT_VISION_PRIORITY_STEP_IDS
    metadata = pack.get("metadata")
    raw_steps = metadata.get("vision_priority_steps") if isinstance(metadata, Mapping) else None
    if not isinstance(raw_steps, list):
        return DEFAULT_VISION_PRIORITY_STEP_IDS
    steps: list[str] = []
    for item in raw_steps:
        if isinstance(item, str) and item and item not in steps:
            steps.append(item)
    return tuple(steps) if steps else DEFAULT_VISION_PRIORITY_STEP_IDS


def _stable_hash_json(data: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_finite_float(value: Any) -> float | None:
    normalized = _coerce_float(value)
    if normalized is None or not math.isfinite(normalized):
        return None
    return normalized


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _is_pressed_delta_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _json_safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return str(value)


def _normalize_vision_mode(value: str) -> str:
    if value not in {"live", "replay"}:
        raise ValueError("vision_mode must be 'live' or 'replay'")
    return value


def _normalize_path_segment(value: Any, *, flag_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{flag_name} must be a non-empty path segment")
    normalized = value.strip()
    path = Path(normalized)
    if (
        path.is_absolute()
        or path.drive
        or path.anchor
        or ":" in normalized
        or len(path.parts) != 1
        or path.parts[0] in {"", ".", ".."}
    ):
        raise ValueError(f"{flag_name} must be a simple path segment without path separators")
    if "\\" in normalized or "/" in normalized:
        raise ValueError(f"{flag_name} must be a simple path segment without path separators")
    return normalized


def _normalize_knowledge_snippet(raw: Mapping[str, Any], fallback_idx: int) -> dict[str, Any]:
    snippet_id_raw = raw.get("snippet_id")
    if not isinstance(snippet_id_raw, str) or not snippet_id_raw:
        snippet_id_raw = raw.get("id")
    snippet_id = (
        snippet_id_raw
        if isinstance(snippet_id_raw, str) and snippet_id_raw
        else f"snippet_{fallback_idx}"
    )

    snippet_raw = raw.get("snippet")
    if snippet_raw is None:
        snippet_raw = raw.get("text")
    snippet = str(_json_safe_scalar(snippet_raw or ""))

    doc_id_raw = raw.get("doc_id")
    if doc_id_raw is None:
        doc_id = "unknown_doc"
    else:
        doc_id = str(_json_safe_scalar(doc_id_raw))

    section_raw = raw.get("section")
    section_scalar = _json_safe_scalar(section_raw)
    section = str(section_scalar) if section_scalar is not None else None

    page_or_heading_raw = raw.get("page_or_heading")
    if page_or_heading_raw is None:
        page_or_heading_raw = raw.get("page")
    if page_or_heading_raw is None:
        page_or_heading_raw = section_raw
    page_or_heading = _json_safe_scalar(page_or_heading_raw)

    normalized: dict[str, Any] = {
        "doc_id": doc_id,
        "section": section,
        "page_or_heading": page_or_heading,
        "snippet": snippet,
        "snippet_id": snippet_id,
    }
    chunk_id_raw = raw.get("chunk_id")
    chunk_id_scalar = _json_safe_scalar(chunk_id_raw)
    if isinstance(chunk_id_scalar, str) and chunk_id_scalar:
        normalized["chunk_id"] = chunk_id_scalar
    score = raw.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        normalized["score"] = float(score) if math.isfinite(float(score)) else str(score)
    line_start = _coerce_int(raw.get("line_start"))
    if line_start is not None and line_start >= 1:
        normalized["line_start"] = line_start
    line_end = _coerce_int(raw.get("line_end"))
    if line_end is not None and line_end >= 1:
        normalized["line_end"] = line_end
    return normalized


def _normalize_retrieve_meta(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "cache_hit": False,
            "grounding_missing": False,
            "grounding_reason": None,
            "snippet_ids": [],
            "source_chunk_refs": [],
            "index_path": None,
            "grounding_error_type": None,
            "index_error_type": None,
            "source_policy_applied": False,
            "source_policy_id": None,
            "source_policy_version": None,
            "source_policy_filtered_out_count": 0,
        }

    def _normalize_error_type(value: Any) -> str | None:
        if isinstance(value, str):
            return value or None
        if value is None:
            return None
        scalar = _json_safe_scalar(value)
        return str(scalar) if scalar is not None else None

    index_path_raw = raw.get("index_path")
    if isinstance(index_path_raw, Path):
        index_path: str | None = str(index_path_raw)
    else:
        index_path_scalar = _json_safe_scalar(index_path_raw)
        index_path = str(index_path_scalar) if index_path_scalar is not None else None

    reason_raw = raw.get("grounding_reason")
    grounding_reason = reason_raw if isinstance(reason_raw, str) and reason_raw else None

    snippet_ids_raw = raw.get("snippet_ids")
    snippet_ids: list[str] = []
    if isinstance(snippet_ids_raw, (list, tuple)):
        for item in snippet_ids_raw:
            scalar = _json_safe_scalar(item)
            if isinstance(scalar, str) and scalar:
                snippet_ids.append(scalar)

    source_chunk_refs_raw = raw.get("source_chunk_refs")
    source_chunk_refs: list[str] = []
    if isinstance(source_chunk_refs_raw, (list, tuple)):
        for item in source_chunk_refs_raw:
            scalar = _json_safe_scalar(item)
            if isinstance(scalar, str) and scalar:
                source_chunk_refs.append(scalar)

    return {
        "cache_hit": bool(raw.get("cache_hit")),
        "grounding_missing": bool(raw.get("grounding_missing")),
        "grounding_reason": grounding_reason,
        "snippet_ids": snippet_ids,
        "source_chunk_refs": source_chunk_refs,
        "index_path": index_path,
        "grounding_error_type": _normalize_error_type(raw.get("grounding_error_type")),
        "index_error_type": _normalize_error_type(raw.get("index_error_type")),
        "source_policy_applied": bool(raw.get("source_policy_applied")),
        "source_policy_id": _normalize_error_type(raw.get("source_policy_id")),
        "source_policy_version": _normalize_error_type(raw.get("source_policy_version")),
        "source_policy_filtered_out_count": _coerce_int(raw.get("source_policy_filtered_out_count")) or 0,
    }


def _normalize_help_report(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        report = dict(raw)
    elif hasattr(raw, "to_dict"):
        report = dict(raw.to_dict())  # type: ignore[call-arg]
    else:
        report = {}
    report.setdefault("executed", [])
    report.setdefault("rejected", [])
    report.setdefault("dropped", [])
    report.setdefault("dry_run", [])
    return report


def _normalize_cached_response_metadata(metadata: dict[str, Any]) -> None:
    metadata.setdefault("retry_count", 0)
    metadata.setdefault("retry_reason", None)
    metadata.setdefault("repair_applied", False)
    metadata.setdefault("repair_details", {})
    metadata.setdefault("fallback_overlay_used", False)
    if metadata.get("fallback_overlay_reason") is None:
        metadata["fallback_overlay_reason"] = "not_needed"


def _normalize_generation_mode(response: TutorResponse) -> str:
    metadata = response.metadata if isinstance(response.metadata, Mapping) else {}
    raw_mode = metadata.get("generation_mode")
    if raw_mode in {"model", "repair", "fallback"}:
        return str(raw_mode)
    if bool(metadata.get("json_repaired")) or bool(metadata.get("repair_applied")):
        return "repair"
    if response.status == "error" or metadata.get("provider") == "fallback":
        return "fallback"
    return "model"


def _coerce_string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item]


def _extract_selected_layout_id(vision_selection: HelpCycleVisionSelection) -> str | None:
    for frame in (vision_selection.trigger_frame, vision_selection.pre_trigger_frame):
        if isinstance(frame, Mapping):
            layout_id = frame.get("layout_id")
            if isinstance(layout_id, str) and layout_id:
                return layout_id
    return None


def _is_terminal_step_hint_complete(hint: Mapping[str, Any] | None) -> bool:
    if not isinstance(hint, Mapping):
        return False
    if hint.get("inferred_step_id") != "S33":
        return False
    missing_conditions = hint.get("missing_conditions")
    normalized_missing = [
        item for item in missing_conditions if isinstance(item, str) and item
    ] if isinstance(missing_conditions, (list, tuple)) else []
    if normalized_missing:
        return False
    gate_blockers = hint.get("gate_blockers")
    normalized_gate_blockers = [
        item for item in gate_blockers if isinstance(item, Mapping) and item
    ] if isinstance(gate_blockers, (list, tuple)) else []
    return not normalized_gate_blockers


def _extract_fused_step_audit(request: TutorRequest) -> tuple[str | None, list[str]]:
    context = request.context if isinstance(request.context, Mapping) else {}
    hint = context.get("deterministic_step_hint")
    if not isinstance(hint, Mapping):
        return None, []
    fused_step_id = hint.get("inferred_step_id")
    if not isinstance(fused_step_id, str) or not fused_step_id:
        fused_step_id = None
    return fused_step_id, _coerce_string_list(hint.get("missing_conditions"))


def _extract_model_next_step_id(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    help_response = metadata.get("help_response")
    if not isinstance(help_response, Mapping):
        return None
    next_payload = help_response.get("next")
    if not isinstance(next_payload, Mapping):
        return None
    step_id = next_payload.get("step_id")
    if not isinstance(step_id, str) or not step_id:
        return None
    return step_id


def _text_claims_step_complete(text: str, step_id: str) -> bool:
    normalized_step = step_id.strip().lower()
    if not normalized_step:
        return False
    normalized_text = " ".join(text.lower().split())
    if not normalized_text:
        return False
    escaped = re.escape(normalized_step)
    negative_patterns = (
        rf"\b{escaped}\b[^.。;；\n]{{0,80}}(?:尚未完成|未完成|not complete|incomplete)",
        rf"\b{escaped}\b[^.。;；\n]{{0,80}}(?:请|(?<![已经])按下|(?<![已经])操作|(?<![已经])点击|press|operate|set|move)[^.。;；\n]{{0,80}}(?:以|to)?\s*完成",
        rf"(?:当前步骤|current step)[^.。;；\n]{{0,80}}(?:尚未完成|未完成|not complete|incomplete)",
    )
    if any(re.search(pattern, normalized_text) is not None for pattern in negative_patterns):
        return False
    patterns = (
        rf"\b{escaped}\b[^.。;；\n]{{0,80}}(?:is complete|step is complete|complete)",
        rf"(?:当前\s*)?\b{escaped}\b[^.。;；\n]{{0,80}}(?:已完成|已经完成|完成)",
        rf"(?:当前步骤|current step)[^.。;；\n]{{0,80}}(?:已完成|已经完成|is complete|complete)",
    )
    return any(re.search(pattern, normalized_text) is not None for pattern in patterns)


def _classify_vision_fallback_reason(
    *,
    vision_selection: HelpCycleVisionSelection,
    vision_fact_context: Mapping[str, Any],
    response_metadata: Mapping[str, Any] | None = None,
    fused_step_id: str | None = None,
) -> str | None:
    if isinstance(response_metadata, Mapping) and bool(response_metadata.get("multimodal_fallback_to_text")):
        return VISION_TEXT_FALLBACK

    vision_fact_status = vision_fact_context.get("status")
    if vision_fact_status == VISION_NOT_REQUIRED:
        return None
    if vision_fact_status == "extractor_failed":
        return VISION_PARSE_FAIL
    if vision_fact_status == "vision_unavailable" and vision_selection.vision_used:
        return VISION_UNAVAILABLE

    model_next_step_id = _extract_model_next_step_id(response_metadata)
    if (
        isinstance(fused_step_id, str)
        and fused_step_id
        and isinstance(model_next_step_id, str)
        and model_next_step_id
        and model_next_step_id != fused_step_id
        and vision_selection.vision_used
    ):
        return VISION_CONFLICT_UNRESOLVED

    if not vision_selection.vision_used:
        if vision_selection.sync_miss_reason == "vision_port_unconfigured":
            return VISION_UNAVAILABLE
        if isinstance(vision_selection.sync_miss_reason, str) and vision_selection.sync_miss_reason:
            return VISION_SYNC_MISS
        return VISION_UNAVAILABLE

    return None


def _build_help_cycle_audit_fields(
    *,
    request: TutorRequest,
    vision_selection: HelpCycleVisionSelection,
    vision_fact_context: Mapping[str, Any],
    response_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fused_step_id, fused_missing_conditions = _extract_fused_step_audit(request)
    vision_fact_metadata = vision_fact_context.get("metadata")
    vision_fact_extractor_used = (
        bool(vision_fact_metadata.get("extractor_used")) if isinstance(vision_fact_metadata, Mapping) else False
    )
    vision_fact_cached_count = (
        vision_fact_metadata.get("cached_fact_count")
        if isinstance(vision_fact_metadata, Mapping) and isinstance(vision_fact_metadata.get("cached_fact_count"), int)
        else 0
    )
    vision_fact_sticky_count = (
        vision_fact_metadata.get("sticky_fact_count")
        if isinstance(vision_fact_metadata, Mapping) and isinstance(vision_fact_metadata.get("sticky_fact_count"), int)
        else 0
    )
    vision_fact_ignored_count = (
        vision_fact_metadata.get("ignored_fact_count")
        if isinstance(vision_fact_metadata, Mapping) and isinstance(vision_fact_metadata.get("ignored_fact_count"), int)
        else 0
    )
    return {
        "vision_used": bool(vision_selection.vision_used),
        "vision_fact_extractor_used": vision_fact_extractor_used,
        "vision_frame_capture_selected": bool(vision_selection.frame_ids),
        "vision_fact_cached_count": vision_fact_cached_count,
        "vision_fact_sticky_count": vision_fact_sticky_count,
        "vision_fact_ignored_count": vision_fact_ignored_count,
        "frame_id": vision_selection.frame_id,
        "sync_delta_ms": vision_selection.sync_delta_ms,
        "vision_fact_summary": dict(vision_fact_context.get("vision_fact_summary", {})),
        "fused_step_id": fused_step_id,
        "fused_missing_conditions": fused_missing_conditions,
        "vision_fallback_reason": _classify_vision_fallback_reason(
            vision_selection=vision_selection,
            vision_fact_context=vision_fact_context,
            response_metadata=response_metadata,
            fused_step_id=fused_step_id,
        ),
        "layout_id": _extract_selected_layout_id(vision_selection),
    }


def _apply_help_cycle_audit_fields(target: dict[str, Any], audit_fields: Mapping[str, Any]) -> None:
    target.update(normalize_help_cycle_audit_fields(audit_fields))


def _attach_help_cycle_trace_to_actions(
    actions: Sequence[Mapping[str, Any] | Any],
    *,
    trace_metadata: Mapping[str, Any],
) -> list[Mapping[str, Any] | Any]:
    normalized_trace = normalize_help_cycle_audit_fields(trace_metadata)
    traced: list[Mapping[str, Any] | Any] = []
    for action in actions:
        if not isinstance(action, Mapping):
            traced.append(action)
            continue
        item = dict(action)
        help_cycle_id = normalized_trace.get("help_cycle_id")
        if isinstance(help_cycle_id, str) and help_cycle_id:
            item["help_cycle_id"] = help_cycle_id
        for key, value in normalized_trace.items():
            if key == "help_cycle_id":
                continue
            item[key] = value
        traced.append(item)
    return traced


def _compact_candidate_for_trace(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    step_id = raw.get("step_id")
    if not isinstance(step_id, str) or not step_id:
        return None
    out: dict[str, Any] = {"step_id": step_id}
    for key in (
        "source",
        "role",
        "observability",
        "requires_visual_confirmation",
        "confidence",
        "rank",
    ):
        if key in raw:
            out[key] = _copy_event_field(raw.get(key))
    for key in ("supporting_evidence_refs", "blocking_evidence_refs", "proposed_next_action_target_ids"):
        value = raw.get(key)
        if isinstance(value, (list, tuple)):
            out[key] = [item for item in value if isinstance(item, str) and item][:8]
    return out


def _overlay_targets_from_actions(actions: Sequence[Mapping[str, Any] | Any]) -> list[str]:
    out: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        target = action.get("target")
        if isinstance(target, str) and target and target not in out:
            out.append(target)
    return out


def _step_id_from_payload(raw: Any) -> str | None:
    if not isinstance(raw, Mapping):
        return None
    step_id = raw.get("step_id")
    return step_id if isinstance(step_id, str) and step_id else None


def _help_response_overlay_targets(raw: Any) -> list[str]:
    if not isinstance(raw, Mapping):
        return []
    overlay = raw.get("overlay")
    if not isinstance(overlay, Mapping):
        return []
    targets = overlay.get("targets")
    if not isinstance(targets, (list, tuple)):
        return []
    return [item for item in targets if isinstance(item, str) and item]


def _help_response_evidence_refs(raw: Any) -> list[str]:
    if not isinstance(raw, Mapping):
        return []
    overlay = raw.get("overlay")
    if not isinstance(overlay, Mapping):
        return []
    evidence = overlay.get("evidence")
    if not isinstance(evidence, (list, tuple)):
        return []
    refs: list[str] = []
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        ref = item.get("ref")
        if isinstance(ref, str) and ref:
            refs.append(ref)
    return refs


def _help_response_step_id(raw: Any) -> str | None:
    if not isinstance(raw, Mapping):
        return None
    next_step = _step_id_from_payload(raw.get("next"))
    if next_step is not None:
        return next_step
    return _step_id_from_payload(raw.get("diagnosis"))


def _message_category_from_response(response: TutorResponse) -> str:
    metadata = response.metadata if isinstance(response.metadata, Mapping) else {}
    if metadata.get("terminal_state_rewritten") is True:
        return "terminal_completed"
    if metadata.get("manual_throttle_guidance_rewritten") is True:
        return "manual_text_guidance"
    if metadata.get("completion_conflict_rewritten") is True:
        return "completion_conflict_repair"
    if metadata.get("harness_guardrail_applied") is True:
        return "harness_guardrail_repair"
    if metadata.get("validator_rejected") is True or metadata.get("repair_applied") is True:
        return "harness_validator_repair"
    if metadata.get("fallback_overlay_used") is True:
        return "fallback_overlay"
    generation_mode = metadata.get("generation_mode")
    if generation_mode == "repair":
        return "llm_repair"
    if generation_mode == "fallback" or response.status == "error":
        return "fallback"
    return "model"


def _vlm_call_trace(
    *,
    vision_selection: HelpCycleVisionSelection,
    vision_fact_context: Mapping[str, Any],
) -> dict[str, Any]:
    status = vision_fact_context.get("status")
    metadata = vision_fact_context.get("metadata")
    extractor_used = bool(metadata.get("extractor_used")) if isinstance(metadata, Mapping) else False
    ignored_fact_count = (
        metadata.get("ignored_fact_count")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("ignored_fact_count"), int)
        else 0
    )
    cached_fact_count = (
        metadata.get("cached_fact_count")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("cached_fact_count"), int)
        else 0
    )
    sticky_fact_count = (
        metadata.get("sticky_fact_count")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("sticky_fact_count"), int)
        else 0
    )
    reason = metadata.get("reason") if isinstance(metadata, Mapping) else None
    if not isinstance(reason, str) or not reason:
        reason = vision_selection.sync_miss_reason if isinstance(vision_selection.sync_miss_reason, str) else None

    if status == VISION_NOT_REQUIRED:
        call_status = "not_required"
    elif status == "extractor_failed":
        call_status = "failed"
    elif extractor_used:
        call_status = "called"
    else:
        call_status = "skipped"

    return {
        "status": call_status,
        "reason": reason,
        "vision_status": vision_selection.status,
        "vision_fact_status": status,
        "extractor_used": extractor_used,
        "extractor_called": extractor_used,
        "frame_capture_selected": bool(vision_selection.frame_ids),
        "cached_fact_count": cached_fact_count,
        "sticky_fact_count": sticky_fact_count,
        "ignored_fact_count": ignored_fact_count,
        "frame_ids": list(vision_selection.frame_ids),
        "sync_status": vision_selection.sync_status,
        "sync_delta_ms": vision_selection.sync_delta_ms,
    }


def _build_harness_trace_metadata(
    *,
    request: TutorRequest,
    response: TutorResponse,
    vision_selection: HelpCycleVisionSelection,
    vision_fact_context: Mapping[str, Any],
) -> dict[str, Any]:
    context = request.context if isinstance(request.context, Mapping) else {}
    response_metadata = response.metadata if isinstance(response.metadata, Mapping) else {}
    raw_candidates = context.get("candidate_steps")
    candidates = [
        candidate
        for candidate in (
            _compact_candidate_for_trace(item)
            for item in (raw_candidates if isinstance(raw_candidates, list) else [])
        )
        if candidate is not None
    ]

    model_help_response = response_metadata.get("model_raw_help_response")
    if not isinstance(model_help_response, Mapping):
        model_help_response = response_metadata.get("help_response")
    model_step_id = _help_response_step_id(model_help_response)
    if model_step_id is None:
        model_step_id = _extract_model_next_step_id(response_metadata)

    final_plan_raw = response_metadata.get("harness_action_plan")
    final_plan = dict(final_plan_raw) if isinstance(final_plan_raw, Mapping) else {}
    final_targets = _overlay_targets_from_actions(response.actions)
    if not final_plan:
        final_plan = {
            "step_id": _step_id_from_payload(response_metadata.get("next"))
            or _step_id_from_payload(response_metadata.get("diagnosis")),
            "overlay_step_id": None,
            "targets": list(final_targets),
            "text_only": not bool(final_targets),
            "source": response_metadata.get("final_action_plan_source") or response_metadata.get("generation_mode"),
        }
    else:
        final_plan.setdefault("targets", list(final_targets))
        final_plan.setdefault("source", response_metadata.get("final_action_plan_source"))

    chosen_step_id = final_plan.get("step_id")
    chosen_candidate = None
    if isinstance(chosen_step_id, str) and chosen_step_id:
        chosen_candidate = next((dict(item) for item in candidates if item.get("step_id") == chosen_step_id), None)
        if chosen_candidate is None:
            chosen_candidate = {"step_id": chosen_step_id, "source": "final_action_plan"}

    rejected_candidate_ids: list[str] = []
    rejected_model_step_id = response_metadata.get("rejected_model_step_id")
    if isinstance(rejected_model_step_id, str) and rejected_model_step_id:
        rejected_candidate_ids.append(rejected_model_step_id)
    for reason in response_metadata.get("harness_validation_reasons", []):
        if not isinstance(reason, str):
            continue
        prefix = "model_step_not_candidate:"
        if reason.startswith(prefix):
            rejected_candidate_ids.append(reason[len(prefix):])
    rejected_candidates = [
        {"step_id": step_id}
        for step_id in _dedupe_strings(rejected_candidate_ids)
    ]

    final_source = final_plan.get("source")
    repair_path = final_source if isinstance(final_source, str) and final_source != "model" else None
    repair_applied = bool(response_metadata.get("repair_applied"))
    if repair_path is None and repair_applied:
        repair_path = "response_repair"
    if repair_path is None and response_metadata.get("fallback_overlay_used") is True:
        fallback_reason = response_metadata.get("fallback_overlay_reason")
        repair_path = fallback_reason if isinstance(fallback_reason, str) and fallback_reason else "fallback_overlay"

    message_category = _message_category_from_response(response)
    vlm_call = _vlm_call_trace(vision_selection=vision_selection, vision_fact_context=vision_fact_context)
    trace = {
        "schema_version": "v1",
        "evidence_packet_summary": dict(context.get("evidence_packet_summary", {})),
        "candidates": candidates,
        "chosen_candidate": chosen_candidate,
        "rejected_candidates": rejected_candidates,
        "model_decision": {
            "step_id": model_step_id,
            "overlay_targets": _help_response_overlay_targets(model_help_response),
            "evidence_refs": _help_response_evidence_refs(model_help_response),
            "status": response.status,
            "generation_mode": response_metadata.get("generation_mode"),
        },
        "validator_result": {
            "rejected": bool(response_metadata.get("validator_rejected")),
            "reasons": list(response_metadata.get("harness_validation_reasons", []))
            if isinstance(response_metadata.get("harness_validation_reasons"), list)
            else [],
            "rejected_model_step_id": rejected_model_step_id if isinstance(rejected_model_step_id, str) else None,
            "fallback_overlay_used": bool(response_metadata.get("fallback_overlay_used")),
            "fallback_overlay_reason": response_metadata.get("fallback_overlay_reason"),
        },
        "repair_result": {
            "applied": repair_applied or repair_path is not None or response_metadata.get("generation_mode") == "repair",
            "path": repair_path,
            "json_repaired": bool(response_metadata.get("json_repaired")),
            "evidence_ref_repair": dict(response_metadata.get("evidence_ref_repair", {}))
            if isinstance(response_metadata.get("evidence_ref_repair"), Mapping)
            else {},
        },
        "final_action_plan": final_plan,
        "final_overlay_targets": final_targets,
        "message_category": message_category,
        "vlm_call": vlm_call,
    }
    response.metadata["message_category"] = message_category
    response.metadata["vlm_call_status"] = vlm_call["status"]
    response.metadata["vlm_call_reason"] = vlm_call["reason"]
    response.metadata["final_overlay_targets"] = list(final_targets)
    response.metadata["final_action_plan"] = dict(final_plan)
    response.metadata["harness_trace"] = trace
    return trace


def _dedupe_strings(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _compact_gate_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": raw.get("status"),
        "reason_code": raw.get("reason_code"),
        "reason": raw.get("reason"),
    }


def _select_gates_for_context(
    all_gates: Mapping[str, Mapping[str, Any]],
    *,
    inferred_step_id: str | None,
    max_items: int = 8,
) -> dict[str, dict[str, Any]]:
    cap = max(0, int(max_items))
    if cap == 0 or not all_gates:
        return {}

    ordered_ids: list[str] = []
    if isinstance(inferred_step_id, str) and inferred_step_id:
        for gate_type in ("precondition", "completion"):
            gate_id = f"{inferred_step_id}.{gate_type}"
            if gate_id in all_gates:
                ordered_ids.append(gate_id)

    blocked_ids = [
        gate_id
        for gate_id, gate in all_gates.items()
        if isinstance(gate, Mapping) and gate.get("status") == "blocked"
    ]
    blocked_id_set = set(blocked_ids)
    allowed_ids = [gate_id for gate_id in all_gates.keys() if gate_id not in blocked_id_set]
    for gate_id in sorted(blocked_ids):
        ordered_ids.append(gate_id)
    for gate_id in sorted(allowed_ids):
        ordered_ids.append(gate_id)

    selected: dict[str, dict[str, Any]] = {}
    for gate_id in ordered_ids:
        if gate_id in selected:
            continue
        gate = all_gates.get(gate_id)
        if not isinstance(gate, Mapping):
            continue
        selected[gate_id] = _compact_gate_payload(gate)
        if len(selected) >= cap:
            break
    return selected


def _extract_step_id(step: Mapping[str, Any]) -> str | None:
    raw_id = step.get("id")
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    raw_step_id = step.get("step_id")
    if isinstance(raw_step_id, str) and raw_step_id:
        return raw_step_id
    return None


def _normalize_step_ui_targets(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _missing_condition_target_hints(
    missing_conditions: Any,
    *,
    allowed_targets: Sequence[str],
) -> list[str]:
    if not isinstance(missing_conditions, (list, tuple)):
        return []
    allowed = {item for item in allowed_targets if isinstance(item, str) and item}
    if not allowed:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in missing_conditions:
        if not isinstance(item, str) or not item:
            continue
        matched = _MISSING_CONDITION_VAR_RE.search(item)
        if matched is None:
            continue
        var_name = matched.group(1)
        for target in _MISSING_CONDITION_TARGET_HINTS.get(var_name, ()):
            if target not in allowed or target in seen:
                continue
            seen.add(target)
            out.append(target)
    return out


def _enforce_s08_ddi_before_ampcd(
    candidate_targets: list[str],
    *,
    step_id: str | None,
) -> list[str]:
    """For S08, ensure DDI brightness selectors precede the AMPCD brightness knob.

    On F/A-18C Lot 20 the AMPCD will not illuminate if no DDI has been
    powered first — the brightness knob alone is insufficient.
    When both DDI and AMPCD targets are candidates, DDIs must come first.
    """
    if step_id != "S08":
        return candidate_targets
    ddi_selectors = ("left_mdi_brightness_selector", "right_mdi_brightness_selector")
    ampcd_target = "ampcd_off_brightness_knob"
    if ampcd_target not in candidate_targets:
        return candidate_targets
    ddi_present = [t for t in ddi_selectors if t in candidate_targets]
    if not ddi_present:
        return candidate_targets
    ampcd_pos = candidate_targets.index(ampcd_target)
    first_ddi_pos = min(candidate_targets.index(t) for t in ddi_present)
    if ampcd_pos >= first_ddi_pos:
        return candidate_targets
    # AMPCD is ahead of all present DDIs — move it after them.
    reordered = [t for t in candidate_targets if t != ampcd_target]
    insert_after = max(reordered.index(t) for t in ddi_present)
    reordered.insert(insert_after + 1, ampcd_target)
    return reordered


def _prefer_navigation_target_from_vision_context(
    *,
    inferred_step_id: str,
    missing_conditions: Sequence[str],
    context: Mapping[str, Any],
    allowed_targets: Sequence[str],
) -> list[str]:
    allowed = {item for item in allowed_targets if isinstance(item, str) and item}
    if inferred_step_id != "S08" or not allowed:
        return []
    if "vision_facts.fcs_page_visible==seen" not in missing_conditions:
        return []
    summary = context.get("vision_fact_summary")
    if not isinstance(summary, Mapping):
        return []
    seen = {
        item for item in summary.get("seen_fact_ids", [])
        if isinstance(item, str) and item
    }
    if "supt_page_visible" in seen and "left_mdi_pb15" in allowed:
        return ["left_mdi_pb15"]
    if "tac_page_visible" in seen and "left_mdi_pb18" in allowed:
        return ["left_mdi_pb18"]
    if "left_mdi_brightness_selector" in allowed:
        return ["left_mdi_brightness_selector"]
    return []


def _prefer_s08_visual_page_targets(
    *,
    missing_conditions: Sequence[str],
    allowed_targets: Sequence[str],
    max_targets: int,
) -> list[str]:
    missing = {item for item in missing_conditions if isinstance(item, str) and item}
    allowed = {item for item in allowed_targets if isinstance(item, str) and item}
    out: list[str] = []
    fcs_missing = "vision_facts.fcs_page_visible==seen" in missing
    bit_missing = "vision_facts.bit_root_page_visible==seen" in missing
    if not (fcs_missing and bit_missing):
        return []
    if bit_missing and "right_mdi_brightness_selector" in allowed:
        out.append("right_mdi_brightness_selector")
    if "left_mdi_pb18" in allowed:
        out.append("left_mdi_pb18")
    if "right_mdi_pb18" in allowed:
        out.append("right_mdi_pb18")
    return out[: max(1, int(max_targets))]


def _state_harness_has_late_vlm_conflict(raw: Any) -> bool:
    if not isinstance(raw, Mapping):
        return False
    conflicts = raw.get("conflicts")
    return isinstance(conflicts, list) and HARNESS_LATE_VLM_CONFLICT in conflicts


def _visual_candidate_steps_from_state_harness(raw: Any) -> list[str]:
    if not isinstance(raw, Mapping):
        return []
    vision_evidence = raw.get("vision_evidence")
    if not isinstance(vision_evidence, Mapping):
        return []
    candidates = vision_evidence.get("visual_candidate_steps")
    if not isinstance(candidates, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _reprioritize_steps_for_state_harness(candidate_steps: Sequence[str], state_harness: Any) -> list[str]:
    steps = [step for step in candidate_steps if isinstance(step, str) and step]
    if not _state_harness_has_late_vlm_conflict(state_harness):
        return steps
    visual_candidates = _visual_candidate_steps_from_state_harness(state_harness)
    prioritized = [step for step in visual_candidates if step in steps]
    return [*prioritized, *[step for step in steps if step not in set(prioritized)]]


def _broaden_overlay_allowlist_for_state_harness(
    overlay_target_allowlist: Sequence[str],
    *,
    state_harness: Any,
    step_fallback_profiles: Mapping[str, Any],
    overlay_allowset: set[str],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for target in overlay_target_allowlist:
        if isinstance(target, str) and target and target not in seen:
            seen.add(target)
            out.append(target)
    if not _state_harness_has_late_vlm_conflict(state_harness):
        return out
    for step_id in _visual_candidate_steps_from_state_harness(state_harness):
        profile = step_fallback_profiles.get(step_id)
        if not isinstance(profile, Mapping):
            continue
        targets = profile.get("ui_targets")
        if not isinstance(targets, list):
            continue
        for target in targets:
            if not isinstance(target, str) or target not in overlay_allowset or target in seen:
                continue
            seen.add(target)
            out.append(target)
    return out


def _build_visual_action_hint(
    *,
    inferred_step_id: str | None,
    missing_conditions: Sequence[str],
    context: Mapping[str, Any],
    allowed_targets: Sequence[str],
) -> dict[str, Any] | None:
    if not isinstance(inferred_step_id, str) or not inferred_step_id:
        return None
    suggested_targets = _prefer_navigation_target_from_vision_context(
        inferred_step_id=inferred_step_id,
        missing_conditions=missing_conditions,
        context=context,
        allowed_targets=allowed_targets,
    )
    if not suggested_targets:
        return None
    target = suggested_targets[0]
    reason_by_target = {
        "left_mdi_pb15": "Right DDI already shows BIT and the left DDI clearly shows the FCS option on PB15; press PB15 to enter the FCS page.",
        "left_mdi_pb18": "The left DDI is still on TAC/root-menu navigation; press PB18 to move to the SUPT page where the FCS option becomes available.",
        "left_mdi_brightness_selector": "Left DDI page cues are not readable yet; restore left DDI visibility first.",
    }
    return {
        "target": target,
        "reason": reason_by_target.get(target, f"Visual state suggests operating {target} next."),
    }


def _s08_power_targets_for_missing_conditions(
    missing_conditions: Sequence[str],
    *,
    allowed_targets: Sequence[str],
) -> list[str]:
    allowed = {item for item in allowed_targets if isinstance(item, str) and item}
    targets: list[str] = []
    for var_name, target, _reason in _S08_POWER_SEQUENCE:
        if target not in allowed:
            continue
        if any(
            isinstance(item, str) and item.strip().startswith(f"vars.{var_name}==")
            for item in missing_conditions
        ):
            targets.append(target)
    return targets


def _build_s08_power_action_hint(
    *,
    vars_selected: Mapping[str, Any],
    allowed_targets: Sequence[str],
) -> dict[str, Any] | None:
    allowed = {item for item in allowed_targets if isinstance(item, str) and item}
    if not allowed:
        return None
    for var_name, target, reason in _S08_POWER_SEQUENCE:
        if target in allowed and vars_selected.get(var_name) is not True:
            return {"target": target, "reason": reason}
    return None


def _normalize_ufc_scratchpad_text(vars_selected: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "ufc_scratchpad_string_1_display",
        "ufc_scratchpad_string_2_display",
        "ufc_scratchpad_number_display",
    ):
        value = vars_selected.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "".join(parts).replace("。", ".").upper()


def _s09_comm1_frequency_complete(vars_selected: Mapping[str, Any]) -> bool:
    if vars_selected.get("comm1_freq_134_000") is True:
        return True
    value = _coerce_int(vars_selected.get("comm1_freq_value"))
    return value == 13400


def _missing_conditions_satisfied_by_vars(
    missing_conditions: Sequence[str],
    vars_selected: Mapping[str, Any],
) -> bool:
    checked = False
    for condition in missing_conditions:
        if not isinstance(condition, str) or not condition:
            continue
        matched = _MISSING_CONDITION_VAR_RE.search(condition)
        if matched is None:
            return False
        var_name = matched.group(1)
        if "==true" in condition:
            checked = True
            if vars_selected.get(var_name) is not True:
                return False
        else:
            return False
    return checked


def _vision_summary_seen_or_fresh(
    summary: Mapping[str, Any] | None,
    fact_id: str,
) -> bool:
    if not isinstance(summary, Mapping) or not fact_id:
        return False
    for key in ("seen_fact_ids", "fresh_fact_ids"):
        raw = summary.get(key)
        if isinstance(raw, (list, tuple, set)) and any(item == fact_id for item in raw):
            return True
    return False


def _s19_final_go_seen_or_fresh(summary: Mapping[str, Any] | None) -> bool:
    return _vision_summary_seen_or_fresh(summary, "fcsmc_final_go_result_visible")


def _s08_visual_hint_fact_id_for_target(target: str | None) -> str | None:
    if target == "left_mdi_pb15":
        return "supt_page_visible"
    if target == "left_mdi_pb18":
        return "tac_page_visible"
    return None


def _s08_power_condition_missing(missing_set: set[str]) -> bool:
    return any(f"vars.{var_name}==true" in missing_set for var_name, _target, _reason in _S08_POWER_SEQUENCE)


def _visual_fact_ref_from_context(context: Mapping[str, Any], fact_id: str | None) -> str | None:
    if not isinstance(fact_id, str) or not fact_id:
        return None
    vision_facts = context.get("vision_facts")
    if isinstance(vision_facts, list):
        for item in vision_facts:
            if not isinstance(item, Mapping) or item.get("fact_id") != fact_id:
                continue
            state = item.get("state")
            if isinstance(state, str) and state not in {"seen", "fresh"}:
                continue
            frame_id = item.get("source_frame_id")
            return (
                f"VISION_FACTS.{fact_id}@{frame_id}"
                if isinstance(frame_id, str) and frame_id
                else f"VISION_FACTS.{fact_id}"
            )
    if _vision_summary_seen_or_fresh(context.get("vision_fact_summary"), fact_id):
        return f"VISION_FACTS.{fact_id}"
    return None


def _s08_visual_hint_from_vision_summary(context: Mapping[str, Any]) -> tuple[str, str, str | None] | None:
    for fact_id, target, reason in (
        (
            "supt_page_visible",
            "left_mdi_pb15",
            "VLM confirms SUPT is visible; press PB15 to enter the FCS page.",
        ),
        (
            "tac_page_visible",
            "left_mdi_pb18",
            "VLM confirms TAC is visible; press PB18 to reach the SUPT page.",
        ),
    ):
        if _vision_summary_seen_or_fresh(context.get("vision_fact_summary"), fact_id):
            return target, reason, _visual_fact_ref_from_context(context, fact_id)
    return None


def _s08_visual_evidence_ref_for_target(evidence_refs: Sequence[str], target: str | None) -> str | None:
    fact_id = _s08_visual_hint_fact_id_for_target(target)
    if fact_id is None:
        return None
    prefix = f"VISION_FACTS.{fact_id}"
    for ref in evidence_refs:
        if isinstance(ref, str) and (ref == prefix or ref.startswith(f"{prefix}@")):
            return ref
    return None


def _s08_page_navigation_fact_id_from_ref(ref: str) -> str | None:
    for fact_id in ("supt_page_visible", "tac_page_visible"):
        prefix = f"VISION_FACTS.{fact_id}"
        if ref == prefix or ref.startswith(f"{prefix}@"):
            return fact_id
    return None


def _s08_filter_unconfirmed_page_navigation_refs(
    context: Mapping[str, Any],
    evidence_refs: Sequence[str],
) -> list[str]:
    filtered: list[str] = []
    vision_summary = context.get("vision_fact_summary")
    for ref in evidence_refs:
        if not isinstance(ref, str) or not ref:
            continue
        fact_id = _s08_page_navigation_fact_id_from_ref(ref)
        if fact_id is not None and not _vision_summary_seen_or_fresh(vision_summary, fact_id):
            continue
        filtered.append(ref)
    return filtered


def _s08_clean_unconfirmed_page_navigation_help_response(
    context: Mapping[str, Any],
    help_response: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(help_response, Mapping):
        return None, False
    overlay = help_response.get("overlay")
    if not isinstance(overlay, Mapping):
        return None, False
    evidence = overlay.get("evidence")
    if not isinstance(evidence, list):
        return None, False

    cleaned_evidence: list[Any] = []
    changed = False
    vision_summary = context.get("vision_fact_summary")
    for item in evidence:
        ref = item.get("ref") if isinstance(item, Mapping) else None
        if isinstance(ref, str):
            fact_id = _s08_page_navigation_fact_id_from_ref(ref)
            if fact_id is not None and not _vision_summary_seen_or_fresh(vision_summary, fact_id):
                changed = True
                continue
        cleaned_evidence.append(item)

    if not changed:
        return None, False
    cleaned_help_response = copy.deepcopy(dict(help_response))
    cleaned_overlay = copy.deepcopy(dict(overlay))
    cleaned_overlay["evidence"] = cleaned_evidence
    cleaned_help_response["overlay"] = cleaned_overlay
    return cleaned_help_response, True


def _s08_visual_fact_ref_for_seen_target(
    context: Mapping[str, Any],
    evidence_refs: Sequence[str],
    target: str | None,
) -> str | None:
    fact_id = _s08_visual_hint_fact_id_for_target(target)
    if fact_id is None or not _vision_summary_seen_or_fresh(context.get("vision_fact_summary"), fact_id):
        return None
    return _s08_visual_evidence_ref_for_target(evidence_refs, target) or _visual_fact_ref_from_context(context, fact_id)


def _s08_visual_hint_from_seen_evidence_refs(
    context: Mapping[str, Any],
    evidence_refs: Sequence[str],
) -> tuple[str, str, str] | None:
    for fact_id, target in (
        ("supt_page_visible", "left_mdi_pb15"),
        ("tac_page_visible", "left_mdi_pb18"),
    ):
        if not _vision_summary_seen_or_fresh(context.get("vision_fact_summary"), fact_id):
            continue
        prefix = f"VISION_FACTS.{fact_id}"
        for ref in evidence_refs:
            if isinstance(ref, str) and (ref == prefix or ref.startswith(f"{prefix}@")):
                return target, f"Visual evidence {ref} confirms S08 page navigation.", ref
    return None


def _state_harness_changed_var(context: Mapping[str, Any], var_name: str) -> Mapping[str, Any] | None:
    state_harness = context.get("state_harness")
    if not isinstance(state_harness, Mapping):
        return None
    digest = state_harness.get("telemetry_window_digest")
    if not isinstance(digest, Mapping):
        return None
    changed_vars = digest.get("changed_vars")
    if not isinstance(changed_vars, (list, tuple)):
        return None
    for item in changed_vars:
        if not isinstance(item, Mapping):
            continue
        if item.get("var") == var_name:
            return item
    return None


def _refuel_probe_motion_state(context: Mapping[str, Any], step_id: str | None) -> str | None:
    vars_selected = context.get("vars")
    vars_map = vars_selected if isinstance(vars_selected, Mapping) else {}
    probe_value = _coerce_float(vars_map.get("ext_refuel_probe_value"))
    switch_value = _coerce_int(vars_map.get("probe_switch_value"))
    changed = _state_harness_changed_var(context, "ext_refuel_probe_value")
    first_value = _coerce_float(changed.get("first_value")) if isinstance(changed, Mapping) else None
    last_value = _coerce_float(changed.get("last_value")) if isinstance(changed, Mapping) else None
    is_extending = first_value is not None and last_value is not None and last_value > first_value
    is_retracting = first_value is not None and last_value is not None and last_value < first_value

    if step_id == "S20":
        if vars_map.get("probe_extended") is True or (probe_value is not None and probe_value >= 60000):
            return "s20_extended"
        if switch_value == 0 and probe_value is not None and 0 < probe_value < 60000 and is_extending:
            return "s20_extending"
    elif step_id == "S21":
        if vars_map.get("probe_retracted") is True or (probe_value is not None and probe_value <= 5000):
            return "s21_retracted"
        near_retract_threshold = probe_value is not None and 5000 < probe_value <= 7000
        if switch_value == 1 and probe_value is not None and probe_value > 5000 and (
            is_retracting or near_retract_threshold
        ):
            return "s21_retracting"
    return None


def _build_procedural_action_hint(
    *,
    inferred_step_id: str | None,
    vars_selected: Mapping[str, Any],
    allowed_targets: Sequence[str],
    step_interacted_targets: Sequence[str] | None = None,
    vision_fact_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if inferred_step_id == "S02":
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None

        def _hint(target: str, reason: str) -> dict[str, Any] | None:
            if target not in allowed:
                return None
            return {"target": target, "reason": reason}

        if vars_selected.get("fire_test_a_complete") is not True:
            return _hint(
                "fire_test_switch",
                "Hold the fire test switch to FIRE TEST A position and observe the aural/visual indications.",
            )
        if vars_selected.get("fire_test_b_complete") is not True:
            return _hint(
                "fire_test_switch",
                "Fire test A is complete. Now hold the fire test switch to FIRE TEST B position and wait ~10 seconds for the second set of indications.",
            )
        return None

    if inferred_step_id == "S03":
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None

        def _hint(target: str, reason: str) -> dict[str, Any] | None:
            if target not in allowed:
                return None
            return {"target": target, "reason": reason}

        if vars_selected.get("apu_on") is not True:
            return _hint("apu_switch", "Set the APU switch to ON first.")
        if vars_selected.get("apu_ready") is not True:
            return _hint(
                "apu_switch",
                "APU switch is already ON — wait for the green APU READY light to illuminate before proceeding.",
            )
        return None

    if inferred_step_id == "S06":
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None

        def _hint(target: str, reason: str) -> dict[str, Any] | None:
            if target not in allowed:
                return None
            return {"target": target, "reason": reason}

        if vars_selected.get("bleed_air_cycle_complete") is True:
            return None
        return _hint(
            "bleed_air_knob",
            "Rotate the BLEED AIR knob 360° clockwise (right-click 4 times) from NORM back to NORM.",
        )

    if inferred_step_id == "S08":
        return _build_s08_power_action_hint(
            vars_selected=vars_selected,
            allowed_targets=allowed_targets,
        )

    four_down_single_step_hints = {
        "S20": ("refuel_probe_switch", "Extend the refueling probe for the four-down check."),
        "S21": ("refuel_probe_switch", "Retract the refueling probe after confirming extension."),
        "S22": ("launch_bar_switch", "Extend the launch bar for the four-down check."),
        "S23": ("launch_bar_switch", "Retract the launch bar after confirming extension."),
        "S24": ("arresting_hook_handle", "Lower the arresting hook for the four-down check."),
        "S25": ("arresting_hook_handle", "Raise the arresting hook after confirming it is down."),
        "S26": ("pitot_heater_switch", "Turn pitot heat ON."),
        "S27": ("flap_switch", "Move the flap switch to AUTO."),
    }
    if inferred_step_id in four_down_single_step_hints:
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None
        target, reason = four_down_single_step_hints[str(inferred_step_id)]
        if target not in allowed:
            return None
        return {"target": target, "reason": reason}

    if inferred_step_id == "S14":
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None

        def _hint(target: str, reason: str) -> dict[str, Any] | None:
            if target not in allowed:
                return None
            return {"target": target, "reason": reason}

        if vars_selected.get("obogs_switch_on") is not True:
            return _hint("obogs_control_switch", "OBOGS control is not yet ON; switch OBOGS on first.")
        if vars_selected.get("obogs_flow_on") is not True:
            return _hint("obogs_flow_knob", "OBOGS control is already ON, but FLOW is not yet ON; set the OXY FLOW knob next.")
        return None

    if inferred_step_id == "S16":
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None

        def _hint(target: str, reason: str) -> dict[str, Any] | None:
            if target not in allowed:
                return None
            return {"target": target, "reason": reason}

        if vars_selected.get("flap_auto") is True:
            return None
        flap_mode = vars_selected.get("flap_mode_value")
        if isinstance(flap_mode, (int, float)) and int(flap_mode) == 0:
            return _hint(
                "flap_switch",
                "Flap switch is at FULL (cold-start default). Move the flap switch to AUTO.",
            )
        return _hint(
            "flap_switch",
            "Set the flap switch to AUTO before continuing.",
        )

    if inferred_step_id == "S12":
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None

        def _hint(target: str, reason: str) -> dict[str, Any] | None:
            if target not in allowed:
                return None
            return {"target": target, "reason": reason}

        if vars_selected.get("ins_fast_align_complete") is True:
            return None
        if vars_selected.get("ins_mode_cv_or_gnd") is True or vars_selected.get("ins_mode_set") is True:
            return _hint(
                "ampcd_pb19",
                "INS mode is set for alignment; press AMPCD PB19 to start the fast alignment self-test.",
            )
        return _hint(
            "ins_mode_knob",
            "Set the INS mode knob to GND for airfield startup or CV for carrier startup.",
        )

    if inferred_step_id == "S19":
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None
        seen_fact_ids = {
            item for item in (vision_fact_summary or {}).get("seen_fact_ids", [])
            if isinstance(item, str) and item
        } if isinstance(vision_fact_summary, Mapping) else set()

        def _hint(target: str, reason: str) -> dict[str, Any] | None:
            if target not in allowed:
                return None
            return {"target": target, "reason": reason}

        if "fcsmc_final_go_result_visible" in seen_fact_ids or _s19_final_go_seen_or_fresh(vision_fact_summary):
            return None
        if "right_mdi_pb5" in allowed and "fcs_bit_switch" in allowed:
            return {
                "targets": ["fcs_bit_switch", "right_mdi_pb5"],
                "reason": "Hold the FCS BIT switch up (Y) while pressing Right DDI PB5 to start the FCS BIT.",
            }
        return _hint(
            "fcs_bit_switch",
            "The right DDI is already on the FCS-MC page. Hold the FCS BIT switch up while pressing Right DDI PB5 to run the BIT.",
        )

    if inferred_step_id == "S18":
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None

        def _hint(target: str, reason: str) -> dict[str, Any] | None:
            if target not in allowed:
                return None
            return {"target": target, "reason": reason}

        return _hint(
            "right_mdi_pb5",
            "On the right DDI BIT FAILURES page, press PB5 to enter the FCS-MC BIT page before holding the FCS BIT switch.",
        )

    if inferred_step_id == "S29":
        allowed = {item for item in allowed_targets if isinstance(item, str) and item}
        if not allowed:
            return None

        def _hint(target: str, reason: str) -> dict[str, Any] | None:
            if target not in allowed:
                return None
            return {"target": target, "reason": reason}

        if vars_selected.get("bingo_fuel_set") is True:
            return None
        if vars_selected.get("ifei_up_pressed") is True or vars_selected.get("ifei_down_pressed") is True:
            return _hint(
                "ifei_up_button",
                "IFEI button press detected — the BINGO fuel value has been set; wait a moment for the updated value to register.",
            )
        return _hint(
            "ifei_up_button",
            "Press the IFEI UP button to set the BINGO fuel value for the mission.",
        )


    if inferred_step_id != "S09":
        return None
    if _s09_comm1_frequency_complete(vars_selected):
        return None

    allowed = {item for item in allowed_targets if isinstance(item, str) and item}
    if not allowed:
        return None

    scratchpad_text = _normalize_ufc_scratchpad_text(vars_selected)
    compact = scratchpad_text.replace(" ", "")
    payload = compact[3:] if compact.startswith("1--") else compact

    def _hint(target: str, reason: str) -> dict[str, Any] | None:
        if target not in allowed:
            return None
        return {"target": target, "reason": reason}

    if payload.endswith("134.000"):
        return _hint("ufc_ent_button", "The UFC scratchpad already shows 134.000 for COMM1 preset 1; press ENT to commit the frequency.")
    if payload.endswith("13.400") or payload.endswith("1.340") or payload.endswith(".134") or payload.endswith("1.34"):
        return _hint("ufc_key_0", "COMM1 preset entry is partway through 134.000; press 0 next.")
    if payload.endswith(".13"):
        return _hint("ufc_key_4", "COMM1 preset entry shows 13; press 4 next.")
    if payload.endswith(".1"):
        return _hint("ufc_key_3", "COMM1 preset entry shows 1; press 3 next.")
    if payload.endswith("305.000") or payload.endswith("305000"):
        return _hint("ufc_key_1", "COMM1 preset 1 is open on the scratchpad with the old 305.000 value; press 1 to begin entering 134.000.")
    if vars_selected.get("ufc_comm1_pull_pressed") is True and "305.000" not in payload:
        return _hint("ufc_key_1", "COMM1 preset entry is already open on the UFC scratchpad; start typing 134.000 with key 1.")
    if vars_selected.get("ufc_comm1_pull_pressed") is True:
        return _hint("ufc_key_1", "COMM1 preset entry has been opened on the UFC scratchpad; press 1 to begin entering 134.000.")
    return _hint("ufc_comm1_channel_selector_pull", "Pull the UFC COMM1 channel selector to open preset 1 in the scratchpad before entering 134.000.")


def _s12_fast_align_action_hint_allowed(
    *,
    context: Mapping[str, Any],
    hint: Mapping[str, Any],
    missing_conditions: set[str],
    action_hint: Any,
) -> bool:
    if not isinstance(action_hint, Mapping) or action_hint.get("target") != "ampcd_pb19":
        return False
    if "vars.ins_fast_align_complete==true" not in missing_conditions:
        return False
    if any("vars.ins_mode" in item for item in missing_conditions):
        return False

    mode_reason_codes = {"s12_requires_ins_mode_gnd", "s12_requires_ins_mode_cv"}
    gates = context.get("gates")
    if isinstance(gates, Mapping):
        gate = gates.get("S12.completion")
        if isinstance(gate, Mapping):
            reason_code = gate.get("reason_code")
            if gate.get("status") == "blocked" and reason_code in mode_reason_codes:
                return False

    gate_blockers = hint.get("gate_blockers")
    if isinstance(gate_blockers, (list, tuple)):
        for blocker in gate_blockers:
            if not isinstance(blocker, Mapping):
                continue
            reason_code = blocker.get("reason_code")
            raw_var = blocker.get("var")
            if reason_code in mode_reason_codes or (
                isinstance(raw_var, str) and "ins_mode" in raw_var
            ):
                return False

    vars_map = context.get("vars")
    vars_selected = vars_map if isinstance(vars_map, Mapping) else {}
    scenario_profile = context.get("scenario_profile")
    if not isinstance(scenario_profile, str) or not scenario_profile:
        scenario_profile = hint.get("scenario_profile")

    ins_mode = vars_selected.get("ins_mode")
    if isinstance(ins_mode, (int, float)) and not isinstance(ins_mode, bool):
        normalized_mode = int(ins_mode)
        if scenario_profile == "carrier":
            return normalized_mode == 1
        return normalized_mode == 2

    return (
        vars_selected.get("ins_mode_cv_or_gnd") is True
        or vars_selected.get("ins_mode_set") is True
    )


def _resolve_overlay_step_id(
    inferred_step_id: str | None,
    *,
    missing_conditions: Sequence[str],
    candidate_steps: Sequence[str],
    step_order_index: Mapping[str, int],
    observability_status: str | None = None,
    requires_visual_confirmation: bool = False,
) -> str | None:
    if not isinstance(inferred_step_id, str) or not inferred_step_id:
        return None
    if requires_visual_confirmation:
        return inferred_step_id
    if isinstance(observability_status, str) and observability_status in {"partial", "unobservable"}:
        return inferred_step_id
    if any(isinstance(item, str) and item for item in missing_conditions):
        return inferred_step_id
    current_idx = step_order_index.get(inferred_step_id)
    if current_idx is None:
        return inferred_step_id
    for next_idx in range(current_idx + 1, len(candidate_steps)):
        next_step_id = candidate_steps[next_idx]
        if isinstance(next_step_id, str) and next_step_id:
            return next_step_id
    return inferred_step_id


def _normalize_rule_var_ref(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    if value.startswith("payload.vars."):
        value = value[len("payload.vars.") :]
    elif value.startswith("vars."):
        value = value[len("vars.") :]
    elif "." in value:
        return None
    if not value:
        return None
    return f"VARS.{value}"


def _collect_step_gate_var_refs(
    step_id: str,
    *,
    precondition_gates: Mapping[str, Any],
    completion_gates: Mapping[str, Any],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for gate_map in (precondition_gates, completion_gates):
        rules_raw = gate_map.get(step_id)
        rules: Iterable[Any]
        if isinstance(rules_raw, Mapping):
            rules = (rules_raw,)
        elif isinstance(rules_raw, Iterable) and not isinstance(rules_raw, (str, bytes)):
            rules = rules_raw
        else:
            continue
        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            ref = _normalize_rule_var_ref(rule.get("var"))
            if ref is None or ref in seen:
                continue
            seen.add(ref)
            out.append(ref)
    return out


def _build_step_fallback_profiles(
    pack_steps: Sequence[Mapping[str, Any]],
    *,
    precondition_gates: Mapping[str, Any],
    completion_gates: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for step in pack_steps:
        if not isinstance(step, Mapping):
            continue
        step_id = _extract_step_id(step)
        if not isinstance(step_id, str) or not step_id or step_id in profiles:
            continue
        ui_targets = _normalize_step_ui_targets(step.get("ui_targets"))
        gate_var_refs = _collect_step_gate_var_refs(
            step_id,
            precondition_gates=precondition_gates,
            completion_gates=completion_gates,
        )
        profiles[step_id] = {
            "ui_targets": ui_targets,
            "gate_var_refs": gate_var_refs,
            "overlay_enabled": bool(step.get("overlay_enabled", True)),
        }
    return profiles


def _resolve_step_overlay_allowlist(
    inferred_step_id: str | None,
    *,
    step_fallback_profiles: Mapping[str, Mapping[str, Any]],
    overlay_allowset: set[str],
    default_allowlist: Sequence[str],
    deterministic_hint: Mapping[str, Any] | None = None,
) -> list[str]:
    if not isinstance(inferred_step_id, str) or not inferred_step_id:
        return list(default_allowlist)
    profile = step_fallback_profiles.get(inferred_step_id)
    if not isinstance(profile, Mapping):
        return list(default_allowlist)
    if profile.get("overlay_enabled") is False:
        return []
    step_targets = _normalize_step_ui_targets(profile.get("ui_targets"))
    hint = deterministic_hint if isinstance(deterministic_hint, Mapping) else {}
    missing_conditions = hint.get("missing_conditions")
    gate_blockers = hint.get("gate_blockers")
    has_hard_blocker = hint_has_hard_blocker(missing_conditions, gate_blockers)
    narrowed = [target for target in step_targets if target in overlay_allowset]
    if has_hard_blocker:
        if narrowed:
            return narrowed
        return list(default_allowlist)
    observability = hint.get("observability_status")
    if not isinstance(observability, str) or not observability:
        observability = hint.get("observability")
    if observability in {"partial", "unobservable"}:
        recent_targets = _normalize_step_ui_targets(hint.get("recent_ui_targets"))
        merged: list[str] = []
        seen: set[str] = set()
        for target in [*step_targets, *recent_targets]:
            if target not in overlay_allowset or target in seen:
                continue
            seen.add(target)
            merged.append(target)
        if merged:
            return merged
    if narrowed:
        return narrowed
    return list(default_allowlist)


def _collect_request_evidence_refs(context: Mapping[str, Any]) -> set[str]:
    return collect_evidence_refs_from_context(context)


@dataclass
class HelpCacheEntry:
    state_key: str
    t_wall: float
    response: TutorResponse


@dataclass
class LiveLoopStats:
    frames: int = 0
    help_cycles: int = 0
    model_calls: int = 0
    cache_hits: int = 0
    vision_cycles: int = 0
    vision_sync_miss_count: int = 0
    vision_text_fallback_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "help_cycles": self.help_cycles,
            "model_calls": self.model_calls,
            "cache_hits": self.cache_hits,
            "vision_cycles": self.vision_cycles,
            "vision_sync_miss_count": self.vision_sync_miss_count,
            "vision_text_fallback_count": self.vision_text_fallback_count,
        }


class ReplayBiosReceiver:
    """
    Replay BIOS frames from JSONL.

    Supported line formats:
    - raw frame object: {"schema_version":"v2","seq":...,"bios":...}
    - Event envelope with observation payload
    - Observation object serialized by Observation.to_dict()
    """

    def __init__(self, path: str | Path, source: str = "dcs_bios_replay", speed: float = 1.0) -> None:
        self.path = Path(path)
        self.source = source
        self.speed = float(speed)
        if not math.isfinite(self.speed) or self.speed < 0:
            raise ValueError("speed must be a finite number >= 0")
        self._fh = self.path.open("r", encoding="utf-8")
        self._lineno = 0
        self.is_exhausted = False
        self._replay_origin_t_wall: float | None = None
        self._wall_start_monotonic: float | None = None

    def _next_item(self) -> dict[str, Any] | None:
        while True:
            line = self._fh.readline()
            if not line:
                self.is_exhausted = True
                self.close()
                return None
            self._lineno += 1
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                self.is_exhausted = True
                self.close()
                raise ValueError(f"{self.path}:{self._lineno} invalid JSON: {exc}") from exc
            if isinstance(obj, Mapping):
                return dict(obj)
            # Ignore non-mapping JSON values and keep scanning the stream.
            continue

    def _extract_frame(self, item: Mapping[str, Any]) -> dict[str, Any] | None:
        if item.get("schema_version") == "v2" and isinstance(item.get("bios"), Mapping):
            return dict(item)

        payload = item.get("payload")
        if isinstance(payload, Mapping):
            if payload.get("schema_version") == "v2" and isinstance(payload.get("bios"), Mapping):
                return dict(payload)
            nested = payload.get("payload")
            if isinstance(nested, Mapping):
                if nested.get("schema_version") == "v2" and isinstance(nested.get("bios"), Mapping):
                    return dict(nested)
        return None

    def get_observation(self) -> Observation | None:
        while True:
            item = self._next_item()
            if item is None:
                return None
            frame = self._extract_frame(item)
            if frame is None:
                continue
            self._pace_by_frame_t_wall(frame)
            seq = _coerce_int(frame.get("seq"))
            meta: dict[str, Any] = {"replay": True}
            if seq is not None:
                meta["seq"] = seq
            return Observation(source=self.source, payload=frame, metadata=meta)

    def _pace_by_frame_t_wall(self, frame: Mapping[str, Any]) -> None:
        if self.speed == 0:
            return
        frame_t_wall = _coerce_float(frame.get("t_wall"))
        if frame_t_wall is None:
            return

        if self._replay_origin_t_wall is None or self._wall_start_monotonic is None:
            self._replay_origin_t_wall = frame_t_wall
            self._wall_start_monotonic = time.monotonic()
            return

        elapsed_replay_s = max(0.0, frame_t_wall - self._replay_origin_t_wall)
        target_elapsed_s = elapsed_replay_s / self.speed
        target_wall = self._wall_start_monotonic + target_elapsed_s
        sleep_s = target_wall - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
        self.is_exhausted = True


class StdinHelpTrigger:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._queue: queue.SimpleQueue[None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._close_wait_timeout_s = 0.2
        self.close_pending_input = False

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            # input() can block; wait briefly, then mark that stdin is still pending.
            self._thread.join(timeout=self._close_wait_timeout_s)
        self.close_pending_input = self._thread.is_alive()

    def poll(self) -> bool:
        try:
            self._queue.get_nowait()
            return True
        except queue.Empty:
            return False

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                line = input()
            except EOFError:
                return
            if self._stop.is_set():
                return
            if line.strip().lower() in {"", "help", "h", "?"}:
                self._queue.put(None)


def _is_help_trigger_payload(text: str) -> bool:
    normalized = text.strip().lower()
    # UDP trigger must be explicit to avoid accidental empty-datagram activation.
    if normalized in {"help", "h", "?"}:
        return True
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, Mapping):
        return False
    intent = obj.get("intent")
    if isinstance(intent, str) and intent.strip().lower() == "help":
        return True
    action = obj.get("action")
    if isinstance(action, str) and action.strip().lower() == "help":
        return True
    event = obj.get("event")
    if isinstance(event, str) and event.strip().lower() == "help":
        return True
    return False


class UdpHelpTrigger:
    def __init__(self, host: str = "127.0.0.1", port: int = 7794, timeout: float = 0.2) -> None:
        if int(port) < 0:
            raise ValueError("port must be >= 0")
        self.host = host
        self.port = int(port)
        self.timeout = max(0.01, float(timeout))
        self._stop = threading.Event()
        self._queue: queue.SimpleQueue[None] = queue.SimpleQueue()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self.timeout)
        self._sock.bind((self.host, self.port))
        self._thread = threading.Thread(target=self._reader, daemon=True)

    @property
    def bound_port(self) -> int:
        return int(self._sock.getsockname()[1])

    def start(self) -> None:
        self._thread.start()

    def poll(self) -> bool:
        try:
            self._queue.get_nowait()
            return True
        except queue.Empty:
            return False

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=self.timeout + 0.2)

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                payload, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            text = payload.decode("utf-8", errors="ignore")
            if _is_help_trigger_payload(text):
                self._queue.put(None)


class CompositeHelpTrigger:
    def __init__(self, triggers: Sequence[HelpTriggerLike]) -> None:
        self._triggers = list(triggers)

    def poll(self) -> bool:
        for trigger in self._triggers:
            if trigger.poll():
                return True
        return False


class UdpVisionCaptureNotifier:
    def __init__(
        self,
        *,
        session_id: str,
        host: str = DEFAULT_VISION_CAPTURE_TRIGGER_HOST,
        port: int = DEFAULT_VISION_CAPTURE_TRIGGER_PORT,
        sock: Any | None = None,
    ) -> None:
        normalized_session_id = str(session_id).strip()
        if not normalized_session_id:
            raise ValueError("session_id must be a non-empty string")
        if int(port) < 0:
            raise ValueError("port must be >= 0")
        self.session_id = normalized_session_id
        self.host = host
        self.port = int(port)
        self._sock = sock if sock is not None else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def notify_help(self) -> None:
        if self.port <= 0:
            return
        payload = build_capture_request_payload(session_id=self.session_id, reason="help")
        try:
            self._sock.sendto(payload, (self.host, self.port))
        except OSError as exc:
            print(
                f"[VISION_CAPTURE] notify_help send failed for {self.host}:{self.port}: "
                f"{type(exc).__name__}: {exc}"
            )

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def _emit_multi_target_overlay_config_warning(
    *,
    max_overlay_targets: int,
    config_path: Path | None = None,
    event_sink: Callable[[Event], None] | None = None,
    prefix: str = "[LIVE_DCS]",
) -> str | None:
    warning = build_multi_target_overlay_config_warning(
        max_overlay_targets=max_overlay_targets,
        config_path=config_path,
    )
    if warning is None:
        return None

    print(f"{prefix} WARNING: {warning}")
    if event_sink is not None:
        event_sink(
            Event(
                kind="system",
                payload={
                    "event": "overlay_config_warning",
                    "warning": warning,
                    "max_overlay_targets": max(0, int(max_overlay_targets)),
                    "config_path": str(config_path) if config_path is not None else None,
                },
            )
        )
    return warning


class LiveDcsTutorLoop:
    def __init__(
        self,
        *,
        source: ObservationSource,
        model: Any,
        action_executor: ActionExecutorLike,
        resolver: VarResolver | None = None,
        mapper: BiosUiMapper | None = None,
        pack_path: str | Path | None = None,
        ui_map_path: str | Path | None = None,
        telemetry_map_path: str | Path | None = None,
        bios_to_ui_path: str | Path | None = None,
        cooldown_s: float = 4.0,
        session_id: str | None = None,
        lang: str = "zh",
        scenario_profile: str = DEFAULT_SCENARIO_PROFILE,
        event_sink: Callable[[Event], None] | None = None,
        dry_run_overlay: bool = False,
        knowledge_adapter: KnowledgePort | None = None,
        knowledge_index_path: str | Path | None = None,
        rag_top_k: int = 5,
        cold_start_production: bool = False,
        knowledge_source_policy_path: str | Path | None = None,
        vision_port: Any | None = None,
        vision_session_id: str | None = None,
        vision_mode: str = "live",
        vision_sync_window_ms: int | None = None,
        vision_trigger_wait_ms: int | None = None,
        vision_fact_extractor: Any | None = None,
        vision_model_name: str | None = None,
        max_overlay_targets: int = 1,
        tutor_text_sender: TutorTextSenderLike | None = None,
        tutor_text_display_time_s: float = 12.0,
        tutor_text_clear_view: bool = False,
    ) -> None:
        self.source = source
        self.model = model
        self.action_executor = action_executor
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.session_id = session_id
        self.lang = "zh" if lang not in {"zh", "en"} else lang
        self.scenario_profile = normalize_scenario_profile(scenario_profile)
        self.event_sink = event_sink
        self.dry_run_overlay = dry_run_overlay
        self.vision_mode = _normalize_vision_mode(vision_mode)
        self.max_overlay_targets = max(0, int(max_overlay_targets))
        self.tutor_text_sender = tutor_text_sender
        self.tutor_text_display_time_s = max(0.1, float(tutor_text_display_time_s))
        self.tutor_text_clear_view = bool(tutor_text_clear_view)

        self.pack_path = Path(pack_path) if pack_path else _default_pack_path()
        self.ui_map_path = Path(ui_map_path) if ui_map_path else _default_ui_map_path()
        self.telemetry_map_path = (
            Path(telemetry_map_path) if telemetry_map_path else _default_telemetry_map_path()
        )
        self.bios_to_ui_path = Path(bios_to_ui_path) if bios_to_ui_path else _default_bios_to_ui_path()
        raw_knowledge_index_path = (
            Path(knowledge_index_path) if knowledge_index_path else _default_knowledge_index_path()
        )
        self.knowledge_index_path = _normalize_fs_path(raw_knowledge_index_path)
        self.rag_top_k = max(0, int(rag_top_k))
        self.cold_start_production = bool(cold_start_production)
        self.knowledge_source_policy_path = (
            Path(knowledge_source_policy_path) if knowledge_source_policy_path else None
        )
        self.knowledge_source_policy: KnowledgeSourcePolicy | None = None
        self.pack_title = _load_pack_title(self.pack_path)
        self.vision_priority_steps = _load_vision_priority_steps(self.pack_path)
        self.vision_priority_step_set = set(self.vision_priority_steps)
        self._load_knowledge_source_policy()

        self.resolver = resolver if resolver is not None else VarResolver.from_yaml(self.telemetry_map_path)
        self.mapper = (
            mapper
            if mapper is not None
            else BiosUiMapper.from_yaml(self.bios_to_ui_path, self.ui_map_path)
        )
        self.knowledge: KnowledgePort | None = knowledge_adapter
        if self.knowledge is None and self.rag_top_k > 0:
            self.knowledge = LocalKnowledgeAdapter(
                index_path=self.knowledge_index_path,
                source_policy=self.knowledge_source_policy,
            )
        self.pack_steps = load_pack_steps(self.pack_path)
        self.step_harness_specs = load_step_harness_specs(
            self.pack_path,
            scenario_profile=self.scenario_profile,
        )
        self.step_signal_profiles = step_signal_profiles_from_specs(self.step_harness_specs)
        gate_config = load_pack_gate_config(
            self.pack_path,
            scenario_profile=self.scenario_profile,
        )
        self.precondition_gates = dict(gate_config.get("precondition_gates", {}))
        self.completion_gates = dict(gate_config.get("completion_gates", {}))
        self.candidate_steps = _load_step_ids(self.pack_path)
        self.overlay_allowlist = _load_overlay_allowlist(self.pack_path, self.ui_map_path)
        self.overlay_allowset = set(self.overlay_allowlist)
        self.step_fallback_profiles = step_fallback_profiles_from_specs(self.step_harness_specs)
        self.recent_ring = RecentDeltaRingBuffer(window_s=8.0, max_items=20)
        self.telemetry_window_ring = RecentDeltaRingBuffer(window_s=8.0, max_items=20)
        self.vision_sync_window_ms = (
            int(vision_sync_window_ms)
            if isinstance(vision_sync_window_ms, int) and vision_sync_window_ms > 0
            else (
                DEFAULT_LIVE_SYNC_WINDOW_MS
                if self.vision_mode == "live"
                else DEFAULT_REPLAY_SYNC_WINDOW_MS
            )
        )
        live_trigger_wait_ms = (
            DEFAULT_LIVE_TRIGGER_WAIT_MS if self.vision_mode == "live" else 0
        )
        self.vision_trigger_wait_ms = (
            int(vision_trigger_wait_ms)
            if isinstance(vision_trigger_wait_ms, int) and vision_trigger_wait_ms >= 0
            else live_trigger_wait_ms
        )
        effective_vision_session_id = vision_session_id or self.session_id
        self.vision_session_id = effective_vision_session_id
        self._vision_session: BufferedVisionSession | None = None
        self.vision_fact_extractor = (
            vision_fact_extractor
            if vision_fact_extractor is not None
            else _build_vision_fact_extractor_from_model(
                model=self.model,
                lang=self.lang,
                pack_path=self.pack_path,
                vision_model_name=vision_model_name,
            )
        )
        self._vision_fact_config = _resolve_vision_fact_config(
            extractor=self.vision_fact_extractor,
            pack_path=self.pack_path,
        )
        if vision_port is not None:
            if not isinstance(effective_vision_session_id, str) or not effective_vision_session_id:
                raise ValueError("vision_session_id or session_id is required when vision_port is configured")
            self._vision_session = BufferedVisionSession(
                vision_port=vision_port,
                session_id=effective_vision_session_id,
                sync_window_ms=self.vision_sync_window_ms,
                trigger_wait_ms=self.vision_trigger_wait_ms,
                live_mode=self.vision_mode == "live",
                observation_sink=lambda observation: _emit_vision_observation_event(
                    observation=observation,
                    event_sink=self.event_sink,
                    fallback_session_id=self.session_id,
                ),
            )

        self._latest_raw_obs: Observation | None = None
        self._latest_enriched_obs: Observation | None = None
        self._help_cache: HelpCacheEntry | None = None
        self._vision_fact_snapshot: dict[str, dict[str, Any]] = {}
        self._accumulated_vars: dict[str, Any] = {}
        self._step_interacted_targets: set[str] = set()
        self._last_inferred_step_id: str | None = None
        self._step_order_index = {
            step_id: idx for idx, step_id in enumerate(self.candidate_steps) if isinstance(step_id, str) and step_id
        }
        self._sticky_inference_step_id: str | None = None
        self._sticky_inference_missing_conditions: tuple[str, ...] = ()
        self._pending_help_trigger_t_wall: float | None = None
        self._stats = LiveLoopStats()

    @property
    def stats(self) -> LiveLoopStats:
        return self._stats

    def close(self) -> None:
        if self._vision_session is not None:
            self._vision_session.close()
        if self.vision_fact_extractor is not None and hasattr(self.vision_fact_extractor, "close"):
            self.vision_fact_extractor.close()
        if self.tutor_text_sender is not None and hasattr(self.tutor_text_sender, "close"):
            self.tutor_text_sender.close()
        if hasattr(self.action_executor, "close"):
            self.action_executor.close()
        if hasattr(self.source, "close"):
            self.source.close()
        if hasattr(self.model, "close"):
            self.model.close()

    def _clear_live_progress_state(self) -> None:
        self._vision_fact_snapshot = {}
        self._step_interacted_targets = set()
        self._last_inferred_step_id = None
        self._sticky_inference_step_id = None
        self._sticky_inference_missing_conditions = ()

    def _remember_step_interactions(self, targets: Sequence[str] | None) -> None:
        if not isinstance(self._last_inferred_step_id, str) or not self._last_inferred_step_id:
            return
        for target in targets or ():
            if isinstance(target, str) and target:
                self._step_interacted_targets.add(target)

    def _ensure_knowledge(self) -> KnowledgePort:
        if self.knowledge is None:
            self.knowledge = LocalKnowledgeAdapter(
                index_path=self.knowledge_index_path,
                source_policy=self.knowledge_source_policy,
            )
        return self.knowledge

    def _load_knowledge_source_policy(self) -> None:
        policy_path = self.knowledge_source_policy_path
        if self.cold_start_production and policy_path is None:
            policy_path = _default_knowledge_source_policy_path()
            if not policy_path.is_file():
                raise ValueError(
                    "cold-start production requires valid knowledge source policy: "
                    f"default policy file {policy_path.name!r} not found in repository checkout. "
                    "Provide --knowledge-source-policy explicitly."
                )
        if policy_path is None:
            return

        try:
            policy = KnowledgeSourcePolicy.from_yaml(
                policy_path,
                index_path=self.knowledge_index_path,
            )
        except KnowledgeSourcePolicyError as exc:
            if self.cold_start_production:
                sanitized = _sanitize_policy_error_for_user(
                    str(exc),
                    path_hints=(
                        policy_path,
                        self.knowledge_index_path,
                    ),
                )
                raise ValueError(
                    "cold-start production requires valid knowledge source policy: "
                    f"{sanitized}"
                ) from exc
            raise

        self.knowledge_source_policy = policy
        if self.cold_start_production:
            print(
                "[KNOWLEDGE_POLICY] 当前仅使用 cold-start 白名单块 "
                f"{policy.public_startup_info()}"
            )

    def _knowledge_store_id(self) -> str | None:
        knowledge = self.knowledge
        if knowledge is None:
            return None
        for attr_name in ("index_path", "store_id", "knowledge_id", "name"):
            raw = getattr(knowledge, attr_name, None)
            if raw is None:
                continue
            if isinstance(raw, Path):
                return str(raw)
            text = str(raw).strip()
            if text:
                return text
        return None

    def _emit_event(
        self,
        *,
        kind: str,
        payload: Mapping[str, Any],
        related_id: str | None = None,
        t_wall: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        vision_refs: Sequence[str] | None = None,
    ) -> None:
        if self.event_sink is None:
            return
        event = Event(
            kind=kind,
            payload=dict(payload),
            related_id=related_id,
            t_wall=t_wall,
            session_id=self.session_id,
            vision_refs=[item for item in (vision_refs or []) if isinstance(item, str) and item],
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
        self.event_sink(event)

    def _poll_vision_sidecar(self) -> None:
        if self._vision_session is not None:
            self._vision_session.poll()

    def _build_vision_selection(
        self,
        *,
        observation: Observation,
        trigger_t_wall: float,
    ) -> HelpCycleVisionSelection:
        normalized_trigger_t_wall = _coerce_finite_float(trigger_t_wall)
        if normalized_trigger_t_wall is None:
            normalized_trigger_t_wall = time.time()
        trigger_wall_ms = int(round(float(normalized_trigger_t_wall) * 1000.0))
        payload = observation.payload if isinstance(observation.payload, Mapping) else {}
        observation_seq = _coerce_int(payload.get("seq"))
        observation_t_wall_s = _coerce_finite_float(payload.get("t_wall"))
        if observation_t_wall_s is None:
            observation_t_wall_s = normalized_trigger_t_wall
        observation_t_wall_ms = int(round(float(observation_t_wall_s) * 1000.0))
        if self._vision_session is None:
            return HelpCycleVisionSelection(
                status="vision_unavailable",
                observation_ref=observation.observation_id,
                observation_seq=observation_seq,
                observation_t_wall_s=observation_t_wall_s,
                observation_t_wall_ms=observation_t_wall_ms,
                trigger_wall_ms=trigger_wall_ms,
                sync_window_ms=self.vision_sync_window_ms,
                vision_used=False,
                frame_id=None,
                sync_status=None,
                sync_delta_ms=None,
                frame_stale=None,
                frame_ids=[],
                selected_frames=[],
                pre_trigger_frame=None,
                trigger_frame=None,
                sync_miss_reason="vision_port_unconfigured",
            )
        selection = self._vision_session.select_for_help(trigger_wall_s=normalized_trigger_t_wall)
        return HelpCycleVisionSelection(
            status=selection.status,
            observation_ref=observation.observation_id,
            observation_seq=observation_seq,
            observation_t_wall_s=observation_t_wall_s,
            observation_t_wall_ms=observation_t_wall_ms,
            trigger_wall_ms=selection.trigger_wall_ms,
            sync_window_ms=selection.sync_window_ms,
            vision_used=selection.vision_used,
            frame_id=selection.frame_id,
            sync_status=selection.sync_status,
            sync_delta_ms=selection.sync_delta_ms,
            frame_stale=selection.frame_stale,
            frame_ids=list(selection.frame_ids),
            selected_frames=[dict(item) for item in selection.selected_frames],
            pre_trigger_frame=dict(selection.pre_trigger_frame) if selection.pre_trigger_frame is not None else None,
            trigger_frame=dict(selection.trigger_frame) if selection.trigger_frame is not None else None,
            sync_miss_reason=selection.sync_miss_reason,
        )

    def _ingest_observation(self, raw_obs: Observation) -> Observation:
        self._latest_raw_obs = raw_obs
        enriched = enrich_bios_observation(
            raw_obs,
            self.resolver,
            mapper=self.mapper,
            delta_stream_id=self.session_id,
        )
        self._latest_enriched_obs = enriched

        enriched_payload = enriched.payload if isinstance(enriched.payload, Mapping) else {}
        enriched_vars = enriched_payload.get("vars")
        if isinstance(enriched_vars, Mapping):
            self._accumulated_vars.update(enriched_vars)
            if self._should_reset_sticky_inference(self._accumulated_vars):
                self._clear_live_progress_state()

        payload = raw_obs.payload if isinstance(raw_obs.payload, Mapping) else {}
        delta = payload.get("delta")
        t_wall = _coerce_float(payload.get("t_wall"))
        seq = _coerce_int(payload.get("seq"))
        if isinstance(delta, Mapping):
            current_delta_targets = project_recent_ui_targets(
                [{"t_wall": t_wall, "seq": seq, "delta": delta}],
                self.mapper,
                max_items=8,
            )
            self._remember_step_interactions(current_delta_targets)
        if isinstance(delta, Mapping) and t_wall is not None:
            self.recent_ring.add_delta(delta, t_wall=t_wall, seq=seq)
        if isinstance(enriched_vars, Mapping) and t_wall is not None:
            self.telemetry_window_ring.add_delta(enriched_vars, t_wall=t_wall, seq=seq)

        self._emit_event(
            kind="observation",
            payload=enriched.to_dict(),
            related_id=enriched.observation_id,
            t_wall=t_wall,
        )
        return enriched

    def _build_grounding_context(
        self,
        deterministic_hint: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        inferred_step_id = deterministic_hint.get("inferred_step_id")
        if not isinstance(inferred_step_id, str) or not inferred_step_id:
            inferred_step_id = None

        missing_conditions_raw = deterministic_hint.get("missing_conditions", [])
        missing_conditions = (
            [item for item in missing_conditions_raw if isinstance(item, str) and item]
            if isinstance(missing_conditions_raw, (list, tuple))
            else []
        )
        recent_targets_raw = deterministic_hint.get("recent_ui_targets", [])
        recent_ui_targets = (
            [item for item in recent_targets_raw if isinstance(item, str) and item]
            if isinstance(recent_targets_raw, list)
            else []
        )

        query = build_grounding_query(
            pack_title=self.pack_title,
            inferred_step=inferred_step_id,
            missing_conditions=missing_conditions,
            recent_ui_targets=recent_ui_targets,
        )

        if self.rag_top_k <= 0:
            return [], {
                "grounding_query": query,
                "grounding_missing": True,
                "grounding_reason": "rag_disabled",
                "grounding_snippet_ids": [],
                "grounding_cache_hit": False,
                "grounding_index_path": str(self.knowledge_index_path),
                "grounding_top_k": self.rag_top_k,
            }

        snippets: list[dict[str, Any]]
        retrieve_meta: dict[str, Any]
        knowledge: KnowledgePort | None = None
        try:
            knowledge = self._ensure_knowledge()
            if isinstance(knowledge, KnowledgeRetrieveWithMetaPort):
                queried, raw_meta = knowledge.retrieve_with_meta(
                    query,
                    top_k=self.rag_top_k,
                    step_id=inferred_step_id,
                )
                snippets = [
                    _normalize_knowledge_snippet(item, idx)
                    for idx, item in enumerate(queried)
                    if isinstance(item, Mapping)
                ]
                retrieve_meta = _normalize_retrieve_meta(raw_meta)
            else:
                queried = knowledge.query(query, k=self.rag_top_k)
                snippets = [
                    _normalize_knowledge_snippet(item, idx)
                    for idx, item in enumerate(queried)
                    if isinstance(item, Mapping)
                ]
                retrieve_meta = _normalize_retrieve_meta(
                    {
                        "cache_hit": False,
                        "grounding_missing": False,
                        "grounding_reason": None,
                        "snippet_ids": [
                            item.get("snippet_id") for item in snippets if isinstance(item.get("snippet_id"), str)
                        ],
                        "index_path": self._knowledge_store_id(),
                    }
                )
        except Exception as exc:
            snippets = []
            retrieve_meta = _normalize_retrieve_meta(
                {
                    "cache_hit": False,
                    "grounding_missing": True,
                    "grounding_reason": "knowledge_retrieve_error",
                    "grounding_error_type": type(exc).__name__,
                    "snippet_ids": [],
                    "index_path": self._knowledge_store_id()
                    if knowledge is not None
                    else str(self.knowledge_index_path),
                }
            )

        policy_filtered_out_count = 0
        policy_id: str | None = None
        policy_version: str | None = None
        source_chunk_refs = list(retrieve_meta.get("source_chunk_refs") or [])
        if self.knowledge_source_policy is not None and not bool(retrieve_meta.get("source_policy_applied")):
            policy_id = self.knowledge_source_policy.policy_id
            policy_version = self.knowledge_source_policy.policy_version
            before_filter_count = len(snippets)
            snippets = self.knowledge_source_policy.filter_snippets(snippets)
            policy_filtered_out_count = max(0, before_filter_count - len(snippets))
            if (
                before_filter_count > 0
                and not snippets
                and not bool(retrieve_meta.get("grounding_missing"))
            ):
                retrieve_meta = dict(retrieve_meta)
                retrieve_meta["grounding_missing"] = True
                retrieve_meta["grounding_reason"] = "policy_filtered_all"
            source_chunk_refs = []
            for item in snippets:
                ref = build_source_chunk_ref(item)
                if ref is not None:
                    source_chunk_refs.append(ref)
        elif bool(retrieve_meta.get("source_policy_applied")):
            policy_id_raw = retrieve_meta.get("source_policy_id")
            policy_version_raw = retrieve_meta.get("source_policy_version")
            policy_id = policy_id_raw if isinstance(policy_id_raw, str) and policy_id_raw else None
            policy_version = (
                policy_version_raw
                if isinstance(policy_version_raw, str) and policy_version_raw
                else None
            )
            policy_filtered_out_count = int(retrieve_meta.get("source_policy_filtered_out_count") or 0)

        snippet_ids = [
            item.get("snippet_id")
            for item in snippets
            if isinstance(item.get("snippet_id"), str) and item.get("snippet_id")
        ]
        grounding_missing = bool(retrieve_meta.get("grounding_missing"))
        grounding_reason = retrieve_meta.get("grounding_reason")
        if grounding_missing and not isinstance(grounding_reason, str):
            grounding_reason = "index_missing"

        grounding_index_path = retrieve_meta.get("index_path")
        if grounding_index_path is None and isinstance(knowledge, LocalKnowledgeAdapter):
            grounding_index_path = str(self.knowledge_index_path)
        elif isinstance(grounding_index_path, Path):
            grounding_index_path = str(grounding_index_path)

        return snippets, {
            "grounding_query": query,
            "grounding_missing": grounding_missing,
            "grounding_reason": grounding_reason,
            "grounding_error_type": retrieve_meta.get("grounding_error_type")
            or retrieve_meta.get("index_error_type"),
            "grounding_snippet_ids": snippet_ids,
            "source_chunk_refs": source_chunk_refs,
            "grounding_cache_hit": bool(retrieve_meta.get("cache_hit")),
            "grounding_index_path": grounding_index_path,
            "grounding_top_k": self.rag_top_k,
            "grounding_policy_id": policy_id,
            "grounding_policy_version": policy_version,
            "grounding_policy_filtered_out_count": policy_filtered_out_count,
        }

    def _build_request(
        self,
        obs: Observation,
        *,
        vision_selection: HelpCycleVisionSelection,
        vision_fact_context: Mapping[str, Any],
        request_id_override: str | None = None,
    ) -> tuple[TutorRequest, dict[str, Any], str]:
        payload = obs.payload if isinstance(obs.payload, Mapping) else {}
        vars_map = payload.get("vars")
        if not isinstance(vars_map, Mapping):
            vars_map = {}
        vars_selected = dict(self._accumulated_vars)
        vars_selected.update(vars_map)
        vision_context = vision_selection.to_dict()
        vision_context["main_help_multimodal_attached"] = False

        now_t_wall = _coerce_float(payload.get("t_wall"))
        recent_frames = self.recent_ring.snapshot(now_t_wall=now_t_wall) if now_t_wall is not None else self.recent_ring.snapshot()
        telemetry_window_frames_raw = (
            self.telemetry_window_ring.snapshot(now_t_wall=now_t_wall)
            if now_t_wall is not None
            else self.telemetry_window_ring.snapshot()
        )
        telemetry_window_frames = _telemetry_window_frames_from_var_snapshots(telemetry_window_frames_raw)
        recent_deltas = build_prompt_recent_deltas(recent_frames, self.mapper, max_items=20)
        recent_actions = build_recent_button_signal(recent_frames, self.mapper, max_items=8)
        recent_buttons = [
            item
            for item in recent_actions.get("recent_buttons", [])
            if isinstance(item, str) and item
        ]
        all_gates = evaluate_pack_gates(
            observations=[obs.to_dict()],
            precondition_gates=self.precondition_gates,
            completion_gates=self.completion_gates,
        )
        inference = infer_step_id(
            self.pack_steps,
            vars_selected,
            recent_buttons,
            gates=all_gates,
            precondition_gates=self.precondition_gates,
            completion_gates=self.completion_gates,
            scenario_profile=self.scenario_profile,
            pack_path=self.pack_path,
            vision_facts=vision_fact_context.get("vision_facts"),
        )
        inference = self._stabilize_live_inference(
            inference,
            vars_selected,
            recent_ui_targets=recent_buttons,
        )

        new_step_id = inference.inferred_step_id
        if new_step_id != self._last_inferred_step_id:
            self._step_interacted_targets = set()
            self._last_inferred_step_id = new_step_id if isinstance(new_step_id, str) else None
        self._remember_step_interactions(recent_buttons)

        gates = _select_gates_for_context(
            all_gates,
            inferred_step_id=inference.inferred_step_id,
            max_items=8,
        )
        inferred_gate_blockers: list[dict[str, str]] = []
        if isinstance(inference.inferred_step_id, str) and inference.inferred_step_id:
            for gate_type in ("precondition", "completion"):
                gate_id = f"{inference.inferred_step_id}.{gate_type}"
                gate_info = all_gates.get(gate_id)
                if not (isinstance(gate_info, Mapping) and gate_info.get("status") == "blocked"):
                    continue
                blocker: dict[str, str] = {"ref": f"GATES.{gate_id}"}
                reason_code = gate_info.get("reason_code")
                if isinstance(reason_code, str) and reason_code:
                    blocker["reason_code"] = reason_code
                reason = gate_info.get("reason")
                if isinstance(reason, str) and reason:
                    blocker["reason"] = reason
                inferred_gate_blockers.append(blocker)

        missing_conditions = list(inference.missing_conditions)
        step_signal_profile = None
        if isinstance(inference.inferred_step_id, str) and inference.inferred_step_id:
            raw_step_signal_profile = self.step_signal_profiles.get(inference.inferred_step_id)
            if isinstance(raw_step_signal_profile, Mapping):
                step_signal_profile = raw_step_signal_profile
        observability_status = None
        requires_visual_confirmation = False
        if isinstance(step_signal_profile, Mapping):
            raw_observability = step_signal_profile.get("observability_status")
            if not isinstance(raw_observability, str) or not raw_observability:
                raw_observability = step_signal_profile.get("observability")
            if isinstance(raw_observability, str) and raw_observability:
                observability_status = raw_observability
            raw_requires_visual = step_signal_profile.get("requires_visual_confirmation")
            if isinstance(raw_requires_visual, bool):
                requires_visual_confirmation = raw_requires_visual
        overlay_step_id = _resolve_overlay_step_id(
            inference.inferred_step_id,
            missing_conditions=missing_conditions,
            candidate_steps=self.candidate_steps,
            step_order_index=self._step_order_index,
            observability_status=observability_status,
            requires_visual_confirmation=requires_visual_confirmation,
        )
        deterministic_hint = {
            "inferred_step_id": inference.inferred_step_id,
            "overlay_step_id": overlay_step_id,
            "missing_conditions": missing_conditions,
            "recent_ui_targets": recent_buttons,
            "gate_blockers": inferred_gate_blockers,
            "scenario_profile": self.scenario_profile,
            "vision_fact_status": vision_fact_context.get("status"),
        }
        if isinstance(inference.inferred_step_id, str) and inference.inferred_step_id:
            step_harness_spec = self.step_harness_specs.get(inference.inferred_step_id)
            if step_harness_spec is not None:
                deterministic_hint["step_harness_spec"] = step_harness_spec.to_dict()
        if isinstance(step_signal_profile, Mapping):
                observability = step_signal_profile.get("observability")
                if isinstance(observability, str) and observability:
                    deterministic_hint["observability"] = observability
                    deterministic_hint["observability_status"] = observability
                evidence_requirements = step_signal_profile.get("evidence_requirements")
                if isinstance(evidence_requirements, list):
                    deterministic_hint["step_evidence_requirements"] = [
                        item for item in evidence_requirements if isinstance(item, str) and item
                    ]
                step_ui_targets = step_signal_profile.get("ui_targets")
                if isinstance(step_ui_targets, list):
                    normalized_targets = [
                        item for item in step_ui_targets if isinstance(item, str) and item
                    ]
                    deterministic_hint["step_ui_targets"] = list(normalized_targets)
                    interacted = [t for t in normalized_targets if t in self._step_interacted_targets]
                    remaining_targets = [t for t in normalized_targets if t not in self._step_interacted_targets]
                    deterministic_hint["step_interacted_targets"] = interacted
                    deterministic_hint["step_remaining_targets"] = remaining_targets
                requires_visual_confirmation = step_signal_profile.get("requires_visual_confirmation")
                if isinstance(requires_visual_confirmation, bool):
                    deterministic_hint["requires_visual_confirmation"] = requires_visual_confirmation
        preliminary_harness_context = {
            "vars": vars_selected,
            "gates": gates,
            "recent_deltas": recent_deltas,
            "recent_actions": recent_actions,
            "telemetry_window_frames": telemetry_window_frames,
            "deterministic_step_hint": deterministic_hint,
            "telemetry": {"t_wall": now_t_wall} if now_t_wall is not None else {},
            "vision": vision_context,
            "vision_facts": list(vision_fact_context.get("vision_facts", [])),
            "vision_fact_summary": dict(vision_fact_context.get("vision_fact_summary", {})),
        }
        evidence_packet = build_evidence_packet(preliminary_harness_context)
        state_harness = evidence_packet.to_state_harness_dict()
        evidence_packet_summary = evidence_packet.compact_summary()
        candidate_step_payload = [
            candidate.to_dict()
            for candidate in build_step_candidates(
                evidence_packet,
                step_harness_specs=self.step_harness_specs,
                ordered_step_ids=_reprioritize_steps_for_state_harness(self.candidate_steps, state_harness),
            )
        ]
        rag_topk, grounding_meta = self._build_grounding_context(deterministic_hint)
        overlay_target_allowlist = _resolve_step_overlay_allowlist(
            overlay_step_id,
            step_fallback_profiles=self.step_fallback_profiles,
            overlay_allowset=self.overlay_allowset,
            default_allowlist=self.overlay_allowlist,
            deterministic_hint=deterministic_hint,
        )
        overlay_target_allowlist = _broaden_overlay_allowlist_for_state_harness(
            overlay_target_allowlist,
            state_harness=state_harness,
            step_fallback_profiles=self.step_fallback_profiles,
            overlay_allowset=self.overlay_allowset,
        )
        visual_action_hint = _build_visual_action_hint(
            inferred_step_id=inference.inferred_step_id,
            missing_conditions=missing_conditions,
            context={
                "vision_fact_summary": vision_fact_context.get("vision_fact_summary", {}),
            },
            allowed_targets=overlay_target_allowlist,
        )
        if isinstance(visual_action_hint, Mapping):
            deterministic_hint["visual_action_hint"] = dict(visual_action_hint)
        action_hint = _build_procedural_action_hint(
            inferred_step_id=overlay_step_id,
            vars_selected=vars_selected,
            allowed_targets=overlay_target_allowlist,
            step_interacted_targets=deterministic_hint.get("step_interacted_targets"),
            vision_fact_summary=vision_fact_context.get("vision_fact_summary", {}),
        )
        if isinstance(action_hint, Mapping):
            deterministic_hint["action_hint"] = dict(action_hint)

        context = {
            "vars": vars_selected,
            "gates": gates,
            "recent_deltas": recent_deltas,
            "recent_actions": recent_actions,
            "pack_path": str(self.pack_path),
            "telemetry_map_path": str(self.telemetry_map_path),
            "candidate_steps": candidate_step_payload,
            "overlay_target_allowlist": overlay_target_allowlist,
            "state_harness": state_harness,
            "evidence_packet_summary": evidence_packet_summary,
            "deterministic_step_hint": deterministic_hint,
            "scenario_profile": self.scenario_profile,
            "rag_topk": rag_topk,
            "grounding_missing": bool(grounding_meta.get("grounding_missing")),
            "grounding_reason": grounding_meta.get("grounding_reason"),
            "grounding_query": grounding_meta.get("grounding_query"),
            "delta_summary": payload.get("delta_summary", {}),
            "delta_dropped_count": obs.metadata.get("delta_dropped_count"),
            "vision": vision_context,
            "vision_facts": list(vision_fact_context.get("vision_facts", [])),
            "vision_fact_summary": dict(vision_fact_context.get("vision_fact_summary", {})),
        }

        prompt_result = build_help_prompt_result(
            context,
            self.lang,
            max_overlay_targets=self.max_overlay_targets,
        )
        prompt_hash = hashlib.sha256(prompt_result.prompt.encode("utf-8")).hexdigest()
        prompt_grounding_missing = bool(prompt_result.metadata.get("grounding_missing"))
        prompt_grounding_reason = prompt_result.metadata.get("grounding_reason")
        req = TutorRequest(
            request_id=(
                request_id_override
                if isinstance(request_id_override, str) and request_id_override
                else str(uuid4())
            ),
            actor="learner",
            intent="help",
            message="help",
            observation_ref=obs.observation_id,
            context=context,
            metadata={
                "prompt_hash": prompt_hash,
                "prompt_tokens_est": int(prompt_result.metadata.get("prompt_tokens_est") or 0),
                "prompt_trimmed": bool(prompt_result.metadata.get("prompt_trimmed")),
                "grounding_query": grounding_meta.get("grounding_query"),
                "grounding_missing": prompt_grounding_missing,
                "grounding_reason": prompt_grounding_reason if isinstance(prompt_grounding_reason, str) else None,
                "grounding_missing_requested": bool(grounding_meta.get("grounding_missing")),
                "grounding_reason_requested": grounding_meta.get("grounding_reason"),
                "grounding_error_type": grounding_meta.get("grounding_error_type"),
                "grounding_snippet_ids": list(prompt_result.metadata.get("rag_snippet_ids") or []),
                "source_chunk_refs": list(grounding_meta.get("source_chunk_refs") or []),
                "grounding_cache_hit": bool(grounding_meta.get("grounding_cache_hit")),
                "grounding_index_path": grounding_meta.get("grounding_index_path"),
                "grounding_policy_id": grounding_meta.get("grounding_policy_id"),
                "grounding_policy_version": grounding_meta.get("grounding_policy_version"),
                "grounding_policy_filtered_out_count": int(
                    grounding_meta.get("grounding_policy_filtered_out_count") or 0
                ),
                "scenario_profile": self.scenario_profile,
                "vision_status": vision_context["status"],
                "vision_frame_ids": list(vision_context["frame_ids"]),
                "vision_fact_status": vision_fact_context.get("status"),
                "vision_fact_seen_ids": list(
                    vision_fact_context.get("vision_fact_summary", {}).get("seen_fact_ids", [])
                ),
                "state_harness_conflicts": list(state_harness.get("conflicts", [])),
                "state_harness_telemetry_status": (
                    state_harness.get("telemetry_evidence", {}).get("source_status")
                    if isinstance(state_harness.get("telemetry_evidence"), Mapping)
                    else None
                ),
                "evidence_packet_summary": evidence_packet_summary,
            },
        )

        state_signature = {
            "vars_discrete": {
                key: value
                for key, value in sorted(vars_selected.items())
                if isinstance(value, bool) or value is None
            },
            "recent_buttons": recent_buttons,
            "candidate_steps": candidate_step_payload,
            "overlay_target_allowlist": self.overlay_allowlist,
            "deterministic_step_hint": deterministic_hint,
            "scenario_profile": self.scenario_profile,
            "vision": {
                "status": vision_context["status"],
                "frame_ids": list(vision_context["frame_ids"]),
                "sync_miss_reason": vision_context["sync_miss_reason"],
            },
            "vision_facts": {
                "status": vision_fact_context.get("status"),
                "seen_fact_ids": list(
                    vision_fact_context.get("vision_fact_summary", {}).get("seen_fact_ids", [])
                ),
                "uncertain_fact_ids": list(
                    vision_fact_context.get("vision_fact_summary", {}).get("uncertain_fact_ids", [])
                ),
            },
            "state_harness": {
                "conflicts": list(state_harness.get("conflicts", [])),
                "telemetry_window_digest": _telemetry_window_signature(state_harness),
                "telemetry_status": (
                    state_harness.get("telemetry_evidence", {}).get("source_status")
                    if isinstance(state_harness.get("telemetry_evidence"), Mapping)
                    else None
                ),
                "visual_candidate_steps": _visual_candidate_steps_from_state_harness(state_harness),
            },
            "evidence_packet_summary": {
                key: value
                for key, value in evidence_packet_summary.items()
                if key != "telemetry_window_digest"
            },
        }
        state_key = _stable_hash_json(state_signature)
        return req, prompt_result.metadata, state_key

    def _should_reset_sticky_inference(self, vars_selected: Mapping[str, Any]) -> bool:
        battery_on = vars_selected.get("battery_on")
        power_available = vars_selected.get("power_available")
        if isinstance(battery_on, bool) and not battery_on:
            return True
        if isinstance(power_available, bool) and not power_available:
            return True
        return False

    def _stabilize_live_inference(
        self,
        inference: StepInferenceResult,
        vars_selected: Mapping[str, Any],
        *,
        recent_ui_targets: Sequence[str] | None = None,
    ) -> StepInferenceResult:
        if self._should_reset_sticky_inference(vars_selected):
            self._clear_live_progress_state()
            return inference
        current_step_id = inference.inferred_step_id
        if not isinstance(current_step_id, str) or not current_step_id:
            return inference
        current_idx = self._step_order_index.get(current_step_id)
        sticky_step_id = self._sticky_inference_step_id
        sticky_idx = self._step_order_index.get(sticky_step_id) if isinstance(sticky_step_id, str) else None
        if current_idx is None:
            return inference
        if sticky_idx is None or current_idx >= sticky_idx:
            self._sticky_inference_step_id = current_step_id
            self._sticky_inference_missing_conditions = tuple(inference.missing_conditions)
            return inference
        if (
            isinstance(sticky_step_id, str)
            and sticky_step_id == "S09"
            and _s09_comm1_frequency_complete(vars_selected)
            and _missing_conditions_satisfied_by_vars(self._sticky_inference_missing_conditions, vars_selected)
        ):
            next_idx = sticky_idx + 1
            if 0 <= next_idx < len(self.pack_steps):
                advanced = infer_step_id(
                    self.pack_steps[next_idx:],
                    vars_selected,
                    recent_ui_targets or (),
                    precondition_gates=self.precondition_gates,
                    completion_gates=self.completion_gates,
                    scenario_profile=self.scenario_profile,
                    pack_path=self.pack_path,
                    vision_facts=None,
                )
                if isinstance(advanced.inferred_step_id, str) and advanced.inferred_step_id:
                    self._sticky_inference_step_id = advanced.inferred_step_id
                    self._sticky_inference_missing_conditions = tuple(advanced.missing_conditions)
                    return advanced
        return StepInferenceResult(
            inferred_step_id=sticky_step_id,
            missing_conditions=self._sticky_inference_missing_conditions,
        )

    def _infer_preliminary_step_for_vision_facts(self, obs: Observation) -> StepInferenceResult:
        payload = obs.payload if isinstance(obs.payload, Mapping) else {}
        vars_map = payload.get("vars")
        if not isinstance(vars_map, Mapping):
            vars_map = {}
        vars_selected = dict(self._accumulated_vars)
        vars_selected.update(vars_map)
        now_t_wall = _coerce_float(payload.get("t_wall"))
        recent_frames = (
            self.recent_ring.snapshot(now_t_wall=now_t_wall)
            if now_t_wall is not None
            else self.recent_ring.snapshot()
        )
        recent_actions = build_recent_button_signal(recent_frames, self.mapper, max_items=8)
        recent_buttons = [
            item
            for item in recent_actions.get("recent_buttons", [])
            if isinstance(item, str) and item
        ]
        all_gates = evaluate_pack_gates(
            observations=[obs.to_dict()],
            precondition_gates=self.precondition_gates,
            completion_gates=self.completion_gates,
        )
        return infer_step_id(
            self.pack_steps,
            vars_selected,
            recent_buttons,
            gates=all_gates,
            precondition_gates=self.precondition_gates,
            completion_gates=self.completion_gates,
            scenario_profile=self.scenario_profile,
            pack_path=self.pack_path,
            vision_facts=None,
        )

    def _next_step_id_after(self, step_id: str | None) -> str | None:
        idx = self._step_order_index.get(step_id) if isinstance(step_id, str) else None
        if idx is None:
            return None
        next_idx = idx + 1
        if next_idx < 0 or next_idx >= len(self.candidate_steps):
            return None
        next_step_id = self.candidate_steps[next_idx]
        return next_step_id if isinstance(next_step_id, str) and next_step_id else None

    def _s08_page_navigation_recently_completed(self) -> bool:
        bit_root_fact = self._vision_fact_snapshot.get("bit_root_page_visible")
        if not (isinstance(bit_root_fact, Mapping) and bit_root_fact.get("state") == "seen"):
            return False
        fcs_fact = self._vision_fact_snapshot.get("fcs_page_visible")
        if isinstance(fcs_fact, Mapping) and fcs_fact.get("state") == "seen":
            return True
        supt_fact = self._vision_fact_snapshot.get("supt_page_visible")
        return (
            isinstance(supt_fact, Mapping)
            and supt_fact.get("state") == "seen"
            and self._recent_target_after_fact("left_mdi_pb15", supt_fact)
        )

    def _recent_target_after_fact(self, target: str, fact: Mapping[str, Any]) -> bool:
        observed_at_wall_ms = _coerce_int(fact.get("observed_at_wall_ms"))
        if observed_at_wall_ms is None:
            return False
        recent_deltas = build_prompt_recent_deltas(self.recent_ring.snapshot(), self.mapper, max_items=20)
        for item in recent_deltas:
            if item.get("mapped_ui_target") != target:
                continue
            t_wall = _coerce_float(item.get("t_wall"))
            if t_wall is None:
                continue
            if int(round(t_wall * 1000.0)) >= observed_at_wall_ms and _is_pressed_delta_value(item.get("to")):
                return True
        return False

    def _active_step_ids_for_vision_facts(
        self,
        preliminary_inference: StepInferenceResult,
        *,
        now_wall_ms: int | None = None,
    ) -> list[str]:
        if isinstance(now_wall_ms, int) and now_wall_ms >= 0:
            self._vision_fact_snapshot = prune_expired_facts(
                self._vision_fact_snapshot,
                now_wall_ms=now_wall_ms,
            )
        current_step_id = preliminary_inference.inferred_step_id
        last_step_id = self._last_inferred_step_id
        sticky_missing_has_visual_hold = any(
            isinstance(item, str) and item.startswith("vision_facts.")
            for item in self._sticky_inference_missing_conditions
        )
        sticky_step_id = self._sticky_inference_step_id
        sticky_idx = self._step_order_index.get(sticky_step_id) if isinstance(sticky_step_id, str) else None

        if (
            current_step_id == "S08"
            and last_step_id == "S08"
            and self._s08_page_navigation_recently_completed()
        ):
            current_step_id = self._next_step_id_after("S08")

        current_idx = self._step_order_index.get(current_step_id) if isinstance(current_step_id, str) else None
        last_idx = self._step_order_index.get(last_step_id) if isinstance(last_step_id, str) else None
        if current_idx is not None and last_idx is not None and current_idx < last_idx:
            if (
                last_step_id == "S19"
                and not (
                    sticky_missing_has_visual_hold
                    and sticky_step_id == last_step_id
                )
            ):
                current_step_id = self._next_step_id_after(last_step_id) or last_step_id
            else:
                current_step_id = last_step_id
            current_idx = self._step_order_index.get(current_step_id) if isinstance(current_step_id, str) else None

        out: list[str] = []
        if isinstance(current_step_id, str) and current_step_id:
            out.append(current_step_id)

        if (
            sticky_missing_has_visual_hold
            and isinstance(sticky_step_id, str)
            and sticky_step_id in self.vision_priority_step_set
            and current_idx is not None
            and sticky_idx is not None
            and current_idx <= sticky_idx
        ):
            if isinstance(sticky_step_id, str) and sticky_step_id not in out:
                out.append(sticky_step_id)
        return out

    def _should_extract_vision_facts_for_steps(self, step_ids: Sequence[str] | None) -> bool:
        if not step_ids:
            return False
        return any(step_id in self.vision_priority_step_set for step_id in step_ids if isinstance(step_id, str))

    def _extract_vision_fact_context(
        self,
        *,
        vision_selection: HelpCycleVisionSelection,
        help_cycle_id: str | None = None,
        active_step_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        trigger_wall_ms = int(vision_selection.trigger_wall_ms)
        self._vision_fact_snapshot = prune_expired_facts(
            self._vision_fact_snapshot,
            now_wall_ms=trigger_wall_ms,
        )
        normalized_active_step_ids = [
            item for item in (active_step_ids or []) if isinstance(item, str) and item
        ]
        if active_step_ids is not None and not self._should_extract_vision_facts_for_steps(normalized_active_step_ids):
            cached_facts = snapshot_to_list(self._vision_fact_snapshot)
            sticky_fact_count = sum(1 for item in cached_facts if item.get("sticky") is True)
            summary = build_vision_fact_summary(
                {},
                status=VISION_NOT_REQUIRED,
                frame_ids=vision_selection.frame_ids,
            )
            return {
                "status": VISION_NOT_REQUIRED,
                "vision_facts": [],
                "vision_fact_summary": summary,
                "metadata": {
                    "reason": "vision_not_required_for_step",
                    "active_step_ids": normalized_active_step_ids,
                    "vision_priority_steps": list(self.vision_priority_steps),
                    "extractor_used": False,
                    "cached_fact_count": len(cached_facts),
                    "sticky_fact_count": sticky_fact_count,
                    "ignored_fact_count": len(cached_facts),
                    "facts_ignored_reason": "current_step_not_visual_priority",
                },
            }
        if self.vision_fact_extractor is None:
            cached_facts = snapshot_to_list(self._vision_fact_snapshot)
            sticky_fact_count = sum(1 for item in cached_facts if item.get("sticky") is True)
            summary = build_vision_fact_summary(
                self._vision_fact_snapshot,
                status="vision_unavailable",
                frame_ids=vision_selection.frame_ids,
            )
            return {
                "status": "vision_unavailable",
                "vision_facts": cached_facts,
                "vision_fact_summary": summary,
                "metadata": {
                    "reason": "vision_fact_extractor_unconfigured",
                    "extractor_used": False,
                    "cached_fact_count": len(cached_facts),
                    "sticky_fact_count": sticky_fact_count,
                    "ignored_fact_count": 0,
                },
            }

        vision_payload = vision_selection.to_dict()
        if isinstance(help_cycle_id, str) and help_cycle_id:
            vision_payload["help_cycle_id"] = help_cycle_id
            vision_payload["request_id"] = help_cycle_id
        cached_facts_before_extract = snapshot_to_list(self._vision_fact_snapshot)
        sticky_fact_count_before_extract = sum(
            1 for item in cached_facts_before_extract if item.get("sticky") is True
        )
        result = self.vision_fact_extractor.extract(
            vision_payload,
            session_id=self.vision_session_id,
            trigger_wall_ms=trigger_wall_ms,
        )
        effective_status = result.status
        merge_error: str | None = None
        if result.observation is not None:
            try:
                self._vision_fact_snapshot = merge_vision_fact_observation(
                    self._vision_fact_snapshot,
                    result.observation,
                    config=self._vision_fact_config,
                    now_wall_ms=trigger_wall_ms,
                )
            except (ValueError, VisionFactsConfigError) as exc:
                merge_error = f"{type(exc).__name__}: {exc}"
                effective_status = "extractor_failed"
            else:
                _emit_vision_fact_observation_event(
                    observation=result.observation,
                    event_sink=self.event_sink,
                    fallback_session_id=self.session_id,
                )

        summary = build_vision_fact_summary(
            self._vision_fact_snapshot,
            status=effective_status,
            frame_ids=vision_selection.frame_ids,
            fresh_fact_ids=(
                [fact.fact_id for fact in result.observation.facts if fact.state == "seen"]
                if result.observation is not None and merge_error is None
                else []
            ),
        )
        metadata = dict(result.metadata)
        metadata["extractor_used"] = True
        metadata["active_step_ids"] = normalized_active_step_ids
        metadata["vision_priority_steps"] = list(self.vision_priority_steps)
        metadata["cached_fact_count"] = len(cached_facts_before_extract)
        metadata["sticky_fact_count"] = sticky_fact_count_before_extract
        metadata["ignored_fact_count"] = 0
        if result.error:
            metadata["error"] = result.error
        if merge_error is not None:
            metadata["vision_fact_merge_error"] = merge_error
            metadata.setdefault("error", merge_error)
        return {
            "status": effective_status,
            "vision_facts": snapshot_to_list(self._vision_fact_snapshot),
            "vision_fact_summary": summary,
            "observation": result.observation.to_dict() if result.observation is not None else None,
            "metadata": metadata,
        }

    def _fallback_message(self, inferred_step_id: str | None, missing_conditions: Sequence[str]) -> str:
        if self.lang == "zh":
            if inferred_step_id and missing_conditions:
                return (
                    f"降级提示：你大概率卡在 {inferred_step_id}，"
                    f"请先满足：{'; '.join(missing_conditions)}。"
                )
            if inferred_step_id:
                return f"降级提示：你大概率卡在 {inferred_step_id}，请先检查并执行该步骤。"
            return "降级提示：暂时无法推断当前卡住步骤，请先检查关键前置条件后再触发 Help。"
        if inferred_step_id and missing_conditions:
            return (
                f"Fallback: likely stuck at {inferred_step_id}; "
                f"please satisfy: {'; '.join(missing_conditions)}."
            )
        if inferred_step_id:
            return f"Fallback: likely stuck at {inferred_step_id}; please check that step."
        return "Fallback: unable to infer current blocked step."

    def _build_terminal_state_response(self, request: TutorRequest | None) -> TutorResponse:
        if self.lang == "zh":
            message = "当前冷启动流程已完成，无需继续操作。"
        else:
            message = "The cold-start procedure is complete. No further action is needed."
        request_id = request.request_id if request is not None else None
        return TutorResponse(
            status="ok",
            in_reply_to=request_id,
            message=message,
            actions=[],
            explanations=[message],
            metadata={
                "provider": "fallback",
                "generation_mode": "fallback",
                "diagnosis": {"step_id": "S33"},
                "next": {"step_id": "S33"},
                "terminal_state_rewritten": True,
                "terminal_state_original_message": message,
                "terminal_state_original_explanations": [],
                "fallback_overlay_used": False,
                "fallback_overlay_reason": "all_steps_complete",
            },
        )

    def _capture_model_raw_help_response(self, response: TutorResponse) -> None:
        metadata = response.metadata if isinstance(response.metadata, Mapping) else {}
        raw_help_response = metadata.get("help_response")
        if not isinstance(raw_help_response, Mapping):
            return
        response.metadata.setdefault("model_raw_help_response", copy.deepcopy(dict(raw_help_response)))
        raw_explanations = raw_help_response.get("explanations")
        if isinstance(raw_explanations, list):
            response.metadata.setdefault(
                "model_raw_explanations",
                [item for item in raw_explanations if isinstance(item, str)],
            )
        raw_next = raw_help_response.get("next")
        if isinstance(raw_next, Mapping):
            response.metadata.setdefault("model_raw_next", dict(raw_next))
        raw_diagnosis = raw_help_response.get("diagnosis")
        if isinstance(raw_diagnosis, Mapping):
            response.metadata.setdefault("model_raw_diagnosis", dict(raw_diagnosis))

    def _annotate_response_audit_metadata(self, response: TutorResponse) -> None:
        self._capture_model_raw_help_response(response)

        final_public_response: dict[str, Any] = {
            "message": response.message,
            "explanations": [item for item in response.explanations if isinstance(item, str)],
        }
        next_payload = response.metadata.get("next")
        if isinstance(next_payload, Mapping):
            final_public_response["next"] = dict(next_payload)
        diagnosis_payload = response.metadata.get("diagnosis")
        if isinstance(diagnosis_payload, Mapping):
            final_public_response["diagnosis"] = dict(diagnosis_payload)
        if response.actions:
            final_public_response["actions"] = [
                dict(action) for action in response.actions if isinstance(action, Mapping)
            ]
        response.metadata["final_public_response"] = final_public_response

    def _normalize_observable_text_only_response(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> None:
        if response.actions or response.status != "ok":
            return
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return
        if bool(hint.get("requires_visual_confirmation")):
            return
        observability = hint.get("observability_status")
        if not isinstance(observability, str) or not observability:
            observability = hint.get("observability")
        if observability != "observable":
            return

        text_parts = [response.message, *response.explanations]
        combined = " ".join(part for part in text_parts if isinstance(part, str) and part).lower()
        if not combined:
            return
        bad_reasons = (
            "视觉不可用",
            "视觉分析不可用",
            "缺乏变量证据",
            "变量证据",
            "vision unavailable",
            "visual analysis unavailable",
            "missing variable evidence",
        )
        if not any(marker in combined for marker in bad_reasons):
            return

        inferred_step_id = hint.get("inferred_step_id") if isinstance(hint.get("inferred_step_id"), str) else None
        missing_conditions = hint.get("missing_conditions")
        normalized_missing = [
            item for item in missing_conditions if isinstance(item, str) and item
        ] if isinstance(missing_conditions, (list, tuple)) else []
        normalized = self._fallback_message(inferred_step_id, normalized_missing)
        response.message = normalized
        response.explanations = [normalized]
        response.metadata["observable_text_rewritten"] = True

    def _rewrite_low_confidence_bootstrap_response(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> bool:
        if response.status != "ok":
            return False
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return False
        inferred_step_id = hint.get("inferred_step_id")
        if inferred_step_id != "S01":
            return False
        state_harness = context.get("state_harness")
        if not isinstance(state_harness, Mapping):
            return False
        telemetry_evidence = state_harness.get("telemetry_evidence")
        if not isinstance(telemetry_evidence, Mapping):
            return False
        if telemetry_evidence.get("source_status") != "low_confidence_bootstrap":
            return False
        missing_count = telemetry_evidence.get("vars_source_missing_count")
        if not isinstance(missing_count, int) or missing_count < 20:
            return False
        vision_fact_summary = context.get("vision_fact_summary")
        if isinstance(vision_fact_summary, Mapping):
            status = vision_fact_summary.get("status")
            frame_ids = vision_fact_summary.get("frame_ids")
            vision_context = context.get("vision")
            vision_used = isinstance(vision_context, Mapping) and bool(vision_context.get("vision_used"))
            if status != VISION_NOT_REQUIRED or not vision_used or not (
                isinstance(frame_ids, (list, tuple)) and any(isinstance(item, str) and item for item in frame_ids)
            ):
                return False
            seen = vision_fact_summary.get("seen_fact_ids")
            fresh = vision_fact_summary.get("fresh_fact_ids")
            has_visual_anchor = any(
                isinstance(items, (list, tuple, set)) and any(isinstance(item, str) and item for item in items)
                for items in (seen, fresh)
            )
            if has_visual_anchor:
                return False
        else:
            return False

        original_actions = copy.deepcopy([dict(action) for action in response.actions if isinstance(action, Mapping)])
        original_message = response.message
        original_explanations = list(response.explanations)
        if self.lang == "zh":
            rewritten = (
                "当前 DCS-BIOS 刚接入，首帧遥测缺失较多，暂不能可靠判断电瓶/早期步骤状态。"
                "请等待一两秒再触发 Help；如果座舱已经上电，我不会在这一帧高亮电瓶开关。"
            )
        else:
            rewritten = (
                "DCS-BIOS has just connected and the first telemetry frame is missing many variables, "
                "so the battery/early-step state is not reliable yet. Wait a second or two and trigger Help again."
            )
        response.actions = []
        response.message = rewritten
        response.explanations = [rewritten]
        response.metadata["bootstrap_low_confidence_guardrail_applied"] = True
        response.metadata["bootstrap_low_confidence_guardrail_step_id"] = inferred_step_id
        response.metadata["bootstrap_low_confidence_vars_source_missing_count"] = missing_count
        if original_actions:
            response.metadata["bootstrap_low_confidence_original_actions"] = original_actions
        if original_message != rewritten:
            response.metadata["bootstrap_low_confidence_original_message"] = original_message
        if original_explanations and original_explanations != [rewritten]:
            response.metadata["bootstrap_low_confidence_original_explanations"] = original_explanations
        return True

    def _rewrite_conflicting_step_completion_response(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> bool:
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return False
        inferred_step_id = hint.get("inferred_step_id")
        if not isinstance(inferred_step_id, str) or not inferred_step_id:
            return False
        missing_conditions = hint.get("missing_conditions")
        normalized_missing = [
            item for item in missing_conditions if isinstance(item, str) and item
        ] if isinstance(missing_conditions, (list, tuple)) else []
        if not normalized_missing:
            return False

        model_next_step_id = _extract_model_next_step_id(response.metadata)
        text_parts = [response.message, *response.explanations]
        combined = " ".join(part for part in text_parts if isinstance(part, str) and part).lower()
        completion_markers = (
            "is complete",
            "step is complete",
            "indicating s08 is complete",
            "已完成",
            "已经完成",
            "当前步骤已完成",
        )
        has_completion_claim = any(marker in combined for marker in completion_markers)
        if model_next_step_id == inferred_step_id and not _text_claims_step_complete(combined, inferred_step_id):
            return False

        original_message = response.message
        original_explanations = list(response.explanations)
        original_next = None
        if isinstance(response.metadata.get("next"), Mapping):
            original_next = dict(response.metadata["next"])
        original_actions = list(response.actions)

        action_target = None
        if response.actions:
            first_action = response.actions[0]
            if isinstance(first_action, Mapping):
                raw_target = first_action.get("target")
                if isinstance(raw_target, str) and raw_target:
                    action_target = raw_target

        missing_set = set(normalized_missing)
        if (
            inferred_step_id == "S08"
            and "vision_facts.fcs_page_visible==seen" in missing_set
            and "vision_facts.bit_root_page_visible==seen" in missing_set
            and action_target == "left_mdi_pb15"
        ):
            if self.lang == "zh":
                rewritten = (
                    "当前仍在 S08。左 DDI 现在只是显示 FCS 按钮，"
                    "还没有真正进入 FCS 页面；请先按左 DDI 的 PB15 进入 FCS 页面。"
                )
            else:
                rewritten = (
                    "You are still on S08. The left DDI is only showing the FCS button, "
                    "but it has not entered the FCS page yet; press Left DDI PB15 to enter the FCS page first."
                )
        elif self.lang == "zh":
            rewritten = f"当前 {inferred_step_id} 尚未完成，请先满足：{'; '.join(normalized_missing)}。"
            if isinstance(action_target, str) and action_target:
                rewritten = f"当前 {inferred_step_id} 尚未完成。请先操作 {action_target}，再满足：{'; '.join(normalized_missing)}。"
        else:
            rewritten = (
                f"{inferred_step_id} is not complete yet. "
                f"Please satisfy: {'; '.join(normalized_missing)}."
            )
            if isinstance(action_target, str) and action_target:
                rewritten = (
                    f"{inferred_step_id} is not complete yet. "
                    f"Please operate {action_target} first, then satisfy: {'; '.join(normalized_missing)}."
                )

        response.message = rewritten
        response.explanations = [rewritten]
        if isinstance(response.metadata.get("next"), Mapping):
            response.metadata["next"] = dict(response.metadata["next"])
            response.metadata["next"]["step_id"] = inferred_step_id
        if isinstance(response.metadata.get("diagnosis"), Mapping):
            response.metadata["diagnosis"] = dict(response.metadata["diagnosis"])
            response.metadata["diagnosis"]["step_id"] = inferred_step_id
        if response.actions:
            response.actions = []
            response.metadata["completion_conflict_overlay_cleared"] = True
        response.metadata["completion_conflict_rewritten"] = True
        if original_message != rewritten:
            response.metadata["completion_conflict_original_message"] = original_message
        if original_explanations and original_explanations != [rewritten]:
            response.metadata["completion_conflict_original_explanations"] = original_explanations
        if original_next is not None:
            response.metadata["completion_conflict_original_next"] = original_next
        if original_actions:
            response.metadata["completion_conflict_original_actions"] = copy.deepcopy(original_actions)
        return True

    def _rewrite_terminal_state_conflict_response(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> bool:
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return False
        help_response = response.metadata.get("help_response")
        if not isinstance(help_response, Mapping):
            return False
        inferred_step_id = hint.get("inferred_step_id")
        if inferred_step_id != "S33":
            return False
        missing_conditions = hint.get("missing_conditions")
        normalized_missing = [
            item for item in missing_conditions if isinstance(item, str) and item
        ] if isinstance(missing_conditions, (list, tuple)) else []
        if normalized_missing:
            return False
        gate_blockers_raw = hint.get("gate_blockers")
        gate_blockers = [
            item for item in gate_blockers_raw if isinstance(item, Mapping) and item
        ] if isinstance(gate_blockers_raw, (list, tuple)) else []
        if gate_blockers:
            return False

        model_next_step_id = _extract_model_next_step_id(response.metadata)
        if model_next_step_id == "S33":
            return False

        response.metadata["terminal_state_rewritten"] = True
        response.metadata["terminal_state_original_message"] = response.message
        response.metadata["terminal_state_original_explanations"] = list(response.explanations)
        if isinstance(response.metadata.get("diagnosis"), Mapping):
            response.metadata["terminal_state_original_diagnosis"] = dict(response.metadata["diagnosis"])
        if isinstance(response.metadata.get("next"), Mapping):
            response.metadata["terminal_state_original_next"] = dict(response.metadata["next"])
        if response.actions:
            response.metadata["terminal_state_original_actions"] = copy.deepcopy(response.actions)

        if self.lang == "zh":
            rewritten = "当前冷启动流程已完成，无需继续操作。"
        else:
            rewritten = "The cold-start procedure is complete. No further action is needed."

        response.message = rewritten
        response.explanations = [rewritten]
        if response.actions:
            response.actions = []
        original_diagnosis = (
            dict(response.metadata["diagnosis"])
            if isinstance(response.metadata.get("diagnosis"), Mapping)
            else {}
        )
        rewritten_diagnosis = {"step_id": "S33"}
        error_category = original_diagnosis.get("error_category")
        if isinstance(error_category, str) and error_category:
            rewritten_diagnosis["error_category"] = error_category
        response.metadata["diagnosis"] = rewritten_diagnosis
        response.metadata["next"] = {"step_id": "S33"}

        return True

    def _should_use_deterministic_overlay_fallback(
        self,
        response: TutorResponse,
        request: TutorRequest,
        mapped_meta: Mapping[str, Any] | None,
    ) -> bool:
        if bool(response.metadata.get("bootstrap_low_confidence_guardrail_applied")):
            return False
        if bool(response.metadata.get("refuel_probe_motion_guidance_rewritten")):
            return False
        if response.actions:
            return False
        if response.status == "error":
            return True
        if isinstance(mapped_meta, Mapping) and (
            mapped_meta.get("overlay_rejected")
            or mapped_meta.get("rejected_targets_by_request_allowlist")
        ):
            return True
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return False
        inferred_step_id = hint.get("inferred_step_id")
        if inferred_step_id == "S19":
            vision_fact_summary = context.get("vision_fact_summary")
            if _s19_final_go_seen_or_fresh(
                vision_fact_summary if isinstance(vision_fact_summary, Mapping) else None
            ):
                return False
        if (
            isinstance(inferred_step_id, str)
            and inferred_step_id
        ):
            model_next_step_id = _extract_model_next_step_id(response.metadata)
            vision_context = context.get("vision")
            vision_used = isinstance(vision_context, Mapping) and bool(vision_context.get("vision_used"))
            if (
                isinstance(model_next_step_id, str)
                and model_next_step_id
                and model_next_step_id != inferred_step_id
                and vision_used
            ):
                return True
        if bool(hint.get("requires_visual_confirmation")):
            action_hint = hint.get("action_hint")
            if not isinstance(action_hint, Mapping):
                return False
            action_target = action_hint.get("target")
            if not isinstance(action_target, str) or not action_target:
                return False
            vision_fact_summary = context.get("vision_fact_summary")
            vision_fact_status = None
            if isinstance(vision_fact_summary, Mapping):
                raw_status = vision_fact_summary.get("status")
                if isinstance(raw_status, str) and raw_status:
                    vision_fact_status = raw_status
            if vision_fact_status != "vision_unavailable":
                return False
            return True
        observability = hint.get("observability_status")
        if not isinstance(observability, str) or not observability:
            observability = hint.get("observability")
        if observability != "observable":
            return False
        action_hint = hint.get("action_hint")
        if isinstance(action_hint, Mapping):
            action_target = action_hint.get("target")
            if isinstance(action_target, str) and action_target:
                if (
                    inferred_step_id == "S09"
                    and response.actions
                    and isinstance(model_next_step_id, str)
                    and model_next_step_id == inferred_step_id
                ):
                    return False
                return True
        missing_conditions = hint.get("missing_conditions")
        return isinstance(missing_conditions, (list, tuple)) and any(
            isinstance(item, str) and item for item in missing_conditions
        )

    def _apply_action_hint_overlay_override(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[bool, str]:
        if not response.actions:
            return False, "missing_actions"
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return False, "missing_deterministic_hint"
        action_target = None
        override_kind = None
        vision_fact_summary = context.get("vision_fact_summary")
        vision_fact_status = None
        seen_fact_ids: set[str] = set()
        if isinstance(vision_fact_summary, Mapping):
            raw_status = vision_fact_summary.get("status")
            if isinstance(raw_status, str) and raw_status:
                vision_fact_status = raw_status
            seen_fact_ids = {
                item for item in vision_fact_summary.get("seen_fact_ids", [])
                if isinstance(item, str) and item
            }
        inferred_step_id = hint.get("inferred_step_id")
        if response.metadata.get("refuel_probe_motion_guidance_rewritten") is True:
            return False, "refuel_probe_motion_wait_already_rewritten"
        action_hint = hint.get("action_hint")
        if bool(hint.get("requires_visual_confirmation")) is True:
            if isinstance(action_hint, Mapping):
                hinted_target = action_hint.get("target")
                if isinstance(hinted_target, str) and hinted_target:
                    if isinstance(vision_fact_status, str) and vision_fact_status in {
                        "vision_unavailable",
                        "extractor_failed",
                    }:
                        action_target = hinted_target
                        override_kind = "action_hint"
                    elif inferred_step_id == "S19" and "fcsmc_page_visible" in seen_fact_ids:
                        action_target = hinted_target
                        override_kind = "action_hint"
        elif inferred_step_id in {"S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27"}:
            if isinstance(action_hint, Mapping):
                hinted_target = action_hint.get("target")
                if isinstance(hinted_target, str) and hinted_target:
                    action_target = hinted_target
                    override_kind = "action_hint"
        elif inferred_step_id == "S08":
            if isinstance(action_hint, Mapping):
                hinted_target = action_hint.get("target")
                if isinstance(hinted_target, str) and hinted_target:
                    action_target = hinted_target
                    override_kind = "action_hint"
        current_targets = [
            target
            for target in (
                action.get("target") if isinstance(action, Mapping) else None
                for action in response.actions
            )
            if isinstance(target, str) and target
        ]
        if not current_targets:
            return False, "missing_action_targets"
        if not isinstance(action_target, str) or not action_target:
            # Check for multi-target action hint (targets field)
            action_hint_for_targets = hint.get("action_hint")
            if isinstance(action_hint_for_targets, Mapping):
                hinted_targets_list = action_hint_for_targets.get("targets")
                if isinstance(hinted_targets_list, list) and hinted_targets_list:
                    if set(current_targets) == set(hinted_targets_list):
                        return False, "already_aligned_multi"
                    action_target = hinted_targets_list[0]
                    override_kind = "action_hint"
        if not isinstance(action_target, str) or not action_target:
            return False, "missing_override_target"
        if (
            self.max_overlay_targets > 1
            and inferred_step_id == "S19"
            and action_target in current_targets
            and set(current_targets).issubset({"fcs_bit_switch", "right_mdi_pb5"})
        ):
            return False, "already_aligned_multi_target_s18"
        if current_targets == [action_target]:
            return False, "already_aligned"

        response.metadata["action_hint_overlay_override_original_actions"] = copy.deepcopy(
            [dict(action) for action in response.actions if isinstance(action, Mapping)]
        )
        response.metadata["action_hint_overlay_override_original_targets"] = list(current_targets)
        response.metadata["action_hint_overlay_override_kind"] = override_kind

        override_used, override_reason = self._apply_safe_fallback_overlay(response, request)
        if not override_used:
            return False, f"override_failed:{override_reason}"

        if (
            inferred_step_id == "S19"
            and override_kind == "action_hint"
            and action_target == "fcs_bit_switch"
            and "fcsmc_page_visible" in seen_fact_ids
        ):
            original_message = response.message
            original_explanations = list(response.explanations)
            if self.lang == "zh":
                rewritten = (
                    "当前右 DDI 已在 FCS-MC 页面。下一步请按住 FCS BIT 开关（Y），"
                    "同时按右 DDI 的 PB5 以启动 FCS BIT 自检。"
                )
            else:
                rewritten = (
                    "The right DDI is already on the FCS-MC page. Hold the FCS BIT switch up (Y) "
                    "while pressing Right DDI PB5 to start the FCS BIT."
                )
            response.message = rewritten
            response.explanations = [rewritten]
            if original_message != rewritten:
                response.metadata["action_hint_overlay_override_original_message"] = original_message
            if original_explanations and original_explanations != [rewritten]:
                response.metadata["action_hint_overlay_override_original_explanations"] = original_explanations
        elif (
            inferred_step_id in {"S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27"}
            and override_kind == "action_hint"
        ):
            original_message = response.message
            original_explanations = list(response.explanations)
            hint_reason = action_hint.get("reason") if isinstance(action_hint, Mapping) else None
            if isinstance(hint_reason, str) and hint_reason:
                rewritten = hint_reason
            elif self.lang == "zh":
                rewritten = "请按系统提示操作当前高亮目标，继续完成四落检查。"
            else:
                rewritten = "Follow the highlighted target to continue the four-down checklist."
            response.message = rewritten
            response.explanations = [rewritten]
            if original_message != rewritten:
                response.metadata["action_hint_overlay_override_original_message"] = original_message
            if original_explanations and original_explanations != [rewritten]:
                response.metadata["action_hint_overlay_override_original_explanations"] = original_explanations
        elif inferred_step_id == "S08" and override_kind == "action_hint":
            original_message = response.message
            original_explanations = list(response.explanations)
            hint_reason = action_hint.get("reason") if isinstance(action_hint, Mapping) else None
            if self.lang == "zh":
                by_target = {
                    "left_mdi_brightness_selector": "左 DDI 选择旋钮还未到 NIGHT/DAY。请先打开左 DDI；DDI 打开后屏幕会有短暂亮起延迟。",
                    "right_mdi_brightness_selector": "右 DDI 选择旋钮还未到 NIGHT/DAY。请打开右 DDI；DDI 打开后屏幕会有短暂亮起延迟。",
                    "ampcd_off_brightness_knob": "左右 DDI 已经上电。下一步请调高 AMPCD 亮度旋钮点亮 AMPCD。",
                    "hud_symbology_brightness_knob": "DDI 和 AMPCD 已经上电。下一步请调高 HUD 亮度。",
                }
                rewritten = by_target.get(action_target, "请按当前高亮目标继续完成显示器上电。")
            elif isinstance(hint_reason, str) and hint_reason:
                rewritten = hint_reason
            else:
                rewritten = "Follow the highlighted display power target; DDI screens can take a moment to illuminate."
            response.message = rewritten
            response.explanations = [rewritten]
            if original_message != rewritten:
                response.metadata["action_hint_overlay_override_original_message"] = original_message
            if original_explanations and original_explanations != [rewritten]:
                response.metadata["action_hint_overlay_override_original_explanations"] = original_explanations

        response.metadata["action_hint_overlay_override_used"] = True
        response.metadata["action_hint_overlay_override_target"] = action_target
        response.metadata["action_hint_overlay_override_reason"] = override_reason
        return True, override_reason

    def _apply_s08_visual_recovery_overlay_override(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[bool, str]:
        if not response.actions:
            return False, "missing_actions"
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return False, "missing_deterministic_hint"
        inferred_step_id = hint.get("inferred_step_id")
        overlay_step_id = hint.get("overlay_step_id", inferred_step_id)
        if inferred_step_id != "S08" or overlay_step_id != "S08":
            return False, "not_s08"
        missing_conditions = hint.get("missing_conditions")
        missing_set = {
            item for item in missing_conditions
            if isinstance(item, str) and item
        } if isinstance(missing_conditions, (list, tuple)) else set()
        current_targets = [
            target
            for target in (
                action.get("target") if isinstance(action, Mapping) else None
                for action in response.actions
            )
            if isinstance(target, str) and target
        ]
        if not current_targets:
            return False, "missing_action_targets"
        visual_summary = context.get("vision_fact_summary")
        visual_seen: set[str] = set()
        visual_fresh: set[str] = set()
        if isinstance(visual_summary, Mapping):
            visual_seen = {
                item for item in visual_summary.get("seen_fact_ids", [])
                if isinstance(item, str) and item
            }
            visual_fresh = {
                item for item in visual_summary.get("fresh_fact_ids", [])
                if isinstance(item, str) and item
            }
        display_page_evidence = {
            "tac_page_visible",
            "supt_page_visible",
            "fcs_page_visible",
            "bit_root_page_visible",
        }
        has_display_page_evidence = bool(display_page_evidence.intersection(visual_seen | visual_fresh))
        if (
            set(current_targets) == {"ampcd_off_brightness_knob"}
            and "vars.mpcd_on==true" in missing_set
            and not has_display_page_evidence
        ):
            ddi_power_missing = (
                "vars.left_ddi_on==true" in missing_set
                or "vars.right_ddi_on==true" in missing_set
            )
            if not ddi_power_missing:
                return False, "ampcd_allowed_after_ddi_power"
            response.metadata["s08_visual_recovery_overlay_override_original_actions"] = copy.deepcopy(
                [dict(action) for action in response.actions if isinstance(action, Mapping)]
            )
            response.metadata["s08_visual_recovery_overlay_override_original_message"] = response.message
            override_used, override_reason = self._apply_safe_fallback_overlay(response, request)
            if not override_used:
                return False, f"override_failed:{override_reason}"
            if self.lang == "zh":
                rewritten = (
                    "当前仍在 S08，但还没有视觉确认 DDI 页面。"
                    "先确认/点亮 DDI，再打开 AMPCD，避免在右 DDI 未建立时提前操作 AMPCD。"
                )
            else:
                rewritten = (
                    "You are still on S08, but no DDI page is visually confirmed yet. "
                    "Confirm or power a DDI before turning on the AMPCD."
                )
            response.message = rewritten
            response.explanations = [rewritten]
            response.metadata["s08_visual_recovery_overlay_override_used"] = True
            response.metadata["s08_visual_recovery_overlay_override_reason"] = override_reason
            response.metadata["s08_visual_recovery_overlay_override_case"] = "ampcd_before_ddi_visual_confirmed"
            return True, override_reason
        if not {
            "vision_facts.fcs_page_visible==seen",
            "vision_facts.bit_root_page_visible==seen",
        }.issubset(missing_set):
            return False, "not_dual_visual_missing"
        if "right_mdi_brightness_selector" in current_targets:
            return False, "already_includes_right_ddi_recovery"
        left_only_nav = {"left_mdi_pb18", "left_mdi_pb15"}
        if not set(current_targets).issubset(left_only_nav):
            return False, "model_action_not_left_only_navigation"

        response.metadata["s08_visual_recovery_overlay_override_original_actions"] = copy.deepcopy(
            [dict(action) for action in response.actions if isinstance(action, Mapping)]
        )
        response.metadata["s08_visual_recovery_overlay_override_original_message"] = response.message
        override_used, override_reason = self._apply_safe_fallback_overlay(response, request)
        if not override_used:
            return False, f"override_failed:{override_reason}"

        if self.lang == "zh":
            rewritten = (
                "当前仍在 S08，左侧页面和右 DDI BIT 页面都还没有确认。"
                "请先恢复/确认右 DDI，再继续左 DDI 的 SUPT/FCS 页面导航。"
            )
        else:
            rewritten = (
                "You are still on S08, and both the left FCS page and the right DDI BIT page "
                "are not confirmed. Recover or confirm the right DDI before continuing left DDI SUPT/FCS navigation."
            )
        response.message = rewritten
        response.explanations = [rewritten]
        response.metadata["s08_visual_recovery_overlay_override_used"] = True
        response.metadata["s08_visual_recovery_overlay_override_reason"] = override_reason
        return True, override_reason

    def _apply_harness_conflict_guardrail(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[bool, str]:
        context = request.context if isinstance(request.context, Mapping) else {}
        state_harness = context.get("state_harness")
        if not _state_harness_has_late_vlm_conflict(state_harness):
            return False, "no_harness_conflict"

        rejected_step_id = _extract_model_next_step_id(response.metadata)
        if not isinstance(rejected_step_id, str) or rejected_step_id not in {"S01", "S02", "S03"}:
            diagnosis = response.metadata.get("diagnosis")
            if isinstance(diagnosis, Mapping) and isinstance(diagnosis.get("step_id"), str):
                rejected_step_id = diagnosis["step_id"]
        if not isinstance(rejected_step_id, str) or rejected_step_id not in {"S01", "S02", "S03"}:
            return False, "model_step_not_early"

        vision_summary = context.get("vision_fact_summary")
        vision_seen_or_fresh: set[str] = set()
        if isinstance(vision_summary, Mapping):
            for key in ("seen_fact_ids", "fresh_fact_ids"):
                raw_ids = vision_summary.get(key)
                if isinstance(raw_ids, (list, tuple, set)):
                    vision_seen_or_fresh.update(item for item in raw_ids if isinstance(item, str) and item)

        if "supt_page_visible" in vision_seen_or_fresh:
            target = "left_mdi_pb15"
            evidence_fact = "supt_page_visible"
            guidance_zh = "VLM 已确认左 DDI 在 SUPT 页面；请按左 DDI PB15 进入 FCS 页面。"
            guidance_en = "The VLM confirms the left DDI is on SUPT; press Left DDI PB15 to enter the FCS page."
        elif "tac_page_visible" in vision_seen_or_fresh:
            target = "left_mdi_pb18"
            evidence_fact = "tac_page_visible"
            guidance_zh = "VLM 已确认左 DDI 仍在 TAC 页面；请先按左 DDI PB18 切到 SUPT，再进入 FCS。"
            guidance_en = "The VLM confirms the left DDI is still on TAC; press Left DDI PB18 to reach SUPT, then enter FCS."
        else:
            target = "right_mdi_brightness_selector"
            evidence_fact = "bit_root_page_visible" if "bit_root_page_visible" in vision_seen_or_fresh else None
            guidance_zh = "VLM 已看到后续显示页面；请先确认 DDI 页面状态，不要退回早期启动步骤。"
            guidance_en = "The VLM sees later display pages; confirm the DDI page state instead of regressing to early startup steps."

        evidence_ref = None
        if isinstance(evidence_fact, str):
            vision_facts = context.get("vision_facts")
            if isinstance(vision_facts, list):
                for item in vision_facts:
                    if not isinstance(item, Mapping) or item.get("fact_id") != evidence_fact:
                        continue
                    frame_id = item.get("source_frame_id")
                    evidence_ref = (
                        f"VISION_FACTS.{evidence_fact}@{frame_id}"
                        if isinstance(frame_id, str) and frame_id
                        else f"VISION_FACTS.{evidence_fact}"
                    )
                    break
            if evidence_ref is None:
                evidence_ref = f"VISION_FACTS.{evidence_fact}"

        guidance = guidance_zh if self.lang == "zh" else guidance_en
        help_obj: dict[str, Any] = {
            "diagnosis": {"step_id": "S08", "error_category": "CO"},
            "next": {"step_id": "S08"},
            "overlay": {"targets": [target], "evidence": []},
            "explanations": [guidance],
        }
        if isinstance(evidence_ref, str) and evidence_ref:
            help_obj["overlay"]["evidence"] = [
                {
                    "target": target,
                    "type": "visual",
                    "ref": evidence_ref,
                    "quote": "VLM visual page state conflicts with early telemetry.",
                    "grounding_confidence": 0.95,
                }
            ]

        mapped = map_help_response_to_tutor_response(
            help_obj,
            request=request,
            status=response.status,
            max_overlay_targets=self.max_overlay_targets,
            ui_map_path=self.ui_map_path,
            lang=self.lang,
        )
        if not mapped.actions:
            return False, "harness_guardrail_mapping_failed"

        response.metadata["harness_conflict_detected"] = True
        response.metadata["harness_guardrail_original_message"] = response.message
        response.metadata["harness_guardrail_original_actions"] = copy.deepcopy(
            [dict(action) for action in response.actions if isinstance(action, Mapping)]
        )
        response.metadata["rejected_model_step_id"] = rejected_step_id
        response.metadata["harness_guardrail_applied"] = True
        response.metadata["harness_guardrail_reason"] = HARNESS_LATE_VLM_CONFLICT
        response.metadata["harness_guardrail_mapping"] = dict(mapped.metadata)
        response.metadata["diagnosis"] = {"step_id": "S08", "error_category": "CO"}
        response.metadata["next"] = {"step_id": "S08"}
        response.actions = list(mapped.actions)
        response.message = guidance
        response.explanations = [guidance]
        return True, HARNESS_LATE_VLM_CONFLICT

    def _apply_harness_validation_action_plan(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[bool, str]:
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return False, "missing_deterministic_hint"

        inferred_step_id = hint.get("inferred_step_id")
        overlay_step_id = hint.get("overlay_step_id")
        if response.metadata.get("refuel_probe_motion_guidance_rewritten") is True:
            return False, "refuel_probe_motion_wait_already_rewritten"
        if response.metadata.get("s09_comm1_completion_guardrail_applied") is True:
            return False, "s09_comm1_completion_already_rewritten"
        if response.metadata.get("completion_conflict_rewritten") is True:
            return False, "completion_conflict_already_rewritten"
        response_mapping_meta = response.metadata.get("response_mapping")
        if (
            response.actions
            and isinstance(response_mapping_meta, Mapping)
            and response_mapping_meta.get("rejected_targets_by_request_allowlist")
        ):
            return False, "response_mapping_already_repaired_allowlist"
        if _state_harness_has_late_vlm_conflict(context.get("state_harness")):
            return False, "late_vlm_conflict_guardrail"
        model_step_id = _extract_model_next_step_id(response.metadata)
        if not isinstance(model_step_id, str) or not model_step_id:
            diagnosis = response.metadata.get("diagnosis")
            if isinstance(diagnosis, Mapping):
                raw_step_id = diagnosis.get("step_id")
                if isinstance(raw_step_id, str) and raw_step_id:
                    model_step_id = raw_step_id

        proposed_targets: list[str] = []
        evidence_refs: list[str] = []
        help_response = response.metadata.get("help_response")
        original_help_response = copy.deepcopy(help_response) if isinstance(help_response, Mapping) else None
        if isinstance(help_response, Mapping):
            overlay = help_response.get("overlay")
            if isinstance(overlay, Mapping):
                raw_targets = overlay.get("targets")
                if isinstance(raw_targets, list):
                    proposed_targets = [item for item in raw_targets if isinstance(item, str) and item]
                raw_evidence = overlay.get("evidence")
                if isinstance(raw_evidence, list):
                    for item in raw_evidence:
                        if not isinstance(item, Mapping):
                            continue
                        ref = item.get("ref")
                        if isinstance(ref, str) and ref:
                            evidence_refs.append(ref)
        if not proposed_targets:
            proposed_targets = [
                target
                for target in (
                    action.get("target") if isinstance(action, Mapping) else None
                    for action in response.actions
                )
                if isinstance(target, str) and target
            ]

        candidate_step_ids: list[str] = []
        raw_candidates = context.get("candidate_steps")
        if isinstance(raw_candidates, list):
            for candidate in raw_candidates:
                if isinstance(candidate, Mapping):
                    step_id = candidate.get("step_id")
                    if isinstance(step_id, str) and step_id:
                        candidate_step_ids.append(step_id)
                elif isinstance(candidate, str) and candidate:
                    candidate_step_ids.append(candidate)
        if not candidate_step_ids:
            candidate_step_ids = list(self.candidate_steps)

        vision_summary = context.get("vision_fact_summary")
        vision_seen_fact_ids: list[str] = []
        vision_fresh_fact_ids: list[str] = []
        vision_not_seen_fact_ids: list[str] = []
        if isinstance(vision_summary, Mapping):
            raw_seen = vision_summary.get("seen_fact_ids")
            if isinstance(raw_seen, (list, tuple, set)):
                vision_seen_fact_ids = [item for item in raw_seen if isinstance(item, str) and item]
            raw_fresh = vision_summary.get("fresh_fact_ids")
            if isinstance(raw_fresh, (list, tuple, set)):
                vision_fresh_fact_ids = [item for item in raw_fresh if isinstance(item, str) and item]
            raw_not_seen = vision_summary.get("not_seen_fact_ids")
            if isinstance(raw_not_seen, (list, tuple, set)):
                vision_not_seen_fact_ids = [item for item in raw_not_seen if isinstance(item, str) and item]

        missing_conditions = hint.get("missing_conditions")
        missing_set = {
            item for item in missing_conditions
            if isinstance(item, str) and item
        } if isinstance(missing_conditions, (list, tuple)) else set()

        action_hint = hint.get("action_hint")
        visual_action_hint = hint.get("visual_action_hint")
        s08_visual_hint_target: str | None = None
        s08_visual_hint_ref: str | None = None
        s08_visual_hint_used = False
        s18_visual_hint_target: str | None = None
        s18_visual_hint_reason: str | None = None
        s18_visual_hint_used = False
        s08_visual_navigation_allowed = (
            inferred_step_id == "S08"
            and not _s08_power_condition_missing(missing_set)
        )
        if inferred_step_id == "S08":
            evidence_refs = _s08_filter_unconfirmed_page_navigation_refs(context, evidence_refs)
        if s08_visual_navigation_allowed and isinstance(visual_action_hint, Mapping):
            visual_target = visual_action_hint.get("target")
            if isinstance(visual_target, str) and visual_target:
                s08_visual_hint_ref = _s08_visual_fact_ref_for_seen_target(
                    context,
                    evidence_refs,
                    visual_target,
                )
                if isinstance(s08_visual_hint_ref, str) and s08_visual_hint_ref:
                    action_hint = visual_action_hint
                    s08_visual_hint_target = visual_target
                    s08_visual_hint_used = True
                    evidence_refs = [s08_visual_hint_ref]
        if s08_visual_navigation_allowed and not s08_visual_hint_used:
            evidence_hint = _s08_visual_hint_from_seen_evidence_refs(context, evidence_refs)
            summary_hint = _s08_visual_hint_from_vision_summary(context)
            if evidence_hint is None and summary_hint is not None:
                evidence_hint = summary_hint
            if evidence_hint is not None:
                evidence_target, evidence_reason, evidence_ref = evidence_hint
                action_hint = {
                    "target": evidence_target,
                    "reason": evidence_reason,
                }
                s08_visual_hint_target = evidence_target
                s08_visual_hint_ref = evidence_ref
                s08_visual_hint_used = True
                if isinstance(evidence_ref, str) and evidence_ref:
                    evidence_refs = [evidence_ref]
        s18_bit_root_to_fcsmc_allowed = (
            inferred_step_id == "S18"
            and (
                "bit_root_page_visible" in set(vision_seen_fact_ids)
                or "bit_root_page_visible" in set(vision_fresh_fact_ids)
            )
            and "fcsmc_page_visible" in set(vision_not_seen_fact_ids)
        )
        if s18_bit_root_to_fcsmc_allowed and isinstance(action_hint, Mapping):
            action_target = action_hint.get("target")
            if action_target == "right_mdi_pb5":
                repaired_hint = dict(action_hint)
                hint_reason = repaired_hint.get("reason")
                if not isinstance(hint_reason, str) or not hint_reason:
                    hint_reason = (
                        "右 DDI 已在 BIT root 页面；请按右 DDI PB5/FCS-MC 进入 FCS-MC BIT 页面。"
                        if self.lang == "zh"
                        else "The right DDI is on the BIT root page; press right DDI PB5/FCS-MC to enter the FCS-MC BIT page."
                    )
                    repaired_hint["reason"] = hint_reason
                action_hint = repaired_hint
                s18_visual_hint_target = "right_mdi_pb5"
                s18_visual_hint_reason = hint_reason
                s18_visual_hint_used = True
                allowed_refs = _collect_request_evidence_refs(context)
                bit_root_ref_prefix = "VISION_FACTS.bit_root_page_visible"
                bit_root_ref = next(
                    (
                        ref for ref in evidence_refs
                        if ref in allowed_refs and ref.startswith(bit_root_ref_prefix)
                    ),
                    None,
                )
                if bit_root_ref is None:
                    frame_ids = vision_summary.get("frame_ids") if isinstance(vision_summary, Mapping) else None
                    candidate_refs: list[str] = []
                    if isinstance(frame_ids, (list, tuple)):
                        candidate_refs.extend(
                            f"{bit_root_ref_prefix}@{frame_id}"
                            for frame_id in frame_ids
                            if isinstance(frame_id, str) and frame_id
                        )
                    candidate_refs.append(bit_root_ref_prefix)
                    bit_root_ref = next((ref for ref in candidate_refs if ref in allowed_refs), None)
                if bit_root_ref is not None:
                    evidence_refs = [bit_root_ref]
        if inferred_step_id == "S18" and not s18_bit_root_to_fcsmc_allowed:
            return False, "legacy_s18_action_hint_guardrail"
        action_hint_step_ids = ["S08", "S09", "S17", "S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27"]
        if inferred_step_id == "S12" and _s12_fast_align_action_hint_allowed(
            context=context,
            hint=hint,
            missing_conditions=missing_set,
            action_hint=action_hint,
        ):
            action_hint_step_ids.append("S12")
        manual_text_guidance_rules: list[HarnessTextGuidanceRule] = []
        include_s05_manual_guidance = (
            "vars.throttle_r_not_off==true" in missing_set
            or "vars.throttle_r_idle_complete==true" in missing_set
        )
        include_s11_manual_guidance = (
            "vars.throttle_l_not_off==true" in missing_set
            or "vars.throttle_l_idle_complete==true" in missing_set
        )
        if self.lang == "zh":
            if include_s05_manual_guidance:
                manual_text_guidance_rules.append(HarnessTextGuidanceRule(
                    step_id="S05",
                    target="*",
                    guidance=(
                        "当前处于 S05（右发油门推进到 IDLE）阶段。右油门杆还没有移出 OFF 卡位。"
                        "该动作不在当前 overlay 布局内，请按 Right Shift+Home 将右油门杆推至 IDLE 位置。"
                    ),
                ))
            if include_s11_manual_guidance:
                manual_text_guidance_rules.append(HarnessTextGuidanceRule(
                    step_id="S11",
                    target="*",
                    guidance=(
                        "当前处于 S11（左发油门推进到 IDLE）阶段。左油门杆还没有移出 OFF 卡位。"
                        "该动作不在当前 overlay 布局内，请按 Right Alt+Home 将左油门杆推至 IDLE 位置。"
                    ),
                ))
        else:
            if include_s05_manual_guidance:
                manual_text_guidance_rules.append(HarnessTextGuidanceRule(
                    step_id="S05",
                    target="*",
                    guidance=(
                        "You are on S05 (move the right throttle to IDLE). The right throttle is still in the OFF detent. "
                        "This control is outside the current overlay layout; press Right Shift+Home to move the right throttle to IDLE."
                    ),
                ))
            if include_s11_manual_guidance:
                manual_text_guidance_rules.append(HarnessTextGuidanceRule(
                    step_id="S11",
                    target="*",
                    guidance=(
                        "You are on S11 (move the left throttle to IDLE). The left throttle is still in the OFF detent. "
                        "This control is outside the current overlay layout; press Right Alt+Home to move the left throttle to IDLE."
                    ),
                ))
        plan = plan_harness_action(
            step_specs=self.step_harness_specs,
            inferred_step_id=inferred_step_id if isinstance(inferred_step_id, str) else None,
            model_step_id=model_step_id if isinstance(model_step_id, str) else None,
            overlay_step_id=overlay_step_id if isinstance(overlay_step_id, str) else None,
            proposed_overlay_targets=proposed_targets,
            candidate_step_ids=candidate_step_ids,
            runtime_overlay_targets=self.overlay_allowlist,
            request_overlay_targets=context.get("overlay_target_allowlist"),
            allowed_evidence_refs=sorted(_collect_request_evidence_refs(context)),
            evidence_refs=evidence_refs,
            max_overlay_targets=self.max_overlay_targets,
            vision_seen_fact_ids=vision_seen_fact_ids,
            vision_fresh_fact_ids=vision_fresh_fact_ids,
            vision_not_seen_fact_ids=vision_not_seen_fact_ids,
            action_hint=action_hint if isinstance(action_hint, Mapping) else None,
            completion_advancements=[
                HarnessCompletionAdvance(
                    step_id="S19",
                    fact_id="fcsmc_final_go_result_visible",
                    next_step_id="S20",
                    source="validator_s19_final_go",
                )
            ],
            action_hint_step_ids=action_hint_step_ids,
            action_hint_fact_rules=[
                HarnessActionHintFactRule(
                    step_id="S18",
                    fact_id="bit_root_page_visible",
                    not_seen_fact_id="fcsmc_page_visible",
                    source="visual_action_hint_repair",
                ),
                HarnessActionHintFactRule(step_id="S19", fact_id="fcsmc_intermediate_result_visible")
            ],
            text_guidance_rules=manual_text_guidance_rules,
        )
        plan_guidance = plan.guidance
        if s18_visual_hint_used and (not isinstance(plan_guidance, str) or not plan_guidance):
            plan_guidance = s18_visual_hint_reason

        response.metadata["validator_rejected"] = bool(plan.validator_rejected)
        response.metadata["repair_applied"] = bool(response.metadata.get("repair_applied")) or bool(plan.repair_applied)
        response.metadata["final_action_plan_source"] = plan.final_action_plan_source
        response.metadata["harness_validation_reasons"] = list(plan.reasons)
        response.metadata["harness_action_plan"] = {
            "step_id": plan.step_id,
            "overlay_step_id": plan.overlay_step_id,
            "targets": list(plan.targets),
            "text_only": plan.text_only,
            "source": plan.final_action_plan_source,
        }
        if s08_visual_hint_used:
            response.metadata["visual_hint_target"] = s08_visual_hint_target
            if isinstance(s08_visual_hint_ref, str) and s08_visual_hint_ref:
                response.metadata["visual_hint_evidence_ref"] = s08_visual_hint_ref
            if plan.repair_applied:
                response.metadata["s08_visual_hint_repair_applied"] = True
            if plan.rejected_model_targets:
                response.metadata["rejected_model_targets"] = list(plan.rejected_model_targets)
                response.metadata["rejected_model_target"] = plan.rejected_model_targets[0]
        if s18_visual_hint_used:
            response.metadata["visual_hint_target"] = s18_visual_hint_target
            if plan.repair_applied:
                response.metadata["s18_visual_hint_repair_applied"] = True
        if plan.rejected_model_targets:
            response.metadata["rejected_model_targets"] = list(plan.rejected_model_targets)
            response.metadata["rejected_model_target"] = plan.rejected_model_targets[0]
        if isinstance(plan.rejected_model_step_id, str) and plan.rejected_model_step_id:
            response.metadata["rejected_model_step_id"] = plan.rejected_model_step_id
        elif "rejected_model_step_id" in response.metadata:
            response.metadata.pop("rejected_model_step_id", None)
        if original_help_response is not None:
            response.metadata.setdefault("model_raw_help_response", original_help_response)

        if plan.text_only:
            original_actions = copy.deepcopy([dict(action) for action in response.actions if isinstance(action, Mapping)])
            if original_actions:
                response.metadata["harness_validator_original_actions"] = original_actions
            response.actions = []
            if isinstance(plan_guidance, str) and plan_guidance:
                original_message = response.message
                original_explanations = list(response.explanations)
                response.message = plan_guidance
                response.explanations = [plan_guidance]
                if original_message != response.message:
                    response.metadata["harness_validator_original_message"] = original_message
                    response.metadata["manual_throttle_guidance_original_message"] = original_message
                if original_explanations and original_explanations != response.explanations:
                    response.metadata["harness_validator_original_explanations"] = original_explanations
                    response.metadata["manual_throttle_guidance_original_explanations"] = original_explanations
            if plan.step_id in {"S05", "S11"}:
                response.metadata["manual_throttle_guidance_rewritten"] = True
                response.metadata["manual_throttle_guidance_step_id"] = plan.step_id
                response.metadata["manual_throttle_guidance_original_actions"] = original_actions
                response.metadata["help_response"] = {
                    "diagnosis": {"step_id": plan.step_id, "error_category": "OM"},
                    "next": {"step_id": plan.step_id},
                    "overlay": {"targets": [], "evidence": []},
                    "explanations": [response.message],
                }
                return True, "manual_throttle_keyboard_guidance"
            return True, plan.final_action_plan_source

        current_targets = [
            target
            for target in (
                action.get("target") if isinstance(action, Mapping) else None
                for action in response.actions
            )
            if isinstance(target, str) and target
        ]
        current_next = response.metadata.get("next")
        current_next_step_id = current_next.get("step_id") if isinstance(current_next, Mapping) else None
        current_diagnosis = response.metadata.get("diagnosis")
        current_diagnosis_step_id = (
            current_diagnosis.get("step_id") if isinstance(current_diagnosis, Mapping) else None
        )
        only_target_repair = bool(plan.repair_applied) and all(
            isinstance(reason, str)
            and (
                reason.startswith("target_not_allowed:")
                or reason.startswith("target_not_in_request_allowlist:")
                or reason.startswith("target_dropped_by_max_overlay_targets:")
            )
            for reason in plan.reasons
        )
        if (
            response.actions
            and tuple(current_targets) == plan.targets
            and (
                (
                    not plan.repair_applied
                    and plan.rejected_model_step_id is None
                )
                or (
                    only_target_repair
                    and plan.rejected_model_step_id is None
                    and current_next_step_id == plan.step_id
                    and current_diagnosis_step_id == plan.step_id
                )
            )
        ):
            cleaned_help_response, evidence_cleaned = _s08_clean_unconfirmed_page_navigation_help_response(
                context,
                response.metadata.get("help_response"),
            )
            if evidence_cleaned and cleaned_help_response is not None:
                response.metadata["help_response"] = cleaned_help_response
                response.metadata["s08_unconfirmed_visual_evidence_filtered"] = True
                return True, "s08_unconfirmed_visual_evidence_filtered"
            return False, "already_valid"
        if not plan.targets:
            if plan.validator_rejected and response.actions:
                response.metadata["harness_validator_original_actions"] = copy.deepcopy(
                    [dict(action) for action in response.actions if isinstance(action, Mapping)]
                )
                response.actions = []
                if isinstance(original_help_response, Mapping):
                    cleaned_help_response = copy.deepcopy(dict(original_help_response))
                    cleaned_help_response["diagnosis"] = {"step_id": plan.step_id, "error_category": "OM"}
                    cleaned_help_response["next"] = {"step_id": plan.step_id}
                    cleaned_help_response["overlay"] = {"targets": [], "evidence": []}
                    response.metadata["help_response"] = cleaned_help_response
                return True, "validator_rejected_no_action"
            return False, "no_planned_targets"

        original_actions = copy.deepcopy([dict(action) for action in response.actions if isinstance(action, Mapping)])
        if original_actions:
            response.metadata["harness_validator_original_actions"] = original_actions

        fallback_help_obj, fallback_reason = self._build_safe_fallback_overlay_help_obj(
            request,
            override_inferred_step_id=plan.step_id,
            override_overlay_step_id=plan.overlay_step_id,
            ignore_request_allowlist=True,
        )
        if isinstance(fallback_help_obj, Mapping):
            planned_help_obj = copy.deepcopy(dict(fallback_help_obj))
        else:
            planned_help_obj = {
                "diagnosis": {"step_id": plan.step_id, "error_category": "OM"},
                "next": {"step_id": plan.step_id},
                "overlay": {"targets": [], "evidence": []},
                "explanations": list(response.explanations) or ([response.message] if response.message else []),
            }
        planned_help_obj["diagnosis"] = {"step_id": plan.step_id, "error_category": "OM"}
        planned_help_obj["next"] = {"step_id": plan.step_id}
        evidence_source = None
        for ref in plan.evidence_refs:
            evidence_type = infer_evidence_type_from_ref(ref)
            if evidence_type is None:
                continue
            evidence_source = {
                "type": evidence_type,
                "ref": ref,
                "quote": "Harness validator planned target.",
                "grounding_confidence": 0.51,
            }
            break
        fallback_overlay = fallback_help_obj.get("overlay") if isinstance(fallback_help_obj, Mapping) else None
        if evidence_source is None and isinstance(fallback_overlay, Mapping):
            fallback_evidence = fallback_overlay.get("evidence")
            if isinstance(fallback_evidence, list):
                for item in fallback_evidence:
                    if not isinstance(item, Mapping):
                        continue
                    ref = item.get("ref")
                    evidence_type = item.get("type")
                    if isinstance(ref, str) and isinstance(evidence_type, str):
                        evidence_source = {
                            "type": evidence_type,
                            "ref": ref,
                            "quote": item.get("quote") if isinstance(item.get("quote"), str) else "Harness validator planned target.",
                            "grounding_confidence": item.get("grounding_confidence", 0.51),
                        }
                        break
        if evidence_source is None:
            return False, "fallback_failed:no_verifiable_evidence_ref"
        planned_help_obj["overlay"] = {
            "targets": list(plan.targets),
            "evidence": [
                {
                    "target": target,
                    "type": evidence_source["type"],
                    "ref": evidence_source["ref"],
                    "quote": evidence_source["quote"],
                    "grounding_confidence": evidence_source["grounding_confidence"],
                }
                for target in plan.targets
            ],
        }
        if isinstance(plan_guidance, str) and plan_guidance:
            planned_help_obj["explanations"] = [plan_guidance]

        mapped = map_help_response_to_tutor_response(
            planned_help_obj,
            request=request,
            status=response.status,
            max_overlay_targets=self.max_overlay_targets,
            ui_map_path=self.ui_map_path,
            lang=self.lang,
        )
        mapped_meta = dict(mapped.metadata)
        response.metadata["harness_validator_mapping"] = mapped_meta
        if not mapped.actions:
            mapping_errors = mapped_meta.get("mapping_errors")
            if isinstance(mapping_errors, list) and mapping_errors:
                return False, f"mapping_failed:{'|'.join(str(item) for item in mapping_errors[:3])}"
            return False, "mapping_failed"
        response.actions = list(mapped.actions)
        response.metadata["diagnosis"] = {"step_id": plan.step_id, "error_category": "OM"}
        response.metadata["next"] = {"step_id": plan.step_id}
        response.metadata["help_response"] = planned_help_obj
        response.metadata["harness_validator_fallback_reason"] = fallback_reason
        if isinstance(plan_guidance, str) and plan_guidance:
            response.message = plan_guidance
            response.explanations = [plan_guidance]
        elif mapped.explanations and response.metadata.get("procedural_guidance_rewritten") is not True:
            response.message = mapped.explanations[0]
            response.explanations = list(mapped.explanations)
        return True, fallback_reason

    def _rewrite_manual_throttle_guidance_response(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[bool, str]:
        if not response.actions:
            return False, "missing_actions"
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return False, "missing_deterministic_hint"
        inferred_step_id = hint.get("inferred_step_id")
        if inferred_step_id not in {"S05", "S11"}:
            return False, "not_manual_throttle_step"

        current_targets = [
            target
            for target in (
                action.get("target") if isinstance(action, Mapping) else None
                for action in response.actions
            )
            if isinstance(target, str) and target
        ]
        if current_targets != ["throttle_quadrant_reference"]:
            return False, "target_not_throttle_reference"

        missing_conditions = hint.get("missing_conditions")
        missing_set = {
            item for item in missing_conditions
            if isinstance(item, str) and item
        } if isinstance(missing_conditions, (list, tuple)) else set()
        if inferred_step_id == "S05" and not (
            "vars.throttle_r_not_off==true" in missing_set
            or "vars.throttle_r_idle_complete==true" in missing_set
        ):
            return False, "missing_condition_not_right_throttle"
        if inferred_step_id == "S11" and not (
            "vars.throttle_l_not_off==true" in missing_set
            or "vars.throttle_l_idle_complete==true" in missing_set
        ):
            return False, "missing_condition_not_left_throttle"

        original_message = response.message
        original_explanations = list(response.explanations)
        original_actions = copy.deepcopy([dict(action) for action in response.actions if isinstance(action, Mapping)])

        if inferred_step_id == "S11":
            if self.lang == "zh":
                rewritten = (
                    "当前处于 S11（左发油门推进到 IDLE）阶段。左油门杆还没有移出 OFF 卡位。"
                    "该动作不在当前 overlay 布局内，请按 Right Alt+Home 将左油门杆推至 IDLE 位置。"
                )
            else:
                rewritten = (
                    "You are on S11 (move the left throttle to IDLE). The left throttle is still in the OFF detent. "
                    "This control is outside the current overlay layout; press Right Alt+Home to move the left throttle to IDLE."
                )
        else:
            if self.lang == "zh":
                rewritten = (
                    "当前处于 S05（右发油门推进到 IDLE）阶段。右油门杆还没有移出 OFF 卡位。"
                    "该动作不在当前 overlay 布局内，请按 Right Shift+Home 将右油门杆推至 IDLE 位置。"
                )
            else:
                rewritten = (
                    "You are on S05 (move the right throttle to IDLE). The right throttle is still in the OFF detent. "
                    "This control is outside the current overlay layout; press Right Shift+Home to move the right throttle to IDLE."
                )

        response.actions = []
        response.message = rewritten
        response.explanations = [rewritten]
        response.metadata["manual_throttle_guidance_rewritten"] = True
        response.metadata["manual_throttle_guidance_step_id"] = inferred_step_id
        response.metadata["manual_throttle_guidance_original_actions"] = original_actions
        if original_message != rewritten:
            response.metadata["manual_throttle_guidance_original_message"] = original_message
        if original_explanations and original_explanations != [rewritten]:
            response.metadata["manual_throttle_guidance_original_explanations"] = original_explanations
        return True, "manual_throttle_keyboard_guidance"

    def _rewrite_procedural_guidance_response(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[bool, str]:
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return False, "missing_deterministic_hint"
        inferred_step_id = hint.get("inferred_step_id")
        missing_conditions = hint.get("missing_conditions")
        missing_set = {
            item for item in missing_conditions
            if isinstance(item, str) and item
        } if isinstance(missing_conditions, (list, tuple)) else set()

        vars_selected = context.get("vars")
        vars_map = vars_selected if isinstance(vars_selected, Mapping) else {}
        vision_fact_summary = context.get("vision_fact_summary")
        vision_summary = vision_fact_summary if isinstance(vision_fact_summary, Mapping) else {}
        vision_seen_or_fresh: set[str] = set()
        for key in ("seen_fact_ids", "fresh_fact_ids"):
            raw_fact_ids = vision_summary.get(key)
            if not isinstance(raw_fact_ids, (list, tuple, set)):
                continue
            vision_seen_or_fresh.update(item for item in raw_fact_ids if isinstance(item, str) and item)
        rewritten: str | None = None
        reason = "not_applicable"

        def _set_text_only_help_response(step_id: str, text: str) -> None:
            help_response = response.metadata.get("help_response")
            rewritten_help_response = dict(help_response) if isinstance(help_response, Mapping) else {}
            rewritten_help_response["diagnosis"] = {"step_id": step_id, "error_category": "OM"}
            rewritten_help_response["next"] = {"step_id": step_id}
            rewritten_help_response["overlay"] = {"targets": [], "evidence": []}
            rewritten_help_response["explanations"] = [text]
            response.metadata["help_response"] = rewritten_help_response

        if inferred_step_id == "S02":
            if "vars.fire_test_a_complete==true" in missing_set:
                reason = "s02_fire_test_a_guidance"
                if self.lang == "zh":
                    rewritten = (
                        "当前处于 S02。请右键按住 Fire and Bleed Air Test Switch 到 TEST A，"
                        "观察火警灯和语音提示；完成后松开并等待约 10 秒，再进行 TEST B。"
                    )
                else:
                    rewritten = (
                        "You are on S02. Right-click and hold the Fire and Bleed Air Test Switch "
                        "to TEST A, observe the fire warning light and aural cues, then release "
                        "and wait about 10 seconds before TEST B."
                    )
            elif "vars.fire_test_b_complete==true" in missing_set:
                reason = "s02_fire_test_b_guidance"
                if self.lang == "zh":
                    rewritten = (
                        "FIRE TEST A 已完成。现在请左键按住 Fire and Bleed Air Test Switch "
                        "到 TEST B，并等待第二组火警灯和语音提示完成。"
                    )
                else:
                    rewritten = (
                        "Fire Test A is complete. Now left-click and hold the Fire and Bleed Air "
                        "Test Switch to TEST B and wait for the second set of fire warning cues."
                    )
        elif inferred_step_id == "S03":
            apu_on = vars_map.get("apu_on")
            apu_ready = vars_map.get("apu_ready")
            if apu_on is True and apu_ready is not True:
                reason = "s03_wait_for_apu_ready"
                if self.lang == "zh":
                    rewritten = "APU 开关已经在 ON。请等待绿色 APU READY 灯亮起，再继续启动右发。"
                else:
                    rewritten = (
                        "The APU switch is already ON. Wait for the green APU READY light "
                        "before continuing to right-engine start."
                    )
            elif apu_on is not True and (
                "vars.apu_start_support_complete==true" in missing_set
                or "vars.apu_on==true" in missing_set
            ):
                reason = "s03_set_apu_on"
                if self.lang == "zh":
                    rewritten = "当前处于 S03。请左键点击 APU 开关将其拨到 ON，然后等待绿色 APU READY 灯亮起。"
                else:
                    rewritten = (
                        "You are on S03. Left-click the APU switch to ON, then wait for "
                        "the green APU READY light."
                    )
        elif inferred_step_id == "S09" and _s09_comm1_frequency_complete(vars_map):
            reason = "s09_comm1_frequency_complete"
            if self.lang == "zh":
                rewritten = "COMM1 预置 1 已经是 134.000 MHz，S09 已完成。下一步进入 S10，启动左发。"
            else:
                rewritten = "COMM1 preset 1 is already set to 134.000 MHz, so S09 is complete. Continue to S10 by starting the left engine."
            response.actions = []
            response.metadata["diagnosis"] = {"step_id": "S10", "error_category": "OM"}
            response.metadata["next"] = {"step_id": "S10"}
            _set_text_only_help_response("S10", rewritten)
            fallback_used, fallback_reason = self._apply_safe_fallback_overlay(
                response,
                request,
                override_inferred_step_id="S10",
                override_overlay_step_id="S10",
                ignore_request_allowlist=True,
            )
            response.metadata["s09_comm1_completion_guardrail_applied"] = True
            response.metadata["s09_comm1_completion_s10_overlay_applied"] = fallback_used
            response.metadata["s09_comm1_completion_s10_overlay_reason"] = fallback_reason
        elif inferred_step_id == "S09" and "vars.comm1_freq_134_000==true" in missing_set:
            reason = "s09_comm1_frequency_guidance"
            if self.lang == "zh":
                rewritten = (
                    "当前处于 S09。请把 COMM1 预置 1 设置为 134.000 MHz：先拉出 UFC 的 COMM1 "
                    "频道选择钮，输入 1-3-4-0-0-0，然后按 ENT 确认。"
                )
            else:
                rewritten = (
                    "You are on S09. Set COMM1 preset 1 to 134.000 MHz: pull the UFC COMM1 "
                    "channel selector, enter 1-3-4-0-0-0, then press ENT."
                )
        elif inferred_step_id == "S12":
            if "vars.ins_fast_align_complete==true" in missing_set and (
                vars_map.get("ins_mode_cv_or_gnd") is True or vars_map.get("ins_mode_set") is True
            ) and _s12_fast_align_action_hint_allowed(
                context=context,
                hint=hint,
                missing_conditions=missing_set,
                action_hint={"target": "ampcd_pb19"},
            ):
                reason = "s12_ampcd_pb19_fast_align_guidance"
                if self.lang == "zh":
                    rewritten = "INS 已设置到对准模式。现在请到 AMPCD 按 PB19，启动快速 INS 校准。"
                else:
                    rewritten = "INS is already set for alignment. Press AMPCD PB19 now to start fast INS alignment."
        elif inferred_step_id in {"S20", "S21"}:
            probe_motion_state = _refuel_probe_motion_state(context, inferred_step_id)
            if probe_motion_state == "s20_extending":
                reason = "s20_refuel_probe_extending_wait"
                if self.lang == "zh":
                    rewritten = "受油管正在伸出。请等待它完全伸出后，再继续四落检查。"
                else:
                    rewritten = "The refueling probe is extending. Wait until it is fully extended before continuing the four-down check."
                response.actions = []
                response.metadata["diagnosis"] = {"step_id": "S20", "error_category": "OM"}
                response.metadata["next"] = {"step_id": "S20"}
                _set_text_only_help_response("S20", rewritten)
            elif probe_motion_state == "s20_extended":
                reason = "s20_refuel_probe_extended_complete"
                if self.lang == "zh":
                    rewritten = "受油管已经完全伸出，S20 已完成。下一步进入 S21，收起受油管。"
                else:
                    rewritten = "The refueling probe is fully extended, so S20 is complete. Continue to S21 by retracting the probe."
                response.actions = []
                response.metadata["diagnosis"] = {"step_id": "S21", "error_category": "OM"}
                response.metadata["next"] = {"step_id": "S21"}
                _set_text_only_help_response("S21", rewritten)
                fallback_used, fallback_reason = self._apply_safe_fallback_overlay(
                    response,
                    request,
                    override_inferred_step_id="S21",
                    override_overlay_step_id="S21",
                    ignore_request_allowlist=True,
                )
                response.metadata["refuel_probe_completion_s21_overlay_applied"] = fallback_used
                response.metadata["refuel_probe_completion_s21_overlay_reason"] = fallback_reason
            elif probe_motion_state == "s21_retracting":
                reason = "s21_refuel_probe_retracting_wait"
                if self.lang == "zh":
                    rewritten = "受油管正在收起，已经接近收起阈值。请等待它完全收好后，再继续下一步。"
                else:
                    rewritten = "The refueling probe is retracting and is near the stowed threshold. Wait until it is fully stowed before continuing."
                response.actions = []
                response.metadata["diagnosis"] = {"step_id": "S21", "error_category": "OM"}
                response.metadata["next"] = {"step_id": "S21"}
                _set_text_only_help_response("S21", rewritten)
            elif probe_motion_state == "s21_retracted":
                reason = "s21_refuel_probe_retracted_complete"
                if self.lang == "zh":
                    rewritten = "受油管已经完全收起，S21 已完成。下一步进入 S22，准备放下 launch bar。"
                else:
                    rewritten = "The refueling probe is fully stowed, so S21 is complete. Continue to S22 by extending the launch bar."
                response.actions = []
                response.metadata["diagnosis"] = {"step_id": "S22", "error_category": "OM"}
                response.metadata["next"] = {"step_id": "S22"}
                _set_text_only_help_response("S22", rewritten)
                fallback_used, fallback_reason = self._apply_safe_fallback_overlay(
                    response,
                    request,
                    override_inferred_step_id="S22",
                    override_overlay_step_id="S22",
                    ignore_request_allowlist=True,
                )
                response.metadata["refuel_probe_completion_s22_overlay_applied"] = fallback_used
                response.metadata["refuel_probe_completion_s22_overlay_reason"] = fallback_reason
        elif inferred_step_id == "S19":
            if "fcsmc_final_go_result_visible" in vision_seen_or_fresh:
                reason = "s19_final_go_complete"
                rejected_model_step_id = _extract_model_next_step_id(response.metadata)
                if self.lang == "zh":
                    rewritten = "FCS BIT 最终 GO 已显示，S19 已完成。下一步进入 S20，先展开受油管开始四落检查。"
                else:
                    rewritten = "The final FCS BIT GO result is visible, so S19 is complete. Continue to S20 by extending the refueling probe."
                response.actions = []
                response.metadata["diagnosis"] = {"step_id": "S20"}
                response.metadata["next"] = {"step_id": "S20"}
                help_response = response.metadata.get("help_response")
                if isinstance(help_response, Mapping):
                    rewritten_help_response = dict(help_response)
                    rewritten_help_response["diagnosis"] = {"step_id": "S20"}
                    rewritten_help_response["next"] = {"step_id": "S20"}
                    rewritten_help_response["overlay"] = {"targets": [], "evidence": []}
                    rewritten_help_response["explanations"] = [rewritten]
                    response.metadata["help_response"] = rewritten_help_response
                fallback_used, fallback_reason = self._apply_safe_fallback_overlay(
                    response,
                    request,
                    override_inferred_step_id="S20",
                    override_overlay_step_id="S20",
                    ignore_request_allowlist=True,
                )
                response.metadata["s19_final_go_guardrail_applied"] = True
                response.metadata["s19_final_go_guardrail_reason"] = reason
                response.metadata["s19_final_go_s20_overlay_applied"] = fallback_used
                response.metadata["s19_final_go_s20_overlay_reason"] = fallback_reason
                if isinstance(rejected_model_step_id, str) and rejected_model_step_id:
                    response.metadata["rejected_model_step_id"] = rejected_model_step_id
            elif "fcsmc_intermediate_result_visible" in vision_seen_or_fresh:
                reason = "s19_intermediate_requires_bit_start"
                if self.lang == "zh":
                    rewritten = (
                        "右 DDI 已显示 FCS-MC 的 FCSA/FCSB PBIT GO 页面。"
                        "请按住 FCS BIT 开关向上，同时按右 DDI PB5 启动测试；看到测试开始后即可松开。"
                    )
                else:
                    rewritten = (
                        "The right DDI shows the FCS-MC FCSA/FCSB PBIT GO page. "
                        "Hold the FCS BIT switch up while pressing Right DDI PB5 to start the BIT; release once the test starts."
                    )
                response.actions = []
                fallback_used, fallback_reason = self._apply_safe_fallback_overlay(
                    response,
                    request,
                    override_inferred_step_id="S19",
                    override_overlay_step_id="S19",
                    ignore_request_allowlist=True,
                )
                response.metadata["s19_intermediate_guardrail_applied"] = True
                response.metadata["s19_intermediate_overlay_applied"] = fallback_used
                response.metadata["s19_intermediate_overlay_reason"] = fallback_reason
                help_response = response.metadata.get("help_response")
                if isinstance(help_response, Mapping):
                    rewritten_help_response = dict(help_response)
                    rewritten_help_response["diagnosis"] = {"step_id": "S19"}
                    rewritten_help_response["next"] = {"step_id": "S19"}
                    rewritten_help_response["explanations"] = [rewritten]
                    if response.actions:
                        rewritten_help_response["overlay"] = {
                            "targets": [
                                action.get("target")
                                for action in response.actions
                                if isinstance(action, Mapping) and isinstance(action.get("target"), str)
                            ],
                            "evidence": [],
                        }
                    response.metadata["help_response"] = rewritten_help_response
            elif "fcsmc_in_test_visible" in vision_seen_or_fresh:
                reason = "s19_fcs_bit_in_test_wait"
                if self.lang == "zh":
                    rewritten = "FCS BIT 已经开始运行。看到 IN TEST 后无需继续保持 FCS BIT 开关向上，请松开并等待最终 GO。"
                else:
                    rewritten = (
                        "The FCS BIT is already running. Once IN TEST is visible, "
                        "you do not need to keep holding the FCS BIT switch; release it and wait for the final GO."
                    )
                response.actions = []
                help_response = response.metadata.get("help_response")
                if isinstance(help_response, Mapping):
                    rewritten_help_response = dict(help_response)
                    rewritten_help_response["diagnosis"] = {"step_id": "S19"}
                    rewritten_help_response["next"] = {"step_id": "S19"}
                    rewritten_help_response["overlay"] = {"targets": [], "evidence": []}
                    rewritten_help_response["explanations"] = [rewritten]
                    response.metadata["help_response"] = rewritten_help_response

        if not rewritten:
            return False, reason

        original_message = response.message
        original_explanations = list(response.explanations)
        response.message = rewritten
        response.explanations = [rewritten]
        response.metadata["procedural_guidance_rewritten"] = True
        response.metadata["procedural_guidance_rewrite_reason"] = reason
        if reason.startswith(("s20_refuel_probe_", "s21_refuel_probe_")):
            response.metadata["refuel_probe_motion_guidance_rewritten"] = True
        if original_message != rewritten:
            response.metadata["procedural_guidance_original_message"] = original_message
        if original_explanations and original_explanations != [rewritten]:
            response.metadata["procedural_guidance_original_explanations"] = original_explanations
        return True, reason

    def _map_response_actions(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        metadata = response.metadata if isinstance(response.metadata, Mapping) else {}
        help_obj = metadata.get("help_response")
        if not isinstance(help_obj, Mapping):
            return list(response.actions), {}

        filtered_help_obj: Mapping[str, Any] = help_obj
        rejected_by_request_allowlist: list[str] = []
        request_allowlist_raw = request.context.get("overlay_target_allowlist")
        if isinstance(request_allowlist_raw, list):
            request_allowlist = {
                item for item in request_allowlist_raw if isinstance(item, str) and item
            }
            overlay = help_obj.get("overlay")
            if request_allowlist and isinstance(overlay, Mapping):
                raw_targets = overlay.get("targets")
                if isinstance(raw_targets, list):
                    allowed_targets: list[str] = []
                    for target in raw_targets:
                        if not isinstance(target, str) or not target:
                            continue
                        if target in request_allowlist:
                            allowed_targets.append(target)
                        else:
                            rejected_by_request_allowlist.append(target)
                    if rejected_by_request_allowlist:
                        allowed_target_set = set(allowed_targets)
                        filtered_overlay = dict(overlay)
                        filtered_overlay["targets"] = allowed_targets
                        evidence_raw = overlay.get("evidence")
                        if isinstance(evidence_raw, list):
                            filtered_evidence: list[Any] = []
                            for item in evidence_raw:
                                if not isinstance(item, Mapping):
                                    continue
                                evidence_target = item.get("target")
                                if not isinstance(evidence_target, str) or evidence_target not in allowed_target_set:
                                    continue
                                filtered_evidence.append(item)
                            filtered_overlay["evidence"] = filtered_evidence
                        filtered_help_obj = dict(help_obj)
                        filtered_help_obj["overlay"] = filtered_overlay

        mapped = map_help_response_to_tutor_response(
            filtered_help_obj,
            request=request,
            status=response.status,
            max_overlay_targets=self.max_overlay_targets,
            ui_map_path=self.ui_map_path,
            lang=self.lang,
        )
        mapped_meta = dict(mapped.metadata)
        if rejected_by_request_allowlist:
            deduped_rejected = _dedupe_strings(rejected_by_request_allowlist)
            mapped_meta["rejected_targets_by_request_allowlist"] = deduped_rejected
            existing_errors = mapped_meta.get("mapping_errors")
            merged_errors: list[str] = []
            if isinstance(existing_errors, list):
                merged_errors = [item for item in existing_errors if isinstance(item, str) and item]
            merged_errors.append("overlay_target_not_in_request_allowlist")
            mapped_meta["mapping_errors"] = _dedupe_strings(merged_errors)
            mapped_meta.setdefault("mapping_error", "overlay_target_not_in_request_allowlist")
        if not response.message and mapped.message:
            response.message = mapped.message
        if (not response.explanations) and mapped.explanations:
            response.explanations = list(mapped.explanations)
        return list(mapped.actions), mapped_meta

    def _backfill_s18_dual_overlay_targets(
        self,
        help_obj: Mapping[str, Any],
        request: TutorRequest,
    ) -> tuple[Mapping[str, Any], bool, str | None]:
        del request
        return help_obj, False, "removed_after_issue_227"

    def _build_safe_fallback_overlay_help_obj(
        self,
        request: TutorRequest,
        *,
        override_inferred_step_id: str | None = None,
        override_overlay_step_id: str | None = None,
        ignore_request_allowlist: bool = False,
    ) -> tuple[dict[str, Any] | None, str]:
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return None, "missing_deterministic_hint"

        inferred_step_id = override_inferred_step_id
        if not isinstance(inferred_step_id, str) or not inferred_step_id:
            inferred_step_id = hint.get("inferred_step_id")
        if not isinstance(inferred_step_id, str) or not inferred_step_id:
            return None, "missing_inferred_step_id"
        overlay_step_id = override_overlay_step_id
        if not isinstance(overlay_step_id, str) or not overlay_step_id:
            overlay_step_id = hint.get("overlay_step_id")
        if not isinstance(overlay_step_id, str) or not overlay_step_id:
            overlay_step_id = inferred_step_id

        step_fallback_profile = self.step_fallback_profiles.get(overlay_step_id)
        if not isinstance(step_fallback_profile, Mapping):
            return None, f"unsupported_step:{overlay_step_id}"
        fallback_targets_raw = step_fallback_profile.get("ui_targets")
        fallback_targets = _normalize_step_ui_targets(fallback_targets_raw)
        if not fallback_targets:
            return None, f"unsupported_step:{overlay_step_id}"
        if step_fallback_profile.get("overlay_enabled") is False:
            return None, f"overlay_disabled:{overlay_step_id}"
        declared_fallback_target = fallback_targets[0]
        missing_conditions_raw = hint.get("missing_conditions")
        missing_conditions = [
            item for item in missing_conditions_raw if isinstance(item, str) and item
        ] if isinstance(missing_conditions_raw, (list, tuple)) else []
        gate_blockers_raw = hint.get("gate_blockers")
        gate_blockers = [
            item for item in gate_blockers_raw if isinstance(item, Mapping) and item
        ] if isinstance(gate_blockers_raw, (list, tuple)) else []

        if inferred_step_id == "S33" and not missing_conditions and not gate_blockers:
            return None, "all_steps_complete"

        _precondition_blocked = any(
            isinstance(b, Mapping)
            and str(b.get("ref", "")).endswith(".precondition")
            for b in gate_blockers
        )

        request_allowlist = context.get("overlay_target_allowlist")
        candidate_targets = list(fallback_targets)
        action_hint = hint.get("action_hint")
        if isinstance(action_hint, Mapping):
            action_target = action_hint.get("target")
            if isinstance(action_target, str) and action_target in candidate_targets:
                candidate_targets = [
                    action_target,
                    *[target for target in candidate_targets if target != action_target],
                ]
            elif isinstance(action_hint.get("targets"), list):
                hinted_targets_list = [t for t in action_hint["targets"] if isinstance(t, str) and t in candidate_targets]
                if hinted_targets_list:
                    candidate_targets = [
                        *hinted_targets_list,
                        *[target for target in candidate_targets if target not in set(hinted_targets_list)],
                    ]
        navigation_targets = _prefer_navigation_target_from_vision_context(
            inferred_step_id=inferred_step_id,
            missing_conditions=missing_conditions,
            context=context,
            allowed_targets=candidate_targets,
        )
        s08_visual_page_targets = _prefer_s08_visual_page_targets(
            missing_conditions=missing_conditions,
            allowed_targets=candidate_targets,
            max_targets=self.max_overlay_targets,
        ) if overlay_step_id == "S08" else []
        if s08_visual_page_targets:
            candidate_targets = [
                *s08_visual_page_targets,
                *[target for target in candidate_targets if target not in set(s08_visual_page_targets)],
            ]
        if navigation_targets and not s08_visual_page_targets:
            candidate_targets = [
                *navigation_targets,
                *[target for target in candidate_targets if target not in set(navigation_targets)],
            ]
        hinted_targets = _missing_condition_target_hints(
            missing_conditions,
            allowed_targets=candidate_targets,
        )
        if hinted_targets:
            candidate_targets = [*hinted_targets, *[target for target in candidate_targets if target not in set(hinted_targets)]]

        # When the step's PRECONDITION gate is blocked and its missing condition
        # does not map to any UI target, remove the step's completion-related
        # fallback targets to avoid misleading highlights.
        # (e.g., RPM < 60% → precondition "rpm_r_gte_60" has no UI target,
        # but step completion target "bleed_air_knob" should not be highlighted yet.)
        _precondition_var_blocked = False
        if _precondition_blocked and not hinted_targets and missing_conditions:
            step_pre_gates = self.precondition_gates.get(inferred_step_id)
            if isinstance(step_pre_gates, (list, tuple)) and step_pre_gates:
                _pre_vars: set[str] = set()
                for rule in step_pre_gates:
                    if isinstance(rule, Mapping):
                        raw_var = rule.get("var")
                        if isinstance(raw_var, str):
                            key = raw_var.replace("payload.vars.", "").replace("vars.", "")
                            _pre_vars.add(key)
                for item in missing_conditions:
                    if isinstance(item, str):
                        m = _MISSING_CONDITION_VAR_RE.search(item)
                        if m and m.group(1) in _pre_vars:
                            _precondition_var_blocked = True
                            break
        if _precondition_var_blocked:
            candidate_targets = [
                t for t in candidate_targets
                if t not in set(fallback_targets)
            ]
            if not candidate_targets:
                _precondition_reason = next(
                    (b.get("reason", "") for b in gate_blockers
                     if isinstance(b, Mapping) and str(b.get("ref", "")).endswith(".precondition")),
                    "precondition_blocked",
                )
                return None, f"precondition_blocked:{_precondition_reason}"

        if (not ignore_request_allowlist) and isinstance(request_allowlist, list):
            allowset = {item for item in request_allowlist if isinstance(item, str) and item}
            if allowset:
                candidate_targets = [target for target in candidate_targets if target in allowset]
                if not candidate_targets:
                    return None, f"target_not_in_request_allowlist:{declared_fallback_target}"
        candidate_targets = [target for target in candidate_targets if target in self.overlay_allowset]
        if not candidate_targets:
            return None, f"target_not_in_runtime_allowlist:{declared_fallback_target}"
        remaining = [t for t in candidate_targets if t not in self._step_interacted_targets]
        if remaining:
            candidate_targets = remaining
        candidate_targets = _enforce_s08_ddi_before_ampcd(
            candidate_targets,
            step_id=overlay_step_id,
        )
        if overlay_step_id == "S08":
            s08_missing_power_targets = _s08_power_targets_for_missing_conditions(
                missing_conditions,
                allowed_targets=candidate_targets,
            )
            if s08_missing_power_targets:
                fallback_targets_list = s08_missing_power_targets[: max(1, int(self.max_overlay_targets))]
            elif s08_visual_page_targets:
                fallback_targets_list = candidate_targets[: max(1, int(self.max_overlay_targets))]
            else:
                s08_power_vars = {"left_ddi_on", "right_ddi_on", "mpcd_on", "hud_on"}
                s08_power_missing_count = sum(
                    1 for item in missing_conditions
                    if isinstance(item, str) and any(
                        item.startswith(f"vars.{v}==") for v in s08_power_vars
                    )
                )
                if s08_power_missing_count > 1:
                    s08_power_order = [
                        "left_mdi_brightness_selector",
                        "right_mdi_brightness_selector",
                        "ampcd_off_brightness_knob",
                        "hud_symbology_brightness_knob",
                    ]
                    candidate_set = set(candidate_targets)
                    s08_power_candidates = [t for t in s08_power_order if t in candidate_set]
                    if len(s08_power_candidates) > 1:
                        fallback_targets_list = s08_power_candidates[: max(1, int(self.max_overlay_targets))]
                    else:
                        fallback_targets_list = [candidate_targets[0]]
                else:
                    fallback_targets_list = [candidate_targets[0]]
        elif overlay_step_id == "S19":
            action_hint_targets = []
            if isinstance(action_hint, Mapping) and isinstance(action_hint.get("targets"), list):
                action_hint_targets = [
                    target for target in action_hint["targets"]
                    if isinstance(target, str) and target in candidate_targets
                ]
            fallback_targets_list = (
                action_hint_targets[: max(1, int(self.max_overlay_targets))]
                if action_hint_targets
                else [candidate_targets[0]]
            )
        else:
            fallback_targets_list = [candidate_targets[0]]
        fallback_target = fallback_targets_list[0]

        candidate_refs: list[str] = []
        gate_blockers = hint.get("gate_blockers")
        if isinstance(gate_blockers, (list, tuple)):
            for blocker in gate_blockers:
                if not isinstance(blocker, Mapping):
                    continue
                ref = blocker.get("ref")
                if isinstance(ref, str) and ref:
                    candidate_refs.append(ref)
        candidate_refs.append(f"GATES.{inferred_step_id}.completion")
        candidate_refs.append(f"GATES.{inferred_step_id}.precondition")
        if overlay_step_id != inferred_step_id:
            candidate_refs.append(f"GATES.{overlay_step_id}.completion")
            candidate_refs.append(f"GATES.{overlay_step_id}.precondition")
        gate_var_refs = step_fallback_profile.get("gate_var_refs")
        if isinstance(gate_var_refs, list):
            for ref in gate_var_refs:
                if isinstance(ref, str) and ref:
                    candidate_refs.append(ref)
        candidate_refs.append(f"RECENT_UI_TARGETS.{fallback_target}")
        rag_topk = context.get("rag_topk")
        if isinstance(rag_topk, list):
            for snippet in rag_topk:
                if not isinstance(snippet, Mapping):
                    continue
                snippet_id = snippet.get("snippet_id")
                if isinstance(snippet_id, str) and snippet_id:
                    candidate_refs.append(f"RAG_SNIPPETS.{snippet_id}")

        allowed_refs = _collect_request_evidence_refs(context)
        step_evidence_requirements = hint.get("step_evidence_requirements")
        if isinstance(override_inferred_step_id, str) and override_inferred_step_id:
            override_profile = self.step_signal_profiles.get(override_inferred_step_id)
            if isinstance(override_profile, Mapping):
                override_requirements = override_profile.get("evidence_requirements")
                if isinstance(override_requirements, list):
                    step_evidence_requirements = override_requirements
        allowed_evidence_types: set[str] | None = None
        if isinstance(step_evidence_requirements, list):
            allowed_evidence_types = set()
            for req in step_evidence_requirements:
                if not isinstance(req, str) or not req:
                    continue
                if req in {"var", "gate", "delta", "rag"}:
                    allowed_evidence_types.add(req)
        selected: tuple[str, str] | None = None
        for ref in _dedupe_strings(candidate_refs):
            if ref in allowed_refs:
                evidence_type = infer_evidence_type_from_ref(ref)
                if evidence_type is None:
                    continue
                if allowed_evidence_types is not None and evidence_type not in allowed_evidence_types:
                    continue
                selected = (ref, evidence_type)
                break
        if selected is None:
            return None, "no_verifiable_evidence_ref"
        selected_ref, evidence_type = selected

        reason_text = None
        if isinstance(gate_blockers, (list, tuple)):
            for blocker in gate_blockers:
                if not isinstance(blocker, Mapping):
                    continue
                if blocker.get("ref") == selected_ref:
                    reason = blocker.get("reason")
                    if isinstance(reason, str) and reason:
                        reason_text = reason
                        break
                    reason_code = blocker.get("reason_code")
                    if isinstance(reason_code, str) and reason_code:
                        reason_text = reason_code
                        break
        if not reason_text:
            reason_text = f"Deterministic blocker indicates target {fallback_target}."
        quote = reason_text.strip()
        if len(quote) > 120:
            quote = quote[:117].rstrip() + "..."

        fallback_help_obj = {
            "diagnosis": {
                "step_id": inferred_step_id,
                "error_category": "OM",
            },
            "next": {
                "step_id": inferred_step_id,
            },
            "overlay": {
                "targets": list(fallback_targets_list),
                "evidence": [
                    {
                        "target": t,
                        "type": evidence_type,
                        "ref": selected_ref,
                        "quote": quote,
                        "grounding_confidence": 0.51,
                    }
                    for t in fallback_targets_list
                ],
            },
            "explanations": [
                (
                    f"请先操作 {', '.join(fallback_targets_list)}。"
                    if self.lang == "zh"
                    else f"Please operate {', '.join(fallback_targets_list)} first."
                )
            ],
        }
        return fallback_help_obj, f"deterministic_step:{inferred_step_id}"

    def _apply_safe_fallback_overlay(
        self,
        response: TutorResponse,
        request: TutorRequest,
        *,
        override_inferred_step_id: str | None = None,
        override_overlay_step_id: str | None = None,
        ignore_request_allowlist: bool = False,
    ) -> tuple[bool, str]:
        fallback_help_obj, fallback_reason = self._build_safe_fallback_overlay_help_obj(
            request,
            override_inferred_step_id=override_inferred_step_id,
            override_overlay_step_id=override_overlay_step_id,
            ignore_request_allowlist=ignore_request_allowlist,
        )
        if not isinstance(fallback_help_obj, Mapping):
            return False, fallback_reason

        mapped = map_help_response_to_tutor_response(
            fallback_help_obj,
            request=request,
            status=response.status,
            max_overlay_targets=self.max_overlay_targets,
            ui_map_path=self.ui_map_path,
            lang=self.lang,
        )
        mapped_meta = dict(mapped.metadata)
        if mapped_meta:
            response.metadata["fallback_response_mapping"] = mapped_meta
        if not mapped.actions:
            mapping_errors = mapped_meta.get("mapping_errors")
            if isinstance(mapping_errors, list) and mapping_errors:
                return False, f"fallback_mapping_failed:{'|'.join(str(item) for item in mapping_errors[:3])}"
            return False, "fallback_mapping_failed"

        response.actions = list(mapped.actions)
        if mapped.message and not response.message:
            response.message = mapped.message
        elif mapped.message and mapped.message != response.message:
            response.metadata["fallback_message"] = mapped.message
        original_explanations = list(response.explanations)
        if mapped.explanations and not response.explanations:
            response.explanations = list(mapped.explanations)
        elif mapped.explanations and original_explanations and list(mapped.explanations) != original_explanations:
            response.metadata["fallback_explanations"] = list(mapped.explanations)
        return True, fallback_reason

    def _rewrite_s18_visual_completion_to_s19(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[bool, str]:
        del response, request
        return False, "removed_after_issue_227"

    def _rewrite_s18_structured_fact_completion_to_s19(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[bool, str]:
        del response, request
        return False, "removed_after_issue_227"

    def _new_response_from_cached(
        self,
        cached_response: TutorResponse,
        *,
        in_reply_to: str | None,
    ) -> TutorResponse:
        return TutorResponse(
            status=cached_response.status,
            in_reply_to=in_reply_to,
            message=cached_response.message,
            actions=copy.deepcopy(list(cached_response.actions)),
            explanations=copy.deepcopy(list(cached_response.explanations)),
            metadata=copy.deepcopy(dict(cached_response.metadata)),
        )

    def _executor_is_configured_dry_run(self) -> bool:
        return bool(getattr(self.action_executor, "dry_run", False))

    def _dry_run_report_from_actions(self, actions: Sequence[Mapping[str, Any] | Any]) -> dict[str, Any]:
        previews: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            previews.append(
                {
                    "type": action.get("type"),
                    "intent": action.get("intent"),
                    "target": action.get("target"),
                    "element_id": action.get("element_id"),
                }
            )
        if previews:
            for preview in previews:
                self._emit_event(kind="overlay_dry_run", payload=preview, t_wall=time.time())
        return {
            "executed": [],
            "rejected": [],
            "dropped": [],
            "dry_run": previews,
        }

    def _execute_or_dry_run_actions(self, actions: Sequence[Mapping[str, Any] | Any]) -> dict[str, Any]:
        if self.dry_run_overlay and self._executor_is_configured_dry_run():
            overlay_raw_report = self.action_executor.execute_actions(actions)
            return _normalize_help_report(overlay_raw_report)
        if self.dry_run_overlay:
            return self._dry_run_report_from_actions(actions)
        overlay_raw_report = self.action_executor.execute_actions(actions)
        return _normalize_help_report(overlay_raw_report)

    def _adjudicate_live_help_response(
        self,
        obs: Observation,
        request: TutorRequest,
        *,
        terminal_state_short_circuited: bool,
        inferred_step_id: str | None,
        fallback_conditions: Sequence[str],
    ) -> TutorResponse:
        if terminal_state_short_circuited:
            return self._build_terminal_state_response(request)
        self._stats.model_calls += 1
        try:
            return self.model.explain_error(obs, request)
        except Exception as exc:
            return TutorResponse(
                status="error",
                in_reply_to=request.request_id,
                message=self._fallback_message(inferred_step_id, fallback_conditions),
                actions=[],
                metadata={
                    "provider": "fallback",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    def _validate_and_repair_live_help_response(
        self,
        response: TutorResponse,
        request: TutorRequest,
        *,
        prompt_meta: Mapping[str, Any],
        state_key: str,
        help_cycle_id: str,
        vision_selection: HelpCycleVisionSelection,
        vision_fact_context: Mapping[str, Any],
        vision_fact_active_step_ids: Sequence[str],
        terminal_state_short_circuited: bool,
    ) -> HelpCycleDecisionResult:
        response.metadata = dict(response.metadata)
        response.metadata.setdefault("provider", "fallback" if response.status == "error" else "unknown")
        response.metadata["prompt_hash"] = request.metadata.get("prompt_hash")
        response.metadata["prompt_tokens_est"] = request.metadata.get("prompt_tokens_est")
        response.metadata["prompt_trimmed"] = request.metadata.get("prompt_trimmed")
        response.metadata.setdefault("generation_prompt_hash", response.metadata.get("prompt_hash"))
        response.metadata.setdefault(
            "generation_prompt_tokens_est",
            response.metadata.get("prompt_tokens_est"),
        )
        response.metadata.setdefault(
            "generation_prompt_trimmed",
            response.metadata.get("prompt_trimmed"),
        )
        response.metadata["request_prompt_hash"] = request.metadata.get("prompt_hash")
        response.metadata["request_prompt_tokens_est"] = request.metadata.get("prompt_tokens_est")
        response.metadata["request_prompt_trimmed"] = request.metadata.get("prompt_trimmed")
        response.metadata["state_key"] = state_key
        response.metadata["prompt_build"] = dict(prompt_meta)
        response.metadata["help_cycle_id"] = help_cycle_id
        response.metadata["vision"] = vision_selection.to_dict()
        response.metadata["vision_status"] = vision_selection.status
        response.metadata["vision_frame_ids"] = list(vision_selection.frame_ids)
        response.metadata["vision_fact_status"] = vision_fact_context["status"]
        response.metadata["vision_fact_active_step_ids"] = list(vision_fact_active_step_ids)
        response.metadata["vision_fact_summary"] = dict(vision_fact_context["vision_fact_summary"])
        response.metadata["vision_facts"] = list(vision_fact_context["vision_facts"])
        response.metadata["generation_mode"] = _normalize_generation_mode(response)
        self._capture_model_raw_help_response(response)

        mapped_actions, mapped_meta = self._map_response_actions(response, request)
        response.actions = mapped_actions
        if mapped_meta:
            response.metadata["response_mapping"] = mapped_meta
        self._normalize_observable_text_only_response(response, request)
        self._rewrite_low_confidence_bootstrap_response(response, request)
        self._rewrite_conflicting_step_completion_response(response, request)
        self._rewrite_terminal_state_conflict_response(response, request)
        self._rewrite_procedural_guidance_response(response, request)

        fallback_overlay_used = False
        fallback_overlay_reason = "all_steps_complete" if terminal_state_short_circuited else "not_needed"
        harness_validation_used, harness_validation_reason = self._apply_harness_validation_action_plan(
            response,
            request,
        )
        if harness_validation_used:
            fallback_overlay_used = bool(response.actions)
            fallback_overlay_reason = harness_validation_reason
        harness_guardrail_used, harness_guardrail_reason = self._apply_harness_conflict_guardrail(
            response,
            request,
        )
        if harness_guardrail_used:
            fallback_overlay_used = True
            fallback_overlay_reason = harness_guardrail_reason
        s08_visual_override_used, s08_visual_override_reason = self._apply_s08_visual_recovery_overlay_override(
            response,
            request,
        )
        if s08_visual_override_used:
            fallback_overlay_used = True
            fallback_overlay_reason = s08_visual_override_reason
        action_hint_override_used, action_hint_override_reason = self._apply_action_hint_overlay_override(
            response,
            request,
        )
        if action_hint_override_used:
            fallback_overlay_used = True
            fallback_overlay_reason = action_hint_override_reason
        should_apply_safe_fallback = self._should_use_deterministic_overlay_fallback(
            response,
            request,
            mapped_meta,
        )
        if should_apply_safe_fallback and not action_hint_override_used and not harness_validation_used:
            fallback_overlay_used, fallback_overlay_reason = self._apply_safe_fallback_overlay(
                response,
                request,
            )
        manual_throttle_rewritten, manual_throttle_reason = self._rewrite_manual_throttle_guidance_response(
            response,
            request,
        )
        if manual_throttle_rewritten:
            fallback_overlay_used = False
            fallback_overlay_reason = manual_throttle_reason

        if mapped_meta:
            mapping_failure_codes = classify_mapping_failure(mapped_meta)
            if mapping_failure_codes:
                response.metadata["response_mapping_failure_codes"] = list(mapping_failure_codes)
                response.metadata["response_mapping_failure_code"] = mapping_failure_codes[0]
                response.metadata.setdefault("response_mapping_failure_stage", "response_mapping")
                if not fallback_overlay_used:
                    response.metadata = merge_failure_metadata(
                        response.metadata,
                        *mapping_failure_codes,
                        stage="response_mapping",
                    )

        response.metadata["fallback_overlay_used"] = fallback_overlay_used
        response.metadata["fallback_overlay_reason"] = fallback_overlay_reason
        self._annotate_response_audit_metadata(response)
        _normalize_cached_response_metadata(response.metadata)
        _build_harness_trace_metadata(
            request=request,
            response=response,
            vision_selection=vision_selection,
            vision_fact_context=vision_fact_context,
        )
        response_audit_fields = _build_help_cycle_audit_fields(
            request=request,
            vision_selection=vision_selection,
            vision_fact_context=vision_fact_context,
            response_metadata=response.metadata,
        )
        _apply_help_cycle_audit_fields(response.metadata, response_audit_fields)
        if isinstance(response_audit_fields.get("vision_fallback_reason"), str):
            response.metadata = merge_failure_metadata(
                response.metadata,
                response_audit_fields["vision_fallback_reason"],
                stage="vision",
            )
        trace_metadata = {
            "help_cycle_id": help_cycle_id,
            "generation_mode": response.metadata["generation_mode"],
            **normalize_help_cycle_audit_fields(response.metadata),
        }
        response.actions = _attach_help_cycle_trace_to_actions(
            response.actions,
            trace_metadata=trace_metadata,
        )
        return HelpCycleDecisionResult(
            response=response,
            fallback_overlay_used=fallback_overlay_used,
            fallback_overlay_reason=fallback_overlay_reason,
        )

    def _execute_final_action_plan_via_orchestrator(
        self,
        *,
        obs: Observation,
        request: TutorRequest,
        response: TutorResponse,
        prompt_meta: Mapping[str, Any],
        state_key: str,
        vision_fact_context: Mapping[str, Any],
        vision_fact_active_step_ids: Sequence[str],
        help_cycle_id: str,
        fallback_overlay_used: bool,
        fallback_overlay_reason: str,
    ) -> tuple[TutorResponse, dict[str, Any]]:
        result = HelpCycleOrchestrator(
            llm=_StaticHelpAdjudicator(response),
            validator=_StaticDecisionValidator(
                fallback_overlay_used=fallback_overlay_used,
                fallback_overlay_reason=fallback_overlay_reason,
            ),
            actions=_CallableActionPlanner(self._execute_or_dry_run_actions),
        ).run_prepared(
            PreparedHelpCycle(
                observation=obs,
                request=request,
                prompt_metadata=prompt_meta,
                state_key=state_key,
                vision_context=vision_fact_context,
                active_step_ids=vision_fact_active_step_ids,
                help_cycle_id=help_cycle_id,
            )
        )
        return result.response, _normalize_help_report(result.action_report)

    def _send_tutor_text(self, response: TutorResponse) -> None:
        if self.tutor_text_sender is None:
            response.metadata["dcs_tutor_text"] = {
                "status": "skipped",
                "reason": "sender_unavailable",
            }
            return
        sanitized_message = sanitize_public_model_text(response.message, lang=self.lang)
        if not isinstance(sanitized_message, str) or not sanitized_message.strip():
            response.metadata["dcs_tutor_text"] = {
                "status": "skipped",
                "reason": "empty_message",
            }
            return
        try:
            result = self.tutor_text_sender.send_text(
                sanitized_message,
                display_time_s=self.tutor_text_display_time_s,
                clear_view=self.tutor_text_clear_view,
                expect_ack=True,
            )
            response.metadata["dcs_tutor_text"] = dict(result) if isinstance(result, Mapping) else {
                "status": "failed",
                "failure_class": "invalid_sender_result",
                "reason": "Tutor text sender returned a non-mapping result",
            }
        except Exception as exc:
            response.metadata["dcs_tutor_text"] = {
                "status": "failed",
                "failure_class": "sender_exception",
                "reason": f"{type(exc).__name__}: {exc}",
                "text": sanitized_message,
                "display_time_s": self.tutor_text_display_time_s,
                "clear_view": self.tutor_text_clear_view,
            }

    def run_help_cycle(self, *, trigger_t_wall: float | None = None) -> tuple[TutorResponse | None, dict[str, Any] | None]:
        obs = self._latest_enriched_obs
        if obs is None:
            return None, None

        resolved_trigger_t_wall = _coerce_finite_float(trigger_t_wall)
        if resolved_trigger_t_wall is None:
            payload = obs.payload if isinstance(obs.payload, Mapping) else {}
            resolved_trigger_t_wall = _coerce_finite_float(payload.get("t_wall"))
        if resolved_trigger_t_wall is None:
            resolved_trigger_t_wall = time.time()

        vision_selection = self._build_vision_selection(
            observation=obs,
            trigger_t_wall=resolved_trigger_t_wall,
        )
        help_cycle_id = str(uuid4())
        preliminary_inference = self._infer_preliminary_step_for_vision_facts(obs)
        vision_fact_active_step_ids = self._active_step_ids_for_vision_facts(
            preliminary_inference,
            now_wall_ms=vision_selection.trigger_wall_ms,
        )
        vision_fact_context = self._extract_vision_fact_context(
            vision_selection=vision_selection,
            help_cycle_id=help_cycle_id,
            active_step_ids=vision_fact_active_step_ids,
        )
        request, prompt_meta, state_key = self._build_request(
            obs,
            vision_selection=vision_selection,
            vision_fact_context=vision_fact_context,
            request_id_override=help_cycle_id,
        )
        help_cycle_id = request.request_id
        request.metadata = dict(request.metadata)
        request.metadata["help_cycle_id"] = help_cycle_id
        request.metadata["vision_status"] = vision_selection.status
        request.metadata["vision_frame_ids"] = list(vision_selection.frame_ids)
        request.metadata["vision_fact_status"] = vision_fact_context["status"]
        request.metadata["vision_fact_active_step_ids"] = list(vision_fact_active_step_ids)
        request.metadata["vision_fact_summary"] = dict(vision_fact_context["vision_fact_summary"])
        request_audit_fields = _build_help_cycle_audit_fields(
            request=request,
            vision_selection=vision_selection,
            vision_fact_context=vision_fact_context,
        )
        _apply_help_cycle_audit_fields(request.metadata, request_audit_fields)
        now_wall = time.time()
        self._emit_event(
            kind="tutor_request",
            payload=_sanitize_request_payload_for_event(request),
            related_id=request.request_id,
            t_wall=now_wall,
            metadata={
                "help_cycle_id": help_cycle_id,
                "vision_status": vision_selection.status,
                "vision_fact_status": vision_fact_context["status"],
                **normalize_help_cycle_audit_fields(request_audit_fields),
            },
            vision_refs=vision_selection.frame_ids,
        )
        self._stats.help_cycles += 1
        if request_audit_fields.get("vision_fact_extractor_used") is True:
            self._stats.vision_cycles += 1
        if request_audit_fields["vision_fallback_reason"] == VISION_SYNC_MISS:
            self._stats.vision_sync_miss_count += 1

        use_cache = False
        cached = self._help_cache
        if cached is not None and cached.state_key == state_key:
            if (now_wall - cached.t_wall) <= self.cooldown_s:
                use_cache = True

        if use_cache and cached is not None:
            self._stats.cache_hits += 1
            response = self._new_response_from_cached(cached.response, in_reply_to=request.request_id)
            response.metadata = dict(response.metadata)
            response.metadata["cached_response_reused"] = True
            response.metadata["cache_age_s"] = round(now_wall - cached.t_wall, 3)
            response.metadata.setdefault("generation_prompt_hash", response.metadata.get("prompt_hash"))
            response.metadata.setdefault(
                "generation_prompt_tokens_est",
                response.metadata.get("prompt_tokens_est"),
            )
            response.metadata.setdefault(
                "generation_prompt_trimmed",
                response.metadata.get("prompt_trimmed"),
            )
            response.metadata["request_prompt_hash"] = request.metadata.get("prompt_hash")
            response.metadata["request_prompt_tokens_est"] = request.metadata.get("prompt_tokens_est")
            response.metadata["request_prompt_trimmed"] = request.metadata.get("prompt_trimmed")
            response.metadata["state_key"] = state_key
            response.metadata["prompt_build"] = dict(prompt_meta)
            _normalize_cached_response_metadata(response.metadata)
            response.metadata["help_cycle_id"] = help_cycle_id
            response.metadata["vision"] = vision_selection.to_dict()
            response.metadata["vision_status"] = vision_selection.status
            response.metadata["vision_frame_ids"] = list(vision_selection.frame_ids)
            response.metadata["vision_fact_status"] = vision_fact_context["status"]
            response.metadata["vision_fact_active_step_ids"] = list(vision_fact_active_step_ids)
            response.metadata["vision_fact_summary"] = dict(vision_fact_context["vision_fact_summary"])
            response.metadata["vision_facts"] = list(vision_fact_context["vision_facts"])
            response.metadata["generation_mode"] = _normalize_generation_mode(response)
            response_audit_fields = _build_help_cycle_audit_fields(
                request=request,
                vision_selection=vision_selection,
                vision_fact_context=vision_fact_context,
                response_metadata=response.metadata,
            )
            _apply_help_cycle_audit_fields(response.metadata, response_audit_fields)
            if isinstance(response_audit_fields.get("vision_fallback_reason"), str):
                response.metadata = merge_failure_metadata(
                    response.metadata,
                    response_audit_fields["vision_fallback_reason"],
                    stage="vision",
                )
            _build_harness_trace_metadata(
                request=request,
                response=response,
                vision_selection=vision_selection,
                vision_fact_context=vision_fact_context,
            )
            trace_metadata = {
                "help_cycle_id": help_cycle_id,
                "generation_mode": response.metadata["generation_mode"],
                **normalize_help_cycle_audit_fields(response.metadata),
            }
            response.actions = _attach_help_cycle_trace_to_actions(
                response.actions,
                trace_metadata=trace_metadata,
            )
            cached_fallback_reason = response.metadata.get("fallback_overlay_reason")
            response, overlay_report = self._execute_final_action_plan_via_orchestrator(
                obs=obs,
                request=request,
                response=response,
                prompt_meta=prompt_meta,
                state_key=state_key,
                vision_fact_context=vision_fact_context,
                vision_fact_active_step_ids=vision_fact_active_step_ids,
                help_cycle_id=help_cycle_id,
                fallback_overlay_used=bool(response.metadata.get("fallback_overlay_used")),
                fallback_overlay_reason=(
                    cached_fallback_reason if isinstance(cached_fallback_reason, str) else "not_needed"
                ),
            )
        else:
            hint = request.context.get("deterministic_step_hint", {})
            inferred_step_id = hint.get("inferred_step_id") if isinstance(hint, Mapping) else None
            missing_conditions = hint.get("missing_conditions", []) if isinstance(hint, Mapping) else []
            if not isinstance(missing_conditions, (list, tuple)):
                missing_conditions = []
            gate_blockers = hint.get("gate_blockers", []) if isinstance(hint, Mapping) else []
            if not isinstance(gate_blockers, (list, tuple)):
                gate_blockers = []
            gate_blocker_conditions: list[str] = []
            for item in gate_blockers:
                if isinstance(item, Mapping):
                    reason = item.get("reason")
                    if isinstance(reason, str) and reason:
                        gate_blocker_conditions.append(reason)
                        continue
                    reason_code = item.get("reason_code")
                    if isinstance(reason_code, str) and reason_code:
                        gate_blocker_conditions.append(reason_code)
                        continue
                    ref = item.get("ref")
                    if isinstance(ref, str) and ref:
                        gate_blocker_conditions.append(ref)
                        continue
                elif isinstance(item, str) and item:
                    # Backward compatibility for legacy string blockers.
                    gate_blocker_conditions.append(item)
            fallback_conditions = _dedupe_strings(
                [
                    item
                    for item in [*missing_conditions, *gate_blocker_conditions]
                    if isinstance(item, str) and item
                ]
            )
            terminal_state_short_circuited = _is_terminal_step_hint_complete(
                hint if isinstance(hint, Mapping) else None
            )
            orchestrated = HelpCycleOrchestrator(
                llm=_CallableHelpAdjudicator(
                    lambda observation, tutor_request: self._adjudicate_live_help_response(
                        observation,
                        tutor_request,
                        terminal_state_short_circuited=terminal_state_short_circuited,
                        inferred_step_id=inferred_step_id if isinstance(inferred_step_id, str) else None,
                        fallback_conditions=fallback_conditions,
                    )
                ),
                validator=_CallableDecisionValidator(
                    lambda tutor_response, tutor_request: self._validate_and_repair_live_help_response(
                        tutor_response,
                        tutor_request,
                        prompt_meta=prompt_meta,
                        state_key=state_key,
                        help_cycle_id=help_cycle_id,
                        vision_selection=vision_selection,
                        vision_fact_context=vision_fact_context,
                        vision_fact_active_step_ids=vision_fact_active_step_ids,
                        terminal_state_short_circuited=terminal_state_short_circuited,
                    )
                ),
                actions=_CallableActionPlanner(self._execute_or_dry_run_actions),
            ).run_prepared(
                PreparedHelpCycle(
                    observation=obs,
                    request=request,
                    prompt_metadata=prompt_meta,
                    state_key=state_key,
                    vision_context=vision_fact_context,
                    active_step_ids=vision_fact_active_step_ids,
                    help_cycle_id=help_cycle_id,
                )
            )
            response = orchestrated.response
            overlay_report = _normalize_help_report(orchestrated.action_report)
            provider = response.metadata.get("provider")
            cacheable = response.status == "ok" and provider != "fallback"
            if cacheable:
                self._help_cache = HelpCacheEntry(
                    state_key=state_key,
                    t_wall=now_wall,
                    response=copy.deepcopy(response),
                )
            else:
                self._help_cache = None

        hint = request.context.get("deterministic_step_hint")
        if isinstance(hint, Mapping):
            observability_status = hint.get("observability_status")
            if isinstance(observability_status, str) and observability_status:
                response.metadata["observability_status"] = observability_status
            requires_visual_confirmation = hint.get("requires_visual_confirmation")
            if isinstance(requires_visual_confirmation, bool):
                response.metadata["requires_visual_confirmation"] = requires_visual_confirmation
        self._annotate_response_audit_metadata(response)
        response.metadata["scenario_profile"] = self.scenario_profile
        if response.metadata.get("vision_fallback_reason") == VISION_TEXT_FALLBACK:
            self._stats.vision_text_fallback_count += 1

        if self.dry_run_overlay and overlay_report.get("dry_run"):
            print(
                json.dumps(
                    {"dry_run_actions": overlay_report["dry_run"]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

        response_mapping = response.metadata.get("response_mapping")
        rejected_overlay = overlay_rejection_payload(
            response_metadata=response.metadata,
            response_mapping=response_mapping if isinstance(response_mapping, Mapping) else None,
        )
        if rejected_overlay is not None:
            rejected_overlay["help_cycle_id"] = help_cycle_id
            self._emit_event(
                kind="overlay_rejected",
                payload=rejected_overlay,
                related_id=request.request_id,
                t_wall=time.time(),
                metadata={
                    "help_cycle_id": help_cycle_id,
                    "generation_mode": response.metadata.get("generation_mode"),
                    **normalize_help_cycle_audit_fields(response.metadata),
                },
                vision_refs=vision_selection.frame_ids,
            )

        self._send_tutor_text(response)
        if bool(getattr(self.model, "print_model_io", False)):
            final_public_response = response.metadata.get("final_public_response")
            if isinstance(final_public_response, Mapping):
                header = f"[MODEL_IO][FINAL_PUBLIC_RESPONSE][request_id={request.request_id}]"
                print(header)
                print(json.dumps(final_public_response, ensure_ascii=False, sort_keys=True))
                print(f"{header}[END]")
        self._emit_event(
            kind="tutor_response",
            payload=_sanitize_response_payload_for_event(response, lang=self.lang),
            related_id=request.request_id,
            t_wall=time.time(),
            metadata={
                "help_cycle_id": help_cycle_id,
                "generation_mode": response.metadata.get("generation_mode"),
                **normalize_help_cycle_audit_fields(response.metadata),
            },
            vision_refs=vision_selection.frame_ids,
        )
        return response, overlay_report

    def run(
        self,
        *,
        max_frames: int = 0,
        duration_s: float = 0.0,
        auto_help_on_first_frame: bool = False,
        auto_help_every_n_frames: int = 0,
        help_trigger: HelpTriggerLike | None = None,
        help_capture_notifier: Any | None = None,
        idle_sleep_s: float = 0.01,
    ) -> dict[str, int]:
        if auto_help_every_n_frames < 0:
            raise ValueError("auto_help_every_n_frames must be >= 0")
        if max_frames < 0:
            raise ValueError("max_frames must be >= 0")
        if duration_s < 0:
            raise ValueError("duration_s must be >= 0")

        start = time.time()
        first_help_done = False

        while True:
            self._poll_vision_sidecar()
            if duration_s > 0 and (time.time() - start) >= duration_s:
                break

            if help_trigger is not None:
                while help_trigger.poll():
                    self._pending_help_trigger_t_wall = time.time() if self.vision_mode == "live" else float("nan")

            obs = self.source.get_observation()
            if obs is not None:
                self._ingest_observation(obs)
                self._stats.frames += 1
                obs_payload = obs.payload if isinstance(obs.payload, Mapping) else {}
                obs_t_wall = _coerce_float(obs_payload.get("t_wall"))
                help_action_t_wall = time.time() if self.vision_mode == "live" else obs_t_wall
                if auto_help_on_first_frame and not first_help_done:
                    if help_capture_notifier is not None and hasattr(help_capture_notifier, "notify_help"):
                        help_capture_notifier.notify_help()
                    self.run_help_cycle(trigger_t_wall=help_action_t_wall)
                    first_help_done = True
                if auto_help_every_n_frames > 0:
                    if self._stats.frames % auto_help_every_n_frames == 0:
                        if help_capture_notifier is not None and hasattr(help_capture_notifier, "notify_help"):
                            help_capture_notifier.notify_help()
                        self.run_help_cycle(trigger_t_wall=help_action_t_wall)
            else:
                exhausted = bool(getattr(self.source, "is_exhausted", False))
                if exhausted:
                    break

            if self._pending_help_trigger_t_wall is not None and self._latest_enriched_obs is not None:
                help_trigger_t_wall = self._pending_help_trigger_t_wall
                self._pending_help_trigger_t_wall = None
                if help_capture_notifier is not None and hasattr(help_capture_notifier, "notify_help"):
                    help_capture_notifier.notify_help()
                self.run_help_cycle(trigger_t_wall=help_trigger_t_wall)

            if max_frames > 0 and self._stats.frames >= max_frames:
                break

            if obs is None:
                time.sleep(max(0.0, idle_sleep_s))

        return self._stats.to_dict()


def _new_default_log_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"live_dcs_{ts}.jsonl"


def _unique_log_path_candidates(requested_path: str | Path) -> Iterable[Path]:
    path = Path(requested_path).expanduser()
    yield path

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = path.stem or "live_dcs"
    suffix = path.suffix
    yield path.with_name(f"{stem}_{timestamp}{suffix}")
    for index in range(2, 1000):
        yield path.with_name(f"{stem}_{timestamp}_{index}{suffix}")


def _open_unique_event_store(requested_path: str | Path) -> tuple[Path, JsonlEventStore]:
    for candidate in _unique_log_path_candidates(requested_path):
        try:
            return candidate, JsonlEventStore(candidate, mode="x")
        except FileExistsError:
            continue
    raise FileExistsError(f"could not resolve unique log path for {requested_path}")


def _build_observation_source_from_args(args: argparse.Namespace) -> ObservationSource:
    if args.replay_bios:
        return ReplayBiosReceiver(args.replay_bios, speed=args.speed)

    bios_source = str(getattr(args, "bios_source", "decoded")).strip().lower()
    if bios_source == "raw":
        aircraft = str(getattr(args, "raw_bios_aircraft", "") or "").strip()
        if not aircraft:
            raise ValueError("--raw-bios-aircraft is required when --bios-source raw")
        return DcsBiosRawReceiver(
            host=args.raw_bios_host,
            port=args.raw_bios_port,
            timeout=args.timeout,
            merge_full_state=bool(args.merge_full_state),
            control_reference_dir=args.raw_bios_control_dir,
            aircraft=aircraft,
        )

    return DcsBiosReceiver(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        merge_full_state=bool(args.merge_full_state),
    )


def _build_vision_port_from_args(
    args: argparse.Namespace,
    *,
    mode: str,
) -> tuple[Any | None, str | None, int | None, int | None]:
    saved_games_dir = getattr(args, "vision_saved_games_dir", None)
    if not isinstance(saved_games_dir, str) or not saved_games_dir.strip():
        return None, None, None, None
    saved_games_dir = saved_games_dir.strip()

    raw_session_id = getattr(args, "vision_session_id", None) or getattr(args, "session_id", None)
    if not isinstance(raw_session_id, str) or not raw_session_id.strip():
        raise ValueError("--vision-session-id or --session-id is required when vision sidecar is enabled")
    session_id = _normalize_path_segment(raw_session_id, flag_name="--vision-session-id")

    channel = _normalize_path_segment(
        getattr(args, "vision_channel", DEFAULT_FRAME_CHANNEL),
        flag_name="--vision-channel",
    )
    layout_id = getattr(args, "vision_layout_id", DEFAULT_LAYOUT_ID)
    sync_window_ms_raw = getattr(args, "vision_sync_window_ms", 0)
    trigger_wait_ms_raw = getattr(args, "vision_trigger_wait_ms", 0)
    sync_window_ms = int(sync_window_ms_raw) if isinstance(sync_window_ms_raw, int) and sync_window_ms_raw > 0 else None
    trigger_wait_ms = (
        int(trigger_wait_ms_raw)
        if isinstance(trigger_wait_ms_raw, int) and trigger_wait_ms_raw > 0
        else None
    )

    return (
        FrameDirectoryVisionPort(
            saved_games_dir=saved_games_dir,
            channel=channel,
            layout_id=layout_id,
        ),
        session_id,
        sync_window_ms,
        trigger_wait_ms,
    )


def _build_model_from_args(args: argparse.Namespace) -> Any:
    provider = args.model_provider
    lang = args.lang
    log_raw_llm_text = bool(getattr(args, "log_raw_llm_text", False))
    print_model_io = bool(getattr(args, "print_model_io", False))
    if provider == "stub":
        return ModelStub(mode=args.stub_mode)

    timeout_s = float(args.model_timeout_s)
    model_max_tokens_value = int(args.model_max_tokens)
    if model_max_tokens_value < 0:
        raise ValueError("--model-max-tokens must be >= 0")
    model_max_tokens = model_max_tokens_value if model_max_tokens_value > 0 else None
    model_enable_multimodal = bool(getattr(args, "model_enable_multimodal", False))
    allowed_local_image_roots: list[Path] = []
    vision_saved_games_dir = getattr(args, "vision_saved_games_dir", None)
    if isinstance(vision_saved_games_dir, str) and vision_saved_games_dir.strip():
        allowed_local_image_roots.append(build_frames_root(vision_saved_games_dir))
    if provider == "openai_compat":
        if not args.model_base_url:
            raise ValueError("--model-base-url is required for openai_compat")
        validate_model_base_url_security(args.model_base_url, provider=provider)
        return OpenAICompatModel(
            model_name=args.model_name,
            base_url=args.model_base_url,
            timeout_s=timeout_s,
            max_tokens=model_max_tokens,
            lang=lang,
            log_raw_llm_text=log_raw_llm_text,
            print_model_io=print_model_io,
            api_key=args.model_api_key,
            enable_multimodal=model_enable_multimodal,
            enable_help_multimodal=False,
            allowed_local_image_roots=allowed_local_image_roots,
            telemetry_map_path=args.telemetry_map,
        )
    if provider == "ollama":
        base_url = args.model_base_url or "http://127.0.0.1:11434"
        validate_model_base_url_security(base_url, provider=provider)
        return OllamaModel(
            model_name=args.model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            lang=lang,
            log_raw_llm_text=log_raw_llm_text,
            print_model_io=print_model_io,
            telemetry_map_path=args.telemetry_map,
        )
    raise ValueError(f"Unsupported model provider: {provider}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live DCS tutor loop (bios -> help -> overlay).")

    parser.add_argument("--host", default="0.0.0.0", help="DCS-BIOS UDP bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7790, help="DCS-BIOS UDP bind port (default 7790)")
    parser.add_argument("--timeout", type=float, default=0.2, help="Receiver socket timeout seconds")
    parser.add_argument(
        "--bios-source",
        choices=["decoded", "raw"],
        default="decoded",
        help="Live BIOS source type: decoded JSON UDP stream or raw DCS-BIOS export stream.",
    )
    parser.add_argument(
        "--raw-bios-host",
        default="239.255.50.10",
        help="DCS-BIOS raw export host when --bios-source raw (default 239.255.50.10)",
    )
    parser.add_argument(
        "--raw-bios-port",
        type=int,
        default=5010,
        help="DCS-BIOS raw export port when --bios-source raw (default 5010)",
    )
    parser.add_argument(
        "--raw-bios-aircraft",
        default="",
        help="Aircraft name for DCS-BIOS raw decoding, e.g. FA-18C_hornet.",
    )
    parser.add_argument(
        "--raw-bios-control-dir",
        default="DCS/Scripts/DCS-BIOS/doc/json",
        help="Control reference directory for DCS-BIOS raw decoding.",
    )
    merge_group = parser.add_mutually_exclusive_group()
    merge_group.add_argument(
        "--merge-full-state",
        dest="merge_full_state",
        action="store_true",
        help="Merge BIOS deltas to full state (default: enabled)",
    )
    merge_group.add_argument(
        "--no-merge-full-state",
        dest="merge_full_state",
        action="store_false",
        help="Disable full-state merge; use delta-only payload as bios state",
    )
    parser.set_defaults(merge_full_state=True)

    parser.add_argument("--pack", default=str(_default_pack_path()), help="pack.yaml path")
    parser.add_argument("--ui-map", default=str(_default_ui_map_path()), help="ui_map.yaml path")
    parser.add_argument("--telemetry-map", default=str(_default_telemetry_map_path()), help="telemetry_map.yaml path")
    parser.add_argument("--bios-to-ui", default=str(_default_bios_to_ui_path()), help="bios_to_ui.yaml path")
    parser.add_argument(
        "--knowledge-index",
        default=str(_default_knowledge_index_path()),
        help="Grounding index.json path (BM25)",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=5,
        help="Grounding snippet top-k for prompt injection (default 5)",
    )
    parser.add_argument(
        "--max-overlay-targets",
        type=int,
        default=4,
        help="Maximum number of overlay targets allowed per help cycle (default 4)",
    )
    cold_start_default = parse_env_bool(ENV_COLD_START_PRODUCTION, default=False)
    cold_start_group = parser.add_mutually_exclusive_group()
    cold_start_group.add_argument(
        "--cold-start-production",
        dest="cold_start_production",
        action="store_true",
        help="Enable cold-start production mode (requires valid knowledge source policy).",
    )
    cold_start_group.add_argument(
        "--no-cold-start-production",
        dest="cold_start_production",
        action="store_false",
        help="Disable cold-start production mode even if env default is enabled.",
    )
    parser.set_defaults(cold_start_production=cold_start_default)
    parser.add_argument(
        "--knowledge-source-policy",
        default=None,
        help=(
            "knowledge_source_policy.yaml path. In cold-start production mode, omitted path "
            "falls back to repository-checkout knowledge_source_policy.yaml when available. "
            "Providing this flag enables policy filtering in any mode."
        ),
    )

    parser.add_argument("--output", help="Event log JSONL output path")
    parser.add_argument("--session-id", default=None, help="Optional event session id")
    parser.add_argument(
        "--vision-saved-games-dir",
        default=None,
        help="Saved Games/<variant> root for frames/<session>/<channel>/frames.jsonl sidecar consumption.",
    )
    parser.add_argument(
        "--vision-session-id",
        default=None,
        help="Frame sidecar session id. Defaults to --session-id when omitted.",
    )
    parser.add_argument("--vision-channel", default=DEFAULT_FRAME_CHANNEL, help="Vision frame channel name")
    parser.add_argument("--vision-layout-id", default=DEFAULT_LAYOUT_ID, help="Expected vision layout id")
    parser.add_argument(
        "--vision-sync-window-ms",
        type=parse_non_negative_int_arg,
        default=0,
        help="Frame selection sync window in milliseconds (0 uses live/replay default).",
    )
    parser.add_argument(
        "--vision-trigger-wait-ms",
        type=parse_non_negative_int_arg,
        default=0,
        help="Extra wait budget for live help-trigger frame arrival (0 uses mode default).",
    )
    parser.add_argument(
        "--vision-capture-trigger-host",
        default=DEFAULT_VISION_CAPTURE_TRIGGER_HOST,
        help="UDP host for notifying the local vision sidecar to capture an immediate help-trigger frame.",
    )
    parser.add_argument(
        "--vision-capture-trigger-port",
        type=int,
        default=DEFAULT_VISION_CAPTURE_TRIGGER_PORT,
        help="UDP port for help-trigger vision capture notifications (0 disables notifier).",
    )

    parser.add_argument("--cooldown-s", type=float, default=4.0, help="Cooldown window for same-state help reuse")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 means unlimited)")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 means unlimited)")

    parser.add_argument("--auto-help-once", action="store_true", help="Auto trigger one help cycle after first frame")
    parser.add_argument("--auto-help-every-n-frames", type=int, default=0, help="Auto help interval by frame count")
    parser.add_argument("--stdin-help", action="store_true", help="Read stdin trigger: Enter/help/h/?")
    parser.add_argument("--help-udp-host", default="127.0.0.1", help="UDP host for help trigger listener")
    parser.add_argument(
        "--help-udp-port",
        type=int,
        default=0,
        help="UDP port for help trigger listener (0 disables UDP help trigger)",
    )
    parser.add_argument(
        "--global-help-hotkey",
        default="",
        help="Windows global help trigger key: ESC, F1-F24, X1/MOUSE4, X2/MOUSE5 (empty disables).",
    )
    parser.add_argument(
        "--global-help-modifiers",
        default="",
        help="Optional Windows global help modifiers joined by +, e.g. Ctrl+Shift.",
    )
    parser.add_argument(
        "--global-help-cooldown-ms",
        type=parse_non_negative_int_arg,
        default=DEFAULT_GLOBAL_HELP_COOLDOWN_MS,
        help="Debounce window for the Windows global help trigger in milliseconds.",
    )
    parser.add_argument(
        "--help-udp-timeout",
        type=float,
        default=0.2,
        help="UDP help trigger socket timeout seconds",
    )
    parser.add_argument(
        "--dry-run-overlay",
        action="store_true",
        help="Do not send UDP overlay commands; print planned actions only",
    )

    parser.add_argument("--replay-bios", help="Replay BIOS JSONL instead of listening UDP")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier for --replay-bios (1.0 realtime, 0 max speed)",
    )

    parser.add_argument("--model-provider", choices=["stub", "openai_compat", "ollama"], default="stub")
    parser.add_argument("--model-name", default=os.getenv("SIMTUTOR_MODEL_NAME", "Qwen3-8B-Instruct"))
    parser.add_argument(
        "--vision-model-name",
        default=os.getenv("SIMTUTOR_VISION_MODEL_NAME", "simtutor-vision"),
        help="Model name for vision fact extraction (LoRA-enabled). Defaults to 'simtutor-vision'.",
    )
    parser.add_argument("--model-base-url", default=os.getenv("SIMTUTOR_MODEL_BASE_URL", ""))
    parser.add_argument("--model-timeout-s", type=float, default=float(os.getenv("SIMTUTOR_MODEL_TIMEOUT_S", "20")))
    parser.add_argument(
        "--model-max-tokens",
        type=parse_non_negative_int_arg,
        default=parse_env_int("SIMTUTOR_MODEL_MAX_TOKENS", default=0, minimum=0),
        help="Max completion tokens for model providers that support it (0 uses provider default).",
    )
    parser.add_argument("--model-api-key", default=os.getenv("SIMTUTOR_MODEL_API_KEY"))
    model_multimodal_default = parse_env_bool("SIMTUTOR_MODEL_ENABLE_MULTIMODAL", default=False)
    model_multimodal_group = parser.add_mutually_exclusive_group()
    model_multimodal_group.add_argument(
        "--model-enable-multimodal",
        dest="model_enable_multimodal",
        action="store_true",
        help="Enable synchronized vision-frame support for structured VLM facts; main help remains text-only.",
    )
    model_multimodal_group.add_argument(
        "--no-model-enable-multimodal",
        dest="model_enable_multimodal",
        action="store_false",
        help="Force text-only requests even when synchronized vision frames are available.",
    )
    parser.set_defaults(model_enable_multimodal=model_multimodal_default)
    parser.add_argument("--stub-mode", default="A", help="ModelStub mode (A/B/C)")
    parser.add_argument("--lang", choices=["zh", "en"], default=os.getenv("SIMTUTOR_LANG", "zh"))
    parser.add_argument(
        "--scenario-profile",
        choices=sorted(SUPPORTED_SCENARIO_PROFILES),
        default=DEFAULT_SCENARIO_PROFILE,
        help="Scenario profile to parameterize pack gate branches (default: airfield).",
    )
    log_raw_default = parse_env_bool("SIMTUTOR_LOG_RAW_LLM_TEXT", default=False)
    log_raw_group = parser.add_mutually_exclusive_group()
    log_raw_group.add_argument(
        "--log-raw-llm-text",
        dest="log_raw_llm_text",
        action="store_true",
        help="Log raw model text into tutor_response.metadata.raw_llm_text(_attempts)",
    )
    log_raw_group.add_argument(
        "--no-log-raw-llm-text",
        dest="log_raw_llm_text",
        action="store_false",
        help="Disable raw model text logging even if SIMTUTOR_LOG_RAW_LLM_TEXT=1",
    )
    parser.set_defaults(log_raw_llm_text=log_raw_default)
    print_model_io_default = parse_env_bool("SIMTUTOR_PRINT_MODEL_IO", default=False)
    print_model_io_group = parser.add_mutually_exclusive_group()
    print_model_io_group.add_argument(
        "--print-model-io",
        dest="print_model_io",
        action="store_true",
        help="Print the full prompt text and decoded raw model reply to the terminal for debugging.",
    )
    print_model_io_group.add_argument(
        "--no-print-model-io",
        dest="print_model_io",
        action="store_false",
        help="Disable terminal model I/O debug printing even if SIMTUTOR_PRINT_MODEL_IO=1.",
    )
    parser.set_defaults(print_model_io=print_model_io_default)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    requested_output = Path(args.output) if args.output else _new_default_log_path()
    output, store = _open_unique_event_store(requested_output)
    print(f"[LIVE_DCS] resolved output path: {output}")

    with store:
        source = _build_observation_source_from_args(args)

        model = _build_model_from_args(args)
        vision_port, vision_session_id, vision_sync_window_ms, vision_trigger_wait_ms = _build_vision_port_from_args(
            args,
            mode="live",
        )
        vision_capture_notifier = (
            UdpVisionCaptureNotifier(
                session_id=vision_session_id,
                host=args.vision_capture_trigger_host,
                port=args.vision_capture_trigger_port,
            )
            if vision_session_id and int(args.vision_capture_trigger_port) > 0
            else None
        )
        store.append(
            Event(
                kind="system",
                payload={
                    "event": "live_dcs_runtime_log",
                    "requested_output_path": str(requested_output),
                    "resolved_output_path": str(output),
                },
                metadata={
                    "requested_output_path": str(requested_output),
                    "resolved_output_path": str(output),
                },
            )
        )
        _emit_multi_target_overlay_config_warning(
            max_overlay_targets=max(0, int(args.max_overlay_targets)),
            config_path=simtutor_config_path_from_saved_games_dir(args.vision_saved_games_dir),
            event_sink=store.append,
        )
        executor = OverlayActionExecutor(
            ui_map_path=args.ui_map,
            pack_path=args.pack,
            max_targets=max(0, int(args.max_overlay_targets)),
            dry_run=bool(args.dry_run_overlay),
            session_id=args.session_id,
            event_sink=store.append,
        )
        tutor_text_sender = DcsTutorTextSender(
            host="127.0.0.1",
            port=7783,
            timeout=0.5,
            enabled=True,
        )
        loop = LiveDcsTutorLoop(
            source=source,
            model=model,
            action_executor=executor,
            pack_path=args.pack,
            ui_map_path=args.ui_map,
            telemetry_map_path=args.telemetry_map,
            bios_to_ui_path=args.bios_to_ui,
            knowledge_index_path=args.knowledge_index,
            rag_top_k=args.rag_top_k,
            cold_start_production=bool(args.cold_start_production),
            knowledge_source_policy_path=args.knowledge_source_policy,
            cooldown_s=args.cooldown_s,
            session_id=args.session_id,
            lang=args.lang,
            scenario_profile=args.scenario_profile,
            event_sink=store.append,
            dry_run_overlay=bool(args.dry_run_overlay),
            vision_port=vision_port,
            vision_session_id=vision_session_id,
            vision_mode="live",
            vision_sync_window_ms=vision_sync_window_ms,
            vision_trigger_wait_ms=vision_trigger_wait_ms,
            vision_model_name=args.vision_model_name,
            max_overlay_targets=max(0, int(args.max_overlay_targets)),
            tutor_text_sender=tutor_text_sender,
        )

        stdin_trigger = StdinHelpTrigger() if args.stdin_help else None
        udp_trigger = (
            UdpHelpTrigger(
                host=args.help_udp_host,
                port=args.help_udp_port,
                timeout=args.help_udp_timeout,
            )
            if args.help_udp_port > 0
            else None
        )
        global_trigger = (
            WindowsGlobalHelpTrigger(
                hotkey=args.global_help_hotkey,
                modifiers=args.global_help_modifiers,
                cooldown_ms=args.global_help_cooldown_ms,
            )
            if str(args.global_help_hotkey).strip()
            else None
        )
        trigger_list: list[HelpTriggerLike] = []
        if stdin_trigger is not None:
            stdin_trigger.start()
            trigger_list.append(stdin_trigger)
            print("[LIVE_DCS] stdin trigger enabled: press Enter/help/h/? to trigger help")
        if udp_trigger is not None:
            udp_trigger.start()
            trigger_list.append(udp_trigger)
            print(
                f"[LIVE_DCS] udp trigger enabled: send 'help' to "
                f"{args.help_udp_host}:{udp_trigger.bound_port}"
            )
        if global_trigger is not None:
            global_trigger.start()
            trigger_list.append(global_trigger)
            print(f"[LIVE_DCS] windows global trigger enabled: {global_trigger.hotkey_label}")
        trigger: HelpTriggerLike | None
        if len(trigger_list) == 1:
            trigger = trigger_list[0]
        elif trigger_list:
            trigger = CompositeHelpTrigger(trigger_list)
        else:
            trigger = None
        try:
            stats = loop.run(
                max_frames=args.max_frames,
                duration_s=args.duration,
                auto_help_on_first_frame=bool(args.auto_help_once),
                auto_help_every_n_frames=args.auto_help_every_n_frames,
                help_trigger=trigger,
                help_capture_notifier=vision_capture_notifier,
            )
        finally:
            if stdin_trigger is not None:
                stdin_trigger.close()
            if udp_trigger is not None:
                udp_trigger.close()
            if global_trigger is not None:
                global_trigger.close()
            if vision_capture_notifier is not None:
                vision_capture_notifier.close()
            loop.close()

    print(f"[LIVE_DCS] wrote events to {output}")
    print(f"[LIVE_DCS] stats={json.dumps(stats, ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
