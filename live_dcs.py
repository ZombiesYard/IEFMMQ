from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import queue
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
from adapters.dcs_bios.bios_ui_map import BiosUiMapper
from adapters.dcs_bios.receiver import DcsBiosReceiver
from adapters.evidence_refs import collect_evidence_refs_from_context, infer_evidence_type_from_ref
from adapters.knowledge_source_policy import KnowledgeSourcePolicy, KnowledgeSourcePolicyError
from adapters.knowledge_local import DEFAULT_INDEX_PATH, LocalKnowledgeAdapter, build_grounding_query
from adapters.model_stub import ModelStub
from adapters.ollama_model import OllamaModel
from adapters.openai_compat_model import OpenAICompatModel
from adapters.pack_gates import (
    DEFAULT_SCENARIO_PROFILE,
    SUPPORTED_SCENARIO_PROFILES,
    evaluate_pack_gates,
    load_pack_gate_config,
    normalize_scenario_profile,
)
from adapters.prompting import build_help_prompt_result
from adapters.recent_actions import (
    RecentDeltaRingBuffer,
    build_prompt_recent_deltas,
    build_recent_button_signal,
)
from adapters.response_mapping import map_help_response_to_tutor_response
from adapters.source_chunk_refs import build_source_chunk_ref
from adapters.step_inference import infer_step_id, load_pack_steps
from adapters.telemetry_pipeline import enrich_bios_observation
from adapters.vision_frames import DEFAULT_FRAME_CHANNEL, FrameDirectoryVisionPort
from adapters.vision_prompting import DEFAULT_LAYOUT_ID
from adapters.vision_sync import (
    DEFAULT_LIVE_SYNC_WINDOW_MS,
    DEFAULT_LIVE_TRIGGER_WAIT_MS,
    DEFAULT_REPLAY_SYNC_WINDOW_MS,
    BufferedVisionSession,
    HelpCycleVisionSelection,
)
from core.constants import ENV_COLD_START_PRODUCTION
from core.env_bool import parse_env_bool
from core.event_store import JsonlEventStore
from core.help_failure import classify_mapping_failure, merge_failure_metadata, overlay_rejection_payload
from core.step_signal_metadata import (
    STEP_EVIDENCE_REQUIREMENT_VALUES,
    STEP_OBSERVABILITY_VALUES,
    compute_requires_visual_confirmation,
    normalize_observability_status,
)
from core.types import Event, Observation, TutorRequest, TutorResponse
from core.vars import VarResolver
from ports.knowledge_port import KnowledgePort, KnowledgeRetrieveWithMetaPort
from simtutor.cli_parsing import parse_env_int, parse_non_negative_int_arg


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


class ObservationSource(Protocol):
    def get_observation(self) -> Observation | None:
        ...


class ActionExecutorLike(Protocol):
    def execute_actions(self, actions: Sequence[Mapping[str, Any] | Any]) -> Any:
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


def _attach_help_cycle_trace_to_actions(
    actions: Sequence[Mapping[str, Any] | Any],
    *,
    help_cycle_id: str,
    generation_mode: str,
) -> list[Mapping[str, Any] | Any]:
    traced: list[Mapping[str, Any] | Any] = []
    for action in actions:
        if not isinstance(action, Mapping):
            traced.append(action)
            continue
        item = dict(action)
        item["help_cycle_id"] = help_cycle_id
        item.setdefault("generation_mode", generation_mode)
        traced.append(item)
    return traced


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
    if not isinstance(raw, list):
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
) -> dict[str, dict[str, list[str]]]:
    profiles: dict[str, dict[str, list[str]]] = {}
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
        }
    return profiles


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

    def to_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "help_cycles": self.help_cycles,
            "model_calls": self.model_calls,
            "cache_hits": self.cache_hits,
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
        self.step_signal_profiles = _load_step_signal_profiles(self.pack_path)
        gate_config = load_pack_gate_config(
            self.pack_path,
            scenario_profile=self.scenario_profile,
        )
        self.precondition_gates = dict(gate_config.get("precondition_gates", {}))
        self.completion_gates = dict(gate_config.get("completion_gates", {}))
        self.candidate_steps = _load_step_ids(self.pack_path)
        self.overlay_allowlist = _load_overlay_allowlist(self.pack_path, self.ui_map_path)
        self.overlay_allowset = set(self.overlay_allowlist)
        self.step_fallback_profiles = _build_step_fallback_profiles(
            self.pack_steps,
            precondition_gates=self.precondition_gates,
            completion_gates=self.completion_gates,
        )
        self.recent_ring = RecentDeltaRingBuffer(window_s=8.0, max_items=20)
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
        self._vision_session: BufferedVisionSession | None = None
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
        self._stats = LiveLoopStats()

    @property
    def stats(self) -> LiveLoopStats:
        return self._stats

    def close(self) -> None:
        if self._vision_session is not None:
            self._vision_session.close()
        if hasattr(self.action_executor, "close"):
            self.action_executor.close()
        if hasattr(self.source, "close"):
            self.source.close()
        if hasattr(self.model, "close"):
            self.model.close()

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
        trigger_wall_ms = int(round(float(trigger_t_wall) * 1000.0))
        payload = observation.payload if isinstance(observation.payload, Mapping) else {}
        observation_seq = _coerce_int(payload.get("seq"))
        observation_t_wall_s = _coerce_finite_float(payload.get("t_wall"))
        if observation_t_wall_s is None:
            observation_t_wall_s = float(trigger_t_wall)
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
        selection = self._vision_session.select_for_help(trigger_wall_s=trigger_t_wall)
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

        payload = raw_obs.payload if isinstance(raw_obs.payload, Mapping) else {}
        delta = payload.get("delta")
        t_wall = _coerce_float(payload.get("t_wall"))
        seq = _coerce_int(payload.get("seq"))
        if isinstance(delta, Mapping) and t_wall is not None:
            self.recent_ring.add_delta(delta, t_wall=t_wall, seq=seq)

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
            if isinstance(missing_conditions_raw, list)
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
    ) -> tuple[TutorRequest, dict[str, Any], str]:
        payload = obs.payload if isinstance(obs.payload, Mapping) else {}
        vars_map = payload.get("vars")
        if not isinstance(vars_map, Mapping):
            vars_map = {}
        vars_selected = dict(vars_map)
        vision_context = vision_selection.to_dict()

        now_t_wall = _coerce_float(payload.get("t_wall"))
        recent_frames = self.recent_ring.snapshot(now_t_wall=now_t_wall) if now_t_wall is not None else self.recent_ring.snapshot()
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
        )
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
        deterministic_hint = {
            "inferred_step_id": inference.inferred_step_id,
            "missing_conditions": missing_conditions,
            "recent_ui_targets": recent_buttons,
            "gate_blockers": inferred_gate_blockers,
            "scenario_profile": self.scenario_profile,
        }
        if isinstance(inference.inferred_step_id, str) and inference.inferred_step_id:
            step_signal_profile = self.step_signal_profiles.get(inference.inferred_step_id)
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
                requires_visual_confirmation = step_signal_profile.get("requires_visual_confirmation")
                if isinstance(requires_visual_confirmation, bool):
                    deterministic_hint["requires_visual_confirmation"] = requires_visual_confirmation
        rag_topk, grounding_meta = self._build_grounding_context(deterministic_hint)

        context = {
            "vars": vars_selected,
            "gates": gates,
            "recent_deltas": recent_deltas,
            "recent_actions": recent_actions,
            "candidate_steps": list(self.candidate_steps),
            "overlay_target_allowlist": list(self.overlay_allowlist),
            "deterministic_step_hint": deterministic_hint,
            "scenario_profile": self.scenario_profile,
            "rag_topk": rag_topk,
            "grounding_missing": bool(grounding_meta.get("grounding_missing")),
            "grounding_reason": grounding_meta.get("grounding_reason"),
            "grounding_query": grounding_meta.get("grounding_query"),
            "delta_summary": payload.get("delta_summary", {}),
            "delta_dropped_count": obs.metadata.get("delta_dropped_count"),
            "vision": vision_context,
        }

        prompt_result = build_help_prompt_result(context, self.lang)
        prompt_hash = hashlib.sha256(prompt_result.prompt.encode("utf-8")).hexdigest()
        prompt_grounding_missing = bool(prompt_result.metadata.get("grounding_missing"))
        prompt_grounding_reason = prompt_result.metadata.get("grounding_reason")
        req = TutorRequest(
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
            },
        )

        state_signature = {
            "vars_discrete": {
                key: value
                for key, value in sorted(vars_selected.items())
                if isinstance(value, bool) or value is None
            },
            "recent_buttons": recent_buttons,
            "candidate_steps": self.candidate_steps,
            "overlay_target_allowlist": self.overlay_allowlist,
            "deterministic_step_hint": deterministic_hint,
            "scenario_profile": self.scenario_profile,
            "vision": {
                "status": vision_context["status"],
                "frame_ids": list(vision_context["frame_ids"]),
                "sync_miss_reason": vision_context["sync_miss_reason"],
            },
        }
        state_key = _stable_hash_json(state_signature)
        return req, prompt_result.metadata, state_key

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
            max_overlay_targets=1,
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

    def _build_safe_fallback_overlay_help_obj(
        self,
        request: TutorRequest,
    ) -> tuple[dict[str, Any] | None, str]:
        context = request.context if isinstance(request.context, Mapping) else {}
        hint = context.get("deterministic_step_hint")
        if not isinstance(hint, Mapping):
            return None, "missing_deterministic_hint"

        inferred_step_id = hint.get("inferred_step_id")
        if not isinstance(inferred_step_id, str) or not inferred_step_id:
            return None, "missing_inferred_step_id"

        step_fallback_profile = self.step_fallback_profiles.get(inferred_step_id)
        if not isinstance(step_fallback_profile, Mapping):
            return None, f"unsupported_step:{inferred_step_id}"
        fallback_targets_raw = step_fallback_profile.get("ui_targets")
        fallback_targets = _normalize_step_ui_targets(fallback_targets_raw)
        if not fallback_targets:
            return None, f"unsupported_step:{inferred_step_id}"
        declared_fallback_target = fallback_targets[0]

        request_allowlist = context.get("overlay_target_allowlist")
        candidate_targets = list(fallback_targets)
        if isinstance(request_allowlist, list):
            allowset = {item for item in request_allowlist if isinstance(item, str) and item}
            if allowset:
                candidate_targets = [target for target in candidate_targets if target in allowset]
                if not candidate_targets:
                    return None, f"target_not_in_request_allowlist:{declared_fallback_target}"
        candidate_targets = [target for target in candidate_targets if target in self.overlay_allowset]
        if not candidate_targets:
            return None, f"target_not_in_runtime_allowlist:{declared_fallback_target}"
        fallback_target = candidate_targets[0]

        candidate_refs: list[str] = []
        gate_blockers = hint.get("gate_blockers")
        if isinstance(gate_blockers, list):
            for blocker in gate_blockers:
                if not isinstance(blocker, Mapping):
                    continue
                ref = blocker.get("ref")
                if isinstance(ref, str) and ref:
                    candidate_refs.append(ref)
        candidate_refs.append(f"GATES.{inferred_step_id}.completion")
        candidate_refs.append(f"GATES.{inferred_step_id}.precondition")
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
        if isinstance(gate_blockers, list):
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
                "targets": [fallback_target],
                "evidence": [
                    {
                        "target": fallback_target,
                        "type": evidence_type,
                        "ref": selected_ref,
                        "quote": quote,
                        "grounding_confidence": 0.51,
                    }
                ],
            },
            "explanations": [
                (
                    f"请先操作 {fallback_target}。"
                    if self.lang == "zh"
                    else f"Please operate {fallback_target} first."
                )
            ],
            "confidence": 0.51,
        }
        return fallback_help_obj, f"deterministic_step:{inferred_step_id}"

    def _apply_safe_fallback_overlay(
        self,
        response: TutorResponse,
        request: TutorRequest,
    ) -> tuple[bool, str]:
        fallback_help_obj, fallback_reason = self._build_safe_fallback_overlay_help_obj(request)
        if not isinstance(fallback_help_obj, Mapping):
            return False, fallback_reason

        mapped = map_help_response_to_tutor_response(
            fallback_help_obj,
            request=request,
            status=response.status,
            max_overlay_targets=1,
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
        if not response.message and mapped.message:
            response.message = mapped.message
        if not response.explanations and mapped.explanations:
            response.explanations = list(mapped.explanations)
        return True, fallback_reason

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

    def run_help_cycle(self, *, trigger_t_wall: float | None = None) -> tuple[TutorResponse | None, dict[str, Any] | None]:
        obs = self._latest_enriched_obs
        if obs is None:
            return None, None

        resolved_trigger_t_wall = trigger_t_wall
        if resolved_trigger_t_wall is None:
            payload = obs.payload if isinstance(obs.payload, Mapping) else {}
            resolved_trigger_t_wall = _coerce_float(payload.get("t_wall"))
        if resolved_trigger_t_wall is None:
            resolved_trigger_t_wall = time.time()

        vision_selection = self._build_vision_selection(
            observation=obs,
            trigger_t_wall=resolved_trigger_t_wall,
        )
        request, prompt_meta, state_key = self._build_request(obs, vision_selection=vision_selection)
        help_cycle_id = request.request_id
        request.metadata = dict(request.metadata)
        request.metadata["help_cycle_id"] = help_cycle_id
        request.metadata["vision_status"] = vision_selection.status
        request.metadata["vision_frame_ids"] = list(vision_selection.frame_ids)
        now_wall = time.time()
        self._emit_event(
            kind="tutor_request",
            payload=request.to_dict(),
            related_id=request.request_id,
            t_wall=now_wall,
            metadata={
                "help_cycle_id": help_cycle_id,
                "vision_status": vision_selection.status,
            },
            vision_refs=vision_selection.frame_ids,
        )
        self._stats.help_cycles += 1

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
            response.metadata["generation_mode"] = _normalize_generation_mode(response)
            response.actions = _attach_help_cycle_trace_to_actions(
                response.actions,
                help_cycle_id=help_cycle_id,
                generation_mode=response.metadata["generation_mode"],
            )
            overlay_report = self._execute_or_dry_run_actions(response.actions)
        else:
            hint = request.context.get("deterministic_step_hint", {})
            inferred_step_id = hint.get("inferred_step_id") if isinstance(hint, Mapping) else None
            missing_conditions = hint.get("missing_conditions", []) if isinstance(hint, Mapping) else []
            if not isinstance(missing_conditions, list):
                missing_conditions = []
            gate_blockers = hint.get("gate_blockers", []) if isinstance(hint, Mapping) else []
            if not isinstance(gate_blockers, list):
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
            self._stats.model_calls += 1
            try:
                response = self.model.explain_error(obs, request)
            except Exception as exc:
                response = TutorResponse(
                    status="error",
                    in_reply_to=request.request_id,
                    message=self._fallback_message(
                        inferred_step_id if isinstance(inferred_step_id, str) else None,
                        fallback_conditions,
                    ),
                    actions=[],
                    metadata={
                        "provider": "fallback",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )

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
            response.metadata["generation_mode"] = _normalize_generation_mode(response)

            mapped_actions, mapped_meta = self._map_response_actions(response, request)
            response.actions = mapped_actions
            if mapped_meta:
                response.metadata["response_mapping"] = mapped_meta
                mapping_failure_codes = classify_mapping_failure(mapped_meta)
                if mapping_failure_codes:
                    response.metadata = merge_failure_metadata(
                        response.metadata,
                        *mapping_failure_codes,
                        stage="response_mapping",
                    )

            fallback_overlay_used = False
            fallback_overlay_reason = "not_needed"
            if response.status == "error" and not response.actions:
                fallback_overlay_used, fallback_overlay_reason = self._apply_safe_fallback_overlay(
                    response,
                    request,
                )

            response.metadata["fallback_overlay_used"] = fallback_overlay_used
            response.metadata["fallback_overlay_reason"] = fallback_overlay_reason
            _normalize_cached_response_metadata(response.metadata)
            response.actions = _attach_help_cycle_trace_to_actions(
                response.actions,
                help_cycle_id=help_cycle_id,
                generation_mode=response.metadata["generation_mode"],
            )

            overlay_report = self._execute_or_dry_run_actions(response.actions)
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
        response.metadata["scenario_profile"] = self.scenario_profile

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
                },
                vision_refs=vision_selection.frame_ids,
            )

        self._emit_event(
            kind="tutor_response",
            payload=response.to_dict(),
            related_id=request.request_id,
            t_wall=time.time(),
            metadata={
                "help_cycle_id": help_cycle_id,
                "generation_mode": response.metadata.get("generation_mode"),
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

            obs = self.source.get_observation()
            if obs is not None:
                self._ingest_observation(obs)
                self._stats.frames += 1
                obs_payload = obs.payload if isinstance(obs.payload, Mapping) else {}
                obs_t_wall = _coerce_float(obs_payload.get("t_wall"))
                if auto_help_on_first_frame and not first_help_done:
                    self.run_help_cycle(trigger_t_wall=obs_t_wall)
                    first_help_done = True
                if auto_help_every_n_frames > 0:
                    if self._stats.frames % auto_help_every_n_frames == 0:
                        self.run_help_cycle(trigger_t_wall=obs_t_wall)
            else:
                exhausted = bool(getattr(self.source, "is_exhausted", False))
                if exhausted:
                    break

            if help_trigger is not None and help_trigger.poll():
                help_trigger_t_wall = time.time() if self.vision_mode == "live" else None
                self.run_help_cycle(trigger_t_wall=help_trigger_t_wall)

            if max_frames > 0 and self._stats.frames >= max_frames:
                break

            if obs is None:
                time.sleep(max(0.0, idle_sleep_s))

        return self._stats.to_dict()


def _new_default_log_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"live_dcs_{ts}.jsonl"


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
    if provider == "stub":
        return ModelStub(mode=args.stub_mode)

    timeout_s = float(args.model_timeout_s)
    model_max_tokens_value = int(args.model_max_tokens)
    if model_max_tokens_value < 0:
        raise ValueError("--model-max-tokens must be >= 0")
    model_max_tokens = model_max_tokens_value if model_max_tokens_value > 0 else None
    if provider == "openai_compat":
        if not args.model_base_url:
            raise ValueError("--model-base-url is required for openai_compat")
        return OpenAICompatModel(
            model_name=args.model_name,
            base_url=args.model_base_url,
            timeout_s=timeout_s,
            max_tokens=model_max_tokens,
            lang=lang,
            log_raw_llm_text=log_raw_llm_text,
            api_key=args.model_api_key,
        )
    if provider == "ollama":
        base_url = args.model_base_url or "http://127.0.0.1:11434"
        return OllamaModel(
            model_name=args.model_name,
            base_url=base_url,
            timeout_s=timeout_s,
            lang=lang,
            log_raw_llm_text=log_raw_llm_text,
        )
    raise ValueError(f"Unsupported model provider: {provider}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live DCS tutor loop (bios -> help -> overlay).")

    parser.add_argument("--host", default="0.0.0.0", help="DCS-BIOS UDP bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7790, help="DCS-BIOS UDP bind port (default 7790)")
    parser.add_argument("--timeout", type=float, default=0.2, help="Receiver socket timeout seconds")
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
    parser.add_argument("--model-base-url", default=os.getenv("SIMTUTOR_MODEL_BASE_URL", ""))
    parser.add_argument("--model-timeout-s", type=float, default=float(os.getenv("SIMTUTOR_MODEL_TIMEOUT_S", "20")))
    parser.add_argument(
        "--model-max-tokens",
        type=parse_non_negative_int_arg,
        default=parse_env_int("SIMTUTOR_MODEL_MAX_TOKENS", default=0, minimum=0),
        help="Max completion tokens for model providers that support it (0 uses provider default).",
    )
    parser.add_argument("--model-api-key", default=os.getenv("SIMTUTOR_MODEL_API_KEY"))
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
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    output = Path(args.output) if args.output else _new_default_log_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    source: ObservationSource
    if args.replay_bios:
        source = ReplayBiosReceiver(args.replay_bios, speed=args.speed)
    else:
        source = DcsBiosReceiver(
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            merge_full_state=bool(args.merge_full_state),
        )

    model = _build_model_from_args(args)
    vision_port, vision_session_id, vision_sync_window_ms, vision_trigger_wait_ms = _build_vision_port_from_args(
        args,
        mode="live",
    )

    with JsonlEventStore(output, mode="w") as store:
        executor = OverlayActionExecutor(
            ui_map_path=args.ui_map,
            pack_path=args.pack,
            dry_run=bool(args.dry_run_overlay),
            session_id=args.session_id,
            event_sink=store.append,
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
            )
        finally:
            if stdin_trigger is not None:
                stdin_trigger.close()
            if udp_trigger is not None:
                udp_trigger.close()
            loop.close()

    print(f"[LIVE_DCS] wrote events to {output}")
    print(f"[LIVE_DCS] stats={json.dumps(stats, ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
