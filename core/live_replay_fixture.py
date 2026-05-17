from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class LiveReplayFixtureError(ValueError):
    pass


class LiveReplayFixtureNotFound(LiveReplayFixtureError):
    pass


def _jsonable(raw: Any) -> Any:
    try:
        json.dumps(raw)
    except TypeError:
        return str(raw)
    return raw


def _as_mapping(raw: Any) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def _as_dict(raw: Any) -> dict[str, Any]:
    return dict(raw) if isinstance(raw, Mapping) else {}


def _as_list(raw: Any) -> list[Any]:
    return list(raw) if isinstance(raw, list) else []


def _nonempty_text(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    return text or None


def _event_kind(event: Mapping[str, Any]) -> str | None:
    return _nonempty_text(event.get("kind") or event.get("type"))


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(event.get("payload"))


def _metadata(event: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(event.get("metadata"))


def _payload_metadata(event: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(_payload(event).get("metadata"))


def _event_help_cycle_id(event: Mapping[str, Any]) -> str | None:
    for raw in (
        _metadata(event).get("help_cycle_id"),
        _payload_metadata(event).get("help_cycle_id"),
    ):
        text = _nonempty_text(raw)
        if text is not None:
            return text
    kind = _event_kind(event)
    payload = _payload(event)
    if kind == "tutor_request":
        return _nonempty_text(payload.get("request_id"))
    if kind == "tutor_response":
        return _nonempty_text(payload.get("in_reply_to"))
    return None


def _event_matches_request_id(event: Mapping[str, Any], request_id: str) -> bool:
    payload = _payload(event)
    metadata = _metadata(event)
    payload_metadata = _payload_metadata(event)
    return any(
        raw == request_id
        for raw in (
            event.get("related_id"),
            payload.get("request_id"),
            payload.get("in_reply_to"),
            metadata.get("help_cycle_id"),
            payload_metadata.get("help_cycle_id"),
        )
    )


def _load_jsonl_lenient(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    events: list[tuple[int, dict[str, Any]]] = []
    malformed: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError as exc:
                malformed.append({"lineno": lineno, "error": str(exc)})
                continue
            if not isinstance(decoded, dict):
                malformed.append({"lineno": lineno, "error": "JSONL row is not an object"})
                continue
            events.append((lineno, decoded))
    return events, malformed


def _first_event(
    events: Iterable[tuple[int, dict[str, Any]]],
    *,
    kind: str,
    request_id: str,
) -> tuple[int, dict[str, Any]] | None:
    for lineno, event in events:
        if _event_kind(event) == kind and _event_matches_request_id(event, request_id):
            return lineno, event
    return None


def _event_belongs_to_cycle(event: Mapping[str, Any], *, request_id: str, help_cycle_id: str) -> bool:
    if _event_matches_request_id(event, request_id):
        return True
    event_help_cycle_id = _event_help_cycle_id(event)
    return event_help_cycle_id == help_cycle_id


def _observation_id_from_event(event: Mapping[str, Any]) -> str | None:
    payload = _payload(event)
    return _nonempty_text(payload.get("observation_id") or payload.get("id"))


def _is_vision_fact_observation_event(event: Mapping[str, Any]) -> bool:
    if _event_kind(event) != "observation":
        return False
    payload = _payload(event)
    payload_metadata = _as_mapping(payload.get("metadata"))
    nested_payload_metadata = _as_mapping(_as_mapping(payload.get("payload")).get("metadata"))
    return any(
        raw == "vision_fact"
        for raw in (
            _metadata(event).get("observation_kind"),
            payload_metadata.get("observation_kind"),
            nested_payload_metadata.get("observation_kind"),
        )
    )


def _event_frame_ids(event: Mapping[str, Any]) -> set[str]:
    payload = _payload(event)
    nested_payload = _as_mapping(payload.get("payload"))
    candidates = [
        event.get("vision_refs"),
        _metadata(event).get("frame_ids"),
        _as_mapping(payload.get("metadata")).get("frame_ids"),
        payload.get("frame_ids"),
        nested_payload.get("frame_ids"),
    ]
    out: set[str] = set()
    for raw in candidates:
        for item in _as_list(raw):
            if isinstance(item, str) and item:
                out.add(item)
    return out


def _nearby_events(
    events: list[tuple[int, dict[str, Any]]],
    *,
    request_lineno: int,
    before: int = 3,
    after: int = 3,
) -> list[tuple[int, dict[str, Any]]]:
    request_index = next(
        (idx for idx, item in enumerate(events) if item[0] == request_lineno),
        -1,
    )
    if request_index < 0:
        return []
    return events[max(0, request_index - before): request_index + after + 1]


def _extract_overlay_targets(response: Mapping[str, Any], trace: Mapping[str, Any]) -> list[str]:
    response_metadata = _as_mapping(response.get("metadata"))
    for raw in (
        trace.get("final_overlay_targets"),
        response_metadata.get("final_overlay_targets"),
    ):
        targets = [item for item in _as_list(raw) if isinstance(item, str) and item]
        if targets:
            return targets
    out: list[str] = []
    for action in _as_list(response.get("actions")):
        if not isinstance(action, Mapping):
            continue
        target = action.get("target")
        if isinstance(target, str) and target and target not in out:
            out.append(target)
    return out


def _extract_final_step_id(response_metadata: Mapping[str, Any], trace: Mapping[str, Any]) -> str | None:
    final_plan = _as_mapping(trace.get("final_action_plan") or response_metadata.get("final_action_plan"))
    step_id = _nonempty_text(final_plan.get("step_id"))
    if step_id is not None:
        return step_id
    final_public = _as_mapping(response_metadata.get("final_public_response"))
    for key in ("next", "diagnosis"):
        step_id = _nonempty_text(_as_mapping(final_public.get(key)).get("step_id"))
        if step_id is not None:
            return step_id
    for key in ("next", "diagnosis"):
        step_id = _nonempty_text(_as_mapping(response_metadata.get(key)).get("step_id"))
        if step_id is not None:
            return step_id
    return None


def _extract_vlm_call_status(response_metadata: Mapping[str, Any], trace: Mapping[str, Any]) -> str | None:
    trace_status = _nonempty_text(_as_mapping(trace.get("vlm_call")).get("status"))
    if trace_status is not None:
        return trace_status
    return _nonempty_text(response_metadata.get("vlm_call_status"))


def _extract_llm_decision_status(trace: Mapping[str, Any], response_metadata: Mapping[str, Any]) -> str:
    validator = _as_mapping(trace.get("validator_result"))
    repair = _as_mapping(trace.get("repair_result"))
    if (
        bool(repair.get("applied"))
        or bool(response_metadata.get("repair_applied"))
        or response_metadata.get("generation_mode") == "repair"
    ):
        return "repaired"
    if bool(validator.get("rejected")) or bool(response_metadata.get("validator_rejected")):
        return "rejected"
    return "accepted"


def _extract_final_response_source(
    response: Mapping[str, Any],
    response_metadata: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> str | None:
    if response_metadata.get("cached_response_reused") is True:
        return "cache"
    provider = _nonempty_text(response_metadata.get("provider"))
    generation_mode = _nonempty_text(response_metadata.get("generation_mode"))
    final_plan_source = _nonempty_text(
        _as_mapping(trace.get("final_action_plan")).get("source")
        or _as_mapping(response_metadata.get("final_action_plan")).get("source")
    )
    if final_plan_source is not None and final_plan_source != "model":
        return final_plan_source
    repair_path = _nonempty_text(_as_mapping(trace.get("repair_result")).get("path"))
    if repair_path is not None:
        return repair_path
    if response_metadata.get("fallback_overlay_used") is True:
        fallback_reason = _nonempty_text(response_metadata.get("fallback_overlay_reason"))
        if fallback_reason and fallback_reason.startswith("deterministic_step:"):
            return "deterministic_fallback"
        return fallback_reason or "fallback_overlay"
    if provider == "fallback" or generation_mode == "fallback" or response.get("status") == "error":
        return "fallback"
    if generation_mode in {"model", "repair"}:
        return generation_mode
    return provider or generation_mode


def _compact_event(lineno: int, event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lineno": lineno,
        "kind": _event_kind(event),
        "related_id": event.get("related_id"),
        "help_cycle_id": _event_help_cycle_id(event),
        "vision_refs": [item for item in _as_list(event.get("vision_refs")) if isinstance(item, str)],
        "payload": _jsonable(event.get("payload")),
        "metadata": _jsonable(event.get("metadata")),
    }


def extract_live_replay_fixture(
    events: list[tuple[int, dict[str, Any]]],
    *,
    request_id: str,
    source_path: str | None = None,
    malformed_lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_request_id = _nonempty_text(request_id)
    if normalized_request_id is None:
        raise LiveReplayFixtureError("request_id must be a non-empty string")

    request_match = _first_event(events, kind="tutor_request", request_id=normalized_request_id)
    if request_match is None:
        raise LiveReplayFixtureNotFound(f"request_id not found in live log: {normalized_request_id}")
    request_lineno, request_event = request_match
    request_payload = _as_dict(request_event.get("payload"))
    help_cycle_id = (
        _event_help_cycle_id(request_event)
        or _nonempty_text(request_payload.get("request_id"))
        or normalized_request_id
    )

    cycle_events = [
        (lineno, event)
        for lineno, event in events
        if _event_belongs_to_cycle(event, request_id=normalized_request_id, help_cycle_id=help_cycle_id)
    ]
    response_match = _first_event(cycle_events, kind="tutor_response", request_id=normalized_request_id)
    if response_match is None:
        raise LiveReplayFixtureNotFound(f"tutor_response not found for request_id: {normalized_request_id}")
    _response_lineno, response_event = response_match
    response_payload = _as_dict(response_event.get("payload"))
    response_metadata = _as_mapping(response_payload.get("metadata"))

    request_context = _as_mapping(request_payload.get("context"))
    evidence_packet_summary = _as_dict(request_context.get("evidence_packet_summary"))
    trace = _as_mapping(response_metadata.get("harness_trace"))
    if not evidence_packet_summary:
        evidence_packet_summary = _as_dict(trace.get("evidence_packet_summary"))
    telemetry_window_digest = _as_dict(evidence_packet_summary.get("telemetry_window_digest"))

    observation_ref = _nonempty_text(request_payload.get("observation_ref"))
    observations: list[Mapping[str, Any]] = []
    if observation_ref is not None:
        observations = [
            _payload(event)
            for _lineno, event in events
            if _event_kind(event) == "observation"
            and _observation_id_from_event(event) == observation_ref
        ]
    if not observations:
        nearby = _nearby_events(events, request_lineno=request_lineno)
        observations = [_payload(event) for _lineno, event in nearby if _event_kind(event) == "observation"]

    frame_ids = set(
        item
        for item in _as_list(
            response_metadata.get("vision_frame_ids")
            or _as_mapping(request_payload.get("metadata")).get("vision_frame_ids")
            or _as_mapping(trace.get("vlm_call")).get("frame_ids")
        )
        if isinstance(item, str) and item
    )
    nearby_line_numbers = {lineno for lineno, _event in _nearby_events(events, request_lineno=request_lineno, before=6, after=3)}
    vision_fact_observations = [
        _payload(event)
        for lineno, event in events
        if _is_vision_fact_observation_event(event)
        and (
            (frame_ids and bool(frame_ids & _event_frame_ids(event)))
            or lineno in nearby_line_numbers
        )
    ]

    final_overlay_targets = _extract_overlay_targets(response_payload, trace)
    expectations = {
        "expected_final_step_id": _extract_final_step_id(response_metadata, trace),
        "expected_overlay_target_ids": final_overlay_targets,
        "vlm_call_status": _extract_vlm_call_status(response_metadata, trace),
        "llm_decision_status": _extract_llm_decision_status(trace, response_metadata),
        "final_response_source": _extract_final_response_source(response_payload, response_metadata, trace),
    }
    actual_request_id = (
        _nonempty_text(request_payload.get("request_id"))
        or _nonempty_text(response_payload.get("in_reply_to"))
    )

    return {
        "schema_version": "live_help_replay_fixture.v1",
        "extraction_id": normalized_request_id,
        "request_id": actual_request_id or normalized_request_id,
        "request_id_missing": actual_request_id is None,
        "help_cycle_id": help_cycle_id,
        "source_log": {
            "path": source_path,
            "event_count": len(events),
            "malformed_lines": list(malformed_lines or []),
        },
        "cycle": {
            "tutor_request": request_payload,
            "tutor_response": response_payload,
            "events": [_compact_event(lineno, event) for lineno, event in cycle_events],
        },
        "context": {
            "observation_ref": observation_ref,
            "observations": [_jsonable(dict(item)) for item in observations if isinstance(item, Mapping)],
            "vision": {
                "request": _as_dict(request_payload.get("metadata")).get("vision"),
                "response": response_metadata.get("vision"),
                "frame_ids": _as_list(
                    response_metadata.get("vision_frame_ids")
                    or _as_mapping(request_payload.get("metadata")).get("vision_frame_ids")
                ),
                "vision_fact_summary": _as_dict(
                    response_metadata.get("vision_fact_summary")
                    or request_context.get("vision_fact_summary")
                ),
                "vision_facts": _as_list(response_metadata.get("vision_facts")),
            },
            "evidence_packet_summary": evidence_packet_summary,
            "telemetry_window_digest": telemetry_window_digest,
            "vision_fact_observations": [
                _jsonable(dict(item)) for item in vision_fact_observations if isinstance(item, Mapping)
            ],
        },
        "model_io": {
            "model_raw_help_response": _jsonable(response_metadata.get("model_raw_help_response")),
            "help_response": _jsonable(response_metadata.get("help_response")),
            "raw_llm_text_present": isinstance(response_metadata.get("raw_llm_text"), str),
            "raw_llm_text_attempt_count": len(_as_list(response_metadata.get("raw_llm_text_attempts"))),
        },
        "harness": {
            "candidate_steps": _as_list(request_context.get("candidate_steps") or trace.get("candidates")),
            "trace": _jsonable(dict(trace)),
            "model_decision": _jsonable(_as_dict(trace.get("model_decision"))),
            "validator_result": _jsonable(_as_dict(trace.get("validator_result"))),
            "repair_result": _jsonable(_as_dict(trace.get("repair_result"))),
            "final_action_plan": _jsonable(
                _as_dict(trace.get("final_action_plan") or response_metadata.get("final_action_plan"))
            ),
        },
        "expectations": expectations,
    }


def extract_live_replay_fixture_from_jsonl(path: str | Path, *, request_id: str) -> dict[str, Any]:
    resolved_path = Path(path).expanduser().resolve()
    events, malformed = _load_jsonl_lenient(resolved_path)
    return extract_live_replay_fixture(
        events,
        request_id=request_id,
        source_path=str(resolved_path),
        malformed_lines=malformed,
    )


def default_fixture_output_path(output_dir: str | Path, *, request_id: str) -> Path:
    safe_request_id = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in request_id)
    if not safe_request_id:
        safe_request_id = "request"
    return Path(output_dir).expanduser() / f"{safe_request_id}.fixture.json"


def write_live_replay_fixture(
    fixture: Mapping[str, Any],
    *,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    if output is None and output_dir is None:
        raise LiveReplayFixtureError("either output or output_dir is required")
    if output is not None and output_dir is not None:
        raise LiveReplayFixtureError("output and output_dir are mutually exclusive")

    if output_dir is not None:
        request_id = _nonempty_text(fixture.get("request_id")) or "request"
        path = default_fixture_output_path(output_dir, request_id=request_id)
    else:
        path = Path(str(output)).expanduser()
        if path.exists() and path.is_dir():
            request_id = _nonempty_text(fixture.get("request_id")) or "request"
            path = default_fixture_output_path(path, request_id=request_id)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "LiveReplayFixtureError",
    "LiveReplayFixtureNotFound",
    "default_fixture_output_path",
    "extract_live_replay_fixture",
    "extract_live_replay_fixture_from_jsonl",
    "write_live_replay_fixture",
]
