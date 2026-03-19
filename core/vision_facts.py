"""
Vision-fact configuration, sticky aggregation, and summary helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml

from core.types_v2 import VisionFactObservation

VISION_FACT_IDS: tuple[str, ...] = (
    "left_ddi_dark",
    "right_ddi_dark",
    "ampcd_dark",
    "left_ddi_menu_root_visible",
    "left_ddi_fcs_option_visible",
    "left_ddi_fcs_page_button_visible",
    "fcs_page_visible",
    "bit_page_visible",
    "bit_root_page_visible",
    "bit_page_failure_visible",
    "right_ddi_fcsmc_page_visible",
    "right_ddi_fcs_option_visible",
    "right_ddi_in_test_visible",
    "fcs_reset_seen",
    "fcs_bit_interaction_seen",
    "fcs_bit_result_visible",
    "takeoff_trim_seen",
    "ins_alignment_page_visible",
    "ins_go",
)
VISION_FACT_STATES: frozenset[str] = frozenset({"seen", "not_seen", "uncertain"})
_SUPPORTED_SCHEMA_VERSIONS = {"v1"}
_STICKY_PRESERVE_STATES = frozenset({"not_seen", "uncertain"})
_S18_RESULT_KINDS = frozenset({"final_go", "intermediate_go", "in_test", "not_ready", "other"})
_S18_GO_NEGATIVE_RE = re.compile(r"\bno\s+go\b", re.IGNORECASE)
_S18_FCSA_GO_RE = re.compile(r"\bfcsa\b\s*[:=]?\s*go\b", re.IGNORECASE)
_S18_FCSB_GO_RE = re.compile(r"\bfcsb\b\s*[:=]?\s*go\b", re.IGNORECASE)


class VisionFactsConfigError(ValueError):
    """Raised when pack-driven vision-facts config resolution fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_vision_facts_path(pack_path: str | Path | None = None) -> Path:
    if pack_path is None:
        return _repo_root() / "packs" / "fa18c_startup" / "vision_facts.yaml"
    resolved_pack_path = Path(pack_path).expanduser().resolve()
    configured = _load_vision_facts_path_from_pack_metadata(resolved_pack_path)
    if configured is not None:
        return configured
    return resolved_pack_path.parent / "vision_facts.yaml"


def _safe_stat(path: Path, *, label: str) -> tuple[int, int]:
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise VisionFactsConfigError(f"{label} not found: {path}") from exc
    except OSError as exc:
        raise VisionFactsConfigError(f"failed to stat {label}: {path}") from exc
    return int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=32)
def _load_vision_facts_path_from_pack_metadata_cached(
    resolved_pack_path: str,
    mtime_ns: int,
    size_bytes: int,
) -> Path | None:
    del mtime_ns, size_bytes
    path = Path(resolved_pack_path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VisionFactsConfigError(f"pack metadata source not found: {path}") from exc
    except OSError as exc:
        raise VisionFactsConfigError(f"failed to read pack metadata source: {path}") from exc
    except yaml.YAMLError as exc:
        raise VisionFactsConfigError(f"failed to parse pack metadata source: {path}") from exc
    if not isinstance(payload, Mapping):
        raise VisionFactsConfigError(f"pack metadata source must be a mapping: {path}")
    metadata = payload.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise VisionFactsConfigError(f"pack metadata must be a mapping: {path}")
    vision_facts_path_raw = metadata.get("vision_facts_path")
    if vision_facts_path_raw is None:
        return None
    if not isinstance(vision_facts_path_raw, str) or not vision_facts_path_raw.strip():
        raise VisionFactsConfigError(f"metadata.vision_facts_path must be a non-empty string: {path}")
    candidate = Path(vision_facts_path_raw.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    return candidate.resolve()


def _load_vision_facts_path_from_pack_metadata(pack_path: Path) -> Path | None:
    return _load_vision_facts_path_from_pack_metadata_cached(
        str(pack_path),
        *_safe_stat(pack_path, label="pack metadata source"),
    )


def _path_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise VisionFactsConfigError(f"vision facts config not found: {path}") from exc
    except OSError as exc:
        raise VisionFactsConfigError(f"failed to stat vision facts config: {path}") from exc
    return (int(stat.st_mtime_ns), int(stat.st_size))


@lru_cache(maxsize=8)
def _load_vision_facts_config_cached(
    resolved_path: str,
    mtime_ns: int,
    size_bytes: int,
) -> dict[str, Any]:
    del mtime_ns, size_bytes
    path = Path(resolved_path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise VisionFactsConfigError(f"vision facts config not found: {path}") from exc
    except OSError as exc:
        raise VisionFactsConfigError(f"failed to read vision facts config: {path}") from exc
    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise VisionFactsConfigError(f"failed to parse vision facts config: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"vision facts config must be a mapping: {path}")
    return _normalize_vision_facts_config(payload)


def load_vision_facts_config(
    path: str | Path | None = None,
    *,
    pack_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = (
        Path(path)
        if path is not None
        else default_vision_facts_path(pack_path)
    ).expanduser().resolve()
    return _load_vision_facts_config_cached(str(resolved), *_path_signature(resolved))


def _normalize_vision_facts_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("vision facts config missing non-empty schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(_SUPPORTED_SCHEMA_VERSIONS))
        raise ValueError(
            f"unsupported vision facts schema_version {schema_version!r}; supported versions: {supported}"
        )
    facts_raw = raw.get("facts")
    if not isinstance(facts_raw, list) or not facts_raw:
        raise ValueError("vision facts config missing non-empty facts list")
    facts_by_id: dict[str, dict[str, Any]] = {}
    for item in facts_raw:
        if not isinstance(item, Mapping):
            raise ValueError("vision facts entries must be mappings")
        fact_id = item.get("fact_id")
        if not isinstance(fact_id, str) or fact_id not in VISION_FACT_IDS:
            raise ValueError(f"unsupported vision fact id: {fact_id!r}")
        if fact_id in facts_by_id:
            raise ValueError(f"duplicate vision fact id: {fact_id}")
        expires_after_ms = item.get("expires_after_ms")
        if isinstance(expires_after_ms, bool) or not isinstance(expires_after_ms, int) or expires_after_ms < 0:
            raise ValueError(f"vision fact {fact_id} expires_after_ms must be a non-negative int")
        sticky = item.get("sticky")
        if not isinstance(sticky, bool):
            raise ValueError(f"vision fact {fact_id} sticky must be a bool")
        intended_regions = _normalize_string_list_field(
            item.get("intended_regions"),
            field_name="intended_regions",
            fact_id=fact_id,
        )
        steps = _normalize_string_list_field(
            item.get("steps"),
            field_name="steps",
            fact_id=fact_id,
        )
        facts_by_id[fact_id] = {
            "fact_id": fact_id,
            "sticky": sticky,
            "expires_after_ms": expires_after_ms,
            "intended_regions": intended_regions,
            "steps": steps,
        }

    bindings_raw = raw.get("step_bindings", {})
    if not isinstance(bindings_raw, Mapping):
        raise ValueError("vision facts config step_bindings must be a mapping")
    step_bindings: dict[str, dict[str, tuple[str, ...]]] = {}
    for step_id, binding in bindings_raw.items():
        if not isinstance(step_id, str) or not step_id:
            raise ValueError("vision facts config step_bindings keys must be non-empty strings")
        if not isinstance(binding, Mapping):
            raise ValueError(f"vision step binding for {step_id} must be a mapping")
        all_of = binding.get("all_of")
        if not isinstance(all_of, list) or not all_of:
            raise ValueError(f"vision step binding for {step_id} requires non-empty all_of")
        normalized_all_of = []
        for fact_id in all_of:
            if not isinstance(fact_id, str) or fact_id not in facts_by_id:
                raise ValueError(f"vision step binding for {step_id} references unsupported fact {fact_id!r}")
            normalized_all_of.append(fact_id)
        any_of = binding.get("any_of")
        normalized_any_of: list[str] = []
        if any_of is not None:
            if not isinstance(any_of, list):
                raise ValueError(f"vision step binding for {step_id} any_of must be a list")
            for fact_id in any_of:
                if not isinstance(fact_id, str) or fact_id not in facts_by_id:
                    raise ValueError(f"vision step binding for {step_id} references unsupported fact {fact_id!r}")
                normalized_any_of.append(fact_id)
        step_bindings[step_id] = {
            "all_of": tuple(normalized_all_of),
            "any_of": tuple(normalized_any_of),
        }

    return {
        "schema_version": schema_version,
        "layout_id": raw.get("layout_id"),
        "facts_by_id": facts_by_id,
        "step_bindings": step_bindings,
    }


def _normalize_string_list_field(
    raw_value: Any,
    *,
    field_name: str,
    fact_id: str,
) -> list[str]:
    if raw_value is None:
        return []
    if not isinstance(raw_value, (list, tuple)):
        raise ValueError(f"vision fact {fact_id} {field_name} must be a list")
    return [value for value in raw_value if isinstance(value, str) and value]


def normalize_fact_state(raw_state: Any) -> str:
    if isinstance(raw_state, str) and raw_state in VISION_FACT_STATES:
        return raw_state
    return "uncertain"


def _normalize_result_kind(
    fact_id: str,
    raw_result_kind: Any,
    evidence_note: str,
) -> str | None:
    if isinstance(raw_result_kind, str):
        normalized = raw_result_kind.strip().lower()
        if normalized in _S18_RESULT_KINDS:
            return normalized
    if fact_id != "fcs_bit_result_visible":
        return None
    note_lower = evidence_note.lower()
    if not note_lower:
        return None
    if "in test" in note_lower:
        return "in_test"
    if "not rdy" in note_lower or "not ready" in note_lower:
        return "not_ready"
    if "pbit go" in note_lower:
        return "intermediate_go"
    if _S18_GO_NEGATIVE_RE.search(evidence_note):
        return "other"
    if _S18_FCSA_GO_RE.search(evidence_note) and _S18_FCSB_GO_RE.search(evidence_note):
        return "final_go"
    return "other"


def normalize_vision_fact(
    raw_fact: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    default_observed_at_wall_ms: int | None = None,
) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("vision fact normalization requires config")
    facts_by_id = config.get("facts_by_id", {})
    fact_id = raw_fact.get("fact_id")
    if not isinstance(fact_id, str) or fact_id not in facts_by_id:
        raise ValueError(f"unsupported vision fact id: {fact_id!r}")
    spec = facts_by_id[fact_id]
    source_frame_id = raw_fact.get("source_frame_id")
    if not isinstance(source_frame_id, str) or not source_frame_id:
        raise ValueError(f"vision fact {fact_id} missing source_frame_id")
    confidence = raw_fact.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"vision fact {fact_id} confidence must be numeric")
    confidence_value = max(0.0, min(1.0, float(confidence)))
    evidence_note = raw_fact.get("evidence_note")
    if not isinstance(evidence_note, str) or not evidence_note.strip():
        raise ValueError(f"vision fact {fact_id} evidence_note must be non-empty")
    observed_at_wall_ms = raw_fact.get("observed_at_wall_ms")
    if observed_at_wall_ms is None:
        observed_at_wall_ms = default_observed_at_wall_ms
    if observed_at_wall_ms is not None:
        if (
            isinstance(observed_at_wall_ms, bool)
            or not isinstance(observed_at_wall_ms, int)
            or observed_at_wall_ms < 0
        ):
            raise ValueError(f"vision fact {fact_id} observed_at_wall_ms must be a non-negative int")

    normalized = {
        "fact_id": fact_id,
        "state": normalize_fact_state(raw_fact.get("state")),
        "source_frame_id": source_frame_id,
        "confidence": confidence_value,
        "expires_after_ms": int(spec["expires_after_ms"]),
        "evidence_note": evidence_note.strip(),
        "observed_at_wall_ms": observed_at_wall_ms,
        "sticky": bool(spec["sticky"]),
    }
    result_kind = _normalize_result_kind(fact_id, raw_fact.get("result_kind"), normalized["evidence_note"])
    if result_kind is not None:
        normalized["result_kind"] = result_kind
    if normalized["observed_at_wall_ms"] is not None:
        normalized["expires_at_wall_ms"] = normalized["observed_at_wall_ms"] + normalized["expires_after_ms"]
    else:
        normalized["expires_at_wall_ms"] = None
    return normalized


def prune_expired_facts(
    snapshot: Mapping[str, Mapping[str, Any]] | None,
    *,
    now_wall_ms: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(snapshot, Mapping):
        return out
    for fact_id, fact in snapshot.items():
        if not isinstance(fact_id, str) or not isinstance(fact, Mapping):
            continue
        expires_at_wall_ms = fact.get("expires_at_wall_ms")
        if isinstance(expires_at_wall_ms, int) and expires_at_wall_ms < now_wall_ms:
            continue
        out[fact_id] = dict(fact)
    return out


def merge_vision_fact_observation(
    snapshot: Mapping[str, Mapping[str, Any]] | None,
    observation: VisionFactObservation | Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    now_wall_ms: int | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise ValueError("vision fact merge requires config")
    payload = observation.to_dict() if isinstance(observation, VisionFactObservation) else dict(observation)
    trigger_wall_ms = payload.get("trigger_wall_ms")
    effective_now_wall_ms = (
        trigger_wall_ms if isinstance(trigger_wall_ms, int) and trigger_wall_ms >= 0 else (now_wall_ms or 0)
    )
    merged = prune_expired_facts(snapshot, now_wall_ms=effective_now_wall_ms)
    facts_raw = payload.get("facts")
    if not isinstance(facts_raw, list):
        return merged
    for raw_fact in facts_raw:
        if not isinstance(raw_fact, Mapping):
            continue
        normalized = normalize_vision_fact(
            raw_fact,
            config=config,
            default_observed_at_wall_ms=effective_now_wall_ms if effective_now_wall_ms > 0 else None,
        )
        fact_id = normalized["fact_id"]
        existing = merged.get(fact_id)
        if (
            existing is not None
            and existing.get("sticky") is True
            and existing.get("state") == "seen"
            and normalized["state"] in _STICKY_PRESERVE_STATES
        ):
            continue
        merged[fact_id] = normalized
    return merged


def snapshot_to_list(snapshot: Mapping[str, Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not isinstance(snapshot, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for fact_id in VISION_FACT_IDS:
        fact = snapshot.get(fact_id)
        if isinstance(fact, Mapping):
            out.append(dict(fact))
    return out


def facts_satisfy_step_binding(
    snapshot: Mapping[str, Mapping[str, Any]] | None,
    *,
    step_id: str,
    config: Mapping[str, Any] | None = None,
) -> bool:
    current_config = config if config is not None else load_vision_facts_config()
    step_bindings = current_config.get("step_bindings", {})
    binding = step_bindings.get(step_id)
    if not isinstance(binding, Mapping):
        return False
    if not isinstance(snapshot, Mapping):
        return False
    required_all_of = binding.get("all_of", ())
    required_any_of = binding.get("any_of", ())
    for fact_id in required_all_of:
        fact = snapshot.get(fact_id)
        if not isinstance(fact, Mapping) or fact.get("state") != "seen":
            return False
    if required_any_of:
        for fact_id in required_any_of:
            fact = snapshot.get(fact_id)
            if isinstance(fact, Mapping) and fact.get("state") == "seen":
                return True
        return False
    return True


def build_vision_fact_summary(
    snapshot: Mapping[str, Mapping[str, Any]] | None,
    *,
    status: str,
    frame_ids: Sequence[str] | None = None,
    fresh_fact_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    seen_fact_ids: list[str] = []
    uncertain_fact_ids: list[str] = []
    not_seen_fact_ids: list[str] = []
    if isinstance(snapshot, Mapping):
        for fact_id in VISION_FACT_IDS:
            fact = snapshot.get(fact_id)
            if not isinstance(fact, Mapping):
                continue
            state = fact.get("state")
            if state == "seen":
                seen_fact_ids.append(fact_id)
            elif state == "not_seen":
                not_seen_fact_ids.append(fact_id)
            else:
                uncertain_fact_ids.append(fact_id)
    segments: list[str] = []
    if seen_fact_ids:
        segments.append("seen=" + ",".join(seen_fact_ids))
    if uncertain_fact_ids:
        segments.append("uncertain=" + ",".join(uncertain_fact_ids))
    if not_seen_fact_ids:
        segments.append("not_seen=" + ",".join(not_seen_fact_ids))
    summary_text = "; ".join(segments) if segments else "no_active_vision_facts"
    return {
        "status": status,
        "frame_ids": [item for item in (frame_ids or []) if isinstance(item, str) and item],
        "fresh_fact_ids": [item for item in (fresh_fact_ids or []) if isinstance(item, str) and item],
        "seen_fact_ids": seen_fact_ids,
        "uncertain_fact_ids": uncertain_fact_ids,
        "not_seen_fact_ids": not_seen_fact_ids,
        "summary_text": summary_text,
    }


def extract_vision_fact_snapshot(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, Mapping):
        if "snapshot" in raw and isinstance(raw.get("snapshot"), list):
            raw = raw.get("snapshot")
        elif "vision_facts" in raw and isinstance(raw.get("vision_facts"), list):
            raw = raw.get("vision_facts")
    if not isinstance(raw, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        fact_id = item.get("fact_id")
        if isinstance(fact_id, str) and fact_id in VISION_FACT_IDS:
            normalized_item = dict(item)
            evidence_note = normalized_item.get("evidence_note")
            evidence_note_text = evidence_note.strip() if isinstance(evidence_note, str) else ""
            result_kind = _normalize_result_kind(
                fact_id,
                normalized_item.get("result_kind"),
                evidence_note_text,
            )
            if result_kind is not None:
                normalized_item["result_kind"] = result_kind
            else:
                normalized_item.pop("result_kind", None)
            out[fact_id] = normalized_item
    return out


__all__ = [
    "VISION_FACT_IDS",
    "VISION_FACT_STATES",
    "VisionFactsConfigError",
    "build_vision_fact_summary",
    "default_vision_facts_path",
    "extract_vision_fact_snapshot",
    "facts_satisfy_step_binding",
    "load_vision_facts_config",
    "merge_vision_fact_observation",
    "normalize_fact_state",
    "normalize_vision_fact",
    "prune_expired_facts",
    "snapshot_to_list",
]
