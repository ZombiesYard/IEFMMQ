from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from adapters.dcs_bios.bios_ui_map import BiosUiMapper
from core.types import Observation
from core.vars import VarResolver

TagHook = Callable[[Observation, Mapping[str, Any]], Sequence[str]]

DEFAULT_SELECTED_VAR_KEYS: tuple[str, ...] = (
    "battery_on",
    "ext_pwr_on",
    "l_gen_on",
    "r_gen_on",
    "power_available",
    "apu_on",
    "apu_ready",
    "engine_crank_right",
    "engine_crank_left",
    "bleed_air_norm",
    "rpm_r",
    "rpm_l",
    "temp_r",
    "temp_l",
    "ff_r",
    "ff_l",
    "oil_r",
    "oil_l",
    "noz_r",
    "noz_l",
    "rpm_r_gte_25",
    "rpm_r_gte_60",
    "rpm_l_gte_60",
    "right_engine_nominal_start_params",
    "vars_source_missing",
)


@dataclass
class TelemetryDebugCache:
    """In-memory cache for raw bios data during live runs (not logged by default)."""

    last_seq: int | None = None
    last_bios: dict[str, Any] = field(default_factory=dict)
    last_delta: dict[str, Any] = field(default_factory=dict)
    last_bios_hash: str | None = None


def _stable_hash_mapping(data: Mapping[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _select_vars(resolved: Mapping[str, Any], keys: Sequence[str] | None) -> dict[str, Any]:
    selected = list(keys) if keys else list(DEFAULT_SELECTED_VAR_KEYS)
    out: dict[str, Any] = {}
    for key in selected:
        if key in resolved:
            out[key] = resolved.get(key)
        elif key == "vars_source_missing":
            out[key] = []
        else:
            out[key] = None
    return out


def _build_delta_summary(delta: Mapping[str, Any], max_keys: int) -> dict[str, Any]:
    keys = [key for key in delta.keys() if isinstance(key, str) and key]
    sample = keys[: max(0, max_keys)]
    return {
        "delta_count": len(keys),
        "changed_keys_sample": sample,
        "truncated": len(sample) < len(keys),
    }


def _merge_tags(existing: Sequence[str], extra: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(existing) + list(extra):
        if not isinstance(raw, str) or not raw:
            continue
        if raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
    return out


def enrich_bios_observation(
    obs: Observation,
    resolver: VarResolver,
    *,
    mapper: BiosUiMapper | None = None,
    selected_var_keys: Sequence[str] | None = None,
    max_delta_keys: int = 8,
    max_recent_ui_targets: int = 8,
    tag_hook: TagHook | None = None,
    debug_cache: TelemetryDebugCache | None = None,
) -> Observation:
    """
    Enrich a DCS-BIOS observation into a compact payload for tutor/prompt pipeline.

    Output payload keeps only:
    - seq / t_wall
    - vars (selected stable vars)
    - delta_summary (small)
    - recent_ui_targets (small)

    Large raw bios state is excluded from payload and can optionally be cached
    in-memory for debug by passing `debug_cache`.
    """
    payload = obs.payload if isinstance(obs.payload, MutableMapping) else {}
    bios = payload.get("bios")
    delta = payload.get("delta")

    bios_map = bios if isinstance(bios, Mapping) else {}
    delta_map = delta if isinstance(delta, Mapping) else {}

    resolved_vars = resolver.resolve(payload)
    selected_vars = _select_vars(resolved_vars, selected_var_keys)

    mapped_targets = mapper.map_delta(delta_map) if mapper else []
    recent_ui_targets = mapped_targets[: max(0, max_recent_ui_targets)]
    delta_summary = _build_delta_summary(delta_map, max_delta_keys)

    compact_payload = {
        "seq": _as_int(payload.get("seq")),
        "t_wall": _as_float(payload.get("t_wall")),
        "vars": selected_vars,
        "delta_summary": delta_summary,
        "recent_ui_targets": recent_ui_targets,
    }

    metadata = dict(obs.metadata or {})
    if "seq" not in metadata and compact_payload["seq"] is not None:
        metadata["seq"] = compact_payload["seq"]
    if "delta_count" not in metadata:
        metadata["delta_count"] = delta_summary["delta_count"]

    bios_hash = _stable_hash_mapping(bios_map)
    metadata["bios_hash"] = bios_hash

    if debug_cache is not None:
        debug_cache.last_seq = compact_payload["seq"]
        debug_cache.last_bios = dict(bios_map)
        debug_cache.last_delta = dict(delta_map)
        debug_cache.last_bios_hash = bios_hash

    extra_tags = tag_hook(obs, compact_payload) if tag_hook else ()
    merged_tags = _merge_tags(obs.tags, extra_tags)

    enriched = Observation(
        source=obs.source,
        payload=compact_payload,
        procedure_hint=obs.procedure_hint,
        tags=merged_tags,
        attachments=list(obs.attachments),
        metadata=metadata,
    )
    enriched.observation_id = obs.observation_id
    enriched.timestamp = obs.timestamp
    enriched.version = obs.version
    return enriched


__all__ = [
    "DEFAULT_SELECTED_VAR_KEYS",
    "TagHook",
    "TelemetryDebugCache",
    "enrich_bios_observation",
]
