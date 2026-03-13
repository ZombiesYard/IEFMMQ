from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from weakref import WeakKeyDictionary

from adapters.delta_aggregator import DeltaAggregator, aggregate_delta_window, emit_delta_sanitized_event
from adapters.delta_sanitizer import DeltaPolicy, DeltaSanitizer
from adapters.dcs_bios.bios_ui_map import BiosUiMapper
from core.types import Event, Observation
from core.vars import VarResolver

TagHook = Callable[[Observation, Mapping[str, Any]], Sequence[str] | None]

DEFAULT_SELECTED_VAR_KEYS: tuple[str, ...] = (
    "battery_on",
    "ext_pwr_on",
    "l_gen_on",
    "r_gen_on",
    "power_available",
    "apu_on",
    "apu_ready",
    "apu_start_support_complete",
    "ins_mode",
    "engine_crank_right",
    "engine_crank_left",
    "engine_crank_right_complete",
    "engine_crank_left_complete",
    "bleed_air_norm",
    "bleed_air_cycle_complete",
    "left_ddi_on",
    "right_ddi_on",
    "mpcd_on",
    "hud_on",
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
    "rpm_l_gte_25",
    "rpm_l_gte_60",
    "left_engine_nominal_start_params",
    "left_engine_idle_ready",
    "throttle_r_not_off",
    "throttle_l_not_off",
    "throttle_r_idle_complete",
    "ins_mode_set",
    "ins_mode_cv_or_gnd",
    "radar_mode_opr",
    "radar_on",
    "right_engine_nominal_start_params",
    "fire_test_active",
    "fire_test_complete",
    "lights_test_active",
    "annunciator_panel_activity",
    "lights_test_complete",
    "comm1_freq_value",
    "comm2_freq_value",
    "comm1_channel_numeric",
    "comm2_channel_numeric",
    "comm1_freq_134_000",
    "ufc_comm1_pull_pressed",
    "ufc_key_1_pressed",
    "ufc_key_3_pressed",
    "ufc_key_4_pressed",
    "ufc_key_0_pressed",
    "ufc_ent_pressed",
    "ufc_scratchpad_number_display",
    "ufc_scratchpad_string_1_display",
    "ufc_scratchpad_string_2_display",
    "obogs_switch_on",
    "obogs_flow_on",
    "obogs_ready",
    "fcs_reset_pressed",
    "fcs_reset_complete",
    "flap_auto",
    "takeoff_trim_pressed",
    "takeoff_trim_set",
    "fcs_bit_switch_up",
    "probe_switch_value",
    "ext_refuel_probe_value",
    "probe_extended",
    "pitot_heat_on",
    "launch_bar_switch_value",
    "hook_handle_value",
    "parking_brake_released",
    "bingo_fuel_set",
    "standby_altimeter_set",
    "radar_altimeter_bug_value",
    "radar_altimeter_bug_set",
    "standby_attitude_uncaged",
    "attitude_source_auto",
    "vars_source_missing",
)

_DEFAULT_DELTA_POLICY: DeltaPolicy | None = None
_DEFAULT_DELTA_SANITIZER: DeltaSanitizer | None = None
_POLICY_SCOPED_SANITIZERS: OrderedDict[tuple[DeltaPolicy, str], DeltaSanitizer] = OrderedDict()
_MAX_POLICY_SCOPED_SANITIZERS = 256
_MOMENTARY_COMPLETION_KEYS: tuple[str, ...] = (
    "fire_test_complete",
    "lights_test_complete",
    "fcs_reset_complete",
    "takeoff_trim_set",
)
_COMPLETION_LATCHES: OrderedDict[str, dict[str, bool | float]] = OrderedDict()
_COMPLETION_LATCHES_LOADED = False
_MAX_COMPLETION_LATCH_STREAMS = 256
_SHARED_DELTA_SANITIZER_LOCKS: WeakKeyDictionary[DeltaSanitizer, threading.Lock] = WeakKeyDictionary()
_DEFAULT_DELTA_STREAM_ID = "__default__"
_DEFAULT_DELTA_POLICY_LOCK = threading.Lock()
_DEFAULT_DELTA_SANITIZER_LOCK = threading.Lock()
_POLICY_SCOPED_SANITIZERS_LOCK = threading.Lock()
_COMPLETION_LATCHES_LOCK = threading.Lock()
_SHARED_DELTA_SANITIZER_LOCKS_LOCK = threading.Lock()


def _default_completion_latches_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if isinstance(appdata, str) and appdata.strip():
        return Path(appdata).expanduser() / "SimTutor" / "completion_latches_v1.json"
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if isinstance(xdg_state_home, str) and xdg_state_home.strip():
        return Path(xdg_state_home).expanduser() / "simtutor" / "completion_latches_v1.json"
    return Path.home() / ".simtutor" / "state" / "completion_latches_v1.json"


_COMPLETION_LATCHES_PATH = _default_completion_latches_path()


def _is_regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return stat.S_ISREG(mode)


def _load_completion_latches_from_disk() -> None:
    global _COMPLETION_LATCHES_LOADED, _COMPLETION_LATCHES
    if _COMPLETION_LATCHES_LOADED:
        return
    loaded: OrderedDict[str, dict[str, bool | float]] = OrderedDict()
    try:
        if _COMPLETION_LATCHES_PATH.exists() and not _is_regular_file(_COMPLETION_LATCHES_PATH):
            _COMPLETION_LATCHES = loaded
            _COMPLETION_LATCHES_LOADED = True
            return
        raw = json.loads(_COMPLETION_LATCHES_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = None
    except (OSError, json.JSONDecodeError):
        raw = None
    if isinstance(raw, Mapping):
        for stream_id, state in raw.items():
            if not isinstance(stream_id, str) or not stream_id.strip() or not isinstance(state, Mapping):
                continue
            normalized_state: dict[str, bool | float] = {}
            for key in _MOMENTARY_COMPLETION_KEYS:
                value = state.get(key)
                if isinstance(value, bool):
                    normalized_state[key] = value
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    normalized_state[key] = float(value)
            if normalized_state:
                loaded[stream_id] = normalized_state
    _COMPLETION_LATCHES = loaded
    _COMPLETION_LATCHES_LOADED = True


def _save_completion_latches_to_disk() -> None:
    payload = {
        stream_id: {
            key: value
            for key, value in state.items()
            if key in _MOMENTARY_COMPLETION_KEYS and isinstance(value, (bool, int, float))
        }
        for stream_id, state in _COMPLETION_LATCHES.items()
        if isinstance(stream_id, str) and stream_id and isinstance(state, Mapping)
    }
    temp_path: Path | None = None
    try:
        if _COMPLETION_LATCHES_PATH.exists() and not _is_regular_file(_COMPLETION_LATCHES_PATH):
            return
        _COMPLETION_LATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=_COMPLETION_LATCHES_PATH.parent,
            prefix=".completion_latches_",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, _COMPLETION_LATCHES_PATH)
    except OSError:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return


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


def _get_default_delta_policy() -> DeltaPolicy:
    global _DEFAULT_DELTA_POLICY
    with _DEFAULT_DELTA_POLICY_LOCK:
        if _DEFAULT_DELTA_POLICY is None:
            _DEFAULT_DELTA_POLICY = DeltaPolicy.from_yaml()
    return _DEFAULT_DELTA_POLICY


def _normalize_delta_stream_id(stream_id: str | None) -> str:
    if isinstance(stream_id, str):
        normalized = stream_id.strip()
        if normalized:
            return normalized
    return _DEFAULT_DELTA_STREAM_ID


def _get_shared_sanitizer_lock(sanitizer: DeltaSanitizer) -> threading.Lock:
    with _SHARED_DELTA_SANITIZER_LOCKS_LOCK:
        lock = _SHARED_DELTA_SANITIZER_LOCKS.get(sanitizer)
        if lock is None:
            lock = threading.Lock()
            _SHARED_DELTA_SANITIZER_LOCKS[sanitizer] = lock
    return lock


def _get_default_delta_sanitizer(*, stream_id: str) -> tuple[DeltaSanitizer, threading.Lock]:
    if stream_id != _DEFAULT_DELTA_STREAM_ID:
        return _get_policy_scoped_sanitizer(_get_default_delta_policy(), stream_id=stream_id)

    global _DEFAULT_DELTA_SANITIZER
    with _DEFAULT_DELTA_SANITIZER_LOCK:
        if _DEFAULT_DELTA_SANITIZER is None:
            _DEFAULT_DELTA_SANITIZER = DeltaSanitizer(_get_default_delta_policy())
        sanitizer = _DEFAULT_DELTA_SANITIZER
    return sanitizer, _get_shared_sanitizer_lock(sanitizer)


def _get_policy_scoped_sanitizer(policy: DeltaPolicy, *, stream_id: str) -> tuple[DeltaSanitizer, threading.Lock]:
    cache_key = (policy, _normalize_delta_stream_id(stream_id))
    with _POLICY_SCOPED_SANITIZERS_LOCK:
        sanitizer = _POLICY_SCOPED_SANITIZERS.get(cache_key)
        if sanitizer is None:
            sanitizer = DeltaSanitizer(policy)
            _POLICY_SCOPED_SANITIZERS[cache_key] = sanitizer
        _POLICY_SCOPED_SANITIZERS.move_to_end(cache_key)
        while len(_POLICY_SCOPED_SANITIZERS) > max(1, int(_MAX_POLICY_SCOPED_SANITIZERS)):
            _, evicted = _POLICY_SCOPED_SANITIZERS.popitem(last=False)
            with _SHARED_DELTA_SANITIZER_LOCKS_LOCK:
                _SHARED_DELTA_SANITIZER_LOCKS.pop(evicted, None)
    return sanitizer, _get_shared_sanitizer_lock(sanitizer)


def _resolve_delta_stream_id(obs: Observation, explicit_stream_id: str | None) -> str:
    if isinstance(explicit_stream_id, str):
        normalized = explicit_stream_id.strip()
        if normalized:
            return normalized

    metadata = obs.metadata if isinstance(obs.metadata, Mapping) else {}
    for key in ("session_id", "stream_id"):
        raw = metadata.get(key)
        if isinstance(raw, str):
            normalized = raw.strip()
            if normalized:
                return normalized
    return _DEFAULT_DELTA_STREAM_ID


def _apply_momentary_completion_latches(
    resolved_vars: Mapping[str, Any],
    *,
    stream_id: str,
    t_wall: float | None,
) -> dict[str, Any]:
    out = dict(resolved_vars)
    if t_wall is None:
        return out

    normalized_stream_id = _normalize_delta_stream_id(stream_id)
    now_s = float(t_wall)
    trigger_sources = {
        "fire_test_complete": bool(resolved_vars.get("fire_test_active")),
        "lights_test_complete": bool(resolved_vars.get("lights_test_active")),
        "fcs_reset_complete": bool(resolved_vars.get("fcs_reset_pressed")),
        "takeoff_trim_set": bool(resolved_vars.get("takeoff_trim_pressed")),
    }
    raw_missing = resolved_vars.get("vars_source_missing")
    missing_sources = {
        item for item in raw_missing if isinstance(item, str) and item
    } if isinstance(raw_missing, list) else set()
    reset_session_completion = (
        "battery_on" not in missing_sources and resolved_vars.get("battery_on") is False
    )

    with _COMPLETION_LATCHES_LOCK:
        _load_completion_latches_from_disk()
        previous_state = dict(_COMPLETION_LATCHES.get(normalized_stream_id, {}))
        state = dict(previous_state)
        state_changed = False
        if reset_session_completion:
            state.clear()
            state_changed = True
        for key in _MOMENTARY_COMPLETION_KEYS:
            stored = state.get(key)
            if trigger_sources.get(key):
                if state.get(key) is not True:
                    state[key] = True
                    state_changed = True
                out[key] = True
                continue
            if stored is True:
                out[key] = True
            elif isinstance(stored, (int, float)) and stored >= now_s:
                out[key] = True
            else:
                if key in state:
                    state.pop(key, None)
                    state_changed = True

        if state:
            _COMPLETION_LATCHES[normalized_stream_id] = state
            _COMPLETION_LATCHES.move_to_end(normalized_stream_id)
            if previous_state != state:
                state_changed = True
        else:
            if normalized_stream_id in _COMPLETION_LATCHES:
                _COMPLETION_LATCHES.pop(normalized_stream_id, None)
                state_changed = True

        while len(_COMPLETION_LATCHES) > max(1, int(_MAX_COMPLETION_LATCH_STREAMS)):
            _COMPLETION_LATCHES.popitem(last=False)
            state_changed = True

        if state_changed:
            _save_completion_latches_to_disk()

    raw_missing = out.get("vars_source_missing")
    if isinstance(raw_missing, list):
        missing_keys = [item for item in raw_missing if isinstance(item, str) and item]
        if missing_keys:
            latched_true = {
                key
                for key in _MOMENTARY_COMPLETION_KEYS
                if bool(out.get(key))
            }
            if latched_true:
                out["vars_source_missing"] = [key for key in missing_keys if key not in latched_true]

    return out


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
    selected = list(DEFAULT_SELECTED_VAR_KEYS) if keys is None else list(keys)
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
    sample_limit = max(0, max_keys)
    sample: list[str] = []
    valid_count = 0
    for key in delta.keys():
        if not isinstance(key, str) or not key:
            continue
        valid_count += 1
        if len(sample) < sample_limit:
            sample.append(key)
    return {
        "delta_count": valid_count,
        "changed_keys_sample": sample,
        "truncated": valid_count > len(sample),
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


def _normalize_hook_tags(raw: Any) -> Sequence[str]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes, bytearray)):
        raise TypeError("tag_hook must return a sequence of strings, not a string/bytes value")
    if not isinstance(raw, Sequence):
        raise TypeError(f"tag_hook must return a sequence of strings or None, got {type(raw).__name__}")
    return raw


def _resolve_delta_policy_and_sanitizer(
    *,
    delta_policy: DeltaPolicy | None,
    delta_sanitizer: DeltaSanitizer | None,
    delta_aggregator: DeltaAggregator | None,
    delta_stream_id: str,
) -> tuple[DeltaPolicy, DeltaSanitizer, threading.Lock]:
    if delta_sanitizer is not None:
        policy = delta_sanitizer.policy
        if delta_policy is not None and delta_policy != policy:
            raise ValueError("delta_policy does not match delta_sanitizer.policy")
        sanitizer = delta_sanitizer
        sanitizer_lock = _get_shared_sanitizer_lock(sanitizer)
    elif delta_policy is not None:
        policy = delta_policy
        # Reuse a sanitizer per policy/stream so debounce state persists per stream.
        sanitizer, sanitizer_lock = _get_policy_scoped_sanitizer(policy, stream_id=delta_stream_id)
    else:
        sanitizer, sanitizer_lock = _get_default_delta_sanitizer(stream_id=delta_stream_id)
        policy = sanitizer.policy

    if delta_aggregator is not None and delta_aggregator.policy != policy:
        raise ValueError("delta_aggregator.policy does not match effective delta policy")

    return policy, sanitizer, sanitizer_lock


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
    include_bios_hash: bool | None = None,
    delta_policy: DeltaPolicy | None = None,
    delta_sanitizer: DeltaSanitizer | None = None,
    delta_aggregator: DeltaAggregator | None = None,
    delta_stream_id: str | None = None,
    delta_event_sink: Callable[[Event], None] | None = None,
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
    BIOS hash calculation is optional; by default it is enabled when `debug_cache`
    is provided, or can be forced with `include_bios_hash=True`.
    Delta sanitizer behavior:
    - If `delta_sanitizer` is provided, that instance is used directly and guarded
      with a per-instance shared lock inside this module.
    - Otherwise, module-level cached sanitizers are reused and state is preserved.
      Cache scope is policy + effective stream id.
      Effective stream id is `delta_stream_id` if provided, else
      `obs.metadata.session_id` / `obs.metadata.stream_id`, else `"__default__"`.
    - Cached sanitizer creation/use is lock-protected for thread safety.
      Policy-scoped cache is LRU-bounded by `_MAX_POLICY_SCOPED_SANITIZERS`.
    To force stateless sanitization, pass a freshly created `DeltaSanitizer`
    per call (and do not reuse it across observations).
    """
    payload = obs.payload if isinstance(obs.payload, MutableMapping) else {}
    bios = payload.get("bios")
    delta = payload.get("delta")

    bios_map = bios if isinstance(bios, Mapping) else {}
    delta_map = delta if isinstance(delta, Mapping) else {}

    resolved_stream_id = _resolve_delta_stream_id(obs, delta_stream_id)
    policy, sanitizer, sanitizer_lock = _resolve_delta_policy_and_sanitizer(
        delta_policy=delta_policy,
        delta_sanitizer=delta_sanitizer,
        delta_aggregator=delta_aggregator,
        delta_stream_id=resolved_stream_id,
    )

    seq = _as_int(payload.get("seq"))
    t_wall = _as_float(payload.get("t_wall"))
    resolved_vars = _apply_momentary_completion_latches(
        resolver.resolve(payload),
        stream_id=resolved_stream_id,
        t_wall=t_wall,
    )
    selected_vars = _select_vars(resolved_vars, selected_var_keys)
    with sanitizer_lock:
        sanitized = sanitizer.sanitize_delta(delta_map, t_wall=t_wall, seq=seq)
    per_obs_summary = aggregate_delta_window([sanitized], policy=policy, mapper=mapper)

    if delta_aggregator is not None:
        window_summary = delta_aggregator.add(sanitized)
    else:
        window_summary = per_obs_summary

    if delta_event_sink is not None:
        emit_delta_sanitized_event(window_summary, related_id=obs.observation_id, event_sink=delta_event_sink)

    recent_ui_targets = window_summary.recent_ui_targets[: max(0, max_recent_ui_targets)]
    fallback_summary = _build_delta_summary(sanitized.kept, max_delta_keys)
    delta_summary = {
        **fallback_summary,
        "delta_count": sanitized.kept_count,
        "raw_delta_count": sanitized.raw_count,
        "dropped_stats": per_obs_summary.dropped_stats,
        "recent_key_changes_topk": per_obs_summary.recent_key_changes_topk,
    }

    compact_payload = {
        "seq": seq,
        "t_wall": t_wall,
        "vars": selected_vars,
        "delta_summary": delta_summary,
        "recent_ui_targets": recent_ui_targets,
    }
    if delta_aggregator is not None:
        compact_payload["delta_window_summary"] = {
            "window_size": window_summary.window_size,
            "raw_changes_total": window_summary.raw_changes_total,
            "kept_changes_total": window_summary.kept_changes_total,
            "dropped_stats": window_summary.dropped_stats,
            "recent_key_changes_topk": window_summary.recent_key_changes_topk,
        }

    metadata = dict(obs.metadata or {})
    if "seq" not in metadata and compact_payload["seq"] is not None:
        metadata["seq"] = compact_payload["seq"]
    if "delta_count" not in metadata:
        metadata["delta_count"] = sanitized.kept_count
    if "raw_delta_count" not in metadata:
        metadata["raw_delta_count"] = sanitized.raw_count
    if "delta_dropped_count" not in metadata:
        metadata["delta_dropped_count"] = sanitized.dropped_count

    should_hash = include_bios_hash if include_bios_hash is not None else debug_cache is not None
    bios_hash: str | None = None
    if should_hash:
        bios_hash = _stable_hash_mapping(bios_map)
        metadata["bios_hash"] = bios_hash

    if debug_cache is not None:
        debug_cache.last_seq = compact_payload["seq"]
        debug_cache.last_bios = dict(bios_map)
        debug_cache.last_delta = dict(delta_map)
        debug_cache.last_bios_hash = bios_hash

    hook_tags = tag_hook(obs, compact_payload) if tag_hook else ()
    extra_tags = _normalize_hook_tags(hook_tags)
    merged_tags = _merge_tags(obs.tags, extra_tags)

    return Observation(
        observation_id=obs.observation_id,
        timestamp=obs.timestamp,
        source=obs.source,
        payload=compact_payload,
        version=obs.version,
        procedure_hint=obs.procedure_hint,
        tags=merged_tags,
        attachments=list(obs.attachments),
        metadata=metadata,
    )


__all__ = [
    "DEFAULT_SELECTED_VAR_KEYS",
    "TagHook",
    "TelemetryDebugCache",
    "enrich_bios_observation",
]
