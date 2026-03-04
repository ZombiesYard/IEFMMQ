"""
Deterministic startup step inference for model fallback and prompt hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from core.step_registry import StepRegistryError, default_step_registry_path, load_step_registry_dicts

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PACK_PATH = _REPO_ROOT / "packs" / "fa18c_startup" / "pack.yaml"
_MAX_MISSING_CONDITIONS = 8
_MAX_RECENT_UI_TARGETS = 12


@dataclass(frozen=True)
class StepInferenceResult:
    inferred_step_id: str | None
    missing_conditions: tuple[str, ...]


def load_pack_steps(pack_path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Load procedure steps from canonical step registry when available.
    Falls back to pack.yaml for compatibility.

    Returns an empty list when file/content is invalid so fallback remains safe.
    """
    path = Path(pack_path) if pack_path else _DEFAULT_PACK_PATH
    registry_path = default_step_registry_path(path)
    cached_registry_steps = _load_registry_steps_cached(str(registry_path.resolve()))
    if cached_registry_steps:
        return [dict(step) for step in cached_registry_steps]

    cached_steps = _load_pack_steps_cached(str(path.resolve()))
    return [dict(step) for step in cached_steps]


@lru_cache(maxsize=8)
def _load_registry_steps_cached(resolved_registry_path: str) -> tuple[dict[str, Any], ...]:
    try:
        entries = load_step_registry_dicts(Path(resolved_registry_path), expected_count=25)
    except (StepRegistryError, OSError, ValueError):
        return ()
    return tuple(dict(step) for step in entries)


@lru_cache(maxsize=8)
def _load_pack_steps_cached(resolved_pack_path: str) -> tuple[dict[str, Any], ...]:
    path = Path(resolved_pack_path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return ()
    if not isinstance(data, Mapping):
        return ()
    steps = data.get("steps")
    if not isinstance(steps, list):
        return ()
    out: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, Mapping):
            out.append(dict(step))
    return tuple(out)


def normalize_recent_ui_targets(raw: Any, *, max_items: int = _MAX_RECENT_UI_TARGETS) -> list[str]:
    candidates: list[Any] = []
    if isinstance(raw, list):
        candidates = list(raw)
    elif isinstance(raw, tuple):
        candidates = list(raw)
    elif isinstance(raw, Mapping):
        value = raw.get("recent_buttons")
        if isinstance(value, (list, tuple)):
            candidates = list(value)
        elif isinstance(value, str):
            candidates = [value]

    out: list[str] = []
    seen: set[str] = set()
    cap = max(0, int(max_items))
    for item in candidates:
        if not isinstance(item, str) or not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= cap:
            break
    return out


def extract_recent_ui_targets(
    context: Mapping[str, Any] | None,
    *,
    max_items: int = _MAX_RECENT_UI_TARGETS,
) -> list[str]:
    if not isinstance(context, Mapping):
        return []

    direct = normalize_recent_ui_targets(context.get("recent_ui_targets"), max_items=max_items)
    if direct:
        return direct

    signal = normalize_recent_ui_targets(context.get("recent_actions"), max_items=max_items)
    if signal:
        return signal

    raw_recent_actions = context.get("recent_actions")
    if isinstance(raw_recent_actions, list):
        extracted: list[str] = []
        for item in raw_recent_actions:
            if not isinstance(item, Mapping):
                continue
            for key in ("ui_target", "mapped_ui_target", "target"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    extracted.append(value)
            ui_targets = item.get("ui_targets")
            if isinstance(ui_targets, (list, tuple)):
                extracted.extend(t for t in ui_targets if isinstance(t, str) and t)
        signal = normalize_recent_ui_targets(extracted, max_items=max_items)
        if signal:
            return signal

    raw_recent_deltas = context.get("recent_deltas")
    if isinstance(raw_recent_deltas, list):
        extracted = []
        for item in raw_recent_deltas:
            if not isinstance(item, Mapping):
                continue
            for key in ("ui_target", "mapped_ui_target", "target"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    extracted.append(value)
        return normalize_recent_ui_targets(extracted, max_items=max_items)
    return []


def infer_step_id(
    pack_steps: Sequence[Mapping[str, Any]],
    vars_map: Mapping[str, Any] | None,
    recent_ui_targets: Sequence[str] | None,
) -> StepInferenceResult:
    """
    Infer likely blocked step for deterministic help fallback.

    Current heuristics prioritize S01-S06 startup gates:
    power -> fire test -> APU -> ENG CRANK -> RPM 25 -> RPM 60/BLEED AIR.
    """
    step_ids = _ordered_step_ids(pack_steps)
    if not step_ids:
        return StepInferenceResult(inferred_step_id=None, missing_conditions=())

    s01 = _pick_step_id(step_ids, "S01", 0)
    s02 = _pick_step_id(step_ids, "S02", 1)
    s03 = _pick_step_id(step_ids, "S03", 2)
    s04 = _pick_step_id(step_ids, "S04", 3)
    s05 = _pick_step_id(step_ids, "S05", 4)
    s06 = _pick_step_id(step_ids, "S06", 5)
    s07 = _pick_step_id(step_ids, "S07", 6)

    vars_safe = vars_map if isinstance(vars_map, Mapping) else {}
    recent = normalize_recent_ui_targets(list(recent_ui_targets or []))
    recent_set = set(recent)

    power_available = _as_bool(vars_safe.get("power_available"))
    battery_on = _as_bool(vars_safe.get("battery_on"))
    l_gen_on = _as_bool(vars_safe.get("l_gen_on"))
    r_gen_on = _as_bool(vars_safe.get("r_gen_on"))
    fire_test_active = _as_bool(vars_safe.get("fire_test_active"))
    apu_on = _as_bool(vars_safe.get("apu_on"))
    apu_ready = _as_bool(vars_safe.get("apu_ready"))
    engine_crank_right = _as_bool(vars_safe.get("engine_crank_right"))
    bleed_air_norm = _as_bool(vars_safe.get("bleed_air_norm"))

    rpm_r = _as_float(vars_safe.get("rpm_r"))
    rpm_r_gte_25 = _as_bool(vars_safe.get("rpm_r_gte_25"))
    rpm_r_gte_60 = _as_bool(vars_safe.get("rpm_r_gte_60"))
    throttle_r_not_off = _as_bool(vars_safe.get("throttle_r_not_off"))
    has_throttle_r_not_off = "throttle_r_not_off" in vars_safe
    if rpm_r_gte_25 is None and rpm_r is not None:
        rpm_r_gte_25 = rpm_r >= 25.0
    if rpm_r_gte_60 is None and rpm_r is not None:
        rpm_r_gte_60 = rpm_r >= 60.0

    blocked_power = power_available is False or battery_on is False or l_gen_on is False or r_gen_on is False
    if blocked_power:
        return _result(
            s01,
            [
                "vars.power_available==true",
                "vars.battery_on==true",
                "vars.l_gen_on==true",
                "vars.r_gen_on==true",
            ],
            actual={
                "vars.power_available==true": power_available,
                "vars.battery_on==true": battery_on,
                "vars.l_gen_on==true": l_gen_on,
                "vars.r_gen_on==true": r_gen_on,
            },
        )

    engine_progressed = engine_crank_right is True or (rpm_r is not None and rpm_r >= 5.0)
    apu_progressed = apu_on is True or apu_ready is True or "apu_switch" in recent_set
    fire_test_recent = "fire_test_switch" in recent_set

    if not apu_progressed and not engine_progressed:
        if fire_test_active is True:
            return _result(s02, ["vars.fire_test_active==false (complete FIRE TEST A/B)"])
        if not fire_test_recent:
            return _result(s02, ["recent_ui_targets has fire_test_switch"])

    if apu_ready is not True and not engine_progressed:
        missing = []
        if apu_on is not True and "apu_switch" not in recent_set:
            missing.append("vars.apu_on==true")
        if apu_ready is not True:
            missing.append("vars.apu_ready==true")
        if not missing:
            missing.append("wait_for_apu_ready")
        return _result(s03, missing)

    if engine_crank_right is not True and (rpm_r is None or rpm_r < 20.0):
        missing = ["vars.engine_crank_right==true"]
        if "eng_crank_switch" not in recent_set:
            missing.append("recent_ui_targets has eng_crank_switch")
        return _result(s04, missing)

    if rpm_r_gte_25 is not True:
        missing = ["vars.rpm_r>=25"]
        if engine_crank_right is not True:
            missing.insert(0, "vars.engine_crank_right==true")
        return _result(s05, missing)
    # Keep S05 inference aligned with pack gate semantics: once the key exists,
    # any non-true value (False/None/unparseable) means the gate is still unmet.
    # Missing key is tolerated because not every telemetry slice carries this signal.
    if has_throttle_r_not_off and throttle_r_not_off is not True:
        return _result(s05, ["vars.throttle_r_not_off==true"])

    if rpm_r_gte_60 is not True:
        return _result(s06, ["vars.rpm_r>=60"])

    bleed_recent = "bleed_air_knob" in recent_set
    if not bleed_recent or bleed_air_norm is not True:
        missing = []
        if not bleed_recent:
            missing.append("recent_ui_targets has bleed_air_knob")
        if bleed_air_norm is not True:
            missing.append("vars.bleed_air_norm==true")
        if not missing:
            missing.append("bleed_air_cycle_complete")
        return _result(s06, missing)

    return _result(s07, [])


def _ordered_step_ids(pack_steps: Sequence[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for step in pack_steps:
        step_id = step.get("id") if isinstance(step, Mapping) else None
        if not isinstance(step_id, str) or not step_id:
            continue
        if step_id in seen:
            continue
        seen.add(step_id)
        out.append(step_id)
    return out


def _pick_step_id(step_ids: Sequence[str], preferred: str, fallback_index: int) -> str | None:
    if preferred in step_ids:
        return preferred
    if 0 <= fallback_index < len(step_ids):
        return step_ids[fallback_index]
    if step_ids:
        return step_ids[-1]
    return None


def _result(
    inferred_step_id: str | None,
    missing_conditions: Sequence[str],
    *,
    actual: Mapping[str, bool | None] | None = None,
) -> StepInferenceResult:
    filtered: list[str] = []
    for item in missing_conditions:
        if not isinstance(item, str) or not item:
            continue
        if actual is not None and item in actual and actual[item] is True:
            continue
        filtered.append(item)
        if len(filtered) >= _MAX_MISSING_CONDITIONS:
            break
    return StepInferenceResult(inferred_step_id=inferred_step_id, missing_conditions=tuple(filtered))


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


__all__ = [
    "StepInferenceResult",
    "extract_recent_ui_targets",
    "infer_step_id",
    "load_pack_steps",
    "normalize_recent_ui_targets",
]
