"""
Canonical step registry loader and validator.

The FA-18C startup step text (official wording + short explanation) must come
from a single machine-readable source: `step_registry.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml

_STEP_ID_RE = re.compile(r"^S(\d{2})$")


class StepRegistryError(ValueError):
    """Raised when canonical step registry data is missing or invalid."""


@dataclass(frozen=True)
class CanonicalStep:
    step_id: str
    phase: str
    phase_label: str | None
    official_description: str
    short_explanation: str
    cockpit_area: str | None
    source_chunk_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.step_id,
            "step_id": self.step_id,
            "phase": self.phase,
            "official_description": self.official_description,
            "short_explanation": self.short_explanation,
            "source_chunk_refs": list(self.source_chunk_refs),
        }
        if self.phase_label is not None:
            out["phase_label"] = self.phase_label
        if self.cockpit_area is not None:
            out["cockpit_area"] = self.cockpit_area
        return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_step_registry_path(pack_path: str | Path | None = None) -> Path:
    if pack_path is None:
        return _repo_root() / "packs" / "fa18c_startup" / "step_registry.yaml"
    return Path(pack_path).resolve().parent / "step_registry.yaml"


def _require_non_empty_str(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StepRegistryError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalize_source_refs(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise StepRegistryError(f"{field_name} must be a non-empty list")
    refs: list[str] = []
    for idx, item in enumerate(value):
        ref = _require_non_empty_str(item, field_name=f"{field_name}[{idx}]")
        refs.append(ref)
    return tuple(refs)


def _validate_step_id_sequence(step_ids: list[str], *, expected_count: int | None) -> None:
    if not step_ids:
        raise StepRegistryError("steps must be non-empty")

    for idx, step_id in enumerate(step_ids):
        if not _STEP_ID_RE.fullmatch(step_id):
            raise StepRegistryError(
                f"steps[{idx}].step_id must match SNN (for example S01), got: {step_id!r}"
            )

    if len(step_ids) != len(set(step_ids)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for step_id in step_ids:
            if step_id in seen and step_id not in duplicates:
                duplicates.append(step_id)
            seen.add(step_id)
        raise StepRegistryError(f"duplicate step_id entries: {duplicates}")

    numeric_ids = [int(step_id[1:]) for step_id in step_ids]
    if expected_count is not None and len(step_ids) != expected_count:
        raise StepRegistryError(
            f"step registry must contain exactly {expected_count} steps, got {len(step_ids)}"
        )

    expected_numbers = list(range(1, len(step_ids) + 1))
    if numeric_ids != expected_numbers:
        expected_ids = [f"S{num:02d}" for num in expected_numbers]
        raise StepRegistryError(
            "step_id sequence must be continuous and ordered from S01 with no gaps; "
            f"expected {expected_ids[0]}..{expected_ids[-1]}, got {step_ids[0]}..{step_ids[-1]}"
        )


def _parse_step(raw_step: Any, *, index: int) -> CanonicalStep:
    if not isinstance(raw_step, dict):
        raise StepRegistryError(f"steps[{index}] must be a mapping")

    step_id = _require_non_empty_str(raw_step.get("step_id"), field_name=f"steps[{index}].step_id")
    phase = _require_non_empty_str(raw_step.get("phase"), field_name=f"steps[{index}].phase")
    official_description = _require_non_empty_str(
        raw_step.get("official_description"),
        field_name=f"steps[{index}].official_description",
    )
    short_explanation = _require_non_empty_str(
        raw_step.get("short_explanation"),
        field_name=f"steps[{index}].short_explanation",
    )
    source_chunk_refs = _normalize_source_refs(
        raw_step.get("source_chunk_refs"),
        field_name=f"steps[{index}].source_chunk_refs",
    )

    phase_label_raw = raw_step.get("phase_label")
    phase_label = (
        _require_non_empty_str(phase_label_raw, field_name=f"steps[{index}].phase_label")
        if phase_label_raw is not None
        else None
    )
    cockpit_area_raw = raw_step.get("cockpit_area")
    cockpit_area = (
        _require_non_empty_str(cockpit_area_raw, field_name=f"steps[{index}].cockpit_area")
        if cockpit_area_raw is not None
        else None
    )

    return CanonicalStep(
        step_id=step_id,
        phase=phase,
        phase_label=phase_label,
        official_description=official_description,
        short_explanation=short_explanation,
        cockpit_area=cockpit_area,
        source_chunk_refs=source_chunk_refs,
    )


@lru_cache(maxsize=8)
def _load_registry_cached(resolved_path: str, expected_count: int | None) -> tuple[CanonicalStep, ...]:
    path = Path(resolved_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StepRegistryError(f"step registry not found: {path}") from exc
    except OSError as exc:
        raise StepRegistryError(f"failed to read step registry: {path}") from exc
    except yaml.YAMLError as exc:
        raise StepRegistryError(f"step registry contains invalid YAML: {path}") from exc

    if not isinstance(raw, dict):
        raise StepRegistryError(f"step registry root must be a mapping: {path}")

    _require_non_empty_str(raw.get("schema_version"), field_name="schema_version")
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise StepRegistryError("steps must be a non-empty list")

    steps = tuple(_parse_step(raw_step, index=idx) for idx, raw_step in enumerate(steps_raw))
    _validate_step_id_sequence([step.step_id for step in steps], expected_count=expected_count)
    return steps


def load_step_registry(
    path: str | Path | None = None,
    *,
    expected_count: int | None = 25,
) -> list[CanonicalStep]:
    registry_path = Path(path).resolve() if path is not None else default_step_registry_path().resolve()
    return list(_load_registry_cached(str(registry_path), expected_count))


def load_step_registry_dicts(
    path: str | Path | None = None,
    *,
    expected_count: int | None = 25,
) -> list[dict[str, Any]]:
    return [step.to_dict() for step in load_step_registry(path, expected_count=expected_count)]


def load_step_ids(
    path: str | Path | None = None,
    *,
    expected_count: int | None = 25,
) -> list[str]:
    return [step.step_id for step in load_step_registry(path, expected_count=expected_count)]


__all__ = [
    "CanonicalStep",
    "StepRegistryError",
    "default_step_registry_path",
    "load_step_ids",
    "load_step_registry",
    "load_step_registry_dicts",
]
