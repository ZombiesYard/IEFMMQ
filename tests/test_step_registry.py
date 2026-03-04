from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.step_registry import (
    StepRegistryError,
    default_step_registry_path,
    load_step_registry,
)


def _write_registry(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "step_registry.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_default_step_registry_contains_full_s01_to_s25() -> None:
    path = default_step_registry_path()
    steps = load_step_registry(path)
    assert len(steps) == 25
    assert [step.step_id for step in steps] == [f"S{i:02d}" for i in range(1, 26)]
    assert all(step.source_chunk_refs for step in steps)


def test_step_registry_rejects_duplicate_step_ids(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        {
            "schema_version": "v1",
            "steps": [
                {
                    "step_id": "S01",
                    "phase": "P1",
                    "official_description": "A",
                    "short_explanation": "a",
                    "source_chunk_refs": ["doc/chunk:1-1"],
                },
                {
                    "step_id": "S01",
                    "phase": "P1",
                    "official_description": "B",
                    "short_explanation": "b",
                    "source_chunk_refs": ["doc/chunk:2-2"],
                },
            ],
        },
    )

    with pytest.raises(StepRegistryError, match="duplicate step_id"):
        load_step_registry(path, expected_count=None)


def test_step_registry_rejects_non_continuous_sequence(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path,
        {
            "schema_version": "v1",
            "steps": [
                {
                    "step_id": "S01",
                    "phase": "P1",
                    "official_description": "A",
                    "short_explanation": "a",
                    "source_chunk_refs": ["doc/chunk:1-1"],
                },
                {
                    "step_id": "S03",
                    "phase": "P1",
                    "official_description": "B",
                    "short_explanation": "b",
                    "source_chunk_refs": ["doc/chunk:2-2"],
                },
            ],
        },
    )

    with pytest.raises(StepRegistryError, match="continuous and ordered from S01"):
        load_step_registry(path, expected_count=None)
