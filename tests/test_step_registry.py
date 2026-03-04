from __future__ import annotations

import os
from pathlib import Path
import time

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


def _bump_mtime(path: Path) -> None:
    now = time.time()
    current = path.stat().st_mtime
    bumped = max(current + 2.0, now + 2.0)
    os.utime(path, (bumped, bumped))


def _single_step_payload(*, schema_version: str = "v1", short_explanation: str = "a") -> dict:
    return {
        "schema_version": schema_version,
        "steps": [
            {
                "step_id": "S01",
                "phase": "P1",
                "official_description": "A",
                "short_explanation": short_explanation,
                "source_chunk_refs": ["doc/chunk:1-1"],
            }
        ],
    }


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


def test_step_registry_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, _single_step_payload(schema_version="v2"))

    with pytest.raises(StepRegistryError, match="unsupported schema_version"):
        load_step_registry(path, expected_count=None)


def test_step_registry_cache_invalidates_when_file_changes(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, _single_step_payload(short_explanation="a"))
    first = load_step_registry(path, expected_count=None)
    assert first[0].short_explanation == "a"

    path.write_text(
        yaml.safe_dump(_single_step_payload(short_explanation="updated"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _bump_mtime(path)
    second = load_step_registry(path, expected_count=None)
    assert second[0].short_explanation == "updated"


def test_default_step_registry_path_reads_pack_metadata_override(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack_dir"
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack_path = pack_dir / "pack.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "pack_id": "test_pack",
                "metadata": {"step_registry_path": "configs/registry.yaml"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    expected = (pack_dir / "configs" / "registry.yaml").resolve()
    assert default_step_registry_path(pack_path) == expected


def test_default_step_registry_path_rejects_invalid_metadata_type(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "pack_id": "test_pack",
                "metadata": {"step_registry_path": 123},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(StepRegistryError, match="metadata.step_registry_path must be a non-empty string"):
        default_step_registry_path(pack_path)
