from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.types_v2 import VisionFact, VisionFactObservation
from core.vision_facts import (
    VisionFactsConfigError,
    build_vision_fact_summary,
    default_vision_facts_path,
    facts_satisfy_step_binding,
    load_vision_facts_config,
    merge_vision_fact_observation,
    prune_expired_facts,
)

_DEFAULT_VISION_FACT_CONFIG = load_vision_facts_config()


def _obs(
    fact_id: str,
    *,
    state: str,
    frame_id: str,
    trigger_wall_ms: int,
) -> VisionFactObservation:
    return VisionFactObservation(
        session_id="sess-live",
        trigger_wall_ms=trigger_wall_ms,
        frame_ids=[frame_id],
        facts=[
            VisionFact(
                fact_id=fact_id,
                state=state,
                source_frame_id=frame_id,
                confidence=0.9 if state == "seen" else 0.2,
                expires_after_ms=600000 if fact_id != "fcs_page_visible" else 2000,
                evidence_note=f"{fact_id}:{state}",
                observed_at_wall_ms=trigger_wall_ms,
            )
        ],
    )


def test_merge_vision_fact_observation_preserves_sticky_seen_after_not_seen() -> None:
    snapshot = merge_vision_fact_observation(
        {},
        _obs(
            "fcs_reset_seen",
            state="seen",
            frame_id="1772872445010_000123",
            trigger_wall_ms=1772872445000,
        ),
        config=_DEFAULT_VISION_FACT_CONFIG,
        now_wall_ms=1772872445000,
    )

    snapshot = merge_vision_fact_observation(
        snapshot,
        _obs(
            "fcs_reset_seen",
            state="not_seen",
            frame_id="1772872447010_000124",
            trigger_wall_ms=1772872447000,
        ),
        config=_DEFAULT_VISION_FACT_CONFIG,
        now_wall_ms=1772872447000,
    )

    assert snapshot["fcs_reset_seen"]["state"] == "seen"
    assert snapshot["fcs_reset_seen"]["source_frame_id"] == "1772872445010_000123"


def test_merge_vision_fact_observation_updates_nonsticky_visibility() -> None:
    snapshot = merge_vision_fact_observation(
        {},
        _obs(
            "fcs_page_visible",
            state="seen",
            frame_id="1772872445010_000123",
            trigger_wall_ms=1772872445000,
        ),
        config=_DEFAULT_VISION_FACT_CONFIG,
        now_wall_ms=1772872445000,
    )

    snapshot = merge_vision_fact_observation(
        snapshot,
        _obs(
            "fcs_page_visible",
            state="not_seen",
            frame_id="1772872447010_000124",
            trigger_wall_ms=1772872447000,
        ),
        config=_DEFAULT_VISION_FACT_CONFIG,
        now_wall_ms=1772872447000,
    )

    assert snapshot["fcs_page_visible"]["state"] == "not_seen"
    assert snapshot["fcs_page_visible"]["source_frame_id"] == "1772872447010_000124"


def test_prune_expired_facts_drops_sticky_after_ttl() -> None:
    snapshot = merge_vision_fact_observation(
        {},
        _obs(
            "takeoff_trim_seen",
            state="seen",
            frame_id="1772872445010_000123",
            trigger_wall_ms=1772872445000,
        ),
        config=_DEFAULT_VISION_FACT_CONFIG,
        now_wall_ms=1772872445000,
    )

    pruned = prune_expired_facts(snapshot, now_wall_ms=1772873045001)

    assert "takeoff_trim_seen" not in pruned


def test_build_vision_fact_summary_reports_uncertain_and_seen_ids() -> None:
    snapshot = merge_vision_fact_observation(
        {},
        VisionFactObservation(
            session_id="sess-live",
            trigger_wall_ms=1772872445000,
            frame_ids=["1772872445010_000123"],
            facts=[
                VisionFact(
                    fact_id="fcs_bit_result_visible",
                    state="uncertain",
                    source_frame_id="1772872445010_000123",
                    confidence=0.33,
                    expires_after_ms=600000,
                    evidence_note="BIT text too blurry.",
                ),
                VisionFact(
                    fact_id="fcs_bit_interaction_seen",
                    state="seen",
                    source_frame_id="1772872445010_000123",
                    confidence=0.91,
                    expires_after_ms=600000,
                    evidence_note="BIT page shows active FCS test.",
                ),
            ],
        ),
        config=_DEFAULT_VISION_FACT_CONFIG,
        now_wall_ms=1772872445000,
    )

    summary = build_vision_fact_summary(
        snapshot,
        status="available",
        frame_ids=["1772872445010_000123"],
        fresh_fact_ids=["fcs_bit_interaction_seen"],
    )

    assert summary["status"] == "available"
    assert summary["seen_fact_ids"] == ["fcs_bit_interaction_seen"]
    assert summary["uncertain_fact_ids"] == ["fcs_bit_result_visible"]
    assert "fcs_bit_interaction_seen" in summary["summary_text"]


def test_default_vision_facts_path_reads_pack_metadata_override(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack_dir"
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack_path = pack_dir / "pack.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "pack_id": "test_pack",
                "metadata": {"vision_facts_path": "configs/vision/custom.yaml"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    expected = (pack_dir / "configs" / "vision" / "custom.yaml").resolve()
    assert default_vision_facts_path(pack_path) == expected


def test_default_vision_facts_path_rejects_invalid_metadata_type(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "pack_id": "test_pack",
                "metadata": {"vision_facts_path": 123},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(VisionFactsConfigError, match="metadata.vision_facts_path must be a non-empty string"):
        default_vision_facts_path(pack_path)


def test_load_vision_facts_config_uses_pack_metadata_path(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack_dir"
    config_dir = pack_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    pack_path = pack_dir / "pack.yaml"
    vision_config_path = config_dir / "vision.yaml"
    pack_path.write_text(
        yaml.safe_dump(
            {
                "pack_id": "test_pack",
                "metadata": {"vision_facts_path": "configs/vision.yaml"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    vision_config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "layout_id": "custom_layout",
                "facts": [
                    {
                        "fact_id": "fcs_page_visible",
                        "sticky": False,
                        "expires_after_ms": 1234,
                    }
                ],
                "step_bindings": {},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    config = load_vision_facts_config(pack_path=pack_path)

    assert config["layout_id"] == "custom_layout"
    assert config["facts_by_id"]["fcs_page_visible"]["expires_after_ms"] == 1234


def test_load_vision_facts_config_rejects_missing_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "vision_facts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "layout_id": "custom_layout",
                "facts": [
                    {
                        "fact_id": "fcs_page_visible",
                        "sticky": False,
                        "expires_after_ms": 1234,
                    }
                ],
                "step_bindings": {},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_vision_facts_config(path)


def test_load_vision_facts_config_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "vision_facts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v2",
                "layout_id": "custom_layout",
                "facts": [
                    {
                        "fact_id": "fcs_page_visible",
                        "sticky": False,
                        "expires_after_ms": 1234,
                    }
                ],
                "step_bindings": {},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported vision facts schema_version"):
        load_vision_facts_config(path)


def test_load_vision_facts_config_wraps_missing_file_in_config_error(tmp_path: Path) -> None:
    path = tmp_path / "missing_vision_facts.yaml"

    with pytest.raises(VisionFactsConfigError, match="vision facts config not found"):
        load_vision_facts_config(path)


def test_load_vision_facts_config_wraps_yaml_error_in_config_error(tmp_path: Path) -> None:
    path = tmp_path / "vision_facts.yaml"
    path.write_text("schema_version: v1\nfacts: [\n", encoding="utf-8")

    with pytest.raises(VisionFactsConfigError, match="failed to parse vision facts config"):
        load_vision_facts_config(path)


def test_load_vision_facts_config_rejects_string_intended_regions(tmp_path: Path) -> None:
    path = tmp_path / "vision_facts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "layout_id": "custom_layout",
                "facts": [
                    {
                        "fact_id": "fcs_page_visible",
                        "sticky": False,
                        "expires_after_ms": 1234,
                        "intended_regions": "left_ddi",
                    }
                ],
                "step_bindings": {},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="intended_regions must be a list"):
        load_vision_facts_config(path)


def test_load_vision_facts_config_rejects_string_steps(tmp_path: Path) -> None:
    path = tmp_path / "vision_facts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "v1",
                "layout_id": "custom_layout",
                "facts": [
                    {
                        "fact_id": "fcs_page_visible",
                        "sticky": False,
                        "expires_after_ms": 1234,
                        "steps": "S08",
                    }
                ],
                "step_bindings": {},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="steps must be a list"):
        load_vision_facts_config(path)


def test_merge_vision_fact_observation_requires_config() -> None:
    with pytest.raises(ValueError, match="vision fact merge requires config"):
        merge_vision_fact_observation(
            {},
            _obs(
                "fcs_page_visible",
                state="seen",
                frame_id="1772872445010_000123",
                trigger_wall_ms=1772872445000,
            ),
        )


def test_facts_satisfy_step_binding_respects_explicit_empty_config() -> None:
    snapshot = {
        "fcs_page_visible": {"fact_id": "fcs_page_visible", "state": "seen"},
        "bit_page_visible": {"fact_id": "bit_page_visible", "state": "seen"},
    }

    assert facts_satisfy_step_binding(snapshot, step_id="S08", config={}) is False
