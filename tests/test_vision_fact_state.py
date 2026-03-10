from __future__ import annotations

from core.types_v2 import VisionFact, VisionFactObservation
from core.vision_facts import build_vision_fact_summary, merge_vision_fact_observation, prune_expired_facts


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
