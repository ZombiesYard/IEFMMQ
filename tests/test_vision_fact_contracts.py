from __future__ import annotations

from core.types_v2 import VisionFact, VisionFactObservation
from simtutor.schemas import SCHEMA_INDEX, load_schema, validate_instance


def test_schema_registry_exposes_vision_fact_observation() -> None:
    assert "vision_fact_observation" in SCHEMA_INDEX
    schema = load_schema("vision_fact_observation")
    assert schema["title"] == "Vision Fact Observation v2"


def test_vision_fact_observation_schema_accepts_uncertain_fact() -> None:
    payload = VisionFactObservation(
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
        frame_ids=["1772872444950_000122", "1772872445010_000123"],
        facts=[
            VisionFact(
                fact_id="fcs_reset_seen",
                state="uncertain",
                source_frame_id="1772872445010_000123",
                confidence=0.42,
                expires_after_ms=600000,
                evidence_note="FCS page content is too blurry to confirm reset state.",
                observed_at_wall_ms=1772872445000,
                sticky=True,
            )
        ],
        summary="uncertain=fcs_reset_seen",
    ).to_dict()

    validate_instance(payload, "vision_fact_observation")


def test_vision_fact_observation_schema_accepts_all_fact_ids() -> None:
    payload = VisionFactObservation(
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
        frame_ids=["1772872445010_000123"],
        facts=[
            VisionFact(
                fact_id=fact_id,
                state="not_seen",
                source_frame_id="1772872445010_000123",
                confidence=0.1,
                expires_after_ms=2000,
                evidence_note=f"{fact_id} not visible in current frame.",
            )
            for fact_id in (
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
        ],
    ).to_dict()

    validate_instance(payload, "vision_fact_observation")
