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
                fact_id="fcsmc_intermediate_result_visible",
                state="uncertain",
                source_frame_id="1772872445010_000123",
                expires_after_ms=2000,
                evidence_note="FCS-MC result text is too blurry to classify as intermediate or final.",
                observed_at_wall_ms=1772872445000,
                sticky=False,
            )
        ],
        summary="uncertain=fcsmc_intermediate_result_visible",
    ).to_dict()

    validate_instance(payload, "vision_fact_observation")


def test_vision_fact_observation_schema_accepts_structured_result_kind() -> None:
    payload = VisionFactObservation(
        session_id="sess-live",
        trigger_wall_ms=1772872445000,
        frame_ids=["1772872445010_000123"],
        facts=[
            VisionFact(
                fact_id="fcsmc_final_go_result_visible",
                state="seen",
                source_frame_id="1772872445010_000123",
                expires_after_ms=600000,
                evidence_note="Right DDI FCS-MC page shows final results: MC1 GO, MC2 GO, FCSA GO, FCSB GO.",
                result_kind="final_go",
            )
        ],
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
                expires_after_ms=2000,
                evidence_note=f"{fact_id} not visible in current frame.",
            )
            for fact_id in (
                "tac_page_visible",
                "supt_page_visible",
                "fcs_page_visible",
                "fcs_page_x_marks_visible",
                "bit_root_page_visible",
                "fcsmc_page_visible",
                "fcsmc_intermediate_result_visible",
                "fcsmc_in_test_visible",
                "fcsmc_final_go_result_visible",
                "hsi_page_visible",
                "hsi_map_layer_visible",
                "ins_grnd_alignment_text_visible",
                "ins_ok_text_visible",
            )
        ],
    ).to_dict()

    validate_instance(payload, "vision_fact_observation")
