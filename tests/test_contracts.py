from core.types import CONTRACT_VERSION, Event, Observation, TutorRequest, TutorResponse
from simtutor.schemas import validate_instance


def test_observation_schema_accepts_defaults():
    obs = Observation(source="mock_env", payload={"rpm": 60}).to_dict()
    assert obs["version"] == CONTRACT_VERSION
    validate_instance(obs, "observation")


def test_tutor_request_schema_accepts_defaults():
    req = TutorRequest(intent="ask_hint", message="what next?").to_dict()
    validate_instance(req, "tutor_request")


def test_tutor_response_schema_accepts_defaults():
    res = TutorResponse(message="Proceed to APU.").to_dict()
    validate_instance(res, "tutor_response")


def test_event_schema_links_payloads():
    obs = Observation(source="mock_env", payload={"rpm": 60}).to_dict()
    evt = Event(kind="observation", payload=obs, related_id=obs["observation_id"]).to_dict()
    validate_instance(evt, "event")


def test_event_schema_accepts_overlay_rejected_event():
    evt = Event(
        kind="overlay_rejected",
        payload={"failure_code": "evidence_fail", "reasons": ["missing_overlay_evidence"]},
    ).to_dict()
    validate_instance(evt, "event")


def test_event_schema_accepts_vision_placeholder_refs():
    evt = Event(
        kind="system",
        payload={"status": "vision_unavailable"},
        vision_refs=["1772872444902_000123"],
    ).to_dict()
    validate_instance(evt, "event")

