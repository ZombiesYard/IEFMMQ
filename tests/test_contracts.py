import json
from pathlib import Path

from jsonschema import Draft202012Validator

from core.types import CONTRACT_VERSION, Event, Observation, TutorRequest, TutorResponse


SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "v1"


def load_schema(name: str) -> dict:
    path = SCHEMA_DIR / f"{name}.schema.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate(obj: dict, schema_name: str) -> None:
    schema = load_schema(schema_name)
    Draft202012Validator(schema).validate(obj)


def test_observation_schema_accepts_defaults():
    obs = Observation(source="mock_env", payload={"rpm": 60}).to_dict()
    assert obs["version"] == CONTRACT_VERSION
    validate(obs, "observation")


def test_tutor_request_schema_accepts_defaults():
    req = TutorRequest(intent="ask_hint", message="what next?").to_dict()
    validate(req, "tutor_request")


def test_tutor_response_schema_accepts_defaults():
    res = TutorResponse(message="Proceed to APU.").to_dict()
    validate(res, "tutor_response")


def test_event_schema_links_payloads():
    obs = Observation(source="mock_env", payload={"rpm": 60}).to_dict()
    evt = Event(kind="observation", payload=obs, related_id=obs["observation_id"]).to_dict()
    validate(evt, "event")

