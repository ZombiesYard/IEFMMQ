import json
from importlib import resources

from jsonschema import Draft202012Validator, FormatChecker

from core.types import CONTRACT_VERSION, Event, Observation, TutorRequest, TutorResponse


SCHEMA_PACKAGE = "simtutor.schemas.v1"


def load_schema(name: str) -> dict:
    schema_path = resources.files(SCHEMA_PACKAGE) / f"{name}.schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate(obj: dict, schema_name: str) -> None:
    schema = load_schema(schema_name)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(obj)


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

