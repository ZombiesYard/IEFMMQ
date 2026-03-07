import json

import pytest

from adapters.help_response_parser import (
    json_extract,
    parse_help_response,
    parse_help_response_with_diagnostics,
    strip_code_fence,
)
from core.help_failure import JSON_EXTRACT_FAIL, SCHEMA_FAIL, exception_failure_code
from tests._fakes import _help_obj_ok


def test_strip_code_fence_removes_balanced_fence() -> None:
    raw = "```json\n{\"k\":1}\n```"
    assert strip_code_fence(raw) == "{\"k\":1}"


def test_strip_code_fence_keeps_malformed_fence_text() -> None:
    raw = "```json\n{\"k\":1}\n"
    assert strip_code_fence(raw) == raw.strip()


def test_json_extract_handles_nested_braces_in_string() -> None:
    raw = 'prefix {"msg":"value { with } braces","n":1} suffix'
    extracted = json_extract(raw)
    assert extracted == '{"msg":"value { with } braces","n":1}'


def test_json_extract_handles_escaped_quotes_and_braces() -> None:
    raw = 'x {"msg":"say \\"{hi}\\" done","nested":{"k":"v"}} y'
    extracted = json_extract(raw)
    obj = json.loads(extracted)
    assert obj["msg"] == 'say "{hi}" done'
    assert obj["nested"]["k"] == "v"


def test_json_extract_parses_json_embedded_in_non_json_text() -> None:
    help_obj = _help_obj_ok()
    raw = f"Model answer:\n{json.dumps(help_obj, ensure_ascii=False)}\nThanks"
    extracted = json_extract(raw)
    assert json.loads(extracted) == help_obj


def test_parse_help_response_parses_embedded_json_text() -> None:
    help_obj = _help_obj_ok()
    raw = f"some text {json.dumps(help_obj, ensure_ascii=False)} trailing text"
    parsed = parse_help_response(raw)
    assert parsed == help_obj


def test_json_extract_raises_on_missing_object() -> None:
    with pytest.raises(ValueError, match="does not contain JSON object/array") as excinfo:
        json_extract("no json here")
    assert exception_failure_code(excinfo.value) is None


def test_parse_help_response_repairs_invalid_evidence_type_from_ref_prefix() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["evidence"][0]["type"] = "status"
    help_obj["overlay"]["evidence"][0]["ref"] = "GATES.S03.completion"
    raw = json.dumps(help_obj, ensure_ascii=False)

    parsed, _extract, repair_meta = parse_help_response_with_diagnostics(raw)

    assert parsed["overlay"]["evidence"][0]["type"] == "gate"
    assert repair_meta["repair_applied"] is True
    assert repair_meta["repaired_evidence_types"] == 1
    assert repair_meta["dropped_unrepairable_evidence"] == 0


def test_parse_help_response_drops_unrepairable_evidence_and_still_rejects_invalid_overlay() -> None:
    help_obj = _help_obj_ok()
    help_obj["overlay"]["evidence"][0]["type"] = "status"
    help_obj["overlay"]["evidence"][0]["ref"] = "UNKNOWN.foo"
    raw = json.dumps(help_obj, ensure_ascii=False)

    with pytest.raises(Exception, match="HelpResponse validation failed") as excinfo:
        parse_help_response_with_diagnostics(raw)
    assert exception_failure_code(excinfo.value) == SCHEMA_FAIL


def test_parse_help_response_drops_non_object_evidence_items_and_keeps_valid_items() -> None:
    help_obj = _help_obj_ok()
    valid_item = dict(help_obj["overlay"]["evidence"][0])
    help_obj["overlay"]["evidence"] = ["noise", 123, valid_item]
    raw = json.dumps(help_obj, ensure_ascii=False)

    parsed, _extract, repair_meta = parse_help_response_with_diagnostics(raw)

    assert parsed["overlay"]["evidence"] == [valid_item]
    assert repair_meta["repair_applied"] is True
    assert repair_meta["repaired_evidence_types"] == 0
    assert repair_meta["dropped_unrepairable_evidence"] == 2
    reasons = [item.get("reason") for item in repair_meta["details"] if isinstance(item, dict)]
    assert reasons.count("non_object_evidence_item") == 2


def test_parse_help_response_marks_json_extract_fail_when_no_json_found() -> None:
    with pytest.raises(ValueError, match="does not contain JSON object/array") as excinfo:
        parse_help_response_with_diagnostics("plain text only")
    assert exception_failure_code(excinfo.value) == JSON_EXTRACT_FAIL


def test_parse_help_response_marks_schema_fail_when_payload_is_not_object() -> None:
    with pytest.raises(ValueError, match="HelpResponse must be a JSON object") as excinfo:
        parse_help_response_with_diagnostics('["not", "an", "object"]')
    assert exception_failure_code(excinfo.value) == SCHEMA_FAIL
