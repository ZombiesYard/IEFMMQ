import json

import pytest

from adapters.help_response_parser import json_extract, parse_help_response, strip_code_fence
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
    with pytest.raises(ValueError, match="does not contain JSON object/array"):
        json_extract("no json here")
