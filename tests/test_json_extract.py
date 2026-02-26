import json

import pytest
from jsonschema.exceptions import ValidationError

from adapters.help_response_parser import parse_help_response_with_meta
from adapters.json_extract import (
    REPAIR_DROP_PREFIX,
    REPAIR_DROP_SUFFIX,
    REPAIR_REMOVE_CODE_FENCE,
    extract_first_json,
    parse_first_json,
)
from tests._fakes import _help_obj_ok


def test_extract_first_json_plain_object_no_repair() -> None:
    result = extract_first_json('{"k":1}')
    assert result.json_text == '{"k":1}'
    assert result.json_repaired is False
    assert result.repair_reasons == ()


def test_extract_first_json_plain_array_no_repair() -> None:
    result = extract_first_json('[{"k":1}]')
    assert result.json_text == '[{"k":1}]'
    assert result.json_repaired is False


def test_extract_first_json_code_fence_records_repair_reason() -> None:
    result = extract_first_json("```json\n{\"k\":1}\n```")
    assert result.json_text == '{"k":1}'
    assert result.json_repaired is True
    assert result.repair_reasons == (REPAIR_REMOVE_CODE_FENCE,)


def test_extract_first_json_inline_code_fence_records_repair_reason() -> None:
    result = extract_first_json('```json {"k":1} ```')
    assert result.json_text == '{"k":1}'
    assert result.json_repaired is True
    assert result.repair_reasons == (REPAIR_REMOVE_CODE_FENCE,)


def test_extract_first_json_prefix_suffix_records_repair_reasons() -> None:
    result = extract_first_json('prefix {"k":1} suffix')
    assert result.json_text == '{"k":1}'
    assert result.json_repaired is True
    assert REPAIR_DROP_PREFIX in result.repair_reasons
    assert REPAIR_DROP_SUFFIX in result.repair_reasons


def test_extract_first_json_multiple_segments_returns_first() -> None:
    result = extract_first_json('x {"a":1} y {"a":2}')
    assert json.loads(result.json_text) == {"a": 1}


def test_extract_first_json_handles_nested_braces_inside_string() -> None:
    result = extract_first_json('x {"msg":"value {with} braces","n":1} y')
    assert json.loads(result.json_text)["msg"] == "value {with} braces"


def test_extract_first_json_handles_nested_arrays_and_objects() -> None:
    result = extract_first_json('x [{"k":{"n":[1,2,3]}}] tail')
    assert json.loads(result.json_text) == [{"k": {"n": [1, 2, 3]}}]


def test_extract_first_json_raises_when_missing_json() -> None:
    with pytest.raises(ValueError, match="does not contain JSON object/array"):
        extract_first_json("no json here")


def test_parse_first_json_raises_on_invalid_json_segment() -> None:
    with pytest.raises(ValueError, match="invalid"):
        parse_first_json("prefix {'k': 1} suffix")


def test_parse_help_response_with_meta_accepts_fenced_payload_and_marks_repaired() -> None:
    payload = _help_obj_ok()
    raw = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
    parsed, meta = parse_help_response_with_meta(raw)
    assert parsed == payload
    assert meta.json_repaired is True
    assert REPAIR_REMOVE_CODE_FENCE in meta.repair_reasons


def test_parse_help_response_with_meta_rejects_typo_field_name() -> None:
    payload = _help_obj_ok()
    payload["diagnozis"] = payload.pop("diagnosis")
    raw = json.dumps(payload, ensure_ascii=False)
    with pytest.raises(ValidationError):
        parse_help_response_with_meta(raw)


def test_parse_help_response_with_meta_rejects_half_legal_json_overlay_not_executable() -> None:
    payload = _help_obj_ok()
    payload["overlay"] = {"targets": ["not_allowed_target"]}
    raw = "Here:\n" + json.dumps(payload, ensure_ascii=False) + "\nThanks."
    with pytest.raises(ValidationError):
        parse_help_response_with_meta(raw)
