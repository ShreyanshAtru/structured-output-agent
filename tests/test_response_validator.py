import json
from datetime import date
import traceback

import pytest
from pydantic import BaseModel, field_validator

from structured_output_agent.exceptions import ResponseParseError, SchemaValidationError
from structured_output_agent.schemas import StrictModel, SupportTicket
from structured_output_agent.validation import ResponseValidator


VALID = dict(category="billing", priority="high", summary="Duplicate charge", requires_human=True)


def test_valid_ticket():
    result = ResponseValidator().validate(json.dumps(VALID), SupportTicket)
    assert isinstance(result, SupportTicket)
    assert result.summary == VALID["summary"]


@pytest.mark.parametrize("text", ["", "  ", "{'x':1}", '{"x":1,}',
    '```json\n{}\n```', 'Here is your JSON: {}', '{} {}',
    '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}',
    '{"x":1,"x":2}', '{"nested":{"x":1,"x":2}}'])
def test_rejects_malformed_or_ambiguous_json(text):
    with pytest.raises(ResponseParseError):
        ResponseValidator().validate(text, SupportTicket)


@pytest.mark.parametrize("text", ['null', '[]', '42', 'true', '"hello"', '{}'])
def test_valid_json_wrong_shape_is_schema_failure(text):
    with pytest.raises(SchemaValidationError):
        ResponseValidator().validate(text, SupportTicket)


@pytest.mark.parametrize("change,code", [
    ({"priority":"urgent"}, "literal_error"),
    ({"requires_human":"true"}, "bool_type"),
    ({"summary":123}, "string_type"),
    ({"extra":"secret-value"}, "extra_forbidden"),
])
def test_schema_issues(change, code):
    with pytest.raises(SchemaValidationError) as caught:
        ResponseValidator().validate(json.dumps(VALID | change), SupportTicket)
    assert code in [issue.code for issue in caught.value.issues]
    assert caught.value.__context__ is None


def test_reusable_with_dates_and_nested_models():
    class Entry(StrictModel):
        when: date
        count: int
    class Batch(StrictModel):
        entries: list[Entry]
    result = ResponseValidator().validate('{"entries":[{"when":"2026-10-15","count":2}]}', Batch)
    assert result.entries[0].when == date(2026, 10, 15)
    with pytest.raises(SchemaValidationError):
        ResponseValidator().validate('{"entries":[{"when":"2026-10-15","count":"2"}]}', Batch)


def test_schema_error_does_not_expose_custom_message_or_unknown_keys():
    class Private(StrictModel):
        value: str
        @field_validator("value")
        @classmethod
        def reject(cls, value):
            raise ValueError(f"private input: {value}")
    with pytest.raises(SchemaValidationError) as caught:
        ResponseValidator().validate('{"value":"private-secret","secret-key":true}', Private)
    rendered = ''.join(traceback.format_exception_only(caught.value)) + repr(caught.value.issues)
    assert "private-secret" not in rendered
    assert "secret-key" not in rendered
    assert "<field>" in rendered


def test_parse_error_is_sanitized_and_has_position():
    with pytest.raises(ResponseParseError) as caught:
        ResponseValidator().parse_json('{"private-secret":}')
    assert caught.value.line == 1
    assert caught.value.column is not None
    assert caught.value.__context__ is None
    assert "private-secret" not in ''.join(traceback.format_exception_only(caught.value))


def test_size_and_nesting_limits():
    with pytest.raises(ResponseParseError, match="character limit"):
        ResponseValidator(max_response_chars=2).parse_json('null')
    with pytest.raises(ResponseParseError):
        ResponseValidator().parse_json('[' * 2000 + ']' * 2000)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        ResponseValidator(max_response_chars=limit)


def test_programmer_errors():
    with pytest.raises(TypeError):
        ResponseValidator().parse_json({})
    with pytest.raises(TypeError):
        ResponseValidator().validate('{}', dict)


def test_model_controls_extra_field_policy():
    class Permissive(BaseModel):
        count: int
    assert ResponseValidator().validate('{"count":1,"extra":2}', Permissive).count == 1


def test_parse_accepts_surrounding_whitespace_and_unicode():
    assert ResponseValidator().parse_json(' \n {"text":"नमस्ते"} \t') == {"text":"नमस्ते"}


def test_custom_depth_boundary():
    validator = ResponseValidator(max_json_depth=2)
    assert validator.parse_json('{"a":[]}') == {"a": []}
    with pytest.raises(ResponseParseError, match="nesting limit"):
        validator.parse_json('{"a":[{}]}')
    with pytest.raises(ValueError):
        ResponseValidator(max_json_depth=0)
