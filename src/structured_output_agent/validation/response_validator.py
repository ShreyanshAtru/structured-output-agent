"""Strict JSON parsing and reusable Pydantic validation without payload logging."""
import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from structured_output_agent.exceptions import (
    ResponseParseError, SchemaValidationError, ValidationIssue,
)

OutputT = TypeVar("OutputT", bound=BaseModel)


class _InvalidJSON(ValueError):
    """Internal marker for duplicate keys and nonstandard numeric constants."""


def _reject_constant(value: str) -> None:
    raise _InvalidJSON("Nonstandard numeric constants are not permitted.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJSON("Duplicate object keys are not permitted.")
        result[key] = value
    return result


class ResponseValidator:
    """Validate a complete JSON response against any Pydantic model.

    No markdown stripping, substring extraction, coercion, or repair is attempted.
    Strict JSON validation preserves Pydantic's JSON handling for dates and UUIDs.
    Unknown fields follow the supplied model's extra policy; use StrictModel to
    forbid them. Input size is bounded before parsing (default one million chars).
    """

    def __init__(self, *, max_response_chars: int = 1_000_000, max_json_depth: int = 100):
        if type(max_response_chars) is not int or max_response_chars < 1:
            raise ValueError("max_response_chars must be a positive integer")
        if type(max_json_depth) is not int or max_json_depth < 1:
            raise ValueError("max_json_depth must be a positive integer")
        self.max_response_chars = max_response_chars
        self.max_json_depth = max_json_depth

    def parse_json(self, response: str) -> object:
        """Parse one JSON value, rejecting ambiguous or nonstandard JSON."""
        if not isinstance(response, str):
            raise TypeError("response must be a string")
        if len(response) > self.max_response_chars:
            raise ResponseParseError("Response exceeds the configured character limit.")
        # Raise outside the handler so exception context cannot expose input text.
        failure = None
        try:
            parsed = json.loads(response, object_pairs_hook=_unique_object,
                                parse_constant=_reject_constant)
            stack = [(parsed, 0)]
            while stack:
                value, depth = stack.pop()
                if isinstance(value, (dict, list)):
                    if depth >= self.max_json_depth:
                        raise _InvalidJSON("JSON exceeds the configured nesting limit.")
                    children = value.values() if isinstance(value, dict) else value
                    stack.extend((child, depth + 1) for child in children)
            return parsed
        except json.JSONDecodeError as exc:
            failure = ResponseParseError("Response is not valid JSON.",
                                         line=exc.lineno, column=exc.colno)
        except _InvalidJSON as exc:
            failure = ResponseParseError(str(exc))
        except (RecursionError, ValueError):
            failure = ResponseParseError("JSON exceeds parser limits.")
        raise failure

    def validate(self, response: str, output_model: type[OutputT]) -> OutputT:
        """Return a typed object or raise a sanitized parse/schema exception."""
        if not isinstance(output_model, type) or not issubclass(output_model, BaseModel):
            raise TypeError("output_model must be a Pydantic BaseModel subclass")
        self.parse_json(response)
        issues = None
        try:
            return output_model.model_validate_json(response, strict=True)
        except ValidationError as exc:
            # Never copy Pydantic messages, input values, context, or unknown keys.
            # Custom validators may embed sensitive input in their messages.
            fields = set(output_model.model_fields)
            fields.update(f.alias for f in output_model.model_fields.values() if f.alias)
            issues = tuple(
                ValidationIssue(
                    location=tuple(
                        part if (index == 0 and isinstance(part, str) and part in fields)
                        else (part if isinstance(part, int) else "<field>")
                        for index, part in enumerate(error["loc"])
                    ),
                    code=error["type"],
                )
                for error in exc.errors(include_input=False, include_context=False,
                                        include_url=False)
            )
        raise SchemaValidationError(issues)
