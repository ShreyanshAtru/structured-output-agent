"""Registry and validation boundary for synchronous tool execution."""
import json
import re
from collections.abc import Sequence
from pydantic import ValidationError
from structured_output_agent.exceptions import (
    ResponseParseError, SchemaValidationError, ToolExecutionError, ToolValidationError,
)
from structured_output_agent.schemas import StrictModel
from structured_output_agent.validation import ResponseValidator
from .base import Tool, ToolCall, ToolDefinition, ToolResult
from structured_output_agent.observability.base import NoOpObservability, Observability


class ToolExecutor:
    """Register once, validate before/after execution, and never retry a tool."""

    def __init__(self, tools: Sequence[Tool], *, validator: ResponseValidator | None = None):
        self.validator = validator if validator is not None else ResponseValidator()
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("registered tools must implement Tool")
            if not isinstance(tool.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", tool.name):
                raise ValueError("tool name must be a lowercase identifier of at most 64 characters")
            if tool.name in self._tools:
                raise ValueError("duplicate tool name")
            if not isinstance(tool.description, str) or not tool.description.strip():
                raise ValueError("tool description must not be blank")
            for model in (tool.input_model, tool.output_model):
                if not isinstance(model, type) or not issubclass(model, StrictModel):
                    raise TypeError("tool schemas must inherit StrictModel")
            self._tools[tool.name] = tool

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(ToolDefinition(
            name=name, description=tool.description,
            input_schema_json=json.dumps(tool.input_model.model_json_schema()),
            output_schema_json=json.dumps(tool.output_model.model_json_schema()),
        ) for name, tool in self._tools.items())

    def execute(self, call: ToolCall, *, observability: Observability | None = None) -> ToolResult:
        observer = observability if observability is not None else NoOpObservability()
        return self._execute(call, observer)

    def _execute(self, call: ToolCall, observer: Observability) -> ToolResult:
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall")
        if not isinstance(call.name, str) or call.name not in self._tools:
            raise ToolExecutionError("Requested tool is not registered.")
        tool = self._tools[call.name]
        invalid = False
        try:
            if not isinstance(call.arguments_json, str):
                invalid = True
            else:
                with observer.span("tool.input.validate", stage="input"):
                    arguments = self.validator.validate(call.arguments_json, tool.input_model)
        except (ResponseParseError, SchemaValidationError) as exc:
            self._emit_issues(observer, exc, tool.input_model, "input")
            invalid = True
        if invalid:
            observer.event("validation.failed", stage="input", error_code="tool_validation_error")
            raise ToolValidationError("Tool input does not match its JSON schema.")
        failed = False
        try:
            with observer.span("tool.invoke"):
                output = tool.execute(arguments)
        except Exception:
            # Tool implementation is an external execution boundary. Do not leak
            # arbitrary exception messages or replay possibly completed effects.
            failed = True
        if failed:
            raise ToolExecutionError("Tool execution failed.")
        invalid = False
        try:
            # Revalidation must not trust model_construct or a mutated instance.
            payload = output.model_dump(mode="python", warnings=False) if isinstance(output, StrictModel) else output
            with observer.span("tool.output.validate", stage="output"):
                checked = tool.output_model.model_validate(payload, strict=True)
                serialized = checked.model_dump_json()
                self.validator.validate(serialized, tool.output_model)
        except (ValidationError, ResponseParseError, SchemaValidationError, TypeError, ValueError) as exc:
            self._emit_issues(observer, exc, tool.output_model, "output")
            invalid = True
        if invalid:
            observer.event("validation.failed", stage="output", error_code="tool_validation_error")
            raise ToolValidationError("Tool output does not match its JSON schema.")
        return ToolResult(name=call.name, output_json=serialized,
                          call_id=call.call_id, provider_items=call.provider_items)

    @staticmethod
    def _emit_issues(observer: Observability, error: Exception, model: type[StrictModel], stage: str) -> None:
        if isinstance(error, ResponseParseError):
            observer.event("validation.issue", stage=stage, error_code="tool_validation_error",
                           issue_code="json_parse_error", line=error.line, column=error.column)
        elif isinstance(error, SchemaValidationError):
            for issue in error.issues[:10]:
                observer.event("validation.issue", stage=stage, error_code="tool_validation_error",
                               field=str(issue.location[0]) if issue.location else "<root>", issue_code=issue.code)
        elif isinstance(error, ValidationError):
            for issue in error.errors(include_input=False, include_context=False, include_url=False)[:10]:
                first = issue["loc"][0] if issue["loc"] else "<root>"
                observer.event("validation.issue", stage=stage, error_code="tool_validation_error",
                               field=first if first in model.model_fields else "<field>", issue_code=issue["type"])
