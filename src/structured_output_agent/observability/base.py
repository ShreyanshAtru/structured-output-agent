"""Payload-free telemetry contract with fail-open instrumentation."""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from uuid import uuid4

from structured_output_agent.exceptions import (
    ProviderError, ResponseParseError, SchemaValidationError, ToolExecutionError, ToolValidationError,
)

_CONTEXT: ContextVar[dict] = ContextVar("agent_telemetry", default={})
_FIELDS = {"run_id", "span_id", "parent_span_id", "attempt", "schema", "tool", "mode",
           "status", "error_code", "duration_ms", "attempts", "tool_calls", "first_attempt_success",
           "initial_validation_failure", "recovered", "stage", "issue_count", "field", "issue_code", "provider", "line", "column"}


def error_code(error: BaseException) -> str:
    for kind, code in ((ResponseParseError, "json_parse_error"),
                       (SchemaValidationError, "schema_validation_error"),
                       (ToolValidationError, "tool_validation_error"),
                       (ToolExecutionError, "tool_execution_error"),
                       (ProviderError, "provider_error")):
        if isinstance(error, kind):
            return code
    return "internal_error"


class Observability(ABC):
    """Core owns correlation and timings; adapters receive allowlisted metadata.

    Backend errors increment telemetry_failures and never change agent behavior.
    Do not put payloads or error messages into allowed metadata fields.
    """

    def __init__(self):
        self.telemetry_failures = 0

    @abstractmethod
    def _emit(self, event: str, attributes: dict) -> None:
        pass

    def _start_span(self, name: str, attributes: dict):
        return None

    def _end_span(self, handle, attributes: dict) -> None:
        pass

    def _safe(self, method, *args):
        try:
            return method(*args)
        except Exception:
            self.telemetry_failures += 1
            return None

    def event(self, event: str, **attributes) -> None:
        combined = {**_CONTEXT.get(), **attributes}
        clean = {key: value for key, value in combined.items()
                 if key in _FIELDS and type(value) in (str, int, float, bool)}
        self._safe(self._emit, event, clean)

    @contextmanager
    def span(self, name: str, **attributes):
        parent = _CONTEXT.get()
        clean = {key: value for key, value in attributes.items()
                 if key in _FIELDS and type(value) in (str, int, float, bool)}
        context = {**parent, **clean, "span_id": str(uuid4())}
        if "span_id" in parent:
            context["parent_span_id"] = parent["span_id"]
        token = _CONTEXT.set(context)
        started = perf_counter()
        handle = self._safe(self._start_span, name, context)
        self.event(name + ".started")
        outcome = {"status": "success"}
        try:
            yield outcome
        except BaseException as exc:
            outcome = {"status": "failure", "error_code": error_code(exc)}
            raise
        finally:
            outcome["duration_ms"] = (perf_counter() - started) * 1000
            self.event(name + ".finished", **outcome)
            if handle is not None:
                self._safe(self._end_span, handle, outcome)
            _CONTEXT.reset(token)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.flush()


class NoOpObservability(Observability):
    def _emit(self, event: str, attributes: dict) -> None:
        pass
