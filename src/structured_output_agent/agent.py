"""Synchronous schema-repair orchestration with typed terminal outcomes."""
import json
from time import perf_counter
from typing import TypeVar
from uuid import uuid4

from structured_output_agent.clients.base import GenerationRequest, LLMClient
from structured_output_agent.exceptions import PROVIDER_FAILURE_MESSAGES, ProviderError, ResponseParseError, SchemaValidationError, ToolExecutionError, ToolValidationError
from structured_output_agent.retry.policy import RetryPolicy
from structured_output_agent.schemas import AgentResult, AttemptRecord, FailureDetail, StrictModel
from structured_output_agent.validation import ResponseValidator
from structured_output_agent.tools import ToolCall, ToolExecutor
from structured_output_agent.observability import Observability, NoOpObservability

OutputT = TypeVar("OutputT", bound=StrictModel)


class StructuredOutputAgent:
    """Compose a client, validator, and retry policy; keep run state local.

    Expected provider failures must be ProviderError. Unexpected programming errors
    propagate rather than being misrepresented as model validation failures.
    Tool failures are terminal; no tool is automatically replayed.
    """

    def __init__(self, client: LLMClient, *, validator: ResponseValidator | None = None,
                 retry_policy: RetryPolicy | None = None,
                 tool_executor: ToolExecutor | None = None, max_tool_calls: int = 5,
                 observability: Observability | None = None):
        if type(max_tool_calls) is not int or not 0 <= max_tool_calls <= 20:
            raise ValueError("max_tool_calls must be between 0 and 20")
        self.observability = observability if observability is not None else NoOpObservability()
        self.tool_executor = tool_executor
        self.max_tool_calls = max_tool_calls
        self.client = client
        self.validator = validator if validator is not None else ResponseValidator()
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()

    def run(self, prompt: str, output_model: type[OutputT]) -> AgentResult[OutputT]:
        mode = self.client.execution_mode
        provider = self.client.provider_name
        run_id = str(uuid4())
        schema_name = output_model.__name__ if isinstance(output_model, type) and issubclass(output_model, StrictModel) else "invalid"
        with self.observability.span("run", run_id=run_id, schema=schema_name, mode=mode, provider=provider) as outcome:
            result = self._run(prompt, output_model, run_id)
            first_invalid = bool(result.attempts and result.attempts[0].error and
                                 result.attempts[0].error.code in ("json_parse_error", "schema_validation_error"))
            outcome["status"] = result.status
            if result.error:
                outcome["error_code"] = result.error.code
            self.observability.event(
                "run.completed", status=result.status,
                error_code=result.error.code if result.error else "none",
                attempts=len(result.attempts), tool_calls=result.tool_calls,
                duration_ms=result.duration_ms,
                first_attempt_success=result.status == "success" and len(result.attempts) == 1,
                initial_validation_failure=first_invalid,
                recovered=first_invalid and result.status == "success",
            )
            return result

    def _run(self, prompt: str, output_model: type[OutputT], run_id: str) -> AgentResult[OutputT]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        if not isinstance(output_model, type) or not issubclass(output_model, StrictModel):
            raise TypeError("agent output_model must inherit StrictModel")
        schema = json.dumps(output_model.model_json_schema())
        result_type = AgentResult[output_model]
        started = perf_counter()
        attempts: list[AttemptRecord] = []
        feedback = None
        tool_calls = 0
        tool_results = []
        definitions = self.tool_executor.definitions if self.tool_executor else ()
        for number in range(1, self.retry_policy.max_attempts + 1):
            attempt_started = perf_counter()
            failure = None
            error = None
            try:
                while True:
                    with self.observability.span("model.generate", attempt=number):
                        response = self.client.generate(GenerationRequest(
                            prompt=prompt, output_schema_json=schema, attempt=number,
                            repair_feedback=feedback, tools=definitions,
                            tool_results=tuple(tool_results),
                        ))
                    if not isinstance(response, ToolCall):
                        break
                    if tool_calls >= self.max_tool_calls:
                        failure = FailureDetail(code="tool_limit_exceeded", message="Tool call limit reached.")
                        break
                    tool_calls += 1
                    if self.tool_executor is None:
                        raise ToolExecutionError("No tools are registered.")
                    tool_name = response.name if any(d.name == response.name for d in definitions) else "unknown"
                    with self.observability.span("tool.execute", attempt=number, tool=tool_name):
                        tool_results.append(self.tool_executor.execute(response, observability=self.observability))
                if failure is None:
                    with self.observability.span("response.validate", attempt=number, stage="final"):
                        data = self.validator.validate(response, output_model)
            except ResponseParseError as exc:
                error = exc
                failure = FailureDetail(code="json_parse_error", message="Response is not valid accepted JSON.", retryable=True)
            except SchemaValidationError as exc:
                error = exc
                failure = FailureDetail(code="schema_validation_error", message="Response does not match the output schema.", retryable=True)
            except ToolValidationError as exc:
                error = exc
                failure = FailureDetail(code="tool_validation_error", message="Tool input or output failed validation.")
            except ToolExecutionError as exc:
                error = exc
                failure = FailureDetail(code="tool_execution_error", message="Tool execution failed.")
            except ProviderError as exc:
                error = exc
                failure = FailureDetail(code="provider_error", message=PROVIDER_FAILURE_MESSAGES.get(exc.reason, "Model generation failed."), retryable=False)
            duration = (perf_counter() - attempt_started) * 1000
            if failure is None:
                attempts.append(AttemptRecord(attempt=number, status="success", duration_ms=duration))
                return result_type(run_id=run_id, status="success", data=data,
                                   attempts=tuple(attempts), tool_calls=tool_calls, duration_ms=(perf_counter() - started) * 1000)
            self.observability.event("attempt.failed", attempt=number, error_code=failure.code, duration_ms=duration)
            if failure.code in ("json_parse_error", "schema_validation_error"):
                self.observability.event("validation.failed", attempt=number, stage="final", error_code=failure.code,
                                         **({"line": error.line, "column": error.column} if isinstance(error, ResponseParseError) else {}))
                if isinstance(error, SchemaValidationError):
                    for issue in error.issues[:10]:
                        field = str(issue.location[0]) if issue.location else "<root>"
                        self.observability.event("validation.issue", attempt=number, stage="final", field=field, issue_code=issue.code, error_code="schema_validation_error")
            attempts.append(AttemptRecord(attempt=number, status="failure", error=failure, duration_ms=duration))
            if self.retry_policy.should_retry(error, attempt=number):
                self.observability.event("retry.scheduled", attempt=number, error_code=failure.code)
                feedback = self.retry_policy.build_feedback(error)
                continue
            terminal = (FailureDetail(code="retry_exhausted", message="Schema-repair attempt limit reached.")
                        if failure.retryable else failure)
            return result_type(run_id=run_id, status="failure", error=terminal,
                               attempts=tuple(attempts), tool_calls=tool_calls, duration_ms=(perf_counter() - started) * 1000)
        raise RuntimeError("Retry policy supplied an invalid attempt budget")
