"""Typed run outcomes and payload-free attempt diagnostics."""
from typing import Generic, Literal, Self, TypeVar
from pydantic import Field, model_validator
from .base import StrictModel

OutputT = TypeVar("OutputT", bound=StrictModel)
ErrorCode = Literal[
    "json_parse_error", "schema_validation_error", "tool_validation_error",
    "tool_execution_error", "tool_limit_exceeded", "provider_error", "configuration_error", "retry_exhausted",
]


class FailureDetail(StrictModel):
    # Messages must be sanitized by their producer; never copy raw provider payloads.
    code: ErrorCode
    message: str = Field(min_length=1, max_length=1000)
    retryable: bool = False


class AttemptRecord(StrictModel):
    attempt: int = Field(ge=1)
    status: Literal["success", "failure"]
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    error: FailureDetail | None = None

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        if (self.status == "failure") != (self.error is not None):
            raise ValueError("only failed attempts must contain an error")
        return self


class AgentResult(StrictModel, Generic[OutputT]):
    run_id: str = Field(min_length=1)
    status: Literal["success", "failure"]
    data: OutputT | None = None
    error: FailureDetail | None = None
    attempts: tuple[AttemptRecord, ...] = ()
    tool_calls: int = Field(default=0, ge=0)
    duration_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.status == "success":
            if self.data is None or self.error is not None:
                raise ValueError("success requires data and forbids an error")
            if not self.attempts or self.attempts[-1].status != "success":
                raise ValueError("success requires a successful final attempt")
        else:
            if self.data is not None or self.error is None:
                raise ValueError("failure requires an error and forbids data")
            if self.attempts and self.attempts[-1].status != "failure":
                raise ValueError("failure requires a failed final attempt")
        if [a.attempt for a in self.attempts] != list(range(1, len(self.attempts) + 1)):
            raise ValueError("attempts must be contiguous and start at one")
        if any(a.status == "success" for a in self.attempts[:-1]):
            raise ValueError("no attempts may follow a successful attempt")
        return self
