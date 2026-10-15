"""Finite schema-repair decisions; transport failures are terminal here."""
import json
from dataclasses import dataclass

from structured_output_agent.exceptions import ResponseParseError, SchemaValidationError


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be an integer between 1 and 10")

    def should_retry(self, error: Exception, *, attempt: int) -> bool:
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        return isinstance(error, (ResponseParseError, SchemaValidationError)) and attempt < self.max_attempts

    def build_feedback(self, error: ResponseParseError | SchemaValidationError) -> str:
        prefix = "Return only one complete JSON value matching the supplied schema. "
        if isinstance(error, ResponseParseError):
            return prefix + "The previous response failed JSON parsing. Avoid prose, fences, duplicate keys, and nonstandard numbers."
        if isinstance(error, SchemaValidationError):
            issues = [{"location": issue.location, "code": issue.code} for issue in error.issues[:10]]
            return prefix + "Correct these schema issues: " + json.dumps(issues, ensure_ascii=True)[:2000]
        raise TypeError("feedback requires a parse or schema error")
