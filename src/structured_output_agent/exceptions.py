"""Application exception categories, separate from result data contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationIssue:
    """Only declared top-level field names and error codes are retained."""

    location: tuple[str | int, ...]
    code: str


class AgentError(Exception):
    """Base class for application failures."""


class ConfigurationError(AgentError):
    """Invalid runtime configuration."""


class ResponseParseError(AgentError):
    """A response is invalid JSON; messages never include the source text."""

    def __init__(self, message: str, *, line: int | None = None, column: int | None = None):
        super().__init__(message)
        self.line = line
        self.column = column


class SchemaValidationError(AgentError):
    """Schema failure with sanitized, payload-free issues."""

    def __init__(self, issues: tuple["ValidationIssue", ...]):
        super().__init__("Response does not match the output schema.")
        self.issues = issues


class ToolValidationError(AgentError):
    """Tool arguments or output violate their schema."""


class ToolExecutionError(AgentError):
    """A tool could not complete execution."""


PROVIDER_FAILURE_MESSAGES = {
    "workspace_routing": "Codex workspace routing failed. Verify Codex sign-in and workspace access, then retry from your terminal.",
    "authentication": "Codex requires ChatGPT sign-in. Run codex login in your project virtual environment.",
    "timeout": "Codex generation timed out. Retry or increase CODEX_TIMEOUT_SECONDS.",
    "runtime": "Codex runtime or scratch directory is unavailable. Check the SDK installation and filesystem permissions.",
}


class ProviderError(AgentError):
    """A provider failure; only allowlisted reasons may reach user-facing results."""

    def __init__(self, message: str, *, reason: str | None = None):
        super().__init__(message)
        self.reason = reason if reason in PROVIDER_FAILURE_MESSAGES else None


class RetryExhaustedError(AgentError):
    """No schema-repair attempts remain."""
