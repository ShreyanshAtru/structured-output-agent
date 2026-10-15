"""Provider-independent generation contract."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from structured_output_agent.tools.base import ToolCall, ToolDefinition, ToolResult


@dataclass(frozen=True, repr=False)
class GenerationRequest:
    """In-memory request; never log its prompt or feedback by default."""

    prompt: str
    output_schema_json: str
    attempt: int
    repair_feedback: str | None = None
    tools: tuple[ToolDefinition, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


class LLMClient(ABC):
    execution_mode = "custom"
    provider_name = "custom"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str | ToolCall:
        """Return final response text or one tool call; raise ProviderError for operational failures."""
