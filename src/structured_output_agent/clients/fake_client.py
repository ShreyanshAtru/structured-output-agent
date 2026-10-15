"""Finite scripted client for repeatable offline scenarios."""
from collections.abc import Sequence

from structured_output_agent.exceptions import ProviderError
from .base import GenerationRequest, LLMClient
from structured_output_agent.tools.base import ToolCall


class FakeLLMClient(LLMClient):
    """Consume one string or ProviderError per call; never cycle responses.

    Create a fresh client for each independent scenario. Captured requests contain
    user input and are for local tests only; they are not telemetry.
    """

    execution_mode = "fake"
    provider_name = "fake"

    def __init__(self, responses: Sequence[str | ToolCall | ProviderError]):
        if isinstance(responses, (str, bytes)) or not responses:
            raise ValueError("responses must be a nonempty sequence of scripted steps")
        if any(not isinstance(step, (str, ToolCall, ProviderError)) for step in responses):
            raise TypeError("scripted steps must be strings, ToolCall, or ProviderError instances")
        self._responses = tuple(responses)
        self._requests: list[GenerationRequest] = []

    @property
    def requests(self) -> tuple[GenerationRequest, ...]:
        return tuple(self._requests)

    def generate(self, request: GenerationRequest) -> str | ToolCall:
        index = len(self._requests)
        self._requests.append(request)
        if index >= len(self._responses):
            raise ProviderError("Fake response script exhausted.")
        step = self._responses[index]
        if isinstance(step, ProviderError):
            raise step
        return step
