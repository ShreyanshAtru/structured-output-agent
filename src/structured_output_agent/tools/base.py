"""Typed tool contracts and provider-neutral tool messages."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar
from structured_output_agent.schemas import StrictModel

InputT = TypeVar("InputT", bound=StrictModel)
OutputT = TypeVar("OutputT", bound=StrictModel)


@dataclass(frozen=True, repr=False)
class ToolCall:
    name: str
    arguments_json: str
    call_id: str | None = None
    provider_items: tuple[str, ...] = ()


@dataclass(frozen=True, repr=False)
class ToolResult:
    name: str
    output_json: str
    call_id: str | None = None
    provider_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema_json: str
    output_schema_json: str


class Tool(ABC, Generic[InputT, OutputT]):
    name: str
    description: str
    input_model: type[InputT]
    output_model: type[OutputT]

    @abstractmethod
    def execute(self, arguments: InputT) -> OutputT:
        """Execute once. The executor independently validates returned data."""
