"""Domain contract for support-ticket classification."""
from typing import Literal
from pydantic import Field, field_validator
from .base import StrictModel


class SupportTicket(StrictModel):
    category: Literal["billing", "technical", "account", "general"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(min_length=1, max_length=500)
    requires_human: bool

    @field_validator("summary")
    @classmethod
    def reject_blank_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must contain non-whitespace characters")
        return value
