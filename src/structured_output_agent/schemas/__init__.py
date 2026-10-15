"""Public schema contracts."""
from .base import StrictModel
from .result import AgentResult, AttemptRecord, FailureDetail
from .ticket import SupportTicket

__all__ = ["StrictModel", "AgentResult", "AttemptRecord", "FailureDetail", "SupportTicket"]
