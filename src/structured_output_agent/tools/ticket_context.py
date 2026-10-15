"""Read-only tool backed solely by synthetic demo records."""
from typing import Literal
from pydantic import Field
from structured_output_agent.exceptions import ToolExecutionError
from structured_output_agent.schemas import StrictModel
from .base import Tool


class TicketContextInput(StrictModel):
    ticket_id: str = Field(pattern=r"^TICKET-[0-9]{3}$")


class TicketContextOutput(StrictModel):
    ticket_id: str = Field(pattern=r"^TICKET-[0-9]{3}$")
    plan: Literal["free", "pro"]
    previous_tickets: int = Field(ge=0)
    duplicate_charge_reported: bool


class TicketContextTool(Tool[TicketContextInput, TicketContextOutput]):
    name = "ticket_context"
    description = "Look up synthetic context for demo ticket TICKET-001 or TICKET-002."
    input_model = TicketContextInput
    output_model = TicketContextOutput

    def execute(self, arguments: TicketContextInput) -> TicketContextOutput:
        records = {
            "TICKET-001": dict(plan="pro", previous_tickets=2, duplicate_charge_reported=True),
            "TICKET-002": dict(plan="free", previous_tickets=0, duplicate_charge_reported=False),
        }
        if arguments.ticket_id not in records:
            raise ToolExecutionError("Synthetic ticket was not found.")
        return TicketContextOutput(ticket_id=arguments.ticket_id, **records[arguments.ticket_id])
