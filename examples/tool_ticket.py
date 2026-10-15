"""Offline validated tool demo: python examples/tool_ticket.py."""
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients import FakeLLMClient
from structured_output_agent.schemas import SupportTicket
from structured_output_agent.tools import ToolCall, ToolExecutor, TicketContextTool


def main() -> None:
    client = FakeLLMClient([
        ToolCall("ticket_context", '{"ticket_id":"TICKET-001"}'),
        '{"category":"billing","priority":"high","summary":"Duplicate charge reported by pro customer","requires_human":true}',
    ])
    agent = StructuredOutputAgent(client, tool_executor=ToolExecutor([TicketContextTool()]))
    print(agent.run("Classify ticket TICKET-001", SupportTicket).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
