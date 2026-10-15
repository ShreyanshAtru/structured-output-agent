"""Opt-in paid API example: configure .env, then python examples/live_ticket.py."""
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.observability import create_observability
from structured_output_agent.clients import create_live_client
from structured_output_agent.config import AgentConfig
from structured_output_agent.retry import RetryPolicy
from structured_output_agent.schemas import SupportTicket
from structured_output_agent.tools import TicketContextTool, ToolExecutor


def main() -> None:
    config = AgentConfig()
    observer = create_observability(config)
    try:
        run(config, observer)
    finally:
        observer.close()


def run(config, observer) -> None:
    with create_live_client(config) as client:
        agent = StructuredOutputAgent(
            client, retry_policy=RetryPolicy(config.max_attempts),
            tool_executor=ToolExecutor([TicketContextTool()]),
            max_tool_calls=config.max_tool_calls, observability=observer,
        )
        result = agent.run(
            "Classify this support ticket: I was charged twice. "
            "Use ticket_context to look up TICKET-001 before classifying.",
            SupportTicket,
        )
        print(result.model_dump_json(indent=2))
        if result.status == "failure":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
