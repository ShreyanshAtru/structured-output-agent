"""Offline recovery demo: python examples/classify_ticket.py."""
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients import FakeLLMClient
from structured_output_agent.schemas import SupportTicket


def main() -> None:
    client = FakeLLMClient([
        '{"category":"billing","priority":"urgent","summary":"Duplicate charge","requires_human":true}',
        '{"category":"billing","priority":"high","summary":"Duplicate charge","requires_human":true}',
    ])
    result = StructuredOutputAgent(client).run("I was charged twice. Please help.", SupportTicket)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
