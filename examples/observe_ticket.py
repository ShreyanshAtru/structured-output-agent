"""Offline reliability scenarios with the configured telemetry backend.

python examples/observe_ticket.py
Local backend writes JSONL; Logfire backend sends sanitized telemetry remotely.
No LLM API calls are made by this script.
"""
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients import FakeLLMClient
from structured_output_agent.config import AgentConfig
from structured_output_agent.exceptions import ProviderError
from structured_output_agent.observability import create_observability
from structured_output_agent.retry import RetryPolicy
from structured_output_agent.schemas import SupportTicket
from structured_output_agent.tools import TicketContextTool, ToolCall, ToolExecutor

VALID = '{"category":"billing","priority":"high","summary":"Synthetic duplicate charge","requires_human":true}'


def main() -> None:
    config = AgentConfig()
    observer = create_observability(config)
    scenarios = {
        "first_attempt": [VALID],
        "recovery": ['not JSON', '{}', VALID],
        "exhaustion": ['{}'] * 3,
        "provider_failure": [ProviderError("Synthetic provider failure")],
        "tool_success": [ToolCall("ticket_context", '{"ticket_id":"TICKET-001"}'), VALID],
        "tool_validation_failure": [ToolCall("ticket_context", '{}')],
    }
    try:
        for name, script in scenarios.items():
            agent = StructuredOutputAgent(
                FakeLLMClient(script), retry_policy=RetryPolicy(3),
                tool_executor=ToolExecutor([TicketContextTool()]), observability=observer,
            )
            result = agent.run("Classify a synthetic support ticket", SupportTicket)
            print(f"{name}: {result.status}; attempts={len(result.attempts)}; run_id={result.run_id}")
    finally:
        observer.close()
    print(f"Telemetry backend failures: {observer.telemetry_failures}")
    if config.observability_backend == "local":
        print(f"Local event log: {config.observability_log_path}")


if __name__ == "__main__":
    main()
