"""Application services for the playground, independent of Streamlit."""
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients import FakeLLMClient, create_live_client
from structured_output_agent.clients.base import GenerationRequest, LLMClient
from structured_output_agent.config import AgentConfig
from structured_output_agent.exceptions import ConfigurationError, ProviderError
from structured_output_agent.observability import create_observability
from structured_output_agent.observability.memory_adapter import MemoryObservability
from structured_output_agent.retry import RetryPolicy
from structured_output_agent.schemas import AgentResult, SupportTicket
from structured_output_agent.tools import TicketContextTool, ToolCall, ToolExecutor

SCENARIOS = (
    "First-attempt success", "Malformed JSON → recovery", "Schema error → recovery",
    "Retry exhaustion", "Validated tool call", "Invalid tool arguments",
    "Invalid tool output", "Tool call limit", "Provider failure", "Custom JSON → recovery",
)
VALID = json.dumps(dict(category="billing", priority="high", summary="Customer reports a duplicate charge", requires_human=True))
TOOL_CALL = ToolCall("ticket_context", '{"ticket_id":"TICKET-001"}')


class PlaygroundError(Exception):
    """Safe user-facing startup failure; no raw configuration is attached."""


class _InvalidOutputTool(TicketContextTool):
    def execute(self, arguments):
        return {"ticket_id": arguments.ticket_id, "plan": "pro", "previous_tickets": "two", "duplicate_charge_reported": True}


def scenario_steps(name: str, max_attempts: int, max_tool_calls: int, custom_json: str):
    scripts = {
        "First-attempt success": [VALID],
        "Malformed JSON → recovery": ['{"category":', VALID],
        "Schema error → recovery": [VALID.replace('"high"', '"urgent"').replace('true', '"true"'), VALID],
        "Retry exhaustion": ['{}'] * max_attempts,
        "Validated tool call": [TOOL_CALL, VALID],
        "Invalid tool arguments": [ToolCall("ticket_context", '{"ticket_id":123}')],
        "Invalid tool output": [TOOL_CALL],
        "Tool call limit": [TOOL_CALL] * (max_tool_calls + 1),
        "Provider failure": [ProviderError("Synthetic provider failure")],
        "Custom JSON → recovery": [custom_json, VALID],
    }
    if name not in scripts:
        raise ValueError("Unknown scenario")
    return scripts[name]


class RecordingClient(LLMClient):
    """Capture bounded session diagnostics separately from telemetry."""

    def __init__(self, client: LLMClient, *, capture_raw: bool):
        self.client = client
        self.execution_mode = client.execution_mode
        self.provider_name = client.provider_name
        self.capture_raw = capture_raw
        self.calls: list[dict] = []

    def generate(self, request: GenerationRequest) -> str | ToolCall:
        record = {"call": len(self.calls) + 1, "attempt": request.attempt,
                  "repair_feedback": request.repair_feedback, "kind": "provider_error"}
        self.calls.append(record)
        response = self.client.generate(request)
        record["kind"] = "tool_call" if isinstance(response, ToolCall) else "final_response"
        if self.capture_raw:
            raw = (json.dumps({"tool": response.name, "arguments_json": response.arguments_json})
                   if isinstance(response, ToolCall) else response)
            record["raw"] = raw[:5000]
            record["truncated"] = len(raw) > 5000
        return response


@dataclass(repr=False)
class PlaygroundReport:
    result: AgentResult[SupportTicket]
    events: tuple[dict, ...]
    calls: tuple[dict, ...]
    mode: str
    scenario: str
    model: str
    backend: str
    raw_captured: bool
    telemetry_failures: int
    dropped_events: int
    provider: str = "gemini"


class PlaygroundRunner:
    def __init__(self, project_root: Path):
        self.project_root = project_root

    def run(self, *, prompt: str, mode: str = "fake", scenario: str = SCENARIOS[1],
            max_attempts: int = 3, max_tool_calls: int = 5, backend: str = "local",
            model: str | None = None, capture_raw: bool = False,
            custom_json: str = "{}", enable_tools: bool = True, provider: str = "gemini") -> PlaygroundReport:
        if not isinstance(prompt, str) or not prompt.strip():
            raise PlaygroundError("Enter a nonblank support message.")
        if mode not in ("fake", "live") or backend not in ("local", "logfire"):
            raise PlaygroundError("Select a supported mode and observability backend.")
        if mode == "fake" and scenario not in SCENARIOS:
            raise PlaygroundError("Select a supported demo scenario.")
        if provider not in ("gemini", "codex"):
            raise PlaygroundError("Select Gemini or Codex for live playground runs.")
        model_overrides = {}
        if provider == "codex" and model and model.strip():
            model_overrides["codex_model"] = model.strip()
        elif provider == "gemini" and model is not None:
            model_overrides["gemini_model"] = model
        config = None
        try:
            config = AgentConfig(
                _env_file=self.project_root / ".env", mode=mode, llm_provider=provider,
                max_attempts=max_attempts, max_tool_calls=max_tool_calls, **model_overrides,
                observability_backend=backend,
            )
        except ValidationError:
            pass
        if config is None:
            raise PlaygroundError("Check settings and .env: live Gemini needs GEMINI_API_KEY; Codex uses codex login; Logfire needs LOGFIRE_TOKEN. Limits and model must be valid.")
        log_path = Path(config.observability_log_path)
        if not log_path.is_absolute():
            config = config.model_copy(update={"observability_log_path": str(self.project_root / log_path)})
        codex_dir = Path(config.codex_work_dir)
        if not codex_dir.is_absolute():
            config = config.model_copy(update={"codex_work_dir": str(self.project_root / codex_dir)})
        observer = None
        try:
            observer = MemoryObservability(create_observability(config))
        except Exception:
            pass
        if observer is None:
            raise PlaygroundError("Could not initialize observability. Check the configured log path or Logfire settings.")
        try:
            with ExitStack() as stack:
                if mode == "fake":
                    client = FakeLLMClient(scenario_steps(scenario, max_attempts, max_tool_calls, custom_json))
                else:
                    client = stack.enter_context(create_live_client(config))
                recorded = RecordingClient(client, capture_raw=capture_raw)
                tool = _InvalidOutputTool() if mode == "fake" and scenario == "Invalid tool output" else TicketContextTool()
                agent = StructuredOutputAgent(
                    recorded, retry_policy=RetryPolicy(config.max_attempts),
                    tool_executor=ToolExecutor([tool]) if enable_tools else None,
                    max_tool_calls=config.max_tool_calls, observability=observer,
                )
                result = agent.run(prompt, SupportTicket)
        finally:
            observer.close()
        return PlaygroundReport(
            result=result, events=tuple(observer.events), calls=tuple(recorded.calls),
            mode=mode, scenario=scenario if mode == "fake" else ("Live Codex" if provider == "codex" else "Live Gemini"),
            model="Scripted demo" if mode == "fake" else ((config.codex_model or "Codex configured default") if provider == "codex" else config.gemini_model),
            backend=backend, raw_captured=capture_raw,
            telemetry_failures=observer.telemetry_failures + observer.backend.telemetry_failures,
            dropped_events=observer.dropped_events, provider=provider if mode == "live" else "fake",
        )


def session_metrics(reports: list[PlaygroundReport], mode: str, provider: str | None = None) -> dict:
    selected = [report for report in reports if report.mode == mode and (provider is None or report.provider == provider)]
    summaries = [next(e for e in report.events if e["event"] == "run.completed") for report in selected]
    count = len(summaries)
    initially_invalid = sum(e["initial_validation_failure"] for e in summaries)
    return {
        "runs": count,
        "first_attempt_success_rate": sum(e["first_attempt_success"] for e in summaries) / count if count else None,
        "final_failure_rate": sum(e["status"] == "failure" for e in summaries) / count if count else None,
        "recovery_rate": sum(e["recovered"] for e in summaries) / initially_invalid if initially_invalid else None,
    }
