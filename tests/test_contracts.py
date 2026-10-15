import json
import pytest
from pydantic import ValidationError
from structured_output_agent.config import AgentConfig
from structured_output_agent.schemas import AgentResult, AttemptRecord, FailureDetail, SupportTicket


@pytest.fixture
def ticket_data():
    return dict(category="billing", priority="high", summary="Duplicate charge", requires_human=True)


def test_ticket_json_round_trip(ticket_data):
    ticket = SupportTicket.model_validate_json(json.dumps(ticket_data))
    assert ticket.model_dump() == ticket_data
    assert SupportTicket.model_validate_json(ticket.model_dump_json()) == ticket
    assert SupportTicket.model_json_schema()["additionalProperties"] is False


@pytest.mark.parametrize("change", [
    {"category": "unknown"}, {"priority": "urgent"}, {"requires_human": "true"},
    {"requires_human": 1}, {"summary": " "}, {"summary": "x" * 501},
    {"summary": 123}, {"unexpected": True},
])
def test_ticket_rejects_invalid_fields(ticket_data, change):
    with pytest.raises(ValidationError):
        SupportTicket.model_validate_json(json.dumps(ticket_data | change))


def test_ticket_requires_all_fields(ticket_data):
    for field in ticket_data:
        with pytest.raises(ValidationError):
            SupportTicket.model_validate({k: v for k, v in ticket_data.items() if k != field})


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in ("AGENT_MODE", "AGENT_MAX_ATTEMPTS", "AGENT_MAX_TOOL_CALLS", "OPENAI_API_KEY", "OPENAI_MODEL", "OBSERVABILITY_BACKEND", "LOGFIRE_TOKEN", "LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_TIMEOUT_SECONDS", "GEMINI_MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(name, raising=False)


def test_fake_defaults():
    config = AgentConfig(_env_file=None)
    assert config.mode == "fake"
    assert config.max_attempts == 3
    assert config.openai_api_key is None


def test_dotenv_and_environment_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AGENT_MODE=fake\nAGENT_MAX_ATTEMPTS=2\nOPENAI_MODEL=\n")
    monkeypatch.setenv("AGENT_MAX_ATTEMPTS", "4")
    config = AgentConfig(_env_file=env)
    assert config.max_attempts == 4
    assert config.openai_model is None


@pytest.mark.parametrize("values", [
    {"max_attempts": 0}, {"max_attempts": 11}, {"max_tool_calls": -1},
    {"mode": "live"}, {"mode": "live", "openai_api_key": "secret"},
    {"observability_backend": "logfire"}, {"mode": "other"},
])
def test_invalid_configuration(values):
    with pytest.raises(ValidationError):
        AgentConfig(_env_file=None, **values)


def test_live_settings_hide_secrets():
    config = AgentConfig(_env_file=None, mode="live", llm_provider="openai", openai_api_key="test-private-key", openai_model="test-model")
    assert "test-private-key" not in repr(config)
    assert "test-private-key" not in config.model_dump_json()


def test_result_recovery_round_trip(ticket_data):
    error = FailureDetail(code="schema_validation_error", message="Invalid priority", retryable=True)
    result = AgentResult[SupportTicket](
        run_id="run-1", status="success", data=SupportTicket(**ticket_data), duration_ms=2.0,
        attempts=(AttemptRecord(attempt=1, status="failure", duration_ms=1.0, error=error),
                  AttemptRecord(attempt=2, status="success", duration_ms=1.0)),
    )
    assert AgentResult[SupportTicket].model_validate_json(result.model_dump_json()) == result


def test_preflight_failure_needs_no_attempts():
    result = AgentResult[SupportTicket](run_id="run-1", status="failure", duration_ms=0.0,
        error=FailureDetail(code="configuration_error", message="Missing model"))
    assert result.attempts == ()


@pytest.mark.parametrize("status,error", [("success", True), ("failure", False)])
def test_attempt_status_error_consistency(status, error):
    with pytest.raises(ValidationError):
        AttemptRecord(attempt=1, status=status, duration_ms=0.0,
            error=FailureDetail(code="provider_error", message="Unavailable") if error else None)


def test_invalid_result_combinations(ticket_data):
    ticket = SupportTicket(**ticket_data)
    error = FailureDetail(code="retry_exhausted", message="Attempt limit reached")
    success = AttemptRecord(attempt=1, status="success", duration_ms=1.0)
    failed = AttemptRecord(attempt=1, status="failure", duration_ms=1.0, error=error)
    invalid = [
        dict(status="success", data=ticket),
        dict(status="success", data=ticket, error=error, attempts=(success,)),
        dict(status="failure", data=ticket, error=error, attempts=(failed,)),
        dict(status="failure", error=error, attempts=(success,)),
        dict(status="success", data=ticket, attempts=(success, success)),
        dict(status="failure"),
    ]
    for values in invalid:
        with pytest.raises(ValidationError):
            AgentResult[SupportTicket](run_id="run-1", duration_ms=1.0, **values)


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_invalid_duration(duration):
    with pytest.raises(ValidationError):
        AttemptRecord(attempt=1, status="success", duration_ms=duration)
