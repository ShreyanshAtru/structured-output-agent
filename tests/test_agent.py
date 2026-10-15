import json
from uuid import UUID

import pytest

from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients import FakeLLMClient, GenerationRequest, LLMClient
from structured_output_agent.exceptions import ProviderError, ResponseParseError, SchemaValidationError, ValidationIssue
from structured_output_agent.retry import RetryPolicy
from structured_output_agent.schemas import AgentResult, StrictModel, SupportTicket

VALID = json.dumps(dict(category="billing", priority="high", summary="Duplicate charge", requires_human=True))


def test_first_attempt_success():
    client = FakeLLMClient([VALID])
    result = StructuredOutputAgent(client).run("Help", SupportTicket)
    assert result.status == "success"
    assert isinstance(result.data, SupportTicket)
    assert len(result.attempts) == 1
    assert result.duration_ms >= result.attempts[0].duration_ms >= 0
    UUID(result.run_id)
    request = client.requests[0]
    assert request.attempt == 1 and request.repair_feedback is None
    assert json.loads(request.output_schema_json) == SupportTicket.model_json_schema()
    assert AgentResult[SupportTicket].model_validate_json(result.model_dump_json()) == result


def test_parse_then_schema_then_success():
    client = FakeLLMClient(['not-json', '{}', VALID])
    result = StructuredOutputAgent(client).run("Help", SupportTicket)
    assert result.status == "success"
    assert [a.status for a in result.attempts] == ["failure", "failure", "success"]
    assert [a.error.code for a in result.attempts[:2]] == ["json_parse_error", "schema_validation_error"]
    assert "JSON parsing" in client.requests[1].repair_feedback
    assert "missing" in client.requests[2].repair_feedback
    assert all(r.prompt == "Help" for r in client.requests)


@pytest.mark.parametrize("budget", [1, 2, 3, 10])
def test_exhaustion_obeys_budget(budget):
    client = FakeLLMClient(['{}'] * (budget + 1))
    result = StructuredOutputAgent(client, retry_policy=RetryPolicy(budget)).run("Help", SupportTicket)
    assert result.status == "failure" and result.data is None
    assert result.error.code == "retry_exhausted"
    assert result.error.retryable is False
    assert len(client.requests) == len(result.attempts) == budget
    assert result.attempts[-1].error.code == "schema_validation_error"


def test_provider_failure_is_terminal_and_sanitized():
    client = FakeLLMClient([ProviderError("private-secret"), VALID])
    result = StructuredOutputAgent(client).run("private-prompt", SupportTicket)
    assert result.error.code == "provider_error"
    assert len(client.requests) == 1
    assert "private" not in result.model_dump_json()


def test_exhausted_script_is_controlled_provider_failure():
    client = FakeLLMClient(['{}'])
    result = StructuredOutputAgent(client).run("Help", SupportTicket)
    assert result.error.code == "provider_error"
    assert len(result.attempts) == 2


def test_no_extra_calls_after_success():
    client = FakeLLMClient([VALID, ProviderError("must not be consumed")])
    assert StructuredOutputAgent(client).run("Help", SupportTicket).status == "success"
    assert len(client.requests) == 1


def test_run_state_does_not_leak():
    client = FakeLLMClient(['{}', VALID, VALID])
    agent = StructuredOutputAgent(client)
    first = agent.run("First", SupportTicket)
    second = agent.run("Second", SupportTicket)
    assert first.run_id != second.run_id
    assert len(second.attempts) == 1
    assert client.requests[-1].attempt == 1
    assert client.requests[-1].repair_feedback is None


def test_generic_second_schema():
    class Count(StrictModel):
        count: int
    result = StructuredOutputAgent(FakeLLMClient(['{"count":"2"}', '{"count":2}'])).run("Count", Count)
    assert result.data.count == 2
    assert len(result.attempts) == 2


def test_unexpected_programming_errors_propagate():
    class BrokenClient(LLMClient):
        def generate(self, request):
            raise RuntimeError("bug")
    with pytest.raises(RuntimeError, match="bug"):
        StructuredOutputAgent(BrokenClient()).run("Help", SupportTicket)


@pytest.mark.parametrize("prompt", ["", "  ", 123])
def test_invalid_input_makes_no_calls(prompt):
    client = FakeLLMClient([VALID])
    with pytest.raises((TypeError, ValueError)):
        StructuredOutputAgent(client).run(prompt, SupportTicket)
    assert client.requests == ()


def test_invalid_schema_makes_no_calls():
    client = FakeLLMClient([VALID])
    with pytest.raises(TypeError):
        StructuredOutputAgent(client).run("Help", dict)
    assert client.requests == ()


@pytest.mark.parametrize("budget", [0, 11, True, 1.5])
def test_policy_invalid_budget(budget):
    with pytest.raises(ValueError):
        RetryPolicy(budget)


def test_policy_retry_categories_and_boundary():
    policy = RetryPolicy(2)
    assert policy.should_retry(ResponseParseError("invalid"), attempt=1)
    assert policy.should_retry(SchemaValidationError(()), attempt=1)
    assert not policy.should_retry(ResponseParseError("invalid"), attempt=2)
    assert not policy.should_retry(ProviderError("fail"), attempt=1)
    with pytest.raises(ValueError):
        policy.should_retry(ResponseParseError("invalid"), attempt=0)
    with pytest.raises(TypeError):
        policy.build_feedback(ProviderError("fail"))


def test_feedback_does_not_copy_exception_message_and_is_bounded():
    policy = RetryPolicy()
    assert "private-secret" not in policy.build_feedback(ResponseParseError("private-secret"))
    error = SchemaValidationError(tuple(ValidationIssue(("summary",), "missing") for _ in range(100)))
    feedback = policy.build_feedback(error)
    assert feedback.count('"code"') == 10
    assert len(feedback) < 2200


@pytest.mark.parametrize("script", [[], "text", [None], [123]])
def test_fake_rejects_invalid_script(script):
    with pytest.raises((ValueError, TypeError)):
        FakeLLMClient(script)


def test_fake_requests_have_no_payload_repr():
    request = GenerationRequest(prompt="private-secret", output_schema_json="{}", attempt=1)
    assert "private-secret" not in repr(request)
