import json
import pytest
from pydantic import Field
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients import FakeLLMClient
from structured_output_agent.exceptions import ToolExecutionError, ToolValidationError
from structured_output_agent.retry import RetryPolicy
from structured_output_agent.schemas import StrictModel, SupportTicket
from structured_output_agent.tools import Tool, ToolCall, ToolExecutor, TicketContextTool

FINAL = '{"category":"billing","priority":"high","summary":"Duplicate charge","requires_human":true}'
CALL = ToolCall('ticket_context', '{"ticket_id":"TICKET-001"}')


class Number(StrictModel):
    count: int = Field(ge=0)


class CountingTool(Tool[Number, Number]):
    name = 'count'
    description = 'Synthetic counter'
    input_model = Number
    output_model = Number

    def __init__(self, output=None, failure=False):
        self.calls = 0
        self.output = output
        self.failure = failure

    def execute(self, arguments):
        self.calls += 1
        if self.failure:
            raise RuntimeError('private-secret')
        return self.output if self.output is not None else Number(count=arguments.count)


def test_valid_execution_and_definitions():
    executor = ToolExecutor([TicketContextTool()])
    result = executor.execute(CALL)
    assert json.loads(result.output_json)['duplicate_charge_reported'] is True
    definition = executor.definitions[0]
    assert definition.name == result.name == 'ticket_context'
    assert json.loads(definition.input_schema_json)['additionalProperties'] is False
    assert 'previous_tickets' in json.loads(definition.output_schema_json)['properties']


@pytest.mark.parametrize('arguments', ['not-json', '{}', '{"count":"1"}', '{"count":-1}', '{"count":1,"extra":2}', '{"count":1,"count":2}', None])
def test_invalid_input_blocks_execution(arguments):
    tool = CountingTool()
    with pytest.raises(ToolValidationError) as caught:
        ToolExecutor([tool]).execute(ToolCall('count', arguments))
    assert tool.calls == 0
    assert caught.value.__context__ is None


@pytest.mark.parametrize('output', [{'count':'1'}, {'count':-1}, {'count':1,'extra':True}, [], Number.model_construct(count='bad')])
def test_invalid_output_is_rejected(output):
    tool = CountingTool(output=output)
    with pytest.raises(ToolValidationError) as caught:
        ToolExecutor([tool]).execute(ToolCall('count', '{"count":1}'))
    assert tool.calls == 1
    assert caught.value.__context__ is None


def test_unknown_tool_and_runtime_errors_are_sanitized():
    with pytest.raises(ToolExecutionError) as caught:
        ToolExecutor([]).execute(ToolCall('private-secret', '{}'))
    assert 'private-secret' not in str(caught.value)
    tool = CountingTool(failure=True)
    with pytest.raises(ToolExecutionError) as caught:
        ToolExecutor([tool]).execute(ToolCall('count', '{"count":1}'))
    assert 'private-secret' not in str(caught.value)
    assert caught.value.__context__ is None
    assert tool.calls == 1


def test_registry_rejects_duplicates_and_bad_contracts():
    with pytest.raises(ValueError, match='duplicate'):
        ToolExecutor([CountingTool(), CountingTool()])
    with pytest.raises(TypeError):
        ToolExecutor([object()])
    bad = CountingTool()
    bad.input_model = dict
    with pytest.raises(TypeError):
        ToolExecutor([bad])
    bad = CountingTool()
    bad.name = 'Bad Name'
    with pytest.raises(ValueError):
        ToolExecutor([bad])


def test_tool_result_reaches_next_model_request():
    client = FakeLLMClient([CALL, FINAL])
    result = StructuredOutputAgent(client, tool_executor=ToolExecutor([TicketContextTool()])).run('Help', SupportTicket)
    assert result.status == 'success' and result.tool_calls == 1
    assert len(result.attempts) == 1
    assert len(client.requests) == 2
    assert client.requests[0].tools[0].name == 'ticket_context'
    assert client.requests[0].tool_results == ()
    assert json.loads(client.requests[1].tool_results[0].output_json)['plan'] == 'pro'


def test_bad_tool_output_never_reaches_model_and_is_not_retried():
    tool = CountingTool(output={'count':'private-secret'})
    client = FakeLLMClient([ToolCall('count', '{"count":1}'), FINAL])
    result = StructuredOutputAgent(client, tool_executor=ToolExecutor([tool])).run('Help', SupportTicket)
    assert result.error.code == 'tool_validation_error'
    assert len(client.requests) == tool.calls == 1
    assert 'private-secret' not in result.model_dump_json()


@pytest.mark.parametrize('limit', [0, 1, 3])
def test_tool_limit_stops_loops_without_extra_execution(limit):
    tool = CountingTool()
    call = ToolCall('count', '{"count":1}')
    client = FakeLLMClient([call] * (limit + 1))
    result = StructuredOutputAgent(client, tool_executor=ToolExecutor([tool]), max_tool_calls=limit).run('Help', SupportTicket)
    assert result.error.code == 'tool_limit_exceeded'
    assert tool.calls == result.tool_calls == limit
    assert len(client.requests) == limit + 1


def test_schema_retry_reuses_context_without_replaying_tool():
    tool = CountingTool()
    client = FakeLLMClient([ToolCall('count', '{"count":1}'), '{}', FINAL])
    result = StructuredOutputAgent(client, tool_executor=ToolExecutor([tool]), retry_policy=RetryPolicy(2)).run('Help', SupportTicket)
    assert result.status == 'success'
    assert tool.calls == 1
    assert len(result.attempts) == 2
    assert client.requests[-1].tool_results == client.requests[-2].tool_results
    assert client.requests[-1].repair_feedback is not None


def test_missing_executor_is_terminal():
    result = StructuredOutputAgent(FakeLLMClient([CALL, FINAL])).run('Help', SupportTicket)
    assert result.error.code == 'tool_execution_error'


def test_tool_budget_is_per_run():
    client = FakeLLMClient([CALL, FINAL, CALL, FINAL])
    agent = StructuredOutputAgent(client, tool_executor=ToolExecutor([TicketContextTool()]), max_tool_calls=1)
    assert agent.run('First', SupportTicket).tool_calls == 1
    assert agent.run('Second', SupportTicket).tool_calls == 1
    assert client.requests[2].tool_results == ()


@pytest.mark.parametrize('limit', [-1, 21, True, 1.5])
def test_invalid_tool_budget(limit):
    with pytest.raises(ValueError):
        StructuredOutputAgent(FakeLLMClient([FINAL]), max_tool_calls=limit)
