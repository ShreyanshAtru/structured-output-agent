import json
import httpx
import pytest
from openai import OpenAI
from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients.base import GenerationRequest
from structured_output_agent.clients.openai_client import OpenAIClient
from structured_output_agent.config import AgentConfig
from structured_output_agent.exceptions import ConfigurationError, ProviderError
from structured_output_agent.schemas import SupportTicket
from structured_output_agent.tools import TicketContextTool, ToolExecutor

VALID = '{"category":"billing","priority":"high","summary":"Duplicate charge","requires_human":true}'


def response(output=None, status='completed'):
    return dict(id='resp_test', object='response', created_at=1, status=status,
                model='test-model', output=output if output is not None else [
                    dict(type='message', id='msg_test', role='assistant', status='completed',
                         content=[dict(type='output_text', text=VALID, annotations=[])])])


def make_client(script):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        step = script[len(requests)-1]
        if isinstance(step, Exception):
            raise step
        if isinstance(step, int):
            return httpx.Response(step, json={'error':{'message':'private-secret','type':'api_error'}})
        return httpx.Response(200, json=step)
    sdk = OpenAI(api_key='test-not-real', http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    return OpenAIClient(model='test-model', sdk_client=sdk), requests, sdk


def test_real_sdk_serializes_structured_request():
    client, requests, sdk = make_client([response()])
    try:
        result = StructuredOutputAgent(client).run('Help', SupportTicket)
        assert result.status == 'success'
        body = requests[0]
        assert body['store'] is False
        assert body['parallel_tool_calls'] is False
        assert body['text']['format']['strict'] is True
        assert body['text']['format']['schema']['additionalProperties'] is False
        assert body['max_output_tokens'] == 2000
    finally:
        sdk.close()


def test_tool_continuation_preserves_call_id_and_reasoning():
    items = [dict(type='reasoning', id='rs_test', summary=[], encrypted_content='encrypted-test'),
             dict(type='function_call', id='fc_test', call_id='call_1', name='ticket_context',
                  arguments='{"ticket_id":"TICKET-001"}', status='completed')]
    client, requests, sdk = make_client([response(items), response()])
    try:
        agent = StructuredOutputAgent(client, tool_executor=ToolExecutor([TicketContextTool()]))
        assert agent.run('Help', SupportTicket).status == 'success'
        body = requests[1]
        assert body['input'][1]['type'] == 'reasoning'
        assert body['input'][1]['encrypted_content'] == 'encrypted-test'
        assert body['input'][2]['call_id'] == 'call_1'
        assert body['input'][3]['type'] == 'function_call_output'
        assert body['input'][3]['call_id'] == 'call_1'
        assert json.loads(body['input'][3]['output'])['plan'] == 'pro'
        assert requests[0]['tools'][0]['strict'] is True
    finally:
        sdk.close()


@pytest.mark.parametrize('step', [401, 429, 500, httpx.ConnectError('private-secret'), httpx.ReadTimeout('private-secret')])
def test_sdk_errors_are_terminal_sanitized_and_not_retried(step):
    client, requests, sdk = make_client([step])
    try:
        result = StructuredOutputAgent(client).run('Help', SupportTicket)
        assert result.error.code == 'provider_error'
        assert len(requests) == 1
        assert 'private-secret' not in result.model_dump_json()
    finally:
        sdk.close()


@pytest.mark.parametrize('payload', [response([], status='incomplete'), response([]),
    response([dict(type='message', id='msg', role='assistant', status='completed',
                   content=[dict(type='refusal', refusal='private-secret')])]),
    response([dict(type='function_call', id='a', call_id='a', name='x', arguments='{}', status='completed'),
              dict(type='function_call', id='b', call_id='b', name='x', arguments='{}', status='completed')])])
def test_refusal_empty_incomplete_parallel_are_provider_failures(payload):
    client, _, sdk = make_client([payload])
    try:
        with pytest.raises(ProviderError) as caught:
            client.generate(GenerationRequest('Help', json.dumps(SupportTicket.model_json_schema()), 1))
        assert caught.value.__context__ is None
        assert 'private-secret' not in str(caught.value)
    finally:
        sdk.close()


def test_local_validation_remains_active_and_feedback_reaches_api():
    bad = response()
    bad['output'][0]['content'][0]['text'] = '{}'
    client, requests, sdk = make_client([bad, response()])
    try:
        result = StructuredOutputAgent(client).run('Help', SupportTicket)
        assert result.status == 'success' and len(result.attempts) == 2
        assert 'missing' in requests[1]['input'][-1]['content']
    finally:
        sdk.close()


def test_adapter_does_not_retain_history_between_runs():
    client, requests, sdk = make_client([response(), response()])
    try:
        agent = StructuredOutputAgent(client)
        agent.run('First', SupportTicket)
        agent.run('Second', SupportTicket)
        assert requests[1]['input'] == [{'role':'user','content':'Second'}]
    finally:
        sdk.close()


@pytest.mark.parametrize('kwargs', [dict(model=''), dict(model='test'), dict(model='test',api_key=' '),
    dict(model='test',api_key='key',timeout_seconds=0), dict(model='test',api_key='key',max_output_tokens=0)])
def test_constructor_validation(kwargs):
    with pytest.raises(ConfigurationError):
        OpenAIClient(**kwargs)


def test_fake_config_cannot_create_live_adapter():
    with pytest.raises(ConfigurationError):
        OpenAIClient.from_config(AgentConfig(_env_file=None))
