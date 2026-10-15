import json
import base64
import httpx
import pytest
from google import genai
from google.genai import types
from pydantic import ValidationError

from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients.gemini_client import GeminiClient
from structured_output_agent.config import AgentConfig
from structured_output_agent.exceptions import ConfigurationError, ProviderError
from structured_output_agent.observability import LocalObservability
from structured_output_agent.schemas import SupportTicket
from structured_output_agent.tools import TicketContextTool, ToolExecutor

VALID = '{"category":"billing","priority":"high","summary":"Duplicate charge","requires_human":true}'


def response(parts=None, finish='STOP'):
    return {'candidates':[{'content':{'role':'model','parts':parts if parts is not None else [{'text':VALID}]}, 'finishReason':finish}]}


def client_for(script):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        step = script[len(requests)-1]
        if isinstance(step, Exception):
            raise step
        if isinstance(step, int):
            return httpx.Response(step, json={'error':{'code':step,'status':'UNKNOWN','message':'private-secret'}})
        return httpx.Response(200, json=step)
    http = httpx.Client(transport=httpx.MockTransport(handle))
    sdk = genai.Client(api_key='test-not-real', vertexai=False, http_options=types.HttpOptions(httpx_client=http))
    return GeminiClient(model='gemini-test', sdk_client=sdk), requests, sdk


def test_structured_output_sdk_serialization():
    client, requests, sdk = client_for([response()])
    try:
        result = StructuredOutputAgent(client).run('Help', SupportTicket)
        assert result.status == 'success'
        config = requests[0]['generationConfig']
        assert config['responseMimeType'] == 'application/json'
        assert config['responseJsonSchema'] == SupportTicket.model_json_schema()
        assert config['maxOutputTokens'] == 4096
    finally:
        sdk.close()


def test_tool_replay_preserves_thought_signature():
    parts = [{'functionCall':{'name':'ticket_context','args':{'ticket_id':'TICKET-001'},'id':'call-1'},'thoughtSignature':'/wE='}]
    client, requests, sdk = client_for([response(parts), response()])
    try:
        result = StructuredOutputAgent(client, tool_executor=ToolExecutor([TicketContextTool()])).run('Help', SupportTicket)
        assert result.status == 'success' and result.tool_calls == 1
        contents = requests[1]['contents']
        assert contents[1]['role'] == 'model'
        assert base64.urlsafe_b64decode(contents[1]['parts'][0]['thoughtSignature']) == b'\xff\x01'
        tool_result = contents[2]['parts'][0]['functionResponse']
        assert tool_result['id'] == 'call-1'
        assert tool_result['response']['plan'] == 'pro'
        assert requests[0]['tools'][0]['functionDeclarations'][0]['name'] == 'ticket_context'
    finally:
        sdk.close()


@pytest.mark.parametrize('step', [400, 401, 403, 429, 500, httpx.ConnectError('private-secret'), httpx.ReadTimeout('private-secret')])
def test_provider_failures_are_sanitized_and_not_retried(step):
    client, requests, sdk = client_for([step])
    try:
        result = StructuredOutputAgent(client).run('Help', SupportTicket)
        assert result.status == 'failure'
        assert result.error.code == 'provider_error'
        assert len(requests) == 1
        assert 'private-secret' not in result.model_dump_json()
    finally:
        sdk.close()


@pytest.mark.parametrize('payload', [
    {}, response([], 'STOP'), response(finish='MAX_TOKENS'), response(finish='SAFETY'),
    response([{'text':'private-thought','thought':True}]),
    response([{'functionCall':{'name':'a','args':{}}},{'functionCall':{'name':'b','args':{}}}]),
])
def test_incomplete_empty_or_parallel_responses_are_terminal(payload):
    client, requests, sdk = client_for([payload])
    try:
        result = StructuredOutputAgent(client).run('Help', SupportTicket)
        assert result.error.code == 'provider_error'
        assert len(requests) == 1
        assert 'private-thought' not in result.model_dump_json()
    finally:
        sdk.close()


def test_local_repair_and_run_isolation():
    client, requests, sdk = client_for([response([{'text':'{}'}]), response(), response()])
    try:
        agent = StructuredOutputAgent(client)
        result = agent.run('First', SupportTicket)
        assert len(result.attempts) == 2 and result.status == 'success'
        assert 'missing' in requests[1]['contents'][-1]['parts'][0]['text']
        agent.run('Second', SupportTicket)
        assert requests[2]['contents'] == [{'parts':[{'text':'Second'}],'role':'user'}]
    finally:
        sdk.close()


def test_gemini_is_live_in_telemetry():
    import io
    stream = io.StringIO()
    client, _, sdk = client_for([response()])
    try:
        StructuredOutputAgent(client, observability=LocalObservability(stream=stream)).run('Help', SupportTicket)
        assert all(record['mode'] == 'live' for record in map(json.loads, stream.getvalue().splitlines()))
    finally:
        sdk.close()


@pytest.mark.parametrize('kwargs', [dict(model=''),dict(model='gemini'),dict(model='gemini',api_key=' '),
    dict(model='gemini',api_key='test',timeout_seconds=0),dict(model='gemini',api_key='test',max_output_tokens=0)])
def test_configuration_errors(kwargs):
    with pytest.raises(ConfigurationError):
        GeminiClient(**kwargs)


def test_gemini_configuration_credentials_and_secret_masking():
    with pytest.raises(ValidationError):
        AgentConfig(_env_file=None,mode='live',llm_provider='gemini')
    config = AgentConfig(_env_file=None,mode='live',gemini_api_key='private-secret')
    assert config.llm_provider == 'gemini'
    assert config.gemini_model == 'gemini-3.8-flash'
    assert 'private-secret' not in repr(config)
    assert 'private-secret' not in config.model_dump_json()
    with pytest.raises(ConfigurationError):
        GeminiClient.from_config(AgentConfig(_env_file=None))


def test_live_factory_selects_configured_provider(monkeypatch):
    from structured_output_agent.clients import create_live_client
    from structured_output_agent.clients.openai_client import OpenAIClient
    gemini = object()
    openai = object()
    monkeypatch.setattr(GeminiClient, 'from_config', classmethod(lambda cls, config: gemini))
    monkeypatch.setattr(OpenAIClient, 'from_config', classmethod(lambda cls, config: openai))
    assert create_live_client(AgentConfig(_env_file=None,mode='live',gemini_api_key='test')) is gemini
    assert create_live_client(AgentConfig(_env_file=None,mode='live',llm_provider='openai',openai_api_key='test',openai_model='test')) is openai
    with pytest.raises(ConfigurationError):
        create_live_client(AgentConfig(_env_file=None))
