import asyncio
import json
from types import SimpleNamespace as NS

import pytest
pytest.importorskip("openai_codex")
from openai_codex.errors import CodexError

from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients.base import GenerationRequest
from structured_output_agent.clients.codex_client import CodexClient
from structured_output_agent.config import AgentConfig
from structured_output_agent.exceptions import ConfigurationError, ProviderError
from structured_output_agent.schemas import StrictModel, SupportTicket
from structured_output_agent.tools import TicketContextTool, ToolExecutor

TICKET=dict(category='billing',priority='medium',summary='Duplicate charge',requires_human=True)
FINAL=json.dumps(TICKET)
ACTION_FINAL=json.dumps(dict(kind='final',output=TICKET,tool=None))
ACTION_TOOL=json.dumps(dict(kind='tool',output=None,tool=dict(name='ticket_context',arguments_json='{"ticket_id":"TICKET-001"}')))


class Factory:
    def __init__(self, steps, *, auth='chatgpt', status='completed', delay=0):
        self.steps=list(steps)
        self.auth=auth
        self.status=status
        self.delay=delay
        self.configs=[]
        self.threads=[]
        self.requests=[]
        self.closed=0
    def __call__(self, *, config):
        self.configs.append(config)
        factory=self
        class SDK:
            async def __aenter__(self): return self
            async def __aexit__(self,*args): factory.closed+=1
            async def account(self):
                if factory.delay: await asyncio.sleep(factory.delay)
                return NS(account=NS(root=NS(type=factory.auth)) if factory.auth else None)
            async def thread_start(self,**kwargs):
                factory.threads.append(kwargs)
                return self
            async def run(self,prompt,**kwargs):
                factory.requests.append((json.loads(prompt),kwargs))
                step=factory.steps.pop(0)
                if isinstance(step,Exception): raise step
                return NS(status=factory.status,final_response=step)
        return SDK()


def client(tmp_path,factory,**kwargs):
    return CodexClient(work_dir=tmp_path/'scratch',sdk_factory=factory,**kwargs)


def test_no_tools_uses_target_schema_and_authenticated_ephemeral_runtime(tmp_path):
    factory=Factory([FINAL])
    result=StructuredOutputAgent(client(tmp_path,factory)).run('Help',SupportTicket)
    assert result.status=='success' and result.data.priority=='medium'
    assert factory.requests[0][1]['output_schema']==SupportTicket.model_json_schema()
    config=factory.configs[0]
    assert config.env['OPENAI_API_KEY']==''
    assert 'forced_login_method="chatgpt"' in config.config_overrides
    assert 'features.shell_tool=false' in config.config_overrides
    assert factory.threads[0]['ephemeral'] is True
    assert factory.threads[0]['sandbox'].value=='read-only'
    assert factory.threads[0]['approval_mode'].value=='deny_all'
    assert factory.closed==1
    assert not list((tmp_path/'scratch').iterdir())


def test_tool_action_is_executed_by_host_and_context_reaches_next_turn(tmp_path):
    factory=Factory([ACTION_TOOL,ACTION_FINAL])
    agent=StructuredOutputAgent(client(tmp_path,factory),tool_executor=ToolExecutor([TicketContextTool()]))
    result=agent.run('Look up TICKET-001',SupportTicket)
    assert result.status=='success' and result.tool_calls==1
    assert len(result.attempts)==1
    first,second=[call[0] for call in factory.requests]
    assert first['validated_tool_results']==[]
    assert second['validated_tool_results'][0]['output']['plan']=='pro'
    schema=factory.requests[0][1]['output_schema']
    assert schema['properties']['output']['anyOf'][0]==SupportTicket.model_json_schema()
    assert schema['properties']['tool']['anyOf'][0]['properties']['name']['enum']==['ticket_context']
    assert factory.closed==2 and len(factory.threads)==2


def test_local_repair_feedback_and_isolated_runs(tmp_path):
    factory=Factory(['{}',FINAL,FINAL])
    agent=StructuredOutputAgent(client(tmp_path,factory))
    assert len(agent.run('First',SupportTicket).attempts)==2
    assert factory.requests[1][0]['repair_feedback']
    assert agent.run('Second',SupportTicket).status=='success'
    assert factory.requests[2][0]['support_message']=='Second'
    assert factory.requests[2][0]['repair_feedback'] is None
    assert factory.requests[2][0]['validated_tool_results']==[]


@pytest.mark.parametrize('auth',['apiKey',None,'amazonBedrock'])
def test_non_chatgpt_auth_blocks_inference(tmp_path,auth):
    factory=Factory([FINAL],auth=auth)
    result=StructuredOutputAgent(client(tmp_path,factory)).run('Help',SupportTicket)
    assert result.error.code=='provider_error'
    assert factory.requests==[] and factory.threads==[]
    assert factory.closed==1


@pytest.mark.parametrize('step,status',[(CodexError('private-secret'),'completed'),(FINAL,'failed'),(None,'completed')])
def test_sdk_errors_are_sanitized_and_terminal(tmp_path,step,status):
    factory=Factory([step],status=status)
    result=StructuredOutputAgent(client(tmp_path,factory)).run('Help',SupportTicket)
    assert result.error.code=='provider_error'
    assert 'private-secret' not in result.model_dump_json()
    assert len(factory.requests)==1 and factory.closed==1


def test_timeout_closes_sdk_without_second_turn(tmp_path):
    factory=Factory([FINAL],delay=.1)
    result=StructuredOutputAgent(client(tmp_path,factory,timeout_seconds=.01)).run('Help',SupportTicket)
    assert result.error.code=='provider_error'
    assert factory.closed==1 and factory.requests==[]


def test_invalid_tool_action_data_is_rejected_by_executor(tmp_path):
    action=json.dumps(dict(kind='tool',output=None,tool=dict(name='ticket_context',arguments_json='{"ticket_id":123}')))
    factory=Factory([action])
    result=StructuredOutputAgent(client(tmp_path,factory),tool_executor=ToolExecutor([TicketContextTool()])).run('Help',SupportTicket)
    assert result.error.code=='tool_validation_error'
    assert len(factory.requests)==1


def test_tool_budget_stops_additional_sdk_requests(tmp_path):
    factory=Factory([ACTION_TOOL,ACTION_TOOL])
    result=StructuredOutputAgent(client(tmp_path,factory),tool_executor=ToolExecutor([TicketContextTool()]),max_tool_calls=1).run('Help',SupportTicket)
    assert result.error.code=='tool_limit_exceeded'
    assert len(factory.requests)==2


@pytest.mark.parametrize('action', ['[]','{}',json.dumps(dict(kind='final',output=TICKET,tool={'name':'x','arguments_json':'{}'}))])
def test_invalid_action_protocol_is_terminal(tmp_path,action):
    factory=Factory([action])
    result=StructuredOutputAgent(client(tmp_path,factory),tool_executor=ToolExecutor([TicketContextTool()])).run('Help',SupportTicket)
    assert result.error.code=='provider_error'


def test_action_schema_preserves_nested_definitions_without_mutating_request(tmp_path):
    class Nested(StrictModel):
        ticket: SupportTicket
    request=GenerationRequest('Help',json.dumps(Nested.model_json_schema()),1,tools=ToolExecutor([TicketContextTool()]).definitions)
    schema=CodexClient._action_schema(request)
    assert 'SupportTicket' in schema['$defs']
    assert '$defs' not in schema['properties']['output']['anyOf'][0]
    assert '$defs' in json.loads(request.output_schema_json)


@pytest.mark.parametrize('kwargs',[{'model':''},{'timeout_seconds':0},{'timeout_seconds':301},{'timeout_seconds':True},{'timeout_seconds':float('nan')}])
def test_invalid_constructor(tmp_path,kwargs):
    with pytest.raises(ConfigurationError): client(tmp_path,Factory([]),**kwargs)


def test_factory_and_config_do_not_require_api_key(monkeypatch):
    from structured_output_agent.clients import create_live_client
    config=AgentConfig(_env_file=None,mode='live',llm_provider='codex')
    sentinel=object()
    monkeypatch.setattr(CodexClient,'from_config',classmethod(lambda cls,config:sentinel))
    assert create_live_client(config) is sentinel


def test_routing_failure_is_actionable_without_copying_raw_sdk_message(tmp_path):
    factory=Factory([CodexError("workspace routing discovery failed private-secret")])
    result=StructuredOutputAgent(client(tmp_path,factory)).run('Help',SupportTicket)
    assert 'workspace routing' in result.error.message.lower()
    assert 'private-secret' not in result.model_dump_json()
