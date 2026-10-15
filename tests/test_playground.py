import json
import pytest
from structured_output_agent.playground import PlaygroundError, PlaygroundRunner, SCENARIOS, session_metrics, VALID
from structured_output_agent.observability.memory_adapter import MemoryObservability
from structured_output_agent.observability.base import Observability


@pytest.mark.parametrize('scenario,status,error', [
    (SCENARIOS[0],'success',None),(SCENARIOS[1],'success',None),(SCENARIOS[2],'success',None),
    (SCENARIOS[3],'failure','retry_exhausted'),(SCENARIOS[4],'success',None),
    (SCENARIOS[5],'failure','tool_validation_error'),(SCENARIOS[6],'failure','tool_validation_error'),
    (SCENARIOS[7],'failure','tool_limit_exceeded'),(SCENARIOS[8],'failure','provider_error'),
    (SCENARIOS[9],'success',None),
])
def test_scenarios(tmp_path,scenario,status,error):
    report = PlaygroundRunner(tmp_path).run(prompt='private-prompt', scenario=scenario)
    assert report.result.status == status
    assert (report.result.error.code if report.result.error else None) == error
    assert report.telemetry_failures == 0
    assert report.mode == 'fake'
    assert all('raw' not in call for call in report.calls)
    assert 'private-prompt' not in json.dumps(report.events)
    assert any(e['event'] == 'run.completed' and e['mode']=='fake' for e in report.events)
    assert (tmp_path/'logs/events.jsonl').exists()
    if scenario == 'Invalid tool output':
        assert any(e['event']=='validation.issue' and e.get('field')=='previous_tickets' for e in report.events)
    if scenario == 'Malformed JSON → recovery':
        assert any(e['event']=='validation.failed' and e.get('line')==1 for e in report.events)


def test_raw_is_session_only_and_diagnostics_match_attempts(tmp_path):
    report = PlaygroundRunner(tmp_path).run(prompt='private-prompt',scenario='Custom JSON → recovery',
        custom_json='{"summary":"private-output","requires_human":"yes"}',capture_raw=True)
    assert report.result.status == 'success'
    assert 'private-output' in report.calls[0]['raw']
    assert report.calls[1]['repair_feedback']
    assert 'private-output' not in json.dumps(report.events)
    assert 'private-output' not in (tmp_path/'logs/events.jsonl').read_text()
    assert any(e['event']=='validation.issue' and e['field']=='requires_human' for e in report.events)


def test_limits_and_disabled_tools(tmp_path):
    runner=PlaygroundRunner(tmp_path)
    assert runner.run(prompt='Help',max_attempts=1).result.error.code=='retry_exhausted'
    assert runner.run(prompt='Help',scenario='Validated tool call',max_tool_calls=0).result.error.code=='tool_limit_exceeded'
    assert runner.run(prompt='Help',scenario='Validated tool call',enable_tools=False).result.error.code=='tool_execution_error'


def test_session_metrics_separate_modes_and_empty_denominators(tmp_path):
    runner=PlaygroundRunner(tmp_path)
    reports=[runner.run(prompt='Help',scenario=name) for name in SCENARIOS[:4]]
    result=session_metrics(reports,'fake')
    assert result['runs']==4
    assert result['first_attempt_success_rate']==.25
    assert result['final_failure_rate']==.25
    assert result['recovery_rate']==pytest.approx(2/3)
    assert session_metrics(reports,'live')['recovery_rate'] is None


@pytest.mark.parametrize('settings', [{'prompt':''},{'prompt':'Help','max_attempts':0},
    {'prompt':'Help','backend':'logfire'},{'prompt':'Help','mode':'live'},
    {'prompt':'Help','scenario':'unknown'}])
def test_startup_errors_are_safe(tmp_path,monkeypatch,settings):
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    monkeypatch.delenv('LOGFIRE_TOKEN',raising=False)
    with pytest.raises(PlaygroundError) as caught:
        PlaygroundRunner(tmp_path).run(**settings)
    assert caught.value.__context__ is None


def test_live_run_uses_factory_without_network(tmp_path,monkeypatch):
    from structured_output_agent.clients.base import LLMClient
    import structured_output_agent.playground as module
    (tmp_path/'.env').write_text('GEMINI_API_KEY=private-test-key\n')
    class Stub(LLMClient):
        execution_mode='live'
        provider_name='gemini'
        def generate(self,request):
            return VALID
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
    monkeypatch.setattr(module,'create_live_client',lambda config: Stub())
    report=PlaygroundRunner(tmp_path).run(prompt='Help',mode='live')
    assert report.result.status=='success'
    assert all(e['mode']=='live' and e['provider']=='gemini' for e in report.events)
    assert 'private-test-key' not in json.dumps(report.events)
    assert report.raw_captured is False


def test_raw_capture_bound_and_event_bound(tmp_path):
    report=PlaygroundRunner(tmp_path).run(prompt='Help',scenario='Custom JSON → recovery',capture_raw=True,custom_json='x'*6000)
    assert len(report.calls[0]['raw'])==5000 and report.calls[0]['truncated']
    memory=MemoryObservability(max_events=1)
    memory.event('first'); memory.event('second')
    assert len(memory.events)==1 and memory.dropped_events==1


def test_memory_backend_failure_does_not_change_agent_result(tmp_path,monkeypatch):
    import structured_output_agent.playground as module
    class Broken(Observability):
        def _emit(self,*args):
            raise RuntimeError('private-failure')
    monkeypatch.setattr(module,'create_observability',lambda config: Broken())
    report=PlaygroundRunner(tmp_path).run(prompt='Help')
    assert report.result.status=='success'
    assert report.telemetry_failures>0
    assert report.events


def test_codex_service_uses_env_model_and_separate_provider_metrics(tmp_path,monkeypatch):
    from structured_output_agent.clients.base import LLMClient
    import structured_output_agent.playground as module
    configs=[]
    class Stub(LLMClient):
        execution_mode='live'
        provider_name='codex'
        def generate(self,request): return VALID
        def __enter__(self): return self
        def __exit__(self,*args): pass
    def create(config):
        configs.append(config)
        return Stub()
    (tmp_path/'.env').write_text('CODEX_MODEL=test-codex-model\n')
    monkeypatch.setattr(module,'create_live_client',create)
    report=PlaygroundRunner(tmp_path).run(prompt='Help',mode='live',provider='codex',model='')
    assert configs[0].codex_model=='test-codex-model'
    assert report.model=='test-codex-model'
    assert session_metrics([report],'live',provider='codex')['runs']==1
    assert session_metrics([report],'live',provider='gemini')['runs']==0
