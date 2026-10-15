from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from structured_output_agent.playground import PlaygroundRunner

APP=Path(__file__).resolve().parents[1]/'ui/app.py'


@pytest.fixture
def app(tmp_path,monkeypatch):
    def init(self,project_root):
        self.project_root=tmp_path
    monkeypatch.setattr(PlaygroundRunner,'__init__',init)
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    monkeypatch.delenv('LOGFIRE_TOKEN',raising=False)
    return AppTest.from_file(str(APP),default_timeout=45).run()


def test_initial_ui_is_offline_without_runs(app):
    assert not app.exception
    assert app.selectbox(key='mode').value=='Offline demo'
    assert app.session_state['reports']==[]


def test_ui_recovery_and_reruns_do_not_repeat_requests(app):
    app.button(key='run').click().run()
    assert not app.exception
    reports=app.session_state['reports']
    assert reports[-1].result.status=='success'
    assert len(reports[-1].result.attempts)==2
    assert len(app.tabs)==4
    assert any('schema-repair retries' in item.value for item in app.markdown)
    run_id=reports[-1].result.run_id
    app.selectbox(key='timeline_filter').select('Failures & retries').run()
    app.run()
    assert len(app.session_state['reports'])==1
    assert app.session_state['reports'][0].result.run_id==run_id


def test_ui_exhaustion_tool_failure_and_clear(app):
    app.selectbox(key='scenario').select('Retry exhaustion').run()
    app.button(key='run').click().run()
    assert not app.exception
    assert app.session_state['reports'][-1].result.error.code=='retry_exhausted'
    app.selectbox(key='scenario').select('Invalid tool output').run()
    app.button(key='run').click().run()
    assert not app.exception
    assert app.session_state['reports'][-1].result.error.code=='tool_validation_error'
    assert len(app.session_state['reports'])==2
    app.button(key='clear_history').click().run()
    assert app.session_state['reports']==[]
    assert not app.exception


def test_live_missing_key_and_logfire_missing_token_display_safe_errors(app):
    app.selectbox(key='mode').select('Live Gemini').run()
    assert not app.checkbox(key='capture_live').value
    assert app.session_state['reports']==[]
    app.button(key='run').click().run()
    assert not app.exception
    assert app.error and app.session_state['reports']==[]
    app.selectbox(key='mode').select('Offline demo').run()
    app.selectbox(key='backend').select('Logfire').run()
    app.button(key='run').click().run()
    assert app.error and not app.exception


def test_custom_json_attempt_limit(app):
    app.selectbox(key='scenario').select('Custom JSON → recovery').run()
    app.text_area(key='custom_json').input('{"requires_human":"true"}').run()
    app.slider(key='max_attempts').set_value(1).run()
    app.button(key='run').click().run()
    assert not app.exception
    report=app.session_state['reports'][-1]
    assert report.result.status=='failure'
    assert len(report.calls)==1
    assert 'requires_human' in report.calls[0]['raw']


def test_live_codex_ui_runs_through_existing_services_without_network(app,monkeypatch):
    from structured_output_agent.clients.base import LLMClient
    import structured_output_agent.playground as playground
    from structured_output_agent.playground import VALID
    captures=[]
    class Stub(LLMClient):
        execution_mode='live'
        provider_name='codex'
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def generate(self,request):
            captures.append(request)
            return VALID
    monkeypatch.setattr(playground,'create_live_client',lambda config:Stub())
    app.selectbox(key='mode').select('Live Codex').run()
    assert not app.exception
    assert app.text_input(key='codex_model').value==''
    assert not app.checkbox(key='capture_live').value
    assert captures==[]
    app.button(key='run').click().run()
    assert not app.exception
    report=app.session_state['reports'][-1]
    assert report.provider=='codex' and report.scenario=='Live Codex'
    assert report.result.status=='success'
    assert all(e['provider']=='codex' for e in report.events)
    app.selectbox(key='metrics_group').select('Live Codex').run()
    app.run()
    assert len(captures)==1
