import io
import json

import logfire
from logfire.testing import TestExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
import pytest

from structured_output_agent.agent import StructuredOutputAgent
from structured_output_agent.clients import FakeLLMClient
from structured_output_agent.exceptions import ProviderError
from structured_output_agent.observability import LocalObservability, LogfireObservability, create_observability
from structured_output_agent.observability.base import Observability
from structured_output_agent.config import AgentConfig
from structured_output_agent.retry import RetryPolicy
from structured_output_agent.schemas import SupportTicket
from structured_output_agent.tools import ToolCall, ToolExecutor, TicketContextTool

VALID = '{"category":"billing","priority":"high","summary":"private-output","requires_human":true}'


def run(observer, script=None, **kwargs):
    return StructuredOutputAgent(FakeLLMClient(script or ['{}', VALID]), observability=observer, **kwargs).run('private-prompt', SupportTicket)


def records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_local_recovery_correlation_privacy_and_metrics():
    stream = io.StringIO()
    observer = LocalObservability(stream=stream)
    result = run(observer)
    data = records(stream)
    assert result.status == 'success'
    assert {e['run_id'] for e in data} == {result.run_id}
    assert any(e['event'] == 'retry.scheduled' for e in data)
    completed = next(e for e in data if e['event'] == 'run.completed')
    assert completed['recovered'] and completed['initial_validation_failure']
    assert not completed['first_attempt_success']
    assert completed['mode'] == 'fake'
    root = next(e for e in data if e['event'] == 'run.started')
    child = next(e for e in data if e['event'] == 'model.generate.started')
    assert child['parent_span_id'] == root['span_id']
    assert 'private' not in stream.getvalue()
    assert observer.telemetry_failures == 0


@pytest.mark.parametrize('script,code', [(['{}'], 'retry_exhausted'), ([ProviderError('private-secret')], 'provider_error')])
def test_terminal_result_and_run_span_are_marked_failed(script, code):
    stream = io.StringIO()
    run(LocalObservability(stream=stream), script, retry_policy=RetryPolicy(1))
    data = records(stream)
    assert data[-1]['event'] == 'run.finished'
    assert data[-1]['status'] == 'failure' and data[-1]['error_code'] == code
    assert 'private-secret' not in stream.getvalue()


def test_tool_stage_spans_and_private_unknown_tool():
    stream = io.StringIO()
    observer = LocalObservability(stream=stream)
    result = run(observer, [ToolCall('ticket_context','{"ticket_id":"TICKET-001"}'), VALID], tool_executor=ToolExecutor([TicketContextTool()]))
    assert result.tool_calls == 1
    events = {e['event'] for e in records(stream)}
    assert {'tool.input.validate.finished','tool.invoke.finished','tool.output.validate.finished'} <= events
    stream.seek(0); stream.truncate()
    run(observer, [ToolCall('private-tool','{}')], tool_executor=ToolExecutor([TicketContextTool()]))
    assert 'private-tool' not in stream.getvalue()


def test_invalid_tool_records_validation_failure():
    stream = io.StringIO()
    run(LocalObservability(stream=stream), [ToolCall('ticket_context','{}')], tool_executor=ToolExecutor([TicketContextTool()]))
    event = next(e for e in records(stream) if e['event'] == 'validation.failed')
    assert event['stage'] == 'input' and event['tool'] == 'ticket_context'


def test_file_factory_and_append(tmp_path):
    path = tmp_path/'logs'/'events.jsonl'
    config = AgentConfig(_env_file=None, observability_log_path=str(path))
    for _ in range(2):
        observer = create_observability(config)
        run(observer, [VALID])
        observer.close()
    data = [json.loads(line) for line in path.read_text().splitlines()]
    assert len({e['run_id'] for e in data}) == 2


def test_backend_errors_do_not_change_result():
    class Broken(Observability):
        def _emit(self, *args):
            raise RuntimeError('backend failed')
        def _start_span(self, *args):
            raise RuntimeError('backend failed')
    observer = Broken()
    assert run(observer).status == 'success'
    assert observer.telemetry_failures > 0


def test_unexpected_exception_logs_no_message_and_resets_context():
    stream = io.StringIO()
    observer = LocalObservability(stream=stream)
    class BrokenValidator:
        def validate(self, *args):
            raise RuntimeError('private-secret')
    with pytest.raises(RuntimeError):
        run(observer, [VALID], validator=BrokenValidator())
    assert 'private-secret' not in stream.getvalue()
    assert records(stream)[-1]['error_code'] == 'internal_error'
    stream.seek(0); stream.truncate()
    observer.event('outside', prompt='private-secret')
    assert 'run_id' not in records(stream)[0]
    assert 'private-secret' not in stream.getvalue()


def test_real_logfire_sdk_spans_metrics_and_no_exception_capture(tmp_path):
    exporter = TestExporter()
    reader = InMemoryMetricReader()
    sdk = logfire.configure(local=True, send_to_logfire=False, console=False,
        inspect_arguments=False, data_dir=tmp_path/'.logfire',
        additional_span_processors=[SimpleSpanProcessor(exporter)],
        metrics=logfire.MetricsOptions(additional_readers=[reader]))
    observer = LogfireObservability(sdk=sdk)
    result = run(observer)
    observer.flush()
    spans = exporter.exported_spans
    assert any(s.name == 'run' for s in spans)
    assert all('private' not in str(s.attributes) for s in spans)
    assert all(not any(e.name == 'exception' for e in s.events) for s in spans)
    assert observer.telemetry_failures == 0
    metric_data = reader.get_metrics_data()
    metrics = [m for resource in metric_data.resource_metrics for scope in resource.scope_metrics for m in scope.metrics]
    assert {'agent.runs','agent.validation_failures','agent.run.duration','agent.run.attempts'} <= {m.name for m in metrics}
    counter = next(m for m in metrics if m.name == 'agent.runs')
    assert sum(point.value for point in counter.data.data_points) == 1
    assert counter.data.data_points[0].attributes['recovered'] is True
    assert 'run_id' not in counter.data.data_points[0].attributes
    assert result.status == 'success'


def test_invalid_local_arguments_and_logfire_token():
    with pytest.raises(ValueError):
        LocalObservability(stream=io.StringIO(), path='unused')
    with pytest.raises(ValueError):
        LogfireObservability(token='')
