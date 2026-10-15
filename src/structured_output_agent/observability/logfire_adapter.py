"""Explicit manual Logfire spans; no payload/exception auto-instrumentation."""
from .base import Observability


class LogfireObservability(Observability):
    def __init__(self, *, token: str | None = None, sdk=None):
        super().__init__()
        if sdk is None:
            if not isinstance(token, str) or not token.strip():
                raise ValueError("Logfire write token is required")
            import logfire
            sdk = logfire.configure(local=True, token=token, send_to_logfire=True,
                                    service_name="structured-output-agent", console=False,
                                    inspect_arguments=False)
        self._sdk = sdk
        self._runs = sdk.metric_counter("agent.runs")
        self._failures = sdk.metric_counter("agent.validation_failures")
        self._duration = sdk.metric_histogram("agent.run.duration", unit="ms")
        self._attempts = sdk.metric_histogram("agent.run.attempts")

    def _emit(self, event: str, attributes: dict) -> None:
        self._sdk.info("agent telemetry", event=event, **attributes)
        # Run IDs belong on traces, never high-cardinality metric dimensions.
        labels = {key: attributes[key] for key in ("mode", "provider", "schema", "status", "error_code", "stage", "tool") if key in attributes}
        if event == "run.completed":
            labels.update({key: attributes[key] for key in
                           ("first_attempt_success", "initial_validation_failure", "recovered")})
            self._runs.add(1, attributes=labels)
            self._duration.record(attributes["duration_ms"], attributes=labels)
            self._attempts.record(attributes["attempts"], attributes=labels)
        if event == "validation.failed":
            self._failures.add(1, attributes=labels)

    def _start_span(self, name: str, attributes: dict):
        handle = self._sdk.span(name, **attributes)
        handle.__enter__()
        return handle

    def _end_span(self, handle, attributes: dict) -> None:
        try:
            for key, value in attributes.items():
                handle.set_attribute(key, value)
        finally:
            # Never hand raw exceptions or traceback locals to Logfire.
            handle.__exit__(None, None, None)

    def flush(self) -> None:
        self._safe(self._sdk.force_flush)
