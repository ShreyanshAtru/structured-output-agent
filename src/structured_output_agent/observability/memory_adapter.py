"""Bounded session telemetry, optionally mirrored to an existing backend."""
from datetime import datetime, timezone
from .base import Observability


class MemoryObservability(Observability):
    def __init__(self, backend: Observability | None = None, *, max_events: int = 2000):
        super().__init__()
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be positive")
        self.backend = backend
        self.max_events = max_events
        self.events: list[dict] = []
        self.dropped_events = 0

    def _emit(self, event: str, attributes: dict) -> None:
        if len(self.events) < self.max_events:
            self.events.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                                "event": event, **attributes})
        else:
            self.dropped_events += 1
        if self.backend is not None:
            self.backend.event(event, **attributes)

    def _start_span(self, name: str, attributes: dict):
        if self.backend is not None:
            return self.backend._safe(self.backend._start_span, name, attributes)
        return None

    def _end_span(self, handle, attributes: dict) -> None:
        if self.backend is not None:
            self.backend._safe(self.backend._end_span, handle, attributes)

    def flush(self) -> None:
        if self.backend is not None:
            self._safe(self.backend.flush)

    def close(self) -> None:
        if self.backend is not None:
            self._safe(self.backend.close)
