"""One JSON object per event, written to a caller stream or local file."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import TextIO
from .base import Observability


class LocalObservability(Observability):
    def __init__(self, *, stream: TextIO | None = None, path: str | Path | None = None):
        super().__init__()
        if stream is not None and path is not None:
            raise ValueError("Choose a stream or path, not both")
        self._owns_stream = path is not None
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._stream = destination.open("a", encoding="utf-8")
        else:
            self._stream = stream if stream is not None else sys.stderr
        self._lock = Lock()

    def _emit(self, event: str, attributes: dict) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **attributes}
        with self._lock:
            self._stream.write(json.dumps(record, allow_nan=False) + "\n")
            self._stream.flush()

    def flush(self) -> None:
        self._safe(self._stream.flush)

    def close(self) -> None:
        if self._owns_stream:
            self._safe(self._stream.close)
        else:
            self.flush()
