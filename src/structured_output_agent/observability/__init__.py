"""Observability adapters and configuration factory."""
from .base import Observability, NoOpObservability
from .local_adapter import LocalObservability
from .logfire_adapter import LogfireObservability


def create_observability(config):
    """Explicit startup: local file or configured remote Logfire project."""
    if config.observability_backend == "logfire":
        return LogfireObservability(token=config.logfire_token.get_secret_value())
    return LocalObservability(path=config.observability_log_path)


__all__ = ["Observability", "NoOpObservability", "LocalObservability",
           "LogfireObservability", "create_observability"]
